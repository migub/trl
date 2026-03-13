# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted for single-GPU multi-turn negotiation training.
# Based on GRPOTrainer from TRL, with multi-turn dialogue generation
# using LoRA enable/disable for Agent vs Opponent.

import os
import textwrap
import warnings
import logging
import copy
import gc
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional, Union, Tuple
from collections.abc import Callable

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available, is_rich_available

from ..chat_template_utils import add_response_schema, get_training_chat_template, parse_response
from ..data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from ..extras.profiling import profiling_context, profiling_decorator
from ..import_utils import is_jmespath_available, is_liger_kernel_available
from ..models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection, disable_gradient_checkpointing
from .base_trainer import _BaseTrainer
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
    use_adapter,
)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class MultiTurnGRPOTrainer(_BaseTrainer):
    """
    GRPO Trainer adapted for multi-turn negotiation on a single GPU.

    Key differences from standard GRPOTrainer:
    - No vLLM: uses model.generate() directly for both Agent and Opponent
    - Agent uses LoRA adapters (enabled), Opponent uses base model (adapters disabled)
    - Multi-turn dialogue generation in _play_negotiation()
    - Assistant mask ensures only agent tokens contribute to the loss

    Additional Args (beyond GRPOConfig):
        max_negotiation_rounds: Number of rounds per agent (default 5)
        max_tokens_per_turn: Max tokens per negotiation message (default 200)
    """

    _tag_names = ["trl", "grpo", "multiturn"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_negotiation_rounds: int = 5,
        max_tokens_per_turn: int = 200,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Force vLLM off - we do local generation
        args.use_vllm = False

        # ---- Model setup ----
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # ---- Reference model ----
        # With LoRA: no separate ref model needed (use disable_adapter)
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_peft_model(model):
            self.ref_model = None
        else:
            # Full fine-tuning: need a separate reference model
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)

        # ---- Processing class ----
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # ---- Reward functions ----
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                self.reward_func_names.append(reward_func.config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_func.__name__)

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match "
                    f"number of reward functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        for i, (rpc, rf) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(rf, PreTrainedModel):
                if rpc is None:
                    rpc = AutoTokenizer.from_pretrained(rf.config._name_or_path)
                if rpc.pad_token_id is None:
                    rpc.pad_token = rpc.eos_token
                rf.config.pad_token_id = rpc.pad_token_id
                reward_processing_classes[i] = rpc
        self.reward_processing_classes = reward_processing_classes

        # ---- Training arguments ----
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations  # G in GRPO
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty

        # Multi-turn config
        self.max_negotiation_rounds = max_negotiation_rounds
        self.max_tokens_per_turn = max_tokens_per_turn

        # Multi-step / clipping
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Scale rewards config (new TRL uses string: "group", "batch", "none")
        self.scale_rewards = args.scale_rewards if hasattr(args, 'scale_rewards') else "group"

        self._step = 0
        self._buffered_inputs = None

        model.warnings_issued["estimate_tokens"] = True

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_tokens_per_turn,
            do_sample=True,
            pad_token_id=processing_class.pad_token_id,
            bos_token_id=processing_class.bos_token_id,
            eos_token_id=processing_class.eos_token_id,
            temperature=self.temperature if self.temperature else 1.0,
            top_p=self.top_p if self.top_p else 1.0,
            top_k=self.top_k if self.top_k else 50,
        )

        # Data collator
        def data_collator(features):
            return features

        # ---- Initialize parent Trainer ----
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Batch size validation
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n for n in range(2, global_batch_size + 1) if global_batch_size % n == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"Global batch size ({num_processes} x {args.per_device_train_batch_size}) must be divisible "
                f"by num_generations ({self.num_generations}). Valid: {possible_values}."
            )

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    # ============================================================
    # Multi-Turn Dialogue Generation
    # ============================================================

    @torch.no_grad()
    def _generate_single_response(self, messages: list) -> str:
        """Generate a single response given a conversation history."""
        input_text = self.processing_class.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processing_class(
            input_text, return_tensors="pt", truncation=True, max_length=1800
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _play_negotiation(
        self, prompt_agent: str, prompt_opponent: str,
        agent_starts: bool = True,
    ) -> Tuple[list, list]:
        """
        Play a full multi-turn negotiation dialogue.
        Agent: model WITH LoRA (trainable policy).
        Opponent: model WITHOUT LoRA (frozen base model).
        """
        agent_history = [{"role": "system", "content": prompt_agent}]
        opponent_history = [{"role": "system", "content": prompt_opponent}]
        conversation = []
        agent_turn_indices = []

        unwrapped = self.accelerator.unwrap_model(self.model)

        for round_num in range(self.max_negotiation_rounds):
            speakers = ["agent", "opponent"] if agent_starts else ["opponent", "agent"]

            for speaker in speakers:
                if speaker == "agent":
                    unwrapped.enable_adapter_layers()
                    response = self._generate_single_response(agent_history)
                    agent_history.append({"role": "assistant", "content": response})
                    opponent_history.append({"role": "user", "content": response})
                    agent_turn_indices.append(len(conversation))
                    conversation.append({"role": "assistant", "content": response})
                else:
                    unwrapped.disable_adapter_layers()
                    response = self._generate_single_response(opponent_history)
                    unwrapped.enable_adapter_layers()
                    opponent_history.append({"role": "assistant", "content": response})
                    agent_history.append({"role": "user", "content": response})
                    conversation.append({"role": "user", "content": response})

        unwrapped.enable_adapter_layers()
        return conversation, agent_turn_indices

    def _tokenize_conversation(
        self, system_prompt: str, conversation: list, agent_turn_indices: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a full conversation and create an assistant_mask.
        Returns: (prompt_ids, completion_ids, assistant_mask)
        """
        device = self.model.device

        # System prompt tokens
        system_messages = [{"role": "system", "content": system_prompt}]
        system_text = self.processing_class.apply_chat_template(
            system_messages, tokenize=False, add_generation_prompt=True
        )
        system_tokens = self.processing_class(system_text, return_tensors="pt", truncation=True, max_length=1800)
        prompt_ids = system_tokens["input_ids"][0]

        # Full conversation tokens
        full_messages = [{"role": "system", "content": system_prompt}] + conversation
        full_text = self.processing_class.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_tokens = self.processing_class(full_text, return_tensors="pt", truncation=True, max_length=2048)
        full_ids = full_tokens["input_ids"][0]

        # Completion = everything after system prompt
        completion_ids = full_ids[len(prompt_ids):]

        # Build assistant_mask
        assistant_mask = torch.zeros(len(completion_ids), dtype=torch.long, device=device)

        current_messages = [{"role": "system", "content": system_prompt}]
        for i, msg in enumerate(conversation):
            current_messages.append(msg)
            if i in agent_turn_indices:
                text_with = self.processing_class.apply_chat_template(
                    current_messages, tokenize=False, add_generation_prompt=False
                )
                text_without = self.processing_class.apply_chat_template(
                    current_messages[:-1], tokenize=False, add_generation_prompt=True
                )
                tokens_with = self.processing_class(text_with, truncation=True, max_length=2048)["input_ids"]
                tokens_without = self.processing_class(text_without, truncation=True, max_length=2048)["input_ids"]

                start = len(tokens_without) - len(prompt_ids)
                end = len(tokens_with) - len(prompt_ids)

                if start >= 0 and end <= len(completion_ids):
                    assistant_mask[start:end] = 1

        return prompt_ids.to(device), completion_ids.to(device), assistant_mask.to(device)

    # ============================================================
    # Standard GRPO methods (adapted from current TRL)
    # ============================================================

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.per_device_train_batch_size,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model, args):
        model.config.use_cache = False
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()
        return model

    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        """Compute per-token log probabilities."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start:start + batch_size]
            attention_mask_batch = attention_mask[start:start + batch_size]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "use_cache": False,
            }

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / (self.temperature if self.temperature else 1.0)

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

        return torch.cat(all_logps, dim=0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    # ============================================================
    # Core: _prepare_inputs and _generate_and_score_completions
    # ============================================================

    @profiling_decorator
    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = generation_batches
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        # ============================================================
        # MULTI-TURN DIALOGUE GENERATION
        # ============================================================
        all_completion_ids = []
        all_assistant_masks = []
        all_prompt_ids = []
        completions_for_reward = []

        logger.info(f"Generating {len(inputs)} multi-turn dialogues...")

        for inp in inputs:
            # Extract prompts
            agent_prompt = inp["prompt"]
            if isinstance(agent_prompt, list):
                agent_prompt = agent_prompt[-1]["content"]
            opponent_prompt = inp.get("prompt_2", agent_prompt)
            if isinstance(opponent_prompt, list):
                opponent_prompt = opponent_prompt[-1]["content"]
            agent_starts = inp.get("starting_agent", True)

            # Play negotiation
            conversation, agent_indices = self._play_negotiation(
                prompt_agent=agent_prompt,
                prompt_opponent=opponent_prompt,
                agent_starts=agent_starts,
            )

            # Tokenize and get assistant mask
            p_ids, c_ids, a_mask = self._tokenize_conversation(
                agent_prompt, conversation, agent_indices
            )
            all_prompt_ids.append(p_ids)
            all_completion_ids.append(c_ids)
            all_assistant_masks.append(a_mask)

            # Store conversation for reward computation
            completions_for_reward.append(conversation)

        # Pad to tensors
        prompt_ids = pad(all_prompt_ids, padding_value=self.processing_class.pad_token_id, padding_side="left")
        prompt_mask = (prompt_ids != self.processing_class.pad_token_id).int()
        completion_ids = pad(all_completion_ids, padding_value=self.processing_class.pad_token_id, padding_side="right")
        assistant_mask = pad(all_assistant_masks, padding_value=0, padding_side="right")
        completion_mask = (completion_ids != self.processing_class.pad_token_id).int()

        # EOS masking
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        eos_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_mask = completion_mask * eos_mask

        # Effective mask: only agent tokens, not padding, before EOS
        effective_mask = completion_mask * assistant_mask

        # Concatenate for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size

        # ============================================================
        # Compute log-probs and reference log-probs
        # ============================================================
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # Old log-probs (for importance sampling if num_iterations > 1)
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            # Reference log-probs (for KL penalty)
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )
                else:
                    # LoRA: disable adapter = reference model
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    with use_adapter(unwrapped, adapter_name="ref" if "ref" in unwrapped.peft_config else None):
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                        )
            else:
                ref_per_token_logps = None

        # ============================================================
        # Compute rewards
        # ============================================================
        # Decode completions for text-based reward functions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": ct}] for ct in completions_text]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions_text)]
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions_for_reward, **reward_kwargs
                )
                output_reward_func = [r if r is not None else torch.nan for r in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather across processes
        rewards_per_func = gather(rewards_per_func)

        # ============================================================
        # Compute advantages (GRPO group normalization)
        # ============================================================
        num_generations = self.num_generations
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)

        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards not in ["none", False]:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice for current process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # ============================================================
        # Logging
        # ============================================================
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        agent_token_ratio = effective_mask.sum().float() / (completion_mask.sum().float() + 1e-8)
        self._metrics[mode]["agent_token_ratio"].append(agent_token_ratio.item())

        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": effective_mask,  # ONLY agent tokens
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    # ============================================================
    # compute_loss (standard GRPO loss)
    # ============================================================

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # PPO-style clipped loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # completion_mask is already effective_mask (agent tokens only)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        is_clipped = (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    # ============================================================
    # Remaining standard methods
    # ============================================================

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs, start_time=None):
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def create_model_card(self, model_name=None, dataset_name=None, tags=None):
        if not self.is_world_process_zero():
            return
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        from .utils import generate_model_card, get_comet_experiment_url
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation="@article{zhihong2024deepseekmath}",
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
