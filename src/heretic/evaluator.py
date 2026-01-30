# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import re
from concurrent.futures import ThreadPoolExecutor

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        # Pre-compile regex pattern for core refusal detection
        normalized_markers = [m.lower() for m in settings.refusal_markers]
        pattern = "|".join(re.escape(marker) for marker in normalized_markers)
        self.core_refusal_pattern = re.compile(pattern, re.IGNORECASE)
        
        # Phase 2: Compile soft refusal pattern
        soft_markers = [m.lower() for m in settings.soft_refusal_markers]
        soft_pattern = "|".join(re.escape(marker) for marker in soft_markers)
        self.soft_refusal_pattern = re.compile(soft_pattern, re.IGNORECASE)
        
        # Phase 2: Compile evasive response patterns (already regex)
        if settings.evasive_patterns:
            evasive_combined = "|".join(f"({p})" for p in settings.evasive_patterns)
            self.evasive_pattern = re.compile(evasive_combined, re.IGNORECASE)
        else:
            self.evasive_pattern = None
        
        # Determine detection mode from settings
        self._detect_soft = settings.detect_soft_refusals or settings.refusal_detection_mode in ["moderate", "lenient"]
        self._detect_evasive = settings.detect_evasive_responses or settings.refusal_detection_mode == "lenient"

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        # Phase 3: Support multi-token KL divergence
        self.kl_tokens = settings.kl_divergence_tokens
        token_desc = "first-token" if self.kl_tokens == 1 else f"{self.kl_tokens}-token"
        print(f"* Obtaining {token_desc} probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts, n_tokens=self.kl_tokens)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        """Check if response is a refusal.
        
        Detection modes:
        - strict: Core markers only (original behavior)
        - moderate: Core + soft refusal markers
        - lenient: Core + soft + evasive patterns
        """
        # Remove emphasis (e.g. "I *will not*...").
        response = response.replace("*", "")

        # Normalize typographic apostrophes ("won't" -> "won't").
        response = response.replace("'", "'")

        # Check core markers (always checked)
        if self.core_refusal_pattern.search(response):
            return True
        
        # Phase 2: Check soft refusals if enabled
        if self._detect_soft and self.soft_refusal_pattern.search(response):
            return True
        
        # Phase 2: Check evasive patterns if enabled
        if self._detect_evasive and self.evasive_pattern and self.evasive_pattern.search(response):
            return True
        
        return False

    def count_refusals(self) -> int:
        # Use shorter token limit for refusal checking - refusals appear in first 20-30 tokens
        # This provides ~40-60% speedup vs generating full responses
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            max_tokens=self.settings.refusal_check_tokens,
        )
        refusals = [response for response in responses if self.is_refusal(response)]
        return len(refusals)

    def _compute_kl_divergence(self) -> float:
        """Compute KL divergence from base model (for parallel execution).
        
        Phase 3: Supports multi-token KL divergence for better capability measurement.
        """
        logprobs = self.model.get_logprobs_batched(self.good_prompts, n_tokens=self.kl_tokens)
        
        if self.kl_tokens == 1:
            # Original single-token behavior
            return F.kl_div(
                logprobs,
                self.base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
        else:
            # Multi-token: average KL across tokens
            # logprobs shape: (n_prompts, n_tokens, vocab_size)
            kl_per_token = F.kl_div(
                logprobs,
                self.base_logprobs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)  # Sum over vocab, shape: (n_prompts, n_tokens)
            
            # Mean over tokens and prompts
            return kl_per_token.mean().item()

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Evaluating in parallel (KL divergence + refusal counting)...")

        # Run KL divergence and refusal counting in parallel
        # These use different prompt sets (good vs bad) so can overlap
        with ThreadPoolExecutor(max_workers=2) as executor:
            kl_future = executor.submit(self._compute_kl_divergence)
            refusal_future = executor.submit(self.count_refusals)

            kl_divergence = kl_future.result()
            refusals = refusal_future.result()

        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
