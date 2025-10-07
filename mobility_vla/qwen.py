"""Integration helpers for using Qwen3-VL as the high-level selector."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer

from .high_level import HighLevelGoalSelector, PromptFormatter
from .tour import DemonstrationTour
from .types import Instruction

_GOAL_PATTERN = re.compile(r"\b(F\d+)\b", re.IGNORECASE)


@dataclass
class Qwen3VLGoalSelector(HighLevelGoalSelector):
    """High-level selector backed by a Qwen3-VL model.

    The selector treats goal finding as a text generation task: MobilityVLA's
    prompt template is passed to the model and the first ``F<index>`` token in
    the decoded response is interpreted as the goal frame identifier.
    """

    model: Any
    tokenizer: Any
    prompt_formatter: PromptFormatter = field(default_factory=PromptFormatter)
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 32,
            "temperature": 0.1,
            "do_sample": False,
        }
    )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device_map: Optional[str] = "auto",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        prompt_formatter: Optional[PromptFormatter] = None,
        **model_kwargs: Any,
    ) -> "Qwen3VLGoalSelector":
        """Initialises the selector from a Hugging Face model id."""

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=dtype,
                **model_kwargs,
            )
        except ValueError as exc:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=dtype,
                **model_kwargs,
            )
            if dtype is None and hasattr(model, "dtype"):
                dtype = model.dtype
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            prompt_formatter=prompt_formatter or PromptFormatter(),
        )

    def select_goal(self, tour: DemonstrationTour, instruction: Instruction) -> str:
        prompt = self.prompt_formatter.format(tour, instruction)
        device = self._get_device()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **self.generation_kwargs,
            )

        generated = output[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        match = _GOAL_PATTERN.search(text)
        frame_ids = [frame.frame_id for frame in tour]
        if match:
            candidate = match.group(1).upper()
            if candidate in frame_ids:
                return candidate

        normalized_text = text.lower()
        for frame_id in frame_ids:
            if frame_id.lower() in normalized_text:
                return frame_id

        raise ValueError(
            "Qwen3-VL response did not contain a recognizable frame identifier. "
            f"Raw output: {text!r}"
        )

    def _get_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
