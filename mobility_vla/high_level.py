"""High-level goal selection policies for MobilityVLA."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .tour import DemonstrationTour, TourFrame
from .types import Instruction, normalize_tokens


class HighLevelGoalSelector:
    """Interface for components that pick a goal frame."""

    def select_goal(self, tour: DemonstrationTour, instruction: Instruction) -> str:
        raise NotImplementedError


@dataclass
class KeywordHighLevelSelector(HighLevelGoalSelector):
    """Keyword-based fallback selector.

    This heuristic approximates the long-context VLM reasoning by computing a
    Jaccard similarity between the user instruction and each frame narrative or
    metadata snippet. It is lightweight yet provides deterministic behaviour for
    the example script and unit tests.
    """

    minimum_overlap: int = 1

    def select_goal(self, tour: DemonstrationTour, instruction: Instruction) -> str:
        tokens_instruction = list(normalize_tokens(instruction.text))
        if not tokens_instruction:
            # fall back to the last frame if the instruction has no useful words.
            return tour.frames[-1].frame_id

        instruction_set = set(tokens_instruction)
        best_score = -1.0
        best_frame_id = tour.frames[-1].frame_id

        for frame in tour:
            frame_tokens = set(self._tokens_for_frame(frame))
            if not frame_tokens:
                continue
            overlap = instruction_set & frame_tokens
            if len(overlap) < self.minimum_overlap:
                continue
            union = instruction_set | frame_tokens
            score = len(overlap) / len(union)
            if score > best_score:
                best_score = score
                best_frame_id = frame.frame_id

        return best_frame_id

    @staticmethod
    def _tokens_for_frame(frame: TourFrame) -> Iterable[str]:
        texts: List[str] = []
        if frame.narrative:
            texts.append(frame.narrative)
        for value in frame.metadata.values():
            texts.append(value)
        if not texts:
            return ()
        return itertools.chain.from_iterable(normalize_tokens(text) for text in texts)


@dataclass
class PromptFormatter:
    """Generates long-context prompts compatible with VLM APIs."""

    max_frames: int = 8

    def format(self, tour: DemonstrationTour, instruction: Instruction) -> str:
        """Returns a textual prompt that mirrors the paper's template."""

        frames = list(tour)
        if len(frames) > self.max_frames:
            frames = self._subsample(frames, self.max_frames)

        lines = [
            "You are a robot operating in a building.",
            "Find the closest frame in the tour video that answers the user.",
            "",
        ]
        for frame in frames:
            lines.append(f"[Frame {frame.frame_id} Image: {frame.image_path or 'N/A'}]")
            lines.append(f"Frame {frame.frame_id}. {frame.narrative or 'No narrative.'}")
            lines.append("")

        if instruction.image_path:
            lines.append("This image is what you see now.")
            lines.append(f"[Current Image: {instruction.image_path}]")
            lines.append("")

        lines.append(f"The user says: {instruction.text}")
        lines.append("Respond with the frame identifier (e.g., F3).")
        return "\n".join(lines)

    @staticmethod
    def _subsample(frames: Sequence[TourFrame], max_frames: int) -> Sequence[TourFrame]:
        """Uniformly samples frames to stay within token budgets."""

        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        if len(frames) <= max_frames:
            return frames

        step = len(frames) / max_frames
        selected: List[TourFrame] = []
        for idx in range(max_frames):
            selected.append(frames[int(idx * step)])
        return selected

