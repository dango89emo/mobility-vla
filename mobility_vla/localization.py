"""Lightweight localization modules for MobilityVLA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .graph import TopologicalGraph
from .tour import TourFrame
from .types import Observation, Pose, normalize_tokens


@dataclass
class LocalizationResult:
    frame_id: str
    pose: Pose
    score: float


class Localizer:
    """Abstract base class for localization strategies."""

    def localize(
        self, observation: Observation, graph: TopologicalGraph
    ) -> LocalizationResult:
        raise NotImplementedError


class NearestNeighborLocalizer(Localizer):
    """Simple descriptor+keyword matching localizer."""

    def __init__(self, minimum_keyword_overlap: int = 1):
        self._minimum_keyword_overlap = minimum_keyword_overlap

    def localize(
        self, observation: Observation, graph: TopologicalGraph
    ) -> LocalizationResult:
        best_result: Optional[LocalizationResult] = None

        for frame in graph.frames.values():
            score = self._score(observation, frame)
            if best_result is None or score > best_result.score:
                best_result = LocalizationResult(frame.frame_id, frame.pose, score)

        if best_result is None:
            # Defensive fallback; this should never trigger because the graph has
            # at least one frame by construction.
            first_frame = next(iter(graph.frames.values()))
            return LocalizationResult(first_frame.frame_id, first_frame.pose, 0.0)

        return best_result

    def _score(self, observation: Observation, frame: TourFrame) -> float:
        descriptor_score = self._descriptor_similarity(observation, frame)
        if descriptor_score is not None:
            return descriptor_score
        return self._keyword_similarity(observation, frame)

    @staticmethod
    def _descriptor_similarity(
        observation: Observation, frame: TourFrame
    ) -> Optional[float]:
        if observation.features is None or frame.global_descriptor is None:
            return None
        if len(observation.features) != len(frame.global_descriptor):
            return None

        distance_sq = 0.0
        for a, b in zip(observation.features, frame.global_descriptor):
            distance_sq += (a - b) ** 2
        return -math.sqrt(distance_sq)

    def _keyword_similarity(self, observation: Observation, frame: TourFrame) -> float:
        tokens_obs = set(normalize_tokens(observation.hint or ""))
        if not tokens_obs:
            return -math.inf

        tokens_frame = set(self._tokens_for_frame(frame))
        overlap = tokens_obs & tokens_frame
        if len(overlap) < self._minimum_keyword_overlap:
            return -math.inf
        union = tokens_obs | tokens_frame
        return len(overlap) / len(union)

    @staticmethod
    def _tokens_for_frame(frame: TourFrame) -> Iterable[str]:
        texts = []
        if frame.narrative:
            texts.append(frame.narrative)
        texts.extend(frame.metadata.values())
        if not texts:
            return ()
        tokens = []
        for text in texts:
            tokens.extend(normalize_tokens(text))
        return tokens

