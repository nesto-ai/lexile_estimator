"""Public calculator class for pure long-text Lexile-like scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import spacy

from .config import (
    DEFAULT_REF_DATA_DIR,
    DEFAULT_TRAINING_FEATURES,
    LEVEL_ORDER,
    MODEL_VERSION,
    SEGMENT_BOUNDARY_MARGIN,
    TO_CONTENT_RANGES,
    TO_GLOBAL_OFFSET,
    TO_LEVEL_OFFSETS,
)
from .features import FeatureExtractor, LexileCalcError, load_resources, load_textstat
from .models import ModelRegistry


@dataclass(frozen=True)
class LevelCandidate:
    level: str
    range_low: float
    range_high: float
    lexile: float
    segment: str
    position: str
    distance_to_range: float
    midpoint_distance: float
    rank: int


@dataclass(frozen=True)
class LexileResult:
    model_lexile: float
    to_calibrated_lexile: float
    top_level: str
    top_segment: str
    candidates: list[LevelCandidate]
    feature_summary: dict[str, Any]
    warnings: list[str]
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return data


def distance_to_range(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 0.0
    return low - value if value < low else value - high


def segment_for_level(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "top"
    if value - low <= SEGMENT_BOUNDARY_MARGIN:
        return "low"
    if high - value <= SEGMENT_BOUNDARY_MARGIN:
        return "top"
    return "core"


def position_in_range(value: float, low: float, high: float) -> str:
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "inside"


def level_candidates(lexile: float) -> list[LevelCandidate]:
    raw: list[LevelCandidate] = []
    for level in LEVEL_ORDER:
        level_range = TO_CONTENT_RANGES[level]
        distance = distance_to_range(lexile, level_range.low, level_range.high)
        raw.append(
            LevelCandidate(
                level=level,
                range_low=level_range.low,
                range_high=level_range.high,
                lexile=round(lexile, 1),
                segment=segment_for_level(lexile, level_range.low, level_range.high),
                position=position_in_range(lexile, level_range.low, level_range.high),
                distance_to_range=round(distance, 1),
                midpoint_distance=round(abs(lexile - level_range.midpoint), 1),
                rank=0,
            )
        )
    ordered = sorted(raw, key=lambda item: (item.distance_to_range, item.midpoint_distance, LEVEL_ORDER.index(item.level)))
    return [
        LevelCandidate(
            level=item.level,
            range_low=item.range_low,
            range_high=item.range_high,
            lexile=item.lexile,
            segment=item.segment,
            position=item.position,
            distance_to_range=item.distance_to_range,
            midpoint_distance=item.midpoint_distance,
            rank=index,
        )
        for index, item in enumerate(ordered, start=1)
    ]


class LexileCalculator:
    """Score pure English text and map it onto TO levels.

    Without `known_level`, the calculator uses a global model plus a single TO
    global offset. With `known_level`, it uses the level-specific route model
    and the empirical TO Content Dev offset for that level.
    """

    def __init__(self, extractor: FeatureExtractor, models: ModelRegistry):
        self.extractor = extractor
        self.models = models

    @classmethod
    def from_project(
        cls,
        project_root: str | Path | None = None,
        *,
        ref_data_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        training_features: str | Path | None = None,
        spacy_model: str = "en_core_web_sm",
    ) -> "LexileCalculator":
        # project_root/data_dir are accepted for older local scripts; defaults
        # now point to package-local CSV resources.
        root = Path(project_root).resolve() if project_root is not None else None

        def resolve_path(path_value: str | Path) -> Path:
            path = Path(path_value)
            if root is not None and not path.is_absolute():
                return (root / path).resolve()
            return path.resolve()

        if ref_data_dir is None and data_dir is not None:
            ref_data_dir = data_dir
        data_path = resolve_path(ref_data_dir) if ref_data_dir is not None else DEFAULT_REF_DATA_DIR
        training_path = (
            resolve_path(training_features)
            if training_features is not None
            else DEFAULT_TRAINING_FEATURES
        )
        try:
            nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise LexileCalcError(
                f"spaCy model {spacy_model!r} is not installed. Run `python -m spacy download {spacy_model}`."
            ) from exc
        resources = load_resources(data_path)
        extractor = FeatureExtractor(nlp=nlp, textstat=load_textstat(), resources=resources)
        models = ModelRegistry.from_feature_csv(training_path)
        return cls(extractor=extractor, models=models)

    def analyze(self, text: str, *, known_level: str | None = None) -> LexileResult:
        feature_row = self.extractor.extract(text)
        warnings = self._warnings(feature_row)

        if known_level is not None:
            level = known_level.upper().strip()
            if level not in LEVEL_ORDER:
                raise ValueError(f"known_level must be one of {', '.join(LEVEL_ORDER)}")
            fitted_model = self.models.route_model(level)
            model_lexile = self.models.predict(feature_row, fitted_model)
            to_lexile = model_lexile + TO_LEVEL_OFFSETS[level]
            if level in {"P", "N"}:
                warnings.append("P/N has no local TO offset calibration yet; offset 0.0 was used.")
        else:
            fitted_model = self.models.global_model()
            model_lexile = self.models.predict(feature_row, fitted_model)
            to_lexile = model_lexile + TO_GLOBAL_OFFSET
            warnings.append("No known_level was provided; TO level candidates use the global TO offset fallback.")

        candidates = level_candidates(to_lexile)
        top = candidates[0]
        return LexileResult(
            model_lexile=round(model_lexile, 1),
            to_calibrated_lexile=round(to_lexile, 1),
            top_level=top.level,
            top_segment=top.segment,
            candidates=candidates,
            feature_summary=self._feature_summary(feature_row),
            warnings=warnings,
        )

    @staticmethod
    def _warnings(feature_row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        word_count = float(feature_row.get("text_words") or 0)
        if word_count < 100:
            warnings.append("Text has fewer than 100 words; prior evaluation marked very short text as unstable.")
        if word_count < 40:
            warnings.append("Text is too short for reliable sentence/vocabulary estimates.")
        if float(feature_row.get("aoa_unknown_token_ratio") or 0.0) > 20.0:
            warnings.append("High AoA unknown-token ratio; domain or proper-noun load may distort scoring.")
        return warnings

    @staticmethod
    def _feature_summary(feature_row: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "text_words",
            "text_sentences",
            "sentence_len_mean",
            "sentence_len_p90",
            "clause_per_sentence",
            "academic_vocab_ratio",
            "domain_academic_vocab_ratio",
            "aoa_mean",
            "aoa_12plus_ratio",
            "aoa_unknown_token_ratio",
            "wordfreq_content_zipf_mean",
            "textstat_flesch_kincaid",
            "textstat_gunning_fog",
            "textstat_dale_chall",
        ]
        return {key: feature_row.get(key) for key in keys}
