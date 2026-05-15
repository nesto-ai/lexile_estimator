"""Pure-text feature extraction for Lexile-like scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_REF_DATA_DIR

try:  # pragma: no cover - optional dependency path
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None


CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj"}
COORDINATION_DEPS = {"cc", "conj"}
QUOTE_CHARS = {'"', "'", "“", "”", "‘", "’"}


class LexileCalcError(RuntimeError):
    """Raised when scoring cannot proceed with the configured resources."""


@dataclass(frozen=True)
class LexicalResources:
    aoa: dict[str, dict[str, float | None]]
    cefr: dict[str, str]
    academic_lemmas: set[str]
    domain_academic_lemmas: set[str]
    subtlex: dict[str, float]


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_textstat():
    try:
        import textstat  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LexileCalcError("textstat is required. Install requirements.txt or activate the dev env.") from exc
    return textstat


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise LexileCalcError(f"{path} is missing required columns: {', '.join(missing)}")


def optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_subtlex(path: Path) -> dict[str, float]:
    if not path.exists():
        raise LexileCalcError(f"Missing SUBTLEX lookup file: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    require_columns(df, path, {"word", "lg10wf"})
    return {
        norm(word): float(freq)
        for word, freq in zip(df["word"], df["lg10wf"], strict=False)
        if norm(word) and not pd.isna(freq)
    }


def load_resources(ref_data_dir: Path | None = None) -> LexicalResources:
    ref_data_path = Path(ref_data_dir) if ref_data_dir is not None else DEFAULT_REF_DATA_DIR
    aoa_path = ref_data_path / "aoa_lookup.csv"
    cefr_path = ref_data_path / "cefr_lookup.csv"
    academic_path = ref_data_path / "academic_lemmas.csv"
    subtlex_path = ref_data_path / "subtlex_lookup.csv"
    missing = [str(path) for path in (aoa_path, cefr_path, academic_path, subtlex_path) if not path.exists()]
    if missing:
        raise LexileCalcError(f"Missing resource files: {', '.join(missing)}")

    aoa_df = pd.read_csv(aoa_path, keep_default_na=False)
    cefr_df = pd.read_csv(cefr_path, keep_default_na=False)
    academic_df = pd.read_csv(academic_path, keep_default_na=False)
    require_columns(aoa_df, aoa_path, {"lemma", "aoa", "freq_pm_sum", "known"})
    require_columns(cefr_df, cefr_path, {"lemma", "level"})
    require_columns(academic_df, academic_path, {"lemma", "is_domain_academic"})

    aoa_lookup: dict[str, dict[str, float | None]] = {}
    for _, row in aoa_df.iterrows():
        lemma = norm(row["lemma"])
        if not lemma:
            continue
        aoa = optional_float(row["aoa"])
        if aoa is None:
            continue
        aoa_lookup[lemma] = {
            "aoa": aoa,
            "freq_pm_sum": optional_float(row["freq_pm_sum"]),
            "known": optional_float(row["known"]),
        }

    cefr_lookup: dict[str, str] = {}
    for _, row in cefr_df.iterrows():
        lemma = norm(row["lemma"])
        level = str(row["level"]).strip()
        if lemma and level:
            cefr_lookup[lemma] = level

    academic_lemmas: set[str] = set()
    domain_academic_lemmas: set[str] = set()
    for _, row in academic_df.iterrows():
        lemma = norm(row["lemma"])
        if not lemma:
            continue
        academic_lemmas.add(lemma)
        if bool_value(row["is_domain_academic"]):
            domain_academic_lemmas.add(lemma)

    return LexicalResources(
        aoa=aoa_lookup,
        cefr=cefr_lookup,
        academic_lemmas=academic_lemmas,
        domain_academic_lemmas=domain_academic_lemmas,
        subtlex=load_subtlex(subtlex_path),
    )


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return None
    index = (len(clean) - 1) * q
    low = int(index)
    high = min(low + 1, len(clean) - 1)
    frac = index - low
    return clean[low] * (1 - frac) + clean[high] * frac


def ratio(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def dep_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(dep_depth(child) for child in children)


def is_finite_verb(token) -> bool:
    if token.pos_ not in {"VERB", "AUX"}:
        return False
    verb_forms = set(token.morph.get("VerbForm"))
    return "Fin" in verb_forms or (token.dep_ == "ROOT" and token.tag_ in {"VBD", "VBP", "VBZ", "MD"})


def sentence_has_finite_verb(sent) -> bool:
    return any(is_finite_verb(token) for token in sent)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class FeatureExtractor:
    def __init__(self, nlp, textstat, resources: LexicalResources):
        self.nlp = nlp
        self.textstat = textstat
        self.resources = resources

    def extract(self, text: str) -> dict[str, Any]:
        text = normalize_text(text)
        if not text:
            raise LexileCalcError("Input text is empty after normalization.")

        doc = self.nlp(text)
        sents = [sent for sent in doc.sents if any(tok.is_alpha for tok in sent)]
        sent_lens = [sum(1 for tok in sent if tok.is_alpha) for sent in sents]
        alpha = [tok for tok in doc if tok.is_alpha]
        content = [
            tok
            for tok in alpha
            if tok.pos_ in CONTENT_POS and tok.pos_ != "PROPN" and tok.ent_type_ == "" and tok.lemma_
        ]
        lemmas = [tok.lemma_.lower() for tok in content]

        row = self._base_features(text, doc, sents, sent_lens, alpha, lemmas)
        self._add_subtlex_and_wordfreq(row, alpha)
        self._add_public_readability(row, text)
        return row

    def _base_features(self, text: str, doc, sents, sent_lens, alpha, lemmas) -> dict[str, Any]:
        resources = self.resources
        aoa_vals = [resources.aoa[lemma]["aoa"] for lemma in lemmas if lemma in resources.aoa]
        aoa_nums = [float(value) for value in aoa_vals if value is not None]
        aoa_unknown_lemmas = [lemma for lemma in lemmas if lemma not in resources.aoa]
        aoa_unknown_counts = Counter(aoa_unknown_lemmas)
        aoa_cefr_unknown_lemmas = [
            lemma for lemma in lemmas if lemma not in resources.aoa and lemma not in resources.cefr
        ]
        unknown_noun_adj_lemmas = [
            tok.lemma_.lower()
            for tok in alpha
            if tok.pos_ in {"NOUN", "ADJ"}
            and tok.ent_type_ == ""
            and tok.lemma_
            and tok.lemma_.lower() not in resources.aoa
            and tok.lemma_.lower() not in resources.cefr
        ]
        unknown_noun_adj_counts = Counter(unknown_noun_adj_lemmas)
        unknown_propn = [
            tok
            for tok in alpha
            if tok.pos_ == "PROPN"
            and tok.lemma_
            and tok.lemma_.lower() not in resources.aoa
            and tok.lemma_.lower() not in resources.cefr
        ]
        freq_vals = [
            resources.aoa[lemma]["freq_pm_sum"]
            for lemma in lemmas
            if lemma in resources.aoa and resources.aoa[lemma]["freq_pm_sum"] is not None
        ]
        freq_nums = [float(value) for value in freq_vals if value is not None]
        cefr_vals = [resources.cefr[lemma] for lemma in lemmas if lemma in resources.cefr]
        unique_lemmas = set(lemmas)
        unknown_repeated_tokens = sum(count for count in aoa_unknown_counts.values() if count >= 2)
        unknown_noun_adj_repeated_tokens = sum(count for count in unknown_noun_adj_counts.values() if count >= 2)

        roots = [sent.root for sent in sents]
        depths = [dep_depth(root) for root in roots]
        clause_tokens = [tok for tok in doc if tok.dep_ in CLAUSE_DEPS]
        relcl_tokens = [tok for tok in doc if tok.dep_ == "relcl"]
        passive_tokens = [
            tok
            for tok in doc
            if tok.dep_ == "auxpass" or (tok.tag_ == "VBN" and any(child.dep_ == "auxpass" for child in tok.children))
        ]
        finite_verb_tokens = [tok for tok in doc if is_finite_verb(tok)]
        dependency_distances = [
            abs(tok.i - tok.head.i)
            for tok in doc
            if not tok.is_punct and not tok.is_space and tok.head is not tok
        ]
        noun_chunk_lengths = [
            sum(1 for tok in chunk if tok.is_alpha)
            for chunk in doc.noun_chunks
            if any(tok.is_alpha for tok in chunk)
        ]
        coordination_tokens = [tok for tok in doc if tok.dep_ in COORDINATION_DEPS]
        dialogue_sents = [sent for sent in sents if any(char in sent.text for char in QUOTE_CHARS)]
        fragment_sents = [sent for sent in sents if not sentence_has_finite_verb(sent)]
        sub_markers = Counter(tok.lemma_.lower() for tok in doc if tok.dep_ == "mark" and tok.lemma_)
        propn = [tok for tok in alpha if tok.pos_ == "PROPN"]
        acronyms = [tok for tok in doc if tok.is_alpha and tok.text.isupper() and len(tok.text) > 1]
        numbers = [tok for tok in doc if tok.like_num]
        named_entity_tokens = [tok for ent in doc.ents for tok in ent if tok.is_alpha]
        multiword_entities = [ent for ent in doc.ents if sum(1 for tok in ent if tok.is_alpha) >= 2]

        words = int(self.textstat.lexicon_count(text, removepunct=True))
        sentence_count = max(1, int(self.textstat.sentence_count(text)))
        msl = words / max(1.0, float(sentence_count))

        return {
            "file": "input_text",
            "lexile": None,
            "lexile_source": "",
            "lexile_candidates": "",
            "band": "",
            "text_words": words,
            "text_sentences": sentence_count,
            "textstat_gunning_fog": round(float(self.textstat.gunning_fog(text)), 2),
            "textstat_flesch_kincaid": round(float(self.textstat.flesch_kincaid_grade(text)), 2),
            "textstat_ari": round(float(self.textstat.automated_readability_index(text)), 2),
            "textstat_coleman_liau": round(float(self.textstat.coleman_liau_index(text)), 2),
            "sentence_len_mean": round(sum(sent_lens) / len(sent_lens), 2) if sent_lens else 0.0,
            "sentence_len_p75": round_or_none(percentile([float(x) for x in sent_lens], 0.75)),
            "sentence_len_p90": round_or_none(percentile([float(x) for x in sent_lens], 0.90)),
            "long_sentence_20_ratio": ratio(sum(length >= 20 for length in sent_lens), len(sent_lens)),
            "dependency_depth_mean": round(sum(depths) / len(depths), 2) if depths else 0.0,
            "avg_dependency_distance": round(sum(dependency_distances) / len(dependency_distances), 2)
            if dependency_distances
            else 0.0,
            "clause_per_sentence": round(len(clause_tokens) / max(1, len(sents)), 3),
            "finite_clause_per_sentence": round(len(finite_verb_tokens) / max(1, len(sents)), 3),
            "relative_clause_per_sentence": round(len(relcl_tokens) / max(1, len(sents)), 3),
            "passive_per_sentence": round(len(passive_tokens) / max(1, len(sents)), 3),
            "noun_phrase_len_mean": round(sum(noun_chunk_lengths) / len(noun_chunk_lengths), 2)
            if noun_chunk_lengths
            else 0.0,
            "coordination_per_sentence": round(len(coordination_tokens) / max(1, len(sents)), 3),
            "dialogue_sentence_ratio": ratio(len(dialogue_sents), len(sents)),
            "fragment_sentence_ratio": ratio(len(fragment_sents), len(sents)),
            "content_token_count": len(lemmas),
            "aoa_coverage": ratio(len(aoa_nums), len(lemmas)),
            "aoa_mean": round(sum(aoa_nums) / len(aoa_nums), 2) if aoa_nums else None,
            "aoa_p90": round_or_none(percentile(aoa_nums, 0.90)),
            "aoa_9plus_ratio": ratio(sum(value >= 9 for value in aoa_nums), len(aoa_nums)),
            "aoa_12plus_ratio": ratio(sum(value >= 12 for value in aoa_nums), len(aoa_nums)),
            "aoa_unknown_token_ratio": ratio(len(aoa_unknown_lemmas), len(lemmas)),
            "aoa_unknown_unique_ratio": ratio(len(set(aoa_unknown_lemmas)), len(unique_lemmas)),
            "aoa_unknown_repeated_token_ratio": ratio(unknown_repeated_tokens, len(lemmas)),
            "aoa_cefr_unknown_token_ratio": ratio(len(aoa_cefr_unknown_lemmas), len(lemmas)),
            "unknown_noun_adj_token_ratio": ratio(len(unknown_noun_adj_lemmas), len(lemmas)),
            "unknown_noun_adj_repeated_token_ratio": ratio(unknown_noun_adj_repeated_tokens, len(lemmas)),
            "unknown_propn_token_ratio": ratio(len(unknown_propn), len(alpha)),
            "named_entity_token_ratio": ratio(len(named_entity_tokens), len(alpha)),
            "multiword_entity_per_100_words": round(100.0 * len(multiword_entities) / max(1, words), 2),
            "aoa_unknown_top": ",".join(f"{lemma}:{count}" for lemma, count in aoa_unknown_counts.most_common(8)),
            "low_freq_lt1pm_ratio": ratio(sum(value < 1 for value in freq_nums), len(freq_nums)),
            "cefr_coverage": ratio(len(cefr_vals), len(lemmas)),
            "cefr_b1plus_ratio": ratio(sum(CEFR_ORDER.get(value, 0) >= 3 for value in cefr_vals), len(cefr_vals)),
            "cefr_b2plus_ratio": ratio(sum(CEFR_ORDER.get(value, 0) >= 4 for value in cefr_vals), len(cefr_vals)),
            "academic_vocab_ratio": ratio(sum(lemma in resources.academic_lemmas for lemma in lemmas), len(lemmas)),
            "domain_academic_vocab_ratio": ratio(
                sum(lemma in resources.domain_academic_lemmas for lemma in lemmas), len(lemmas)
            ),
            "proper_noun_ratio": ratio(len(propn), len(alpha)),
            "acronym_ratio": ratio(len(acronyms), len(alpha)),
            "number_ratio": ratio(len(numbers), len(doc)),
            "subordination_markers": ",".join(f"{key}:{value}" for key, value in sub_markers.most_common(8)),
            "msl": round(msl, 4),
            "lmsl_ln": round(math.log(max(msl, 1e-6)), 4),
            "lmsl_log10": round(math.log10(max(msl, 1e-6)), 4),
        }

    def _add_subtlex_and_wordfreq(self, row: dict[str, Any], alpha) -> None:
        lower_tokens = [tok.text.lower() for tok in alpha]
        content_lemmas = [
            tok.lemma_.lower()
            for tok in alpha
            if tok.pos_ in CONTENT_POS and tok.pos_ != "PROPN" and tok.ent_type_ == "" and tok.lemma_
        ]
        content_noun_adj = [
            tok.lemma_.lower()
            for tok in alpha
            if tok.pos_ in {"NOUN", "ADJ"} and tok.ent_type_ == "" and tok.lemma_
        ]

        def subtlex_stats(tokens: list[str], prefix: str) -> None:
            known = [self.resources.subtlex[token] for token in tokens if token in self.resources.subtlex]
            floor_values = [self.resources.subtlex.get(token, 0.0) for token in tokens]
            row[f"{prefix}_coverage"] = round(ratio(len(known), len(tokens)), 2)
            row[f"{prefix}_mlwf_known"] = round(mean(known), 4)
            row[f"{prefix}_mlwf_floor"] = round(mean(floor_values), 4)
            row[f"{prefix}_mlwf_p10"] = round_or_none(percentile(floor_values, 0.10), 4) or 0.0
            row[f"{prefix}_low_lg10wf_lt2_ratio"] = round(ratio(sum(value < 2.0 for value in floor_values), len(floor_values)), 2)
            row[f"{prefix}_low_lg10wf_lt3_ratio"] = round(ratio(sum(value < 3.0 for value in floor_values), len(floor_values)), 2)

        subtlex_stats(lower_tokens, "subtlex_all")
        subtlex_stats(content_lemmas, "subtlex_content")
        subtlex_stats(content_noun_adj, "subtlex_noun_adj")

        if zipf_frequency is not None:
            def zipf_stats(tokens: list[str], prefix: str) -> None:
                values = [float(zipf_frequency(token, "en", minimum=0.0)) for token in tokens]
                known = [value for value in values if value > 0.0]
                row[f"{prefix}_coverage"] = round(ratio(len(known), len(tokens)), 2)
                row[f"{prefix}_zipf_mean"] = round(mean(values), 4)
                row[f"{prefix}_zipf_p10"] = round_or_none(percentile(values, 0.10), 4) or 0.0
                row[f"{prefix}_zipf_p25"] = round_or_none(percentile(values, 0.25), 4) or 0.0
                row[f"{prefix}_low_zipf_lt3_ratio"] = round(ratio(sum(value < 3.0 for value in values), len(values)), 2)
                row[f"{prefix}_low_zipf_lt4_ratio"] = round(ratio(sum(value < 4.0 for value in values), len(values)), 2)

            zipf_stats(lower_tokens, "wordfreq_all")
            zipf_stats(content_lemmas, "wordfreq_content")
            zipf_stats(content_noun_adj, "wordfreq_noun_adj")

        row["lexile_spec_subtlex_all_ln"] = round(
            9.82247 * row["lmsl_ln"] - 2.14634 * row["subtlex_all_mlwf_floor"], 4
        )
        row["lexile_spec_subtlex_content_ln"] = round(
            9.82247 * row["lmsl_ln"] - 2.14634 * row["subtlex_content_mlwf_floor"], 4
        )
        row["lexile_spec_subtlex_all_log10"] = round(
            9.82247 * row["lmsl_log10"] - 2.14634 * row["subtlex_all_mlwf_floor"], 4
        )

    def _add_public_readability(self, row: dict[str, Any], text: str) -> None:
        methods = {
            "textstat_dale_chall": "dale_chall_readability_score",
            "textstat_spache": "spache_readability",
            "textstat_linsear": "linsear_write_formula",
            "textstat_smog": "smog_index",
            "textstat_lix": "lix",
            "textstat_rix": "rix",
            "textstat_difficult_words": "difficult_words",
        }
        for feature_name, method_name in methods.items():
            try:
                row[feature_name] = round(float(getattr(self.textstat, method_name)(text)), 4)
            except Exception:
                row[feature_name] = 0.0
        word_count = max(1.0, float(row.get("text_words") or 0.0))
        row["textstat_difficult_words_ratio"] = round(ratio(float(row["textstat_difficult_words"]), word_count), 4)
