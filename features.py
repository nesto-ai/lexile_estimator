"""Pure-text feature extraction for Lexile-like scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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


def load_subtlex(path: Path) -> dict[str, float]:
    if not path.exists():
        raise LexileCalcError(f"Missing SUBTLEX file: {path}")
    df = pd.read_excel(path, sheet_name="out1g", usecols=["Word", "Lg10WF"])
    return {
        str(word).strip().lower(): float(freq)
        for word, freq in zip(df["Word"], df["Lg10WF"], strict=False)
        if str(word).strip()
    }


def load_resources(data_dir: Path) -> LexicalResources:
    aoa_path = data_dir / "AoA_51715_words.xlsx"
    cefr_path = data_dir / "CEFR,CEFR_J 데이터.xlsx"
    families_path = data_dir / "families.xlsx"
    subtlex_path = data_dir / "SUBTLEXusExcel2007.xlsx"
    missing = [str(path) for path in (aoa_path, cefr_path, families_path, subtlex_path) if not path.exists()]
    if missing:
        raise LexileCalcError(f"Missing resource files: {', '.join(missing)}")

    aoa_df = pd.read_excel(aoa_path, sheet_name="Sheet1")
    cefr_df = pd.read_excel(cefr_path, sheet_name="CEFR")
    cefrj_df = pd.read_excel(cefr_path, sheet_name="CEFR_J")
    fam_df = pd.read_excel(families_path, sheet_name="data")

    aoa_lookup: dict[str, dict[str, float | None]] = {}
    lemma_series = aoa_df["Lemma_highest_PoS"].map(norm)
    for lemma, group in aoa_df.dropna(subset=["Lemma_highest_PoS"]).groupby(lemma_series):
        if not lemma:
            continue
        aoa_vals = group["AoA_Kup_lem"].dropna().astype(float)
        freq_vals = group["Freq_pm"].dropna().astype(float)
        known_vals = group["Perc_known_lem"].dropna().astype(float)
        if len(aoa_vals) == 0:
            continue
        aoa_lookup[lemma] = {
            "aoa": float(aoa_vals.iloc[0]),
            "freq_pm_sum": float(freq_vals.sum()) if len(freq_vals) else None,
            "known": float(known_vals.iloc[0]) if len(known_vals) else None,
        }

    cefr_lookup: dict[str, str] = {}
    for level_col, df in (("CEFR", cefr_df), ("CEFR_J", cefrj_df)):
        for _, row in df.iterrows():
            level = row.get(level_col)
            if pd.isna(level):
                continue
            for form in re.split(r"[/,;]", norm(row.get("headword"))):
                form = form.strip()
                if not form:
                    continue
                current = cefr_lookup.get(form)
                if current is None or CEFR_ORDER.get(str(level), 99) < CEFR_ORDER.get(str(current), 99):
                    cefr_lookup[form] = str(level)

    academic_lemmas = set(fam_df["family"].dropna().map(norm)) | set(fam_df["word"].dropna().map(norm))
    domain_rows = fam_df[fam_df["domain"].notna()]
    domain_academic_lemmas = set(domain_rows["family"].dropna().map(norm)) | set(
        domain_rows["word"].dropna().map(norm)
    )

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

