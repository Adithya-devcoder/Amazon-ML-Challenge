"""
evaluation.py
=============
Evaluation metrics and diagnostics for Amazon ML Challenge 2026:
Business Entity Resolution.

Team Role & Scope
-----------------
* Member 1: Data + EDA + Evaluation + Integration Lead
* Competition Objective: Resolve Source 1 entities to matching Source 2 and Source 3 entities.
* Official Evaluation Metric: Macro-averaged F0.5 per Source 1 entity (beta = 0.5).
  Precision is weighted twice as heavily as recall:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
* Candidate Recall: Explicitly evaluated for candidate generation/blocking stages,
  kept completely distinct from final prediction F0.5.

Design Principles & Constraints
-------------------------------
* Zero dataset loading on import: operates strictly on passed mappings or DataFrames.
* Hardware & memory conscious: Intel i3 / 16 GB RAM friendly.
  Accumulates streaming running totals to prevent memory exhaustion on multi-million row datasets.
* Robust parsing: gracefully handles nulls, NaNs, bracketed lists, whitespace, and delimited strings.
* Strict validation: rejects malformed structural inputs without arbitrary silent corruption.
* Zero external network/API dependencies, zero hardcoded country assumptions.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA: float = 0.5
BETA_SQ: float = BETA * BETA  # 0.25
BETA_WEIGHT: float = 1.0 + BETA_SQ  # 1.25

NULL_LITERALS: frozenset[str] = frozenset({"nan", "none", "null", "nat", "<na>", ""})


# ---------------------------------------------------------------------------
# Input Parsing & Extraction
# ---------------------------------------------------------------------------
def parse_match_ids(value: Any) -> set[str]:
    """Safely convert ground-truth or prediction representations into a clean set of entity IDs.

    Handles:
    - Missing / null indicators: ``None``, ``float('nan')``, ``pd.NA``, empty string, whitespace.
    - Comma-separated strings: ``"S2-101,S3-202"`` or ``"S2-101, S3-202"``.
    - Stringified lists: ``"['S2-101', 'S3-202']"`` or ``"[S2-101, S3-202]"``.
    - Python collections: ``set``, ``list``, ``tuple``, or any non-string sequence.
    - Trims whitespace from individual ID tokens and eliminates duplicates.

    Parameters
    ----------
    value : Any
        Ground-truth or prediction entry.

    Returns
    -------
    set[str]
        Deduplicated set of non-empty entity ID strings.

    Raises
    ------
    TypeError
        If ``value`` is of an unsupported type (e.g., non-NaN numbers, dictionaries).
    """
    if value is None:
        return set()

    # Handle float NaN
    if isinstance(value, float):
        if math.isnan(value):
            return set()
        raise TypeError(
            f"Numeric float ({value!r}) cannot be parsed as entity ID set."
        )

    # Collections (set, list, tuple) must be handled before pd.isna scalar checks
    if isinstance(value, set):
        cleaned: set[str] = set()
        for elem in value:
            if elem is None:
                continue
            if isinstance(elem, float) and math.isnan(elem):
                continue
            token = str(elem).strip()
            if token and token.lower() not in NULL_LITERALS:
                cleaned.add(token)
        return cleaned

    if isinstance(value, (list, tuple)):
        cleaned = set()
        for elem in value:
            if elem is None:
                continue
            if isinstance(elem, float) and math.isnan(elem):
                continue
            token = str(elem).strip().strip("'\"").strip()
            if token and token.lower() not in NULL_LITERALS:
                cleaned.add(token)
        return cleaned

    # Handle pandas / numpy NA indicators (scalars only)
    try:
        if pd.isna(value):
            return set()
    except (ValueError, TypeError):
        pass

    # String representations
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in NULL_LITERALS:
            return set()

        # Strip enclosing square or curly brackets if present
        if (text.startswith("[") and text.endswith("]")) or (
            text.startswith("{") and text.endswith("}")
        ):
            text = text[1:-1].strip()
            if not text:
                return set()

        tokens = text.split(",")
        cleaned = set()
        for tok in tokens:
            token = tok.strip().strip("'\"").strip()
            if token and token.lower() not in NULL_LITERALS:
                cleaned.add(token)
        return cleaned

    raise TypeError(
        f"Unsupported type for match IDs: {type(value).__name__} (value: {value!r})"
    )


def _to_id_mapping(data: Any) -> Mapping[str, Any]:
    """Convert input dictionary, pandas Series, or pandas DataFrame to an ID mapping.

    Parameters
    ----------
    data : Any
        Input mapping or DataFrame.

    Returns
    -------
    Mapping[str, Any]
        Mapping from Source 1 entity ID to its corresponding match representation.

    Raises
    ------
    TypeError
        If input data cannot be converted to a mapping.
    ValueError
        If DataFrame has insufficient columns.
    """
    if isinstance(data, Mapping):
        return data

    if isinstance(data, pd.Series):
        return data.to_dict()

    if isinstance(data, pd.DataFrame):
        if data.empty:
            return {}

        # Detect Source 1 ID column
        id_col: Optional[str] = None
        for candidate in ["source1_entity_id", "entity_id", "s1_id", "id"]:
            if candidate in data.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = str(data.columns[0])

        # Detect matched IDs column
        match_col: Optional[str] = None
        for candidate in [
            "matched_entity_ids",
            "matched_ids",
            "matches",
            "predictions",
            "candidate_ids",
        ]:
            if candidate in data.columns:
                match_col = candidate
                break
        if match_col is None:
            if len(data.columns) > 1:
                match_col = str(data.columns[1])
            else:
                match_col = id_col

        return dict(zip(data[id_col].astype(str), data[match_col]))

    raise TypeError(
        f"Expected dict, Mapping, pd.Series, or pd.DataFrame; received {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Per-Entity Metric Calculation
# ---------------------------------------------------------------------------
def calculate_entity_metrics(
    ground_truth_ids: Any,
    predicted_ids: Any,
) -> dict[str, Any]:
    """Compute TP, FP, FN, Precision, Recall, and F0.5 for a single Source 1 entity.

    Formulas:
    ---------
    * TP = len(predicted & ground_truth)
    * FP = len(predicted - ground_truth)
    * FN = len(ground_truth - predicted)
    * Precision = TP / (TP + FP)
    * Recall = TP / (TP + FN)
    * F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Special Case Handling:
    ----------------------
    * Both empty (True no-match): Precision = 1.0, Recall = 1.0, F0.5 = 1.0
    * Ground truth non-empty, prediction empty: Precision = 1.0, Recall = 0.0, F0.5 = 0.0
    * Ground truth empty, prediction non-empty: Precision = 0.0, Recall = 1.0, F0.5 = 0.0

    Parameters
    ----------
    ground_truth_ids : Any
        Ground-truth matched IDs (set, list, string, or NaN).
    predicted_ids : Any
        Predicted matched IDs (set, list, string, or NaN).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "tp": int
        - "fp": int
        - "fn": int
        - "precision": float
        - "recall": float
        - "f0_5": float
    """
    gt_set = parse_match_ids(ground_truth_ids)
    pred_set = parse_match_ids(predicted_ids)

    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    gt_empty = len(gt_set) == 0
    pred_empty = len(pred_set) == 0

    if gt_empty and pred_empty:
        # True no-match: model correctly predicted no matches
        precision = 1.0
        recall = 1.0
        f0_5 = 1.0
    elif not gt_empty and pred_empty:
        # Missed matches: 0/0 precision convention = 1.0, recall = 0.0
        precision = 1.0
        recall = 0.0
        f0_5 = 0.0
    elif gt_empty and not pred_empty:
        # Hallucinated matches on a true no-match entity: precision = 0.0, recall = 1.0
        precision = 0.0
        recall = 1.0
        f0_5 = 0.0
    else:
        # General case where both sets contain IDs
        pred_count = tp + fp
        gt_count = tp + fn
        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_count if gt_count > 0 else 0.0

        denom = BETA_SQ * precision + recall
        if denom > 0.0:
            f0_5 = (BETA_WEIGHT * precision * recall) / denom
        else:
            f0_5 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
    }


# ---------------------------------------------------------------------------
# Full Evaluation Routine (Macro-Averaged per Source 1 Entity)
# ---------------------------------------------------------------------------
def evaluate_predictions(
    ground_truth_map: Any,
    prediction_map: Any,
    *,
    return_per_entity: bool = False,
) -> dict[str, Any]:
    """Evaluate predictions against ground truth using macro-averaging per Source 1 entity.

    Memory Safety:
    --------------
    Operates via single-pass streaming aggregation (running sums and counters).
    Does NOT construct large intermediate comparison tables in memory, keeping
    the footprint O(1) beyond the input structures (safe for 16 GB RAM / i3 CPU).

    Parameters
    ----------
    ground_truth_map : Any
        Ground-truth mapping (dict, Series, or DataFrame).
    prediction_map : Any
        Prediction mapping (dict, Series, or DataFrame).
        Entities in ground_truth_map that are absent in prediction_map are treated
        as having empty predictions (predicted no match).
    return_per_entity : bool, default False
        If True, includes per-entity metric records in the returned dictionary
        under the key "per_entity". Note: set to False for multi-million entity
        runs to save memory.

    Returns
    -------
    dict[str, Any]
        Structured evaluation dictionary containing:
        - Macro metrics: "macro_precision", "macro_recall", "macro_f0_5"
        - Entity counts: "total_entities", "entities_with_ground_truth_matches",
          "true_no_match_entities", "predicted_no_match_entities"
        - Match aggregates: "total_true_matches", "total_predicted_matches",
          "total_true_positives", "total_false_positives", "total_false_negatives"
        - Diagnostics: "exact_match_entities", "exact_match_rate",
          "correct_zero_match_entities", "false_positive_on_zero_match",
          "false_negative_on_matches", "entities_with_false_positives",
          "entities_with_false_negatives", "prediction_coverage"
        - Source breakdowns: "source_breakdown" for "s2" and "s3"
        - Optionally "per_entity" if return_per_entity=True.
    """
    gt_dict = _to_id_mapping(ground_truth_map)
    pred_dict = _to_id_mapping(prediction_map)

    total_entities = len(gt_dict)
    if total_entities == 0:
        logger.warning("Empty ground truth provided to evaluate_predictions.")
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f0_5": 0.0,
            "total_entities": 0,
            "entities_with_ground_truth_matches": 0,
            "true_no_match_entities": 0,
            "predicted_no_match_entities": 0,
            "total_true_matches": 0,
            "total_predicted_matches": 0,
            "total_true_positives": 0,
            "total_false_positives": 0,
            "total_false_negatives": 0,
            "exact_match_entities": 0,
            "exact_match_rate": 0.0,
            "correct_zero_match_entities": 0,
            "false_positive_on_zero_match": 0,
            "false_negative_on_matches": 0,
            "entities_with_false_positives": 0,
            "entities_with_false_negatives": 0,
            "prediction_coverage": 0.0,
            "source_breakdown": {
                "s2": {
                    "true_matches": 0,
                    "predicted_matches": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "micro_precision": 0.0,
                    "micro_recall": 0.0,
                    "micro_f0_5": 0.0,
                },
                "s3": {
                    "true_matches": 0,
                    "predicted_matches": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "micro_precision": 0.0,
                    "micro_recall": 0.0,
                    "micro_f0_5": 0.0,
                },
            },
        }

    sum_precision: float = 0.0
    sum_recall: float = 0.0
    sum_f0_5: float = 0.0

    total_true_matches: int = 0
    total_predicted_matches: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0

    entities_with_gt_matches: int = 0
    true_no_match_entities: int = 0
    predicted_no_match_entities: int = 0

    exact_match_entities: int = 0
    correct_zero_match_entities: int = 0
    false_positive_on_zero_match: int = 0
    false_negative_on_matches: int = 0
    entities_with_fp: int = 0
    entities_with_fn: int = 0
    entities_in_pred_count: int = 0

    # Source-specific counters
    s2_gt_matches: int = 0
    s3_gt_matches: int = 0
    s2_pred_matches: int = 0
    s3_pred_matches: int = 0
    s2_tp: int = 0
    s3_tp: int = 0
    s2_fp: int = 0
    s3_fp: int = 0
    s2_fn: int = 0
    s3_fn: int = 0

    per_entity_store: Optional[dict[str, dict[str, Any]]] = (
        {} if return_per_entity else None
    )

    for s1_id, gt_raw in gt_dict.items():
        gt_set = parse_match_ids(gt_raw)

        if s1_id in pred_dict:
            entities_in_pred_count += 1
            pred_set = parse_match_ids(pred_dict[s1_id])
        else:
            pred_set = set()

        tp_set = pred_set & gt_set
        fp_set = pred_set - gt_set
        fn_set = gt_set - pred_set

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        gt_len = len(gt_set)
        pred_len = len(pred_set)

        total_true_matches += gt_len
        total_predicted_matches += pred_len
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Entity presence categorisation
        if gt_len > 0:
            entities_with_gt_matches += 1
        else:
            true_no_match_entities += 1

        if pred_len == 0:
            predicted_no_match_entities += 1

        # Per-entity metric evaluation
        if gt_len == 0 and pred_len == 0:
            precision = 1.0
            recall = 1.0
            f0_5 = 1.0
            exact_match_entities += 1
            correct_zero_match_entities += 1
        elif gt_len > 0 and pred_len == 0:
            precision = 1.0
            recall = 0.0
            f0_5 = 0.0
            false_negative_on_matches += 1
            entities_with_fn += 1
        elif gt_len == 0 and pred_len > 0:
            precision = 0.0
            recall = 1.0
            f0_5 = 0.0
            false_positive_on_zero_match += 1
            entities_with_fp += 1
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = BETA_SQ * precision + recall
            f0_5 = (BETA_WEIGHT * precision * recall) / denom if denom > 0.0 else 0.0

            if pred_set == gt_set:
                exact_match_entities += 1
            if fp > 0:
                entities_with_fp += 1
            if fn > 0:
                entities_with_fn += 1

        sum_precision += precision
        sum_recall += recall
        sum_f0_5 += f0_5

        # Source breakdown
        for m in gt_set:
            if m.startswith("S2-"):
                s2_gt_matches += 1
            elif m.startswith("S3-"):
                s3_gt_matches += 1

        for m in pred_set:
            if m.startswith("S2-"):
                s2_pred_matches += 1
            elif m.startswith("S3-"):
                s3_pred_matches += 1

        for m in tp_set:
            if m.startswith("S2-"):
                s2_tp += 1
            elif m.startswith("S3-"):
                s3_tp += 1

        for m in fp_set:
            if m.startswith("S2-"):
                s2_fp += 1
            elif m.startswith("S3-"):
                s3_fp += 1

        for m in fn_set:
            if m.startswith("S2-"):
                s2_fn += 1
            elif m.startswith("S3-"):
                s3_fn += 1

        if per_entity_store is not None:
            per_entity_store[str(s1_id)] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f0_5": f0_5,
            }

    macro_precision = sum_precision / total_entities
    macro_recall = sum_recall / total_entities
    macro_f0_5 = sum_f0_5 / total_entities

    def _calc_micro(tp_val: int, fp_val: int, fn_val: int) -> tuple[float, float, float]:
        p = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        r = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        d = BETA_SQ * p + r
        f = (BETA_WEIGHT * p * r) / d if d > 0.0 else 0.0
        return p, r, f

    s2_p, s2_r, s2_f = _calc_micro(s2_tp, s2_fp, s2_fn)
    s3_p, s3_r, s3_f = _calc_micro(s3_tp, s3_fp, s3_fn)

    results: dict[str, Any] = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0_5": macro_f0_5,
        "total_entities": total_entities,
        "entities_with_ground_truth_matches": entities_with_gt_matches,
        "true_no_match_entities": true_no_match_entities,
        "predicted_no_match_entities": predicted_no_match_entities,
        "total_true_matches": total_true_matches,
        "total_predicted_matches": total_predicted_matches,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "exact_match_entities": exact_match_entities,
        "exact_match_rate": exact_match_entities / total_entities,
        "correct_zero_match_entities": correct_zero_match_entities,
        "false_positive_on_zero_match": false_positive_on_zero_match,
        "false_negative_on_matches": false_negative_on_matches,
        "entities_with_false_positives": entities_with_fp,
        "entities_with_false_negatives": entities_with_fn,
        "prediction_coverage": entities_in_pred_count / total_entities,
        "source_breakdown": {
            "s2": {
                "true_matches": s2_gt_matches,
                "predicted_matches": s2_pred_matches,
                "true_positives": s2_tp,
                "false_positives": s2_fp,
                "false_negatives": s2_fn,
                "micro_precision": s2_p,
                "micro_recall": s2_r,
                "micro_f0_5": s2_f,
            },
            "s3": {
                "true_matches": s3_gt_matches,
                "predicted_matches": s3_pred_matches,
                "true_positives": s3_tp,
                "false_positives": s3_fp,
                "false_negatives": s3_fn,
                "micro_precision": s3_p,
                "micro_recall": s3_r,
                "micro_f0_5": s3_f,
            },
        },
    }

    if per_entity_store is not None:
        results["per_entity"] = per_entity_store

    return results


# ---------------------------------------------------------------------------
# Candidate Recall Calculation (Blocking Set Evaluation)
# ---------------------------------------------------------------------------
def calculate_candidate_recall(
    ground_truth_map: Any,
    candidate_map: Any,
) -> dict[str, Any]:
    """Measure candidate retrieval recall against true ground-truth matches.

    Purpose:
    --------
    Evaluates blocking / candidate pair generation quality.
    Measures the upper bound of recall achievable by any downstream classification stage.
    This metric is strictly candidate hit rate, kept distinct from final prediction F0.5.

    Formula:
    --------
    Candidate Recall = (Total true matched IDs present in candidate set) / (Total true matched IDs)

    Parameters
    ----------
    ground_truth_map : Any
        Ground-truth mapping (dict, Series, or DataFrame).
    candidate_map : Any
        Candidate mapping (dict, Series, or DataFrame) of S1 ID -> candidate IDs.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "candidate_recall": float
        - "total_true_matches": int
        - "matches_in_candidates": int
        - "missed_matches": int
        - "total_entities": int
        - "entities_with_true_matches": int
        - "true_no_match_entities": int
        - "entities_with_full_candidate_recall": int
        - "entities_with_zero_candidate_recall": int
        - "entities_with_partial_candidate_recall": int
        - "full_candidate_recall_rate": float
        - "mean_candidates_per_entity": float
        - "source_breakdown": source-specific candidate recall metrics for S2 and S3.
    """
    gt_dict = _to_id_mapping(ground_truth_map)
    cand_dict = _to_id_mapping(candidate_map)

    total_entities = len(gt_dict)
    total_true_matches = 0
    total_matches_in_candidates = 0

    entities_with_gt_matches = 0
    true_no_match_entities = 0
    entities_full_recall = 0
    entities_zero_recall = 0
    entities_partial_recall = 0

    total_candidate_pairs = 0

    s2_true = 0
    s2_in_cand = 0
    s3_true = 0
    s3_in_cand = 0

    for s1_id, gt_raw in gt_dict.items():
        gt_set = parse_match_ids(gt_raw)
        gt_count = len(gt_set)

        cand_set = parse_match_ids(cand_dict.get(s1_id, set()))
        total_candidate_pairs += len(cand_set)

        if gt_count == 0:
            true_no_match_entities += 1
            continue

        entities_with_gt_matches += 1
        total_true_matches += gt_count

        recalled_set = gt_set & cand_set
        recalled_count = len(recalled_set)
        total_matches_in_candidates += recalled_count

        if recalled_count == gt_count:
            entities_full_recall += 1
        elif recalled_count == 0:
            entities_zero_recall += 1
        else:
            entities_partial_recall += 1

        for m in gt_set:
            if m.startswith("S2-"):
                s2_true += 1
                if m in cand_set:
                    s2_in_cand += 1
            elif m.startswith("S3-"):
                s3_true += 1
                if m in cand_set:
                    s3_in_cand += 1

    candidate_recall = (
        total_matches_in_candidates / total_true_matches if total_true_matches > 0 else 1.0
    )
    missed_matches = total_true_matches - total_matches_in_candidates

    s2_recall = s2_in_cand / s2_true if s2_true > 0 else 1.0
    s3_recall = s3_in_cand / s3_true if s3_true > 0 else 1.0

    return {
        "candidate_recall": candidate_recall,
        "total_true_matches": total_true_matches,
        "matches_in_candidates": total_matches_in_candidates,
        "missed_matches": missed_matches,
        "total_entities": total_entities,
        "entities_with_true_matches": entities_with_gt_matches,
        "true_no_match_entities": true_no_match_entities,
        "entities_with_full_candidate_recall": entities_full_recall,
        "entities_with_zero_candidate_recall": entities_zero_recall,
        "entities_with_partial_candidate_recall": entities_partial_recall,
        "full_candidate_recall_rate": (
            entities_full_recall / entities_with_gt_matches
            if entities_with_gt_matches > 0
            else 1.0
        ),
        "mean_candidates_per_entity": (
            total_candidate_pairs / total_entities if total_entities > 0 else 0.0
        ),
        "source_breakdown": {
            "s2": {
                "true_matches": s2_true,
                "matches_in_candidates": s2_in_cand,
                "missed_matches": s2_true - s2_in_cand,
                "candidate_recall": s2_recall,
            },
            "s3": {
                "true_matches": s3_true,
                "matches_in_candidates": s3_in_cand,
                "missed_matches": s3_true - s3_in_cand,
                "candidate_recall": s3_recall,
            },
        },
    }


# ---------------------------------------------------------------------------
# Diagnostic Report Formatters
# ---------------------------------------------------------------------------
def format_evaluation_report(results: dict[str, Any]) -> str:
    """Format the dictionary returned by evaluate_predictions into a readable report string.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by evaluate_predictions.

    Returns
    -------
    str
        Multi-line formatted summary.
    """
    s2 = results.get("source_breakdown", {}).get("s2", {})
    s3 = results.get("source_breakdown", {}).get("s3", {})

    lines = [
        "=" * 68,
        "               BUSINESS ENTITY RESOLUTION EVALUATION REPORT",
        "=" * 68,
        f"  Total S1 Entities Evaluated : {results.get('total_entities', 0):,}",
        f"  Prediction Coverage         : {results.get('prediction_coverage', 0.0) * 100:.2f}%",
        "-" * 68,
        "  OFFICIAL COMPETITION METRICS (Macro-Averaged per Source 1 Entity):",
        f"    Macro F0.5 (Primary Metric): {results.get('macro_f0_5', 0.0):.6f}",
        f"    Macro Precision            : {results.get('macro_precision', 0.0):.6f}",
        f"    Macro Recall               : {results.get('macro_recall', 0.0):.6f}",
        "-" * 68,
        "  ENTITY BREAKDOWN:",
        f"    Entities with GT Matches   : {results.get('entities_with_ground_truth_matches', 0):,}",
        f"    True No-Match Entities     : {results.get('true_no_match_entities', 0):,}",
        f"    Predicted No-Match Entities: {results.get('predicted_no_match_entities', 0):,}",
        f"    Exact-Match Entities       : {results.get('exact_match_entities', 0):,} ({results.get('exact_match_rate', 0.0) * 100:.2f}%)",
        f"    Correct Zero-Match Entities: {results.get('correct_zero_match_entities', 0):,}",
        "-" * 68,
        "  ERROR DIAGNOSTICS:",
        f"    FP on Zero-Match Entities  : {results.get('false_positive_on_zero_match', 0):,}",
        f"    FN on Matched Entities     : {results.get('false_negative_on_matches', 0):,}",
        f"    Entities with any FP       : {results.get('entities_with_false_positives', 0):,}",
        f"    Entities with any FN       : {results.get('entities_with_false_negatives', 0):,}",
        "-" * 68,
        "  MATCH PAIR AGGREGATES:",
        f"    Total True Matches         : {results.get('total_true_matches', 0):,}",
        f"    Total Predicted Matches    : {results.get('total_predicted_matches', 0):,}",
        f"    Total True Positives (TP)  : {results.get('total_true_positives', 0):,}",
        f"    Total False Positives (FP) : {results.get('total_false_positives', 0):,}",
        f"    Total False Negatives (FN) : {results.get('total_false_negatives', 0):,}",
        "-" * 68,
        "  SOURCE-SPECIFIC BREAKDOWN:",
        f"    Source 2 -> True: {s2.get('true_matches', 0):,} | Pred: {s2.get('predicted_matches', 0):,} | TP: {s2.get('true_positives', 0):,} | FP: {s2.get('false_positives', 0):,} | FN: {s2.get('false_negatives', 0):,}",
        f"                Micro P: {s2.get('micro_precision', 0.0):.4f} | Micro R: {s2.get('micro_recall', 0.0):.4f} | Micro F0.5: {s2.get('micro_f0_5', 0.0):.4f}",
        f"    Source 3 -> True: {s3.get('true_matches', 0):,} | Pred: {s3.get('predicted_matches', 0):,} | TP: {s3.get('true_positives', 0):,} | FP: {s3.get('false_positives', 0):,} | FN: {s3.get('false_negatives', 0):,}",
        f"                Micro P: {s3.get('micro_precision', 0.0):.4f} | Micro R: {s3.get('micro_recall', 0.0):.4f} | Micro F0.5: {s3.get('micro_f0_5', 0.0):.4f}",
        "=" * 68,
    ]
    return "\n".join(lines)


def format_candidate_recall_report(results: dict[str, Any]) -> str:
    """Format the dictionary returned by calculate_candidate_recall into a readable string.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by calculate_candidate_recall.

    Returns
    -------
    str
        Multi-line formatted summary.
    """
    s2 = results.get("source_breakdown", {}).get("s2", {})
    s3 = results.get("source_breakdown", {}).get("s3", {})

    lines = [
        "=" * 68,
        "            BLOCKING / CANDIDATE SET RECALL REPORT",
        "=" * 68,
        f"  Candidate Recall (Overall)    : {results.get('candidate_recall', 0.0):.6f} ({results.get('candidate_recall', 0.0) * 100:.2f}%)",
        f"  Total True Matches            : {results.get('total_true_matches', 0):,}",
        f"  Matches Inside Candidate Sets : {results.get('matches_in_candidates', 0):,}",
        f"  Missed Ground-Truth Matches   : {results.get('missed_matches', 0):,}",
        "-" * 68,
        "  ENTITY COVERAGE:",
        f"  Total Evaluated Entities      : {results.get('total_entities', 0):,}",
        f"  Entities with True Matches    : {results.get('entities_with_true_matches', 0):,}",
        f"  True No-Match Entities        : {results.get('true_no_match_entities', 0):,}",
        f"  Entities with 100% Recall     : {results.get('entities_with_full_candidate_recall', 0):,} ({results.get('full_candidate_recall_rate', 0.0) * 100:.2f}%)",
        f"  Entities with Partial Recall  : {results.get('entities_with_partial_candidate_recall', 0):,}",
        f"  Entities with Zero Recall     : {results.get('entities_with_zero_candidate_recall', 0):,}",
        "-" * 68,
        "  EFFICIENCY & CARDINALITY:",
        f"  Mean Candidates per Entity    : {results.get('mean_candidates_per_entity', 0.0):.2f}",
        "-" * 68,
        "  SOURCE-SPECIFIC CANDIDATE RECALL:",
        f"    Source 2 -> True: {s2.get('true_matches', 0):,} | In Candidates: {s2.get('matches_in_candidates', 0):,} | Missed: {s2.get('missed_matches', 0):,} | Recall: {s2.get('candidate_recall', 0.0):.4f}",
        f"    Source 3 -> True: {s3.get('true_matches', 0):,} | In Candidates: {s3.get('matches_in_candidates', 0):,} | Missed: {s3.get('missed_matches', 0):,} | Recall: {s3.get('candidate_recall', 0.0):.4f}",
        "=" * 68,
    ]
    return "\n".join(lines)
