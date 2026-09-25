"""
data_loader.py
==============
Memory-conscious, reusable data-loading utilities for the Amazon ML Challenge 2026
Business Entity Resolution project.

Design principles
-----------------
* Nothing is loaded at import time — the caller decides what to load.
* All files are read as TSV (sep="\\t").
* Chunked-reading helpers are provided for large source files.
* Column selection (usecols) is exposed everywhere so callers can avoid
  pulling unused columns into RAM.
* Validation is strict and fails fast with actionable error messages.
* pathlib.Path is used for all filesystem operations.
* No external network/API calls, no modification of the original TSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level path anchor
# ---------------------------------------------------------------------------
# Resolve the project root relative to this file's location:
#   src/business_entity_resolution/data_loader.py  ->  ../../  (project root)
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DATA_ROOT: Path = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Known schema constants
# ---------------------------------------------------------------------------

#: Columns that every source file (S1 / S2 / S3) must contain.
SOURCE_REQUIRED_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

#: Columns that the training ground-truth file must contain.
GROUND_TRUTH_REQUIRED_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

#: Expected entity-ID prefixes per source.
SOURCE_ID_PREFIX: dict[str, str] = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_data_path(relative: str) -> Path:
    """Return an absolute Path inside the project's ``data/`` directory.

    Parameters
    ----------
    relative:
        A path string relative to ``data/``, e.g. ``"train/train_source1.tsv"``.

    Returns
    -------
    Path
        Absolute path to the requested file.
    """
    return _DATA_ROOT / relative


def _validate_file_exists(path: Path) -> None:
    """Raise :class:`FileNotFoundError` with a clear message if *path* is absent.

    Parameters
    ----------
    path:
        Absolute path to check.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")


def _validate_columns(df: pd.DataFrame, required: List[str], source_label: str) -> None:
    """Raise :class:`ValueError` if *df* is missing any of *required*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required:
        Column names that must be present.
    source_label:
        Human-readable name for the dataset, used in the error message.

    Raises
    ------
    ValueError
        If one or more required columns are absent from *df*.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source_label}] Missing required column(s): {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )


def _validate_entity_id_prefix(
    series: pd.Series,
    expected_prefix: str,
    source_label: str,
    sample_size: int = 1_000,
) -> None:
    """Raise :class:`ValueError` if any entity ID lacks the expected prefix.

    To avoid scanning the entire (potentially millions-of-rows) column, a
    random sample of up to *sample_size* rows is checked.  Pass
    ``sample_size=0`` to skip the check entirely, or set it to a large number
    for exhaustive validation.

    Parameters
    ----------
    series:
        The ``entity_id`` column as a pandas Series.
    expected_prefix:
        The prefix every ID must start with, e.g. ``"S1-"``.
    source_label:
        Human-readable name used in error messages.
    sample_size:
        Maximum number of rows to examine.  Defaults to 1 000.

    Raises
    ------
    ValueError
        If any sampled ID does not start with *expected_prefix*.
    """
    if sample_size <= 0:
        return

    n = min(sample_size, len(series))
    sample = series.dropna().sample(n=n, random_state=42) if n < len(series) else series.dropna()

    bad = sample[~sample.str.startswith(expected_prefix)]
    if not bad.empty:
        examples = bad.head(5).tolist()
        raise ValueError(
            f"[{source_label}] Unexpected entity_id prefix. "
            f"Expected '{expected_prefix}' but found: {examples}"
        )


# ---------------------------------------------------------------------------
# Chunked-reading schema guard
# ---------------------------------------------------------------------------


def _validated_chunk_iter(
    raw_iter: "Iterator[pd.DataFrame]",
    required_columns: List[str],
    source_label: str,
    expected_id_prefix: Optional[str] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "Iterator[pd.DataFrame]":
    """Wrap a raw chunk iterator, validating the **first** chunk before yielding.

    Only the first chunk is inspected so that the rest of the file is never
    materialised into RAM.  Subsequent chunks are yielded as-is — they share
    the same header so column presence is guaranteed by pandas itself.

    Parameters
    ----------
    raw_iter:
        The :class:`pandas.io.parsers.TextFileReader` returned by
        :func:`load_tsv` when ``chunksize`` is set.
    required_columns:
        Columns that must be present in the first chunk.
    source_label:
        Human-readable dataset name used in error messages.
    expected_id_prefix:
        If given, the ``entity_id`` column in the first chunk is sampled to
        check this prefix.
    validate_ids:
        When ``False``, the ID-prefix check is skipped even if
        *expected_id_prefix* is provided.
    id_sample_size:
        Rows sampled for ID-prefix validation.

    Yields
    ------
    pd.DataFrame
        First the validated first chunk, then all remaining chunks.

    Raises
    ------
    ValueError
        If the first chunk fails column or ID-prefix validation.
    StopIteration
        If the file is empty (no chunks produced).
    """
    first = True
    for chunk in raw_iter:
        if first:
            first = False
            _validate_columns(chunk, required_columns, source_label)
            if validate_ids and expected_id_prefix is not None:
                _validate_entity_id_prefix(
                    chunk["entity_id"], expected_id_prefix,
                    source_label, id_sample_size,
                )
        yield chunk


# ---------------------------------------------------------------------------
# Core generic loader
# ---------------------------------------------------------------------------


def load_tsv(
    path: "Path | str",
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    chunksize: Optional[int] = None,
    encoding: str = "utf-8",
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load a TSV file into a :class:`pandas.DataFrame`.

    This is the lowest-level loader.  All higher-level loaders delegate here.

    Parameters
    ----------
    path:
        Absolute or relative path to the TSV file.
    usecols:
        If given, only these columns are read into memory.  All other columns
        are ignored, reducing RAM usage.
    dtype:
        Optional mapping of column name -> dtype forwarded to
        :func:`pandas.read_csv`.
    chunksize:
        If set, returns a :class:`pandas.io.parsers.TextFileReader` iterator
        that yields DataFrames of *chunksize* rows at a time.  Useful for
        files too large to fit in RAM.
    encoding:
        File encoding.  Defaults to ``"utf-8"``.

    Returns
    -------
    pd.DataFrame
        The loaded data (when *chunksize* is ``None``).
    Iterator[pd.DataFrame]
        A chunk iterator (when *chunksize* is not ``None``).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as a TSV.
    """
    path = Path(path)
    _validate_file_exists(path)

    read_kwargs: dict = {
        "sep": "\t",
        "encoding": encoding,
        "usecols": usecols,
        "dtype": dtype,
        "on_bad_lines": "error",
    }
    if chunksize is not None:
        read_kwargs["chunksize"] = chunksize

    try:
        result = pd.read_csv(path, **read_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse '{path}' as a TSV file. "
            f"Ensure the file uses tab separation. Original error: {exc}"
        ) from exc

    logger.debug("Loaded '%s' (chunksize=%s, usecols=%s)", path.name, chunksize, usecols)
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Source loaders — training set
# ---------------------------------------------------------------------------


def load_source1(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **training** Source-1 dataset.

    Source 1 is the reference source.  Every entity_id must start with ``S1-``.

    Parameters
    ----------
    usecols:
        Subset of columns to load (e.g. ``["entity_id", "business_name"]``).
        Pass ``None`` to load all columns.
    chunksize:
        If set, return a chunk iterator instead of a full DataFrame.
        Useful when RAM is tight.  Example: ``chunksize=50_000``.
    validate_ids:
        When ``True`` (default) and *chunksize* is ``None``, verify that a
        random sample of entity IDs all start with ``S1-``.  Set to ``False``
        to skip this check.
    id_sample_size:
        Number of rows sampled for ID-prefix validation.  Ignored when
        *validate_ids* is ``False`` or *chunksize* is not ``None``.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]
        Loaded data, or a chunk iterator.

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or entity IDs have an unexpected prefix.
    """
    path = _resolve_data_path("train/train_source1.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "train_source1")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source1"],
                "train_source1", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "train_source1",
            expected_id_prefix=SOURCE_ID_PREFIX["source1"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


def load_source2(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **training** Source-2 dataset.

    Every entity_id must start with ``S2-``.

    Parameters
    ----------
    usecols:
        Subset of columns to load.
    chunksize:
        If set, return a chunk iterator.
    validate_ids:
        When ``True`` (default) and not chunked, validate ID prefixes.
    id_sample_size:
        Number of rows sampled for ID validation.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or IDs have an unexpected prefix.
    """
    path = _resolve_data_path("train/train_source2.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "train_source2")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source2"],
                "train_source2", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "train_source2",
            expected_id_prefix=SOURCE_ID_PREFIX["source2"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


def load_source3(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **training** Source-3 dataset.

    Every entity_id must start with ``S3-``.

    Parameters
    ----------
    usecols:
        Subset of columns to load.
    chunksize:
        If set, return a chunk iterator.
    validate_ids:
        When ``True`` (default) and not chunked, validate ID prefixes.
    id_sample_size:
        Number of rows sampled for ID validation.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or IDs have an unexpected prefix.
    """
    path = _resolve_data_path("train/train_source3.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "train_source3")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source3"],
                "train_source3", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "train_source3",
            expected_id_prefix=SOURCE_ID_PREFIX["source3"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


# ---------------------------------------------------------------------------
# Ground-truth loader — training set
# ---------------------------------------------------------------------------


def load_train_ground_truth(
    *,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the **training** ground-truth file.

    The ground-truth TSV has two columns:

    * ``source1_entity_id`` — the Source-1 entity.
    * ``matched_entity_ids`` — a comma-separated string of matching
      Source-2 and/or Source-3 IDs.  May represent zero, one, or many
      matches.

    This file is small relative to the source files so chunked reading is
    not offered.

    Parameters
    ----------
    usecols:
        Subset of columns to load.  Pass ``None`` to load all columns.

    Returns
    -------
    pd.DataFrame
        Ground-truth DataFrame.

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing.
    """
    path = _resolve_data_path("train/train_ground_truth.tsv")
    df = load_tsv(path, usecols=usecols)
    assert isinstance(df, pd.DataFrame)
    _validate_columns(df, GROUND_TRUTH_REQUIRED_COLUMNS, "train_ground_truth")
    return df


# ---------------------------------------------------------------------------
# Test-set loaders
# ---------------------------------------------------------------------------


def load_test_source1(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **test** Source-1 dataset.

    Parameters
    ----------
    usecols:
        Subset of columns to load.
    chunksize:
        If set, return a chunk iterator.
    validate_ids:
        When ``True`` (default) and not chunked, validate ID prefixes.
    id_sample_size:
        Number of rows sampled for ID validation.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or IDs have an unexpected prefix.
    """
    path = _resolve_data_path("test/test_source1.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "test_source1")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source1"],
                "test_source1", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "test_source1",
            expected_id_prefix=SOURCE_ID_PREFIX["source1"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


def load_test_source2(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **test** Source-2 dataset.

    Parameters
    ----------
    usecols:
        Subset of columns to load.
    chunksize:
        If set, return a chunk iterator.
    validate_ids:
        When ``True`` (default) and not chunked, validate ID prefixes.
    id_sample_size:
        Number of rows sampled for ID validation.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or IDs have an unexpected prefix.
    """
    path = _resolve_data_path("test/test_source2.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "test_source2")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source2"],
                "test_source2", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "test_source2",
            expected_id_prefix=SOURCE_ID_PREFIX["source2"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


def load_test_source3(
    *,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate_ids: bool = True,
    id_sample_size: int = 1_000,
) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """Load the **test** Source-3 dataset.

    Parameters
    ----------
    usecols:
        Subset of columns to load.
    chunksize:
        If set, return a chunk iterator.
    validate_ids:
        When ``True`` (default) and not chunked, validate ID prefixes.
    id_sample_size:
        Number of rows sampled for ID validation.

    Returns
    -------
    pd.DataFrame | Iterator[pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the TSV file is absent.
    ValueError
        If required columns are missing or IDs have an unexpected prefix.
    """
    path = _resolve_data_path("test/test_source3.tsv")
    result = load_tsv(path, usecols=usecols, chunksize=chunksize)

    if chunksize is None:
        assert isinstance(result, pd.DataFrame)
        _validate_columns(result, SOURCE_REQUIRED_COLUMNS, "test_source3")
        if validate_ids:
            _validate_entity_id_prefix(
                result["entity_id"], SOURCE_ID_PREFIX["source3"],
                "test_source3", id_sample_size,
            )
    else:
        return _validated_chunk_iter(
            result, SOURCE_REQUIRED_COLUMNS, "test_source3",
            expected_id_prefix=SOURCE_ID_PREFIX["source3"],
            validate_ids=validate_ids,
            id_sample_size=id_sample_size,
        )

    return result


# ---------------------------------------------------------------------------
# Standalone validation helpers  (re-usable by other modules / tests)
# ---------------------------------------------------------------------------


def validate_source_columns(df: pd.DataFrame, source_label: str) -> None:
    """Check that *df* contains all columns required for a source file.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    source_label:
        Human-readable name used in error messages (e.g. ``"train_source1"``).

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    _validate_columns(df, SOURCE_REQUIRED_COLUMNS, source_label)


def validate_ground_truth_columns(df: pd.DataFrame) -> None:
    """Check that *df* contains all columns required for the ground-truth file.

    Parameters
    ----------
    df:
        DataFrame to inspect.

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    _validate_columns(df, GROUND_TRUTH_REQUIRED_COLUMNS, "ground_truth")


def validate_entity_id_prefix(
    df: pd.DataFrame,
    expected_prefix: str,
    source_label: str,
    sample_size: int = 1_000,
) -> None:
    """Public wrapper for entity-ID prefix validation.

    Useful for validating a DataFrame loaded externally (e.g. inside a
    Jupyter notebook or by another module).

    Parameters
    ----------
    df:
        DataFrame containing an ``entity_id`` column.
    expected_prefix:
        The prefix that every ID must start with, e.g. ``"S1-"``.
    source_label:
        Human-readable name used in error messages.
    sample_size:
        Number of rows sampled for the check.

    Raises
    ------
    ValueError
        If *df* has no ``entity_id`` column, or if any sampled ID does not
        start with *expected_prefix*.
    """
    if "entity_id" not in df.columns:
        raise ValueError(
            f"[{source_label}] DataFrame has no 'entity_id' column. "
            f"Found: {df.columns.tolist()}"
        )
    _validate_entity_id_prefix(df["entity_id"], expected_prefix, source_label, sample_size)


# ---------------------------------------------------------------------------
# Chunked-reading convenience iterator
# ---------------------------------------------------------------------------


def iter_source_chunks(
    source_name: str,
    split: str = "train",
    *,
    chunksize: int = 50_000,
    usecols: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Iterate over a source file in fixed-size chunks.

    This is the recommended pattern when processing a large source file
    without loading it entirely into RAM::

        for chunk in iter_source_chunks("source2", "train", chunksize=100_000):
            # process chunk -- a regular pd.DataFrame
            ...

    Parameters
    ----------
    source_name:
        One of ``"source1"``, ``"source2"``, or ``"source3"``.
    split:
        Dataset split: ``"train"`` or ``"test"``.
    chunksize:
        Number of rows per chunk.  Defaults to 50 000.
    usecols:
        Subset of columns to read.

    Yields
    ------
    pd.DataFrame
        DataFrames of at most *chunksize* rows each.

    Raises
    ------
    ValueError
        If *source_name* or *split* is not recognised.
    FileNotFoundError
        If the corresponding TSV file does not exist.
    """
    valid_sources = {"source1", "source2", "source3"}
    valid_splits = {"train", "test"}

    if source_name not in valid_sources:
        raise ValueError(
            f"Unknown source_name '{source_name}'. Expected one of {sorted(valid_sources)}."
        )
    if split not in valid_splits:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of {sorted(valid_splits)}."
        )

    filename = f"{split}_{source_name}.tsv"
    path = _resolve_data_path(f"{split}/{filename}")

    yield from load_tsv(path, usecols=usecols, chunksize=chunksize)
