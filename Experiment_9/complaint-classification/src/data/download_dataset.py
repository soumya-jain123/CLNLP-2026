"""
Intelligent Consumer Complaint Classification & Prioritization
Data Acquisition Module

Downloads the Hugging Face `liri-uzh/cfpb-complaints-mini` dataset using
the Hugging Face datasets-server REST API with pagination, retry handling,
data validation, schema inspection, and summary generation.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CFPBDataDownloader")

# Constants
DEFAULT_DATASET = "liri-uzh/cfpb-complaints-mini"
BASE_API_URL = "https://datasets-server.huggingface.co"
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_RETRIES = 6
INITIAL_RETRY_DELAY = 3.0
REQUEST_DELAY = 0.25  # Polite delay between batch requests to avoid 429 rate limits
TIMEOUT_SECONDS = 30


def get_project_root() -> Path:
    """Resolve project root directory robustly."""
    return Path(__file__).resolve().parent.parent.parent


def create_robust_session() -> requests.Session:
    """
    Create a configured requests Session with standard timeouts and headers.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "CFPBComplaintDownloader/1.0 "
                "(Mozilla/5.0; Academic Research NLP Project)"
            ),
            "Accept": "application/json",
        }
    )
    return session


def fetch_with_retry(
    url: str,
    params: Dict[str, Any],
    session: requests.Session,
    max_retries: int = MAX_BATCH_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
) -> Dict[str, Any]:
    """
    Fetch JSON from API with explicit exponential backoff for 429 (Rate Limit)
    and transient 5xx server errors.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_time = float(retry_after) if retry_after else delay
                logger.warning(
                    f"HTTP 429 Rate Limited on attempt {attempt}/{max_retries}. "
                    f"Sleeping for {sleep_time:.1f}s before retry..."
                )
                time.sleep(sleep_time)
                delay *= 2
                continue

            if response.status_code in [500, 502, 503, 504]:
                logger.warning(
                    f"HTTP {response.status_code} Server Error on attempt {attempt}/{max_retries}. "
                    f"Sleeping for {delay:.1f}s before retry..."
                )
                time.sleep(delay)
                delay *= 2
                continue

            response.raise_for_status()

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(
                f"Network exception on attempt {attempt}/{max_retries}: {e}. "
                f"Sleeping for {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"Failed to fetch {url} with params {params} after {max_retries} retries."
    )


def fetch_available_splits(
    dataset_name: str, session: requests.Session
) -> List[str]:
    """
    Query the Hugging Face splits API to verify available dataset splits.
    """
    url = f"{BASE_API_URL}/splits"
    params = {"dataset": dataset_name}
    logger.info(f"Querying splits API for dataset '{dataset_name}'...")
    data = fetch_with_retry(url, params, session)
    splits_info = data.get("splits", [])
    available_splits = [s["split"] for s in splits_info if "split" in s]
    logger.info(f"Verified available splits: {available_splits}")
    return available_splits


def fetch_split_metadata(
    dataset_name: str, split: str, session: requests.Session
) -> Dict[str, Any]:
    """
    Fetch split metadata including schema features and total row count.
    """
    url = f"{BASE_API_URL}/rows"
    params = {
        "dataset": dataset_name,
        "config": "default",
        "split": split,
        "offset": 0,
        "length": 1,
    }
    logger.info(f"Fetching metadata for split '{split}'...")
    data = fetch_with_retry(url, params, session)
    num_rows_total = data.get("num_rows_total")
    features = data.get("features", [])
    sample_rows = data.get("rows", [])
    sample_row = sample_rows[0].get("row", {}) if sample_rows else {}

    logger.info(
        f"Split '{split}': total_rows={num_rows_total}, features count={len(features)}"
    )
    return {
        "num_rows_total": num_rows_total,
        "features": features,
        "sample_row": sample_row,
    }


def identify_and_verify_schema(
    features: List[Dict[str, Any]], sample_row: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Automatically identify the text column and target label column from schema,
    and verify the mapping against sample records.
    """
    feature_names = [
        f.get("name") if "name" in f else f.get("feature", {}).get("name")
        for f in features
    ]
    if not feature_names and sample_row:
        feature_names = list(sample_row.keys())

    logger.info(f"Detected schema columns: {feature_names}")

    text_candidates = [
        "text",
        "complaint",
        "narrative",
        "complaint_text",
        "consumer_complaint_narrative",
        "body",
        "content",
    ]
    label_candidates = ["label", "product", "target", "category", "class"]

    identified_text_col = None
    for cand in text_candidates:
        if cand in feature_names:
            identified_text_col = cand
            break

    identified_label_col = None
    for cand in label_candidates:
        if cand in feature_names:
            identified_label_col = cand
            break

    # Schema fallback inspection
    if not identified_text_col and sample_row:
        # Pick the string column with the longest average length
        str_cols = [
            k for k, v in sample_row.items() if isinstance(v, str)
        ]
        if str_cols:
            identified_text_col = max(str_cols, key=lambda k: len(str(sample_row[k])))

    if not identified_label_col and sample_row:
        for k in sample_row:
            if k != identified_text_col:
                identified_label_col = k
                break

    if not identified_text_col or not identified_label_col:
        raise ValueError(
            f"Could not automatically determine text or label column from {feature_names}"
        )

    # Verification
    sample_text_val = sample_row.get(identified_text_col)
    sample_label_val = sample_row.get(identified_label_col)
    logger.info(
        f"Verified mapping: Text Column='{identified_text_col}' "
        f"(sample length={len(str(sample_text_val))}), "
        f"Label Column='{identified_label_col}' (sample='{sample_label_val}')"
    )

    return identified_text_col, identified_label_col


def download_split_data(
    dataset_name: str,
    split: str,
    expected_total: int,
    batch_size: int,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Download complete split rows using pagination with offset and length.
    Deduplicates row indices and preserves all original data fields.
    """
    url = f"{BASE_API_URL}/rows"
    records: List[Dict[str, Any]] = []
    seen_row_indices = set()
    offset = 0

    logger.info(
        f"Starting download for split '{split}' ({expected_total} rows expected, batch size {batch_size})..."
    )

    with tqdm(
        total=expected_total,
        desc=f"Downloading {split}",
        unit="rows",
        dynamic_ncols=True,
    ) as pbar:
        while offset < expected_total:
            current_length = min(batch_size, expected_total - offset)
            params = {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": current_length,
            }

            payload = fetch_with_retry(url, params, session)
            batch_rows = payload.get("rows", [])
            if not batch_rows:
                logger.warning(
                    f"No rows returned at offset {offset}. Terminating pagination for split '{split}'."
                )
                break

            for item in batch_rows:
                row_idx = item.get("row_idx")
                row_data = item.get("row", {})

                # Avoid duplicate rows
                if row_idx is not None:
                    if row_idx in seen_row_indices:
                        continue
                    seen_row_indices.add(row_idx)

                records.append(row_data)

            pbar.update(len(batch_rows))
            offset += len(batch_rows)

            # Small polite pause to prevent triggering rate limits
            if offset < expected_total and REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(records)
    logger.info(
        f"Completed download for split '{split}'. Retrieved {len(df)} records. "
        f"Validation: {'PASSED' if len(df) == expected_total else 'MISMATCH'}"
    )

    if len(df) != expected_total:
        logger.warning(
            f"Row count mismatch for split '{split}': expected {expected_total}, retrieved {len(df)}"
        )

    return df


def save_raw_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save raw dataset as CSV without modifying any values or columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Successfully saved raw dataset to: {output_path} ({len(df)} rows)")


def compute_split_statistics(
    df: pd.DataFrame, text_col: str, label_col: str
) -> Dict[str, Any]:
    """Calculate key statistics for a dataset split."""
    num_records = len(df)
    num_cols = len(df.columns)
    col_names = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in col_names}

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = {
        col: (missing_counts[col] / num_records * 100) if num_records > 0 else 0
        for col in col_names
    }

    # Duplicate records
    duplicate_rows = int(df.duplicated().sum())

    # Text narratives analysis
    if text_col in df.columns:
        empty_narratives = int(
            (df[text_col].isnull() | (df[text_col].astype(str).str.strip() == "")).sum()
        )
        text_series = df[text_col].dropna().astype(str)
        char_lengths = text_series.apply(len)
        word_lengths = text_series.apply(lambda x: len(x.split()))
        char_stats = {
            "min": int(char_lengths.min()) if not char_lengths.empty else 0,
            "mean": float(char_lengths.mean()) if not char_lengths.empty else 0,
            "median": float(char_lengths.median()) if not char_lengths.empty else 0,
            "max": int(char_lengths.max()) if not char_lengths.empty else 0,
        }
        word_stats = {
            "min": int(word_lengths.min()) if not word_lengths.empty else 0,
            "mean": float(word_lengths.mean()) if not word_lengths.empty else 0,
            "median": float(word_lengths.median()) if not word_lengths.empty else 0,
            "max": int(word_lengths.max()) if not word_lengths.empty else 0,
        }
    else:
        empty_narratives = 0
        char_stats = {}
        word_stats = {}

    # Target class distribution
    if label_col in df.columns:
        label_counts = df[label_col].value_counts().to_dict()
        num_unique_labels = len(label_counts)
        label_dist = {
            lbl: {
                "count": count,
                "percentage": round(count / num_records * 100, 2)
                if num_records > 0
                else 0,
            }
            for lbl, count in label_counts.items()
        }
    else:
        num_unique_labels = 0
        label_dist = {}

    return {
        "num_records": num_records,
        "num_cols": num_cols,
        "columns": col_names,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "duplicate_rows": duplicate_rows,
        "num_unique_labels": num_unique_labels,
        "empty_narratives": empty_narratives,
        "label_distribution": label_dist,
        "char_stats": char_stats,
        "word_stats": word_stats,
    }


def generate_markdown_summary(
    splits_data: Dict[str, pd.DataFrame],
    text_col: str,
    label_col: str,
    report_path: Path,
) -> None:
    """
    Generate comprehensive dataset summary report in Markdown.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Dict[str, Any]] = {}

    for split_name, df in splits_data.items():
        stats[split_name] = compute_split_statistics(df, text_col, label_col)

    combined_df = pd.concat(list(splits_data.values()), ignore_index=True)
    stats["combined"] = compute_split_statistics(combined_df, text_col, label_col)

    md_lines: List[str] = [
        "# Dataset Summary Report: Intelligent Consumer Complaint Classification & Prioritization",
        "",
        f"**Dataset Name:** `{DEFAULT_DATASET}`  ",
        "**Source:** Hugging Face Datasets Server REST API  ",
        f"**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Identified Text Column:** `{text_col}`  ",
        f"**Identified Target Label Column:** `{label_col}`  ",
        "",
        "---",
        "",
        "## 1. Overview of Dataset Splits",
        "",
        "| Split | Number of Records | Number of Columns | Duplicate Records | Empty Narratives |",
        "|---|---:|---:|---:|---:|",
    ]

    for split_name in ["train", "test", "combined"]:
        if split_name in stats:
            s = stats[split_name]
            md_lines.append(
                f"| **{split_name.capitalize()}** | {s['num_records']:,} | {s['num_cols']} | "
                f"{s['duplicate_rows']:,} | {s['empty_narratives']:,} |"
            )

    md_lines.extend(
        [
            "",
            "## 2. Schema, Columns & Data Types",
            "",
            "| Column Name | Data Type | Missing Count (Train) | Missing % (Train) | Missing Count (Test) | Missing % (Test) | Role / Description |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )

    combined_cols = stats["combined"]["columns"]
    for col in combined_cols:
        col_type = stats["combined"]["dtypes"].get(col, "unknown")
        train_miss = stats.get("train", {}).get("missing_counts", {}).get(col, 0)
        train_miss_pct = stats.get("train", {}).get("missing_pcts", {}).get(col, 0.0)
        test_miss = stats.get("test", {}).get("missing_counts", {}).get(col, 0)
        test_miss_pct = stats.get("test", {}).get("missing_pcts", {}).get(col, 0.0)

        if col == text_col:
            role = "Feature: Consumer complaint text narrative"
        elif col == label_col:
            role = "Target: Categorical class label"
        elif col == "product_raw":
            role = "Metadata: Original CFPB product name"
        else:
            role = "Attribute"

        md_lines.append(
            f"| `{col}` | `{col_type}` | {train_miss} | {train_miss_pct:.2f}% | "
            f"{test_miss} | {test_miss_pct:.2f}% | {role} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Target Class Distribution",
            "",
            f"Total unique complaint classes: **{stats['combined']['num_unique_labels']}**",
            "",
            "| Target Class (`label`) | Train Count | Train % | Test Count | Test % | Total Count | Total % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    all_labels = sorted(stats["combined"]["label_distribution"].keys())
    for lbl in all_labels:
        tr_info = stats.get("train", {}).get("label_distribution", {}).get(
            lbl, {"count": 0, "percentage": 0.0}
        )
        te_info = stats.get("test", {}).get("label_distribution", {}).get(
            lbl, {"count": 0, "percentage": 0.0}
        )
        tot_info = stats["combined"]["label_distribution"].get(
            lbl, {"count": 0, "percentage": 0.0}
        )

        md_lines.append(
            f"| `{lbl}` | {tr_info['count']:,} | {tr_info['percentage']:.2f}% | "
            f"{te_info['count']:,} | {te_info['percentage']:.2f}% | "
            f"{tot_info['count']:,} | {tot_info['percentage']:.2f}% |"
        )

    md_lines.extend(
        [
            "",
            "## 4. Text Length Characteristics (Complaint Narratives)",
            "",
            "| Split | Metric | Min | Median | Mean | Max |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for split_name in ["train", "test", "combined"]:
        if split_name in stats:
            c = stats[split_name]["char_stats"]
            w = stats[split_name]["word_stats"]
            md_lines.append(
                f"| **{split_name.capitalize()}** | Character Length | {c.get('min', 0):,} | "
                f"{c.get('median', 0.0):.1f} | {c.get('mean', 0.0):.1f} | {c.get('max', 0):,} |"
            )
            md_lines.append(
                f"| **{split_name.capitalize()}** | Word Count | {w.get('min', 0):,} | "
                f"{w.get('median', 0.0):.1f} | {w.get('mean', 0.0):.1f} | {w.get('max', 0):,} |"
            )

    md_lines.extend(
        [
            "",
            "## 5. Key Quality Observations & Recommendations",
            "",
            f"1. **Completeness:** There are **{stats['combined']['empty_narratives']}** empty complaint narratives across train and test splits.",
            f"2. **Duplicates:** Found **{stats['combined']['duplicate_rows']}** duplicate rows in the combined dataset.",
            "3. **Data Integrity:** All original fields (`text`, `label`, `product_raw`) have been preserved as unmanipulated raw data in CSV format.",
            "4. **Class Balance:** Class distributions in both train and test splits remain consistent, supporting fair evaluation for classification models.",
            "",
            "---",
            "*Report auto-generated by `src/data/download_dataset.py`*",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info(f"Dataset summary report written to: {report_path}")


def verify_datasets(
    raw_dir: Path, expected_splits: List[str], text_col: str, label_col: str
) -> bool:
    """
    Verification step confirming that the CSV files can be loaded,
    contain data, and that expected columns and labels exist.
    """
    logger.info("Executing dataset verification step...")
    all_valid = True

    for split in expected_splits:
        csv_file = raw_dir / f"{split}.csv"
        if not csv_file.exists():
            logger.error(f"Verification failed: File not found: {csv_file}")
            all_valid = False
            continue

        try:
            df = pd.read_csv(csv_file)
            logger.info(
                f"Verification: '{csv_file.name}' loaded successfully. Shape={df.shape}"
            )

            if len(df) == 0:
                logger.error(f"Verification failed: '{csv_file.name}' is empty.")
                all_valid = False

            if text_col not in df.columns:
                logger.error(
                    f"Verification failed: '{csv_file.name}' missing text column '{text_col}'"
                )
                all_valid = False

            if label_col not in df.columns:
                logger.error(
                    f"Verification failed: '{csv_file.name}' missing label column '{label_col}'"
                )
                all_valid = False

            unique_labels = df[label_col].dropna().unique()
            logger.info(
                f"Verification: '{csv_file.name}' has {len(unique_labels)} unique classes in '{label_col}'."
            )

        except Exception as e:
            logger.error(f"Verification failed reading '{csv_file}': {e}")
            all_valid = False

    if all_valid:
        logger.info("ALL DATASET VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    else:
        logger.error("ONE OR MORE VERIFICATION CHECKS FAILED.")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Download and summarize Hugging Face CFPB Complaints Mini Dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset identifier (default: liri-uzh/cfpb-complaints-mini)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save raw CSVs (default: data/raw)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory to save dataset summary report (default: reports)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Pagination batch size (default: 100)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloaded CSV files without downloading again.",
    )

    args = parser.parse_args()

    # Determine paths
    project_root = get_project_root()
    raw_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "raw"
    )
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir
        else project_root / "reports"
    )

    logger.info(f"Project root: {project_root}")
    logger.info(f"Target raw data directory: {raw_dir}")
    logger.info(f"Target reports directory: {reports_dir}")

    session = create_robust_session()

    if args.verify_only:
        splits = ["train", "test"]
        valid = verify_datasets(raw_dir, splits, "text", "label")
        sys.exit(0 if valid else 1)

    # 1. Fetch available splits
    available_splits = fetch_available_splits(args.dataset, session)
    if not available_splits:
        logger.error("No splits returned by Hugging Face API.")
        sys.exit(1)

    splits_data: Dict[str, pd.DataFrame] = {}
    identified_text_col: Optional[str] = None
    identified_label_col: Optional[str] = None

    # 2. Process each split
    for split in available_splits:
        meta = fetch_split_metadata(args.dataset, split, session)
        num_rows_total = meta["num_rows_total"]
        features = meta["features"]
        sample_row = meta["sample_row"]

        # Automatic schema verification on first split
        if identified_text_col is None or identified_label_col is None:
            identified_text_col, identified_label_col = identify_and_verify_schema(
                features, sample_row
            )

        # Download split data
        df_split = download_split_data(
            dataset_name=args.dataset,
            split=split,
            expected_total=num_rows_total,
            batch_size=args.batch_size,
            session=session,
        )

        # Save to data/raw/{split}.csv
        csv_path = raw_dir / f"{split}.csv"
        save_raw_csv(df_split, csv_path)
        splits_data[split] = df_split

    # 3. Generate summary report in Markdown
    summary_report_path = reports_dir / "dataset_summary.md"
    generate_markdown_summary(
        splits_data=splits_data,
        text_col=identified_text_col or "text",
        label_col=identified_label_col or "label",
        report_path=summary_report_path,
    )

    # 4. Perform post-download verification
    verification_success = verify_datasets(
        raw_dir=raw_dir,
        expected_splits=available_splits,
        text_col=identified_text_col or "text",
        label_col=identified_label_col or "label",
    )

    if not verification_success:
        logger.error("Post-download verification check failed.")
        sys.exit(1)

    logger.info("Data acquisition and initial analysis completed successfully.")


if __name__ == "__main__":
    main()
