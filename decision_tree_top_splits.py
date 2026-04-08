from __future__ import annotations

import ast
import json
import os
from openai import OpenAI
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

Criterion = Literal["gini", "entropy"]


def _encode_target(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(df[target_column].astype(str))
    return y, le


def left_mask_for_split(df: pd.DataFrame, split: dict[str, Any]) -> np.ndarray:
    """Boolean mask for the left branch of a split dict produced by :func:`top_single_split_options_df`."""
    col = split["feature"]
    kind = split["kind"]
    if kind == "numeric":
        x = df[col].to_numpy(dtype=float, copy=False)
        finite = np.isfinite(x)
        thr = float(split["threshold"])
        return finite & (x <= thr)
    if kind == "boolean":
        return (df[col] == split["value"]).fillna(False).to_numpy(dtype=bool)
    if kind == "list_len":
        lengths = _list_lengths(df[col])
        finite = np.isfinite(lengths)
        thr = float(split["threshold"])
        return finite & (lengths <= thr)
    if kind == "list_contains":
        elem = split["value"]
        out = np.zeros(len(df), dtype=bool)
        for i, v in enumerate(df[col]):
            if not _is_missing_cell(v) and hasattr(v, "__contains__"):
                out[i] = elem in v
        return out
    vals = df[col].astype("string").fillna("__nan__")
    v = split["value"]
    if v is None:
        return vals.to_numpy() == "__nan__"
    return vals.to_numpy() == v


def _class_counts_str(y_leaf: np.ndarray, le: LabelEncoder, n_classes: int) -> str:
    counts = np.bincount(y_leaf.astype(np.int64, copy=False), minlength=n_classes)
    parts = [f"{le.classes_[i]}={counts[i]}" for i in range(n_classes) if counts[i] > 0]
    return ", ".join(parts)


def impure_leaf_examples(
    df: pd.DataFrame,
    target_column: str,
    split: dict[str, Any],
) -> dict[str, Any]:
    """
    Rows in each leaf whose true label is **not** the leaf majority class (impure / minority rows).

    Returns
    -------
    dict with keys ``left``, ``right`` (DataFrames), ``majority_left``, ``majority_right`` (str labels),
    and ``impurity_left`` / ``impurity_right`` (fraction of minority rows in each leaf).
    """
    y, le = _encode_target(df, target_column)
    n_classes = int(y.max()) + 1
    left = left_mask_for_split(df, split)
    y_l, y_r = y[left], y[~left]
    maj_l = int(np.argmax(np.bincount(y_l, minlength=n_classes)))
    maj_r = int(np.argmax(np.bincount(y_r, minlength=n_classes)))
    maj_l_s, maj_r_s = str(le.classes_[maj_l]), str(le.classes_[maj_r])

    wrong_l = y[left] != maj_l
    wrong_r = y[~left] != maj_r

    out_l = df.loc[left][wrong_l].copy()
    out_r = df.loc[~left][wrong_r].copy()
    out_l["_leaf"] = "left"
    out_r["_leaf"] = "right"
    out_l["_leaf_majority"] = maj_l_s
    out_r["_leaf_majority"] = maj_r_s

    n_l, n_r = int(left.sum()), int((~left).sum())
    return {
        "left": out_l,
        "right": out_r,
        "combined": pd.concat([out_l, out_r], axis=0) if len(out_l) or len(out_r) else pd.DataFrame(),
        "majority_left": maj_l_s,
        "majority_right": maj_r_s,
        "impurity_left": float(wrong_l.sum() / n_l) if n_l else 0.0,
        "impurity_right": float(wrong_r.sum() / n_r) if n_r else 0.0,
    }


def plot_split_stump(
    df: pd.DataFrame,
    target_column: str,
    split: dict[str, Any],
    criterion: Criterion = "entropy",
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (11, 6.5),
) -> plt.Figure:
    """
    Draw a depth-1 decision stump (root + two leaves) with class counts and impurity per leaf.
    """
    y, le = _encode_target(df, target_column)
    n_classes = int(y.max()) + 1
    left = left_mask_for_split(df, split)
    y_l, y_r = y[left], y[~left]
    g_l = _impurity(y_l, criterion, n_classes)
    g_r = _impurity(y_r, criterion, n_classes)
    maj_l = int(np.argmax(np.bincount(y_l, minlength=n_classes)))
    maj_r = int(np.argmax(np.bincount(y_r, minlength=n_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    def _box(xy: tuple[float, float], w: float, h: float, text: str, fc: str) -> None:
        x, y0 = xy
        patch = FancyBboxPatch(
            (x - w / 2, y0 - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#333",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x, y0, text, ha="center", va="center", fontsize=9, family="monospace")

    root_rule = split.get("split_rule", "?")
    root_txt = f"Split\n{root_rule}\n\nn={len(df)}"
    _box((0.5, 0.78), 0.52, 0.18, root_txt, "#E8DCC8")

    left_txt = (
        f"LEFT\nn={len(y_l)}\n{criterion}={g_l:.4f}\n"
        f"majority: {le.classes_[maj_l]}\n{_class_counts_str(y_l, le, n_classes)}"
    )
    _box((0.28, 0.32), 0.42, 0.36, left_txt, "#C8E6D5")

    right_txt = (
        f"RIGHT\nn={len(y_r)}\n{criterion}={g_r:.4f}\n"
        f"majority: {le.classes_[maj_r]}\n{_class_counts_str(y_r, le, n_classes)}"
    )
    _box((0.72, 0.32), 0.42, 0.36, right_txt, "#C8D5E8")

    ax.annotate(
        "",
        xy=(0.28, 0.52),
        xytext=(0.45, 0.68),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
    )
    ax.text(0.32, 0.62, "True", fontsize=9)
    ax.annotate(
        "",
        xy=(0.72, 0.52),
        xytext=(0.55, 0.68),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
    )
    ax.text(0.68, 0.62, "False", fontsize=9)

    fig.tight_layout()
    return fig


def _impurity(y: np.ndarray, criterion: Criterion, n_classes: int) -> float:
    """Node impurity for labels in {0, ..., n_classes-1}."""
    n = y.size
    if n == 0:
        return 0.0
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=n_classes)
    p = counts[counts > 0] / n
    if criterion == "gini":
        return float(1.0 - np.sum(p * p))
    # entropy (log2, same as sklearn tree export convention)
    return float(-np.sum(p * np.log2(p)))


def _impurity_decrease(
    y: np.ndarray,
    left_mask: np.ndarray,
    criterion: Criterion,
    n_classes: int,
) -> float:
    """Weighted impurity decrease (same as sklearn's impurity improvement for a split)."""
    n = y.size
    n_l = int(left_mask.sum())
    n_r = n - n_l
    if n_l == 0 or n_r == 0:
        return 0.0
    parent = _impurity(y, criterion, n_classes)
    y_l, y_r = y[left_mask], y[~left_mask]
    child = (n_l / n) * _impurity(y_l, criterion, n_classes) + (n_r / n) * _impurity(
        y_r, criterion, n_classes
    )
    return parent - child


def _precision_recall_per_class_majority_leaf(
    y: np.ndarray,
    left_mask: np.ndarray,
    class_names: np.ndarray,
    n_classes: int,
) -> dict[str, dict[str, float | int]]:
    """
    Predict each row by the majority class in its leaf (left vs right), then per-class
    precision and recall on the full sample (same convention as a depth-1 decision stump).
    """
    y_l = y[left_mask]
    y_r = y[~left_mask]
    maj_l = int(np.argmax(np.bincount(y_l.astype(np.int64, copy=False), minlength=n_classes)))
    maj_r = int(np.argmax(np.bincount(y_r.astype(np.int64, copy=False), minlength=n_classes)))
    y_pred = np.where(left_mask, maj_l, maj_r)
    prec, rec, _, sup = precision_recall_fscore_support(
        y,
        y_pred,
        labels=np.arange(n_classes),
        zero_division=0,
    )
    return {
        str(class_names[i]): {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "support": int(sup[i]),
        }
        for i in range(n_classes)
    }


def _numeric_threshold_candidates(x: np.ndarray, max_candidates: int) -> np.ndarray:
    """Midpoints between consecutive sorted unique values (sklearn-style for continuous features)."""
    v = np.unique(x[~np.isnan(x)])
    if v.size < 2:
        return np.array([], dtype=float)
    mids = (v[:-1] + v[1:]) / 2.0
    if mids.size <= max_candidates:
        return mids
    idx = np.linspace(0, mids.size - 1, max_candidates, dtype=int)
    return np.unique(mids[idx])


def _is_list_column(s: pd.Series) -> bool:
    """Return True if the series holds list values (ArrowDtype list or object dtype of lists)."""
    if isinstance(s.dtype, pd.ArrowDtype):
        try:
            import pyarrow as pa
            return pa.types.is_list(s.dtype.pyarrow_dtype)
        except ImportError:
            return False
    if s.dtype == object:
        return _series_values_are_all_lists(s)
    return False


def _list_lengths(s: pd.Series) -> np.ndarray:
    """Return a float array of list lengths; NaN where the cell is missing."""
    out = np.empty(len(s), dtype=float)
    for i, v in enumerate(s):
        if _is_missing_cell(v):
            out[i] = np.nan
        elif hasattr(v, "__len__"):
            out[i] = float(len(v))
        else:
            out[i] = np.nan
    return out


def _list_unique_elements(s: pd.Series) -> list[Any | None]:
    """Collect unique scalar elements across all lists in the series."""
    seen: set[Any] = set()
    for v in s:
        if v is None or _is_missing_cell(v):
            continue
        if not hasattr(v, "__iter__") or isinstance(v, str):
            continue
        for elem in v:
            if isinstance(elem, (dict, list)):  # 👈 skip dicts AND nested lists
                continue
            seen.add(elem)

    elems = sorted(seen, key=str)
    return elems


def top_single_split_options_df(
    df: pd.DataFrame,
    target_column: str,
    *,
    criterion: Criterion = "entropy",
    k: int = 5,
    max_thresholds_per_numeric: int = 256,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    """Best one-shot binary splits from an in-memory DataFrame (same logic as CSV path helper).

    Each split dict includes ``metrics_per_class``: for each target label, precision and recall
    when predictions are the leaf majority class (depth-1 stump).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} not in columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise ValueError("No feature columns after removing the target column.")

    rng = np.random.default_rng(random_state)

    y, le_y = _encode_target(df, target_column)
    n_classes = int(y.max()) + 1

    results: list[tuple[float, dict[str, Any]]] = []

    for col in feature_cols:
        s = df[col]
        # Numeric path
        if pd.api.types.is_numeric_dtype(s):
            x = s.to_numpy(dtype=float, copy=False)
            finite = np.isfinite(x)
            if not finite.any():
                continue
            x_eff = x[finite]
            for thr in _numeric_threshold_candidates(x_eff, max_thresholds_per_numeric):
                left = finite & (x <= thr)
                gain = _impurity_decrease(y, left, criterion, n_classes)
                if gain > 0:
                    row: dict[str, Any] = {
                        "feature": col,
                        "kind": "numeric",
                        "threshold": float(thr),
                        "operator": "<=",
                        "split_rule": f"{col} <= {thr:g}",
                        "criterion": criterion,
                        "impurity_decrease": float(gain),
                        "n_left": int(left.sum()),
                        "n_right": int((~left).sum()),
                        "metrics_per_class": _precision_recall_per_class_majority_leaf(
                            y, left, le_y.classes_, n_classes
                        ),
                    }
                    results.append((gain, row))
            continue

        # Boolean path
        if pd.api.types.is_bool_dtype(s):
            left = (s == True).fillna(False).to_numpy(dtype=bool)  # noqa: E712
            if left.sum() > 0 and (~left).sum() > 0:
                gain = _impurity_decrease(y, left, criterion, n_classes)
                if gain > 0:
                    row = {
                        "feature": col,
                        "kind": "boolean",
                        "value": True,
                        "operator": "==",
                        "split_rule": f"{col} == True",
                        "criterion": criterion,
                        "impurity_decrease": float(gain),
                        "n_left": int(left.sum()),
                        "n_right": int((~left).sum()),
                        "metrics_per_class": _precision_recall_per_class_majority_leaf(
                            y, left, le_y.classes_, n_classes
                        ),
                    }
                    results.append((gain, row))
            continue

        # List path: (1) length threshold, (2) element-in-list
        if _is_list_column(s):
            lengths = _list_lengths(s)
            finite = np.isfinite(lengths)
            if finite.any():
                for thr in _numeric_threshold_candidates(lengths[finite], max_thresholds_per_numeric):
                    left = finite & (lengths <= thr)
                    gain = _impurity_decrease(y, left, criterion, n_classes)
                    if gain > 0:
                        row = {
                            "feature": col,
                            "kind": "list_len",
                            "threshold": float(thr),
                            "operator": "len <=",
                            "split_rule": f"len({col}) <= {thr:g}",
                            "criterion": criterion,
                            "impurity_decrease": float(gain),
                            "n_left": int(left.sum()),
                            "n_right": int((~left).sum()),
                            "metrics_per_class": _precision_recall_per_class_majority_leaf(
                                y, left, le_y.classes_, n_classes
                            ),
                        }
                        results.append((gain, row))
            unqiue_elems = _list_unique_elements(s)
            if unqiue_elems:
                for elem in unqiue_elems:
                    if elem =='13934402797805':
                        pass

                    def safe_contains(v):
                        if v is None or v is pd.NA:
                            return False
                        try:
                            return elem in v
                        except Exception:
                            return False

                    left = s.apply(safe_contains).to_numpy(dtype=bool)
                    if left.sum() == 0 or (~left).sum() == 0:
                        continue
                    gain = _impurity_decrease(y, left, criterion, n_classes)
                    if gain > 0:
                        row = {
                            "feature": col,
                            "kind": "list_contains",
                            "value": elem,
                            "operator": "in",
                            "split_rule": f"{repr(elem)} in {col}",
                            "criterion": criterion,
                            "impurity_decrease": float(gain),
                            "n_left": int(left.sum()),
                            "n_right": int((~left).sum()),
                            "metrics_per_class": _precision_recall_per_class_majority_leaf(
                                y, left, le_y.classes_, n_classes
                            ),
                        }
                        results.append((gain, row))
            continue

        # Categorical / string: binary split value vs rest
        vals = s.astype("string").fillna("__nan__")
        uniques = vals.unique()
        # Limit categories to keep runtime reasonable
        if uniques.size > 200:
            uniques = rng.choice(uniques, size=200, replace=False)

        for cat in uniques:
            left = vals.to_numpy() == cat
            if left.sum() == 0 or (~left).sum() == 0:
                continue
            gain = _impurity_decrease(y, left, criterion, n_classes)
            if gain > 0:
                row = {
                    "feature": col,
                    "kind": "categorical",
                    "value": str(cat) if cat != "__nan__" else None,
                    "split_rule": (
                        f"{col} == {repr(cat)}" if cat != "__nan__" else f"{col} is null"
                    ),
                    "criterion": criterion,
                    "impurity_decrease": float(gain),
                    "n_left": int(left.sum()),
                    "n_right": int((~left).sum()),
                    "metrics_per_class": _precision_recall_per_class_majority_leaf(
                        y, left, le_y.classes_, n_classes
                    ),
                }
                results.append((gain, row))

    results.sort(key=lambda t: t[0], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for gain, row in results:
        key = (row["feature"], row.get("threshold"), row.get("value"), row.get("split_rule"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= k:
            break

    return deduped


def top_single_split_options(
    csv_path: str,
    target_column: str,
    *,
    criterion: Criterion = "entropy",
    k: int = 5,
    max_thresholds_per_numeric: int = 256,
    random_state: int | None = 0,
    **read_csv_kw: Any,
) -> list[dict[str, Any]]:
    """
    Load a CSV and return up to ``k`` best one-shot binary splits for purity of ``target_column``.
    Same as :func:`top_single_split_options_df` after ``pandas.read_csv``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target_column : str
        Name of the column to predict (classification target).
    criterion : {"gini", "entropy"}, default "entropy"
        Same meaning as ``criterion`` in ``sklearn.tree.DecisionTreeClassifier``.
        Use ``"entropy"`` for classic information-gain style splits (entropy reduction).
    k : int, default 5
        Number of top split options to return.
    max_thresholds_per_numeric : int
        Cap on threshold candidates per numeric column (uniform subsample of midpoints).
    random_state : int or None
        RNG seed when subsampling many categorical levels.
    **read_csv_kw
        Forwarded to ``pandas.read_csv`` (e.g. ``sep``, ``encoding``).

    Returns
    -------
    list of dict
        Each dict describes one split: feature, rule, impurity decrease, and counts.
    """
    df = pd.read_csv(csv_path, **read_csv_kw)
    return top_single_split_options_df(
        df,
        target_column,
        criterion=criterion,
        k=k,
        max_thresholds_per_numeric=max_thresholds_per_numeric,
        random_state=random_state,
    )


@dataclass(frozen=True)
class TopSplitsResult:
    splits: list[dict[str, Any]]
    target_column: str
    criterion: Criterion


def top_single_split_options_result(
    csv_path: str,
    target_column: str,
    **kwargs: Any,
) -> TopSplitsResult:
    """Same as ``top_single_split_options`` but returns a small dataclass wrapper."""
    criterion = kwargs.get("criterion", "entropy")
    splits = top_single_split_options(csv_path, target_column, **kwargs)
    return TopSplitsResult(splits=splits, target_column=target_column, criterion=criterion)

# New function for running the top splits pipeline
def run_top_splits_pipeline(
    df: pd.DataFrame,
    target_column: str,
    splits: list[dict[str, Any]],
    criterion: Criterion,
) -> None:
    out_dir = Path(__file__).resolve().parent

    best = splits[0]

    # --- Plot and save tree ---
    fig = plot_split_stump(
        df,
        target_column,
        best,
        criterion=criterion,
        title=f"Best single split ({criterion})",
    )
    tree_png = out_dir / "split_stump.png"
    fig.savefig(tree_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tree figure: {tree_png}")
    print()

    # --- Impurity analysis ---
    imp = impure_leaf_examples(df, target_column, best)
    print(
        "Leaf impurity (fraction of rows not matching leaf majority class): "
        f"left={imp['impurity_left']:.2%}, right={imp['impurity_right']:.2%}"
    )

    n_imp = len(imp["combined"])
    print(f"Non-pure example rows (total): {n_imp}")

    if n_imp:
        impure_csv = out_dir / "split_impure_examples.csv"
        imp["combined"].to_csv(impure_csv, index=False)
        print(f"Wrote: {impure_csv}")
        print(imp["combined"].head(15).to_string())

    print()
    # --- Print splits summary ---
    def save_splits_report(splits, target_column, output_path="best_splits_report.txt"):
        output_path = Path(output_path)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"Best splits for target column: {target_column!r}\n\n")

            for i, split in enumerate(splits, start=1):
                f.write(f"Split {i}\n")
                f.write(f"  Rule: {split['split_rule']}\n")
                f.write(f"  Score improvement: {split['impurity_decrease']:.6f}\n")
                f.write(f"  Left rows: {split['n_left']}\n")
                f.write(f"  Right rows: {split['n_right']}\n")
                f.write("  Precision / recall per target class (majority class per leaf):\n")

                for cls, m in split["metrics_per_class"].items():
                    f.write(
                        f"    {cls!r}: precision={m['precision']:.4f}  "
                        f"recall={m['recall']:.4f}  support={m['support']}\n"
                    )

                f.write("\n")

        print(f"Saved report to: {output_path.resolve()}")
    save_splits_report(splits, target_column)


def top_single_split_options_result_df(
    df: pd.DataFrame,
    target_column: str,
    **kwargs: Any,
) -> TopSplitsResult:
    criterion = kwargs.get("criterion", "entropy")
    splits = top_single_split_options_df(df, target_column, **kwargs)
    return TopSplitsResult(splits=splits, target_column=target_column, criterion=criterion)
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    original_shape = cleaned_df.shape

    # Remove rows that are all null
    all_null_rows_mask = cleaned_df.isna().all(axis=1)
    removed_null_rows = cleaned_df.index[all_null_rows_mask].tolist()
    cleaned_df = cleaned_df.loc[~all_null_rows_mask].copy()

    # Remove columns that are all null
    all_null_cols = cleaned_df.columns[cleaned_df.isna().all(axis=0)].tolist()
    cleaned_df = cleaned_df.drop(columns=all_null_cols)

    # Remove constant columns (same value across all non-null rows)
    constant_cols = {}
    for col in cleaned_df.columns:
        if cleaned_df[col].nunique(dropna=False) == 1:
            constant_cols[col] = cleaned_df[col].iloc[0]

    cleaned_df = cleaned_df.drop(columns=list(constant_cols.keys()))

    # Remove fixed list of columns explicitly
    FIXED_DROP_COLUMNS = [
        "Workflow title","Intent decision","Simulated Decision","Args Resolved At","Selected intent ids","Comment inquiry date.1","All the ticket comments decisions"
    ]

    removed_fixed_cols = []
    for col in FIXED_DROP_COLUMNS:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=[col])
            removed_fixed_cols.append(col)

    # Summary
    print("Dataset cleaning summary")
    print("-" * 30)
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print()
    print(f"Removed all-null rows: {len(removed_null_rows)}")
    if removed_null_rows:
        print(f"Row indices removed: {removed_null_rows}")
    print()
    print(f"Removed all-null columns: {len(all_null_cols)}")
    if all_null_cols:
        print(f"Columns removed: {all_null_cols}")
    print()
    print(f"Removed constant-value columns: {len(constant_cols)}")
    if constant_cols:
        print("Columns removed with constant values:")
        for col, val in constant_cols.items():
            print(f"  - {col}: {val}")

    print()
    print(f"Removed fixed columns: {len(removed_fixed_cols)}")
    if removed_fixed_cols:
        print(f"Columns removed (fixed list): {removed_fixed_cols}")

    return cleaned_df

def filter_primary_secondary_decisions(
    df: pd.DataFrame,
    prim_idx: int | None = None,
    second_idx: int | None = None,
) -> pd.DataFrame:
    Decision = df['Decision'].drop_duplicates().reset_index(drop=True)
    print('Choose 2 indexes 1 at a time of Primary and Second decision')
    print(Decision)
    prim = prim_idx if prim_idx is not None else int(input("Choose Primary: "))
    primary_decision = Decision[prim]
    second = second_idx if second_idx is not None else int(input("Choose second: "))
    secondary_decision = Decision[second]
    df_filtered = df[df["Decision"].isin([primary_decision, secondary_decision])]
    return df_filtered
# ===== Run as script: edit these =====
def run_llm_QA_on_splits(txt_path: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY (never commit API keys to the repository).")
    with open(txt_path, "r", encoding="utf-8") as f:
        text_from_file = f.read()
    user_prompt = f"""
    {text_from_file}   
    """
    system_prompt = """
    We found a split as part of a decision tree iteration ,
     Asses the best 5 options Each splits contains the following format: 
     Split Index 
     Rule: The feature will be used along with a predicate condition 
     Score improvement: Z - Decimal number
     Left rows: X 
     Right rows: Y 
     Precision / recall per target class (majority class per leaf): 
     '1st target': precision=A recall=C support=E '
     2nd target': precision=B recall=D support=F 

     we are trying to distinguish between two targets by applying a Rule. 
     Your task is to asses if the Rule makes sense to be applied. 
     Analyze the Rule logic and the two groups devided in order to make a decision
     
     YOUR OUTPUT:
      Return a dict split_index:rule,valid_rule,explanation
      The rule is the variable along with a mathematical logic used to seperate between the groups.
      valid_rule is a boolean sign for the applicability of the rule.
      The explanation is the explanation for why you choose it.
      Return true for a split that makes sense to be applied
      Return false for a split that doesnt make sense 
      Attach an explanation to each split and why you choose it.

       Here are the splits:
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    print("SYSTEM PROMPT")
    print(system_prompt)
    print("--------------------------------")
    print("USER PROMPT")
    print(user_prompt)
    print("--------------------------------")
    print("----Running LLM-----")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


_BOOL_TRUE = frozenset({"True", "true", "TRUE"})
_BOOL_FALSE = frozenset({"False", "false", "FALSE"})


def _parse_cell_to_dict(val: Any) -> dict | None:
    """Return a Python dict if ``val`` looks like a dict literal; else ``None``."""
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Strip one layer of wrapping quotes (CSV / Excel sometimes double-wrap JSON)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1].strip()
        if inner.startswith("{"):
            s = inner.replace('\\"', '"')
    if not s.startswith("{"):
        return None
    import re

    s = re.sub(r'(\b\w+\b)\s*:', r'"\1":', s)

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _try_dict_series(s: pd.Series) -> pd.Series | None:
    """
    Detect dict-like JSON / Python literals. Uses per-cell parsing so a few bad rows
    do not block the whole column; requires most non-null cells to parse as dicts.
    """
    raw = s.dropna()
    if raw.empty:
        return None
    as_str = raw.astype(str).str.strip()
    looks = as_str.str.match(r"^\{", na=False)
    if sum(looks)==0:
        return None

    out_cells: list[Any] = []
    n_in = 0
    n_ok = 0
    for v in s:
        if pd.isna(v):
            out_cells.append(np.nan)
            continue
        n_in += 1
        pd_ = _parse_cell_to_dict(v)
        if pd_ is not None:
            out_cells.append(pd_)
            n_ok += 1
        else:
            out_cells.append(v)

    if n_in == 0 or n_ok / n_in < 0.72:
        return None

    return pd.Series(out_cells, index=s.index, dtype=object, name=s.name)


def _parse_cell_to_list(val: Any) -> list[Any] | None:
    """Return a Python list if ``val`` looks like a list literal; else ``None``."""
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Strip one layer of wrapping quotes (CSV / Excel sometimes double-wrap JSON)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1].strip()
        if inner.startswith("["):
            s = inner.replace('\\"', '"')
    if not s.startswith("["):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, tuple):
            return list(obj)
    return None


def _is_missing_cell(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def _series_values_are_all_lists(s: pd.Series) -> bool:
    for v in s:
        if _is_missing_cell(v):
            continue
        if not isinstance(v, list):
            return False
    return True


def _maybe_arrow_list_dtype(s: pd.Series) -> pd.Series:
    """
    If every non-null value is a ``list`` of strings, use PyArrow ``list<string>`` so
    ``.dtype`` is not plain ``object``. Falls back to ``object`` on import or type errors.
    """
    if s.dropna().empty:
        return s
    try:
        import pyarrow as pa
    except ImportError:
        return s
    vals: list[Any] = []
    for v in s:
        if _is_missing_cell(v):
            vals.append(None)
            continue
        if not isinstance(v, list):
            return s
        if not all(isinstance(x, str) for x in v):
            return s
        vals.append(v)
    try:
        arr = pa.array(vals, type=pa.list_(pa.string()))
    except (pa.ArrowInvalid, TypeError, ValueError):
        return s
    return pd.Series(arr, index=s.index, dtype=pd.ArrowDtype(arr.type), name=s.name)


def _try_bool_series(s: pd.Series) -> pd.Series | None:
    """If values look like booleans, return ``BooleanDtype`` series; else ``None``."""

    def cell(v: Any) -> Any:
        if pd.isna(v):
            return pd.NA
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer, float, np.floating)) and not isinstance(
            v, (bool, np.bool_)
        ):
            if v == 1:
                return True
            if v == 0:
                return False
        t = str(v).strip().lower()
        if t in _BOOL_TRUE:
            return True
        if t in _BOOL_FALSE:
            return False
        return pd.NA

    if s.notna().sum() == 0:
        return None
    mapped = s.map(cell)
    nn = s.notna()
    recognized = nn & mapped.notna()
    if recognized.sum() < 0.85 * nn.sum():
        return None
    return mapped.astype("boolean")


def _try_datetime_series(s: pd.Series) -> pd.Series | None:
    if s.dropna().empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
    nn = s.notna()
    if nn.sum() == 0:
        return None
    if parsed.notna().sum() / nn.sum() < 0.75:
        return None
    return parsed


def _try_list_series(s: pd.Series) -> pd.Series | None:
    """
    Detect list-like JSON / Python literals. Uses per-cell parsing so a few bad rows
    do not block the whole column; requires most non-null cells to parse as lists.
    """
    raw = s.dropna()
    if raw.empty:
        return None
    as_str = raw.astype(str).str.strip()
    looks = as_str.str.match(r"^\[", na=False)
    if looks.mean() < 0.55:
        return None

    out_cells: list[Any] = []
    n_in = 0
    n_ok = 0
    for v in s:
        if pd.isna(v):
            out_cells.append(np.nan)
            continue
        n_in += 1
        pl = _parse_cell_to_list(v)
        if pl is not None:
            out_cells.append(pl)
            n_ok += 1
        else:
            out_cells.append(v)

    if n_in == 0 or n_ok / n_in < 0.72:
        return None

    out = pd.Series(out_cells, index=s.index, dtype=object, name=s.name)
    if _series_values_are_all_lists(out):
        out = _maybe_arrow_list_dtype(out)
    return out


def parse_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort coercion of object/string columns to booleans, datetimes, list, or dict values.

    Order per column: **dict-like strings** → **list-like strings** → **booleans** → **datetimes**.
    Numeric columns are left unchanged. Already-typed boolean/datetime columns are unchanged.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue

        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
            out[col] = s

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            dct = _try_dict_series(s)
            if dct is not None:
                out[col] = dct
                continue
            lst = _try_list_series(s)
            if lst is not None:
                out[col] = lst
                continue
            b = _try_bool_series(s)
            if b is not None:
                out[col] = b
                continue
            dt = _try_datetime_series(s)
            if dt is not None:
                out[col] = dt
                continue

    return out


def _series_to_naive_datetime(s: pd.Series) -> pd.Series:
    """Coerce to datetime64[ns], dropping timezone for safe subtraction."""
    try:
        t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        t = pd.to_datetime(s, errors="coerce", utc=True)
    if hasattr(t.dt, "tz") and t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t


def add_days_ago_columns_for_all_datetimes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every **unordered pair** of distinct ``datetime64`` columns ``(a, b)`` (names sorted
    lexicographically so each pair appears once), append ``"{a} - {b} | days"`` with
    ``(a - b).dt.days`` — positive when ``a`` is after ``b``. Missing datetimes yield NA.
    """
    out = df.copy()
    dt_cols = sorted(
        c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])
    )
    if len(dt_cols) < 2:
        return out

    for i, a in enumerate(dt_cols):
        for b in dt_cols[i + 1 :]:
            new_name = f"{a} - {b} | days"
            if new_name in out.columns:
                continue
            ca = _series_to_naive_datetime(out[a])
            cb = _series_to_naive_datetime(out[b])
            out[new_name] = (ca - cb).dt.days

    return out


def add_Object_dtype_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every plain-object column that is not numeric, bool, datetime, list, or a pure-string
    column, append a boolean column ``"{col} Exists"`` that is True where the original cell is
    non-null and False where it is null.
    """
    out = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == 'O':
            non_null = s.dropna()
            if non_null.empty:
                continue
            if non_null.apply(lambda x: isinstance(x, str)).all():
                continue
            out[f"{col} Exists"] = s.notna()
    return out


if __name__ == "__main__":
    FILE_PATH = "/Users/oranfrydman/CaludeProj/data_set_3009.csv"
    TARGET_COLUMN = "Decision"
    NUM_SPLITS = 5
    CRITERION: Criterion = "entropy"
    csv_path = Path(FILE_PATH)
    df = pd.read_csv(csv_path, encoding="latin1")
    df = clean_dataset(df)
    df = parse_dataframe_dtypes(df)
    df = add_days_ago_columns_for_all_datetimes_features(df)
    df = add_Object_dtype_missing(df)
    df = filter_primary_secondary_decisions(df)
    print(df.dtypes)

    splits = top_single_split_options_df(
        df,
        TARGET_COLUMN,
        criterion=CRITERION,
        k=NUM_SPLITS,
    )

    if not splits:
        print("No splits found.")
        raise SystemExit(0)

    run_top_splits_pipeline(df, TARGET_COLUMN, splits, CRITERION)
    with open("best_splits_report.txt", "r", encoding="utf-8") as f:
        print(f.read())
    pass
    #print(run_llm_QA_on_splits("best_splits_report.txt"))


#todo:
# - input ( csv , target_col)
#- 2 decisions (try to devide) str : In , ==

# SYstem output - 5 best splits
# LLM iteration - applicable sensabilty




