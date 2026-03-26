from __future__ import annotations

from typing import Sequence, Optional
import numpy as np
import pandas as pd


class CheckError(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise CheckError(msg)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def require_columns(df: pd.DataFrame, cols: Sequence[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"Missing columns {missing}. {where}".strip())


def require_unique_key(df: pd.DataFrame, key_cols: Sequence[str], where: str = "") -> None:
    require_columns(df, key_cols, where=where)
    dup = df.duplicated(list(key_cols)).sum()
    if dup:
        fail(f"Duplicate key rows: {int(dup)} on {list(key_cols)}. {where}".strip())


def require_no_nan(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, where: str = "") -> None:
    """
    Fail nếu có NaN/Inf.
    - cols=None: check toàn bộ cột
    - cols=list: chỉ check các cột đó
    """
    if cols is None:
        cols = list(df.columns)
    require_columns(df, cols, where=where)

    sub = df.loc[:, list(cols)]
    na = sub.isna().sum()
    na_total = int(na.sum())

    # inf check cho numeric
    num_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
    inf_total = 0
    if num_cols:
        arr = sub[num_cols].to_numpy(dtype=float, copy=False)
        inf_total = int(np.isinf(arr).sum())

    if na_total or inf_total:
        top_na = na[na > 0].sort_values(ascending=False).head(10).to_dict()
        fail(
            f"NaN/Inf detected. na_total={na_total}, inf_total={inf_total}. "
            f"Top NaN cols={top_na}. {where}".strip()
        )