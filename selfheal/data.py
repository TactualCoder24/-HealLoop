"""Stage 1 - INGEST & VALIDATE: read NASA C-MAPSS rows and check schema, chronology, ranges, duplicates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS

# Plain-language names of the 21 C-MAPSS sensors (NASA documentation) used in work-order evidence.
SENSOR_NAMES = {
    "s1": "T2 fan inlet temp", "s2": "T24 LPC outlet temp", "s3": "T30 HPC outlet temp", "s4": "T50 LPT outlet temp",
    "s5": "P2 fan inlet pressure", "s6": "P15 bypass-duct pressure", "s7": "P30 HPC outlet pressure",
    "s8": "Nf physical fan speed", "s9": "Nc physical core speed", "s10": "epr engine pressure ratio",
    "s11": "Ps30 HPC static pressure", "s12": "phi fuel flow / Ps30", "s13": "NRf corrected fan speed",
    "s14": "NRc corrected core speed", "s15": "BPR bypass ratio", "s16": "farB burner fuel-air ratio",
    "s17": "htBleed bleed enthalpy", "s18": "Nf_dmd demanded fan speed", "s19": "PCNfR_dmd demanded corrected fan speed",
    "s20": "W31 HPT coolant bleed", "s21": "W32 LPT coolant bleed",
}


class DataValidationError(ValueError):
    """Raised when a frame violates a hard ingest rule (schema, duplicates, order)."""


@dataclass
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    name: str
    n_rows: int
    n_units: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def raise_if_fatal(self) -> None:
        if self.errors:
            raise DataValidationError(
                f"{self.name}: " + "; ".join(f"[{i.code}] {i.message}" for i in self.errors)
            )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "rows": self.n_rows,
            "units": self.n_units,
            "errors": [(i.code, i.count) for i in self.errors],
            "warnings": [(i.code, i.count) for i in self.warnings],
        }


def read_cmapss(path: str | Path) -> pd.DataFrame:
    """Parse one whitespace-separated C-MAPSS file (26 columns) preserving row order."""
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != len(COLUMNS):
        raise DataValidationError(
            f"{Path(path).name}: expected {len(COLUMNS)} columns, found {raw.shape[1]}"
        )
    raw.columns = COLUMNS
    raw["unit"] = raw["unit"].astype("int64")
    raw["cycle"] = raw["cycle"].astype("int64")
    for c in SETTINGS + SENSORS:
        raw[c] = raw[c].astype("float64")
    return raw


def read_rul(path: str | Path) -> pd.Series:
    """Official true RUL at the last observed cycle of each test engine (index = unit id)."""
    s = pd.read_csv(path, header=None)[0].astype("float64")
    s.index = np.arange(1, len(s) + 1)
    s.index.name = "unit"
    s.name = "rul"
    return s


def load_subset(raw_dir: str | Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_dir = Path(raw_dir)
    files = {k: raw_dir / f"{k}_{subset}.txt" for k in ("train", "test", "RUL")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "C-MAPSS files missing: " + ", ".join(missing) + "  (run `python -m selfheal download`)"
        )
    return read_cmapss(files["train"]), read_cmapss(files["test"]), read_rul(files["RUL"])


def train_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {c: (float(df[c].min()), float(df[c].max())) for c in SETTINGS + SENSORS}


def validate_frame(
    df: pd.DataFrame,
    name: str,
    bounds: dict[str, tuple[float, float]] | None = None,
    bounds_margin: float = 0.5,
) -> ValidationReport:
    """Schema / type / missingness / chronology / duplicate / range checks.

    Hard errors: wrong schema, non-numeric columns, duplicate (unit, cycle), out-of-order cycles,
    interleaved engines.  Warnings: cycle gaps, NaN/inf values, engines not starting at cycle 1,
    values outside `bounds` (train range widened by `bounds_margin`).
    """
    rep = ValidationReport(name=name, n_rows=len(df), n_units=int(df["unit"].nunique()) if "unit" in df else 0)
    if list(df.columns) != COLUMNS:
        rep.issues.append(Issue("error", "schema", f"columns differ from the 26-column C-MAPSS schema"))
        return rep
    bad_types = [c for c in COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_types:
        rep.issues.append(Issue("error", "dtype", f"non-numeric columns: {bad_types}", len(bad_types)))
        return rep

    vals = df[SETTINGS + SENSORS]
    n_inf = int(np.isinf(vals.to_numpy()).sum())
    if n_inf:
        rep.issues.append(Issue("warning", "inf", "infinite sensor values", n_inf))
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        rep.issues.append(Issue("warning", "missing", "NaN values present (handled causally in cleaning)", n_nan))
    if df[["unit", "cycle"]].isna().any().any():
        rep.issues.append(Issue("error", "id_missing", "unit/cycle contain NaN"))
        return rep
    if (df["cycle"] < 1).any() or (df["unit"] < 1).any():
        rep.issues.append(Issue("error", "id_range", "unit/cycle must be >= 1"))

    unit = df["unit"].to_numpy()
    cyc = df["cycle"].to_numpy()
    n_blocks = int(np.count_nonzero(np.diff(unit) != 0)) + 1 if len(unit) else 0
    if n_blocks != len(np.unique(unit)):
        rep.issues.append(Issue("error", "order", "engine rows are not contiguous (interleaved engines)"))
    same = unit[1:] == unit[:-1]
    d = np.diff(cyc)
    dup = int(np.count_nonzero(same & (d == 0)))
    back = int(np.count_nonzero(same & (d < 0)))
    gaps = int(np.count_nonzero(same & (d > 1)))
    if dup:
        rep.issues.append(Issue("error", "duplicate", "duplicate (unit, cycle) rows", dup))
    if back:
        rep.issues.append(Issue("error", "out_of_order", "cycles decrease within an engine", back))
    if gaps:
        rep.issues.append(Issue("warning", "gap", "cycle gaps inside an engine", gaps))
    starts = df.groupby("unit")["cycle"].min()
    if (starts != 1).any():
        rep.issues.append(Issue("warning", "start", "engines not starting at cycle 1", int((starts != 1).sum())))

    if bounds:
        n_out = 0
        for c, (lo, hi) in bounds.items():
            span = max(hi - lo, 1e-9)
            n_out += int(((df[c] < lo - bounds_margin * span) | (df[c] > hi + bounds_margin * span)).sum())
        if n_out:
            rep.issues.append(Issue("warning", "range", "values outside widened train range", n_out))
    return rep


class CycleValidator:
    """Streaming ingest check: reject duplicate / out-of-order cycles, report gaps (one cycle at a time)."""

    def __init__(self) -> None:
        self.last: dict[int, int] = {}

    def check(self, unit: int, cycle: int) -> tuple[bool, str, int]:
        """Return (accepted, reason, n_missing_cycles_before_this_one)."""
        prev = self.last.get(unit)
        if prev is None:
            self.last[unit] = cycle
            return True, "ok", 0
        if cycle == prev:
            return False, "duplicate", 0
        if cycle < prev:
            return False, "out_of_order", 0
        self.last[unit] = cycle
        return True, "ok", cycle - prev - 1

    def reset(self) -> None:
        self.last.clear()
