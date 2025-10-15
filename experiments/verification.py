from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl


class DataVerifier:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.patients = pl.read_parquet(self.data_dir / "patients.parquet")
        self.medications = pl.read_parquet(self.data_dir / "medications.parquet")
        self.labs = pl.read_parquet(self.data_dir / "labs.parquet")

    def analyze_treatment_effect(
        self,
        drug: str,
        lab_test: str,
        sex: str | None = None,
        age_range: tuple[int, int] | None = None,
    ) -> Mapping[str, Any]:
        meds_filtered = self.medications.filter(
            pl.col("drug").str.contains(f"(?i){drug}")
        )
        labs_filtered = self.labs.filter(
            pl.col("lab_test").str.contains(f"(?i){lab_test}")
        )

        if sex:
            meds_filtered = meds_filtered.filter(pl.col("sex") == sex)
            labs_filtered = labs_filtered.filter(pl.col("sex") == sex)

        if age_range:
            min_age, max_age = age_range
            meds_filtered = meds_filtered.filter(
                (pl.col("anchor_age") >= min_age) & (pl.col("anchor_age") <= max_age)
            )
            labs_filtered = labs_filtered.filter(
                (pl.col("anchor_age") >= min_age) & (pl.col("anchor_age") <= max_age)
            )

        if len(meds_filtered) == 0 or len(labs_filtered) == 0:
            return self._empty_result()

        combined = (
            meds_filtered.join(
                labs_filtered, on=["subject_id", "hadm_id"], how="inner", suffix="_lab"
            )
            .with_columns(
                [
                    pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
                    pl.col("charttime_lab").str.strptime(
                        pl.Datetime, "%Y-%m-%d %H:%M:%S"
                    ),
                ]
            )
            .with_columns(
                [(pl.col("charttime_lab") - pl.col("charttime")).alias("time_diff_ns")]
            )
        )

        group_cols = [
            "subject_id",
            "hadm_id",
            "drug",
            "charttime",
            "lab_test",
            "sex",
            "anchor_age",
        ]

        labs_before = (
            combined.filter(pl.col("time_diff_ns") < datetime.timedelta(0))
            .sort(
                [*group_cols, "time_diff_ns"],
                descending=[False] * len(group_cols) + [True],
            )
            .group_by(group_cols)
            .agg(
                [
                    pl.col("valuenum").first().alias("value_before"),
                    pl.col("time_diff_ns").first().alias("time_before"),
                ]
            )
        )

        labs_after = (
            combined.filter(pl.col("time_diff_ns") > datetime.timedelta(0))
            .sort([*group_cols, "time_diff_ns"])
            .group_by(group_cols)
            .agg(
                [
                    pl.col("valuenum").first().alias("value_after"),
                    pl.col("time_diff_ns").first().alias("time_after"),
                ]
            )
        )

        effects = labs_before.join(labs_after, on=group_cols, how="inner").with_columns(
            [
                (pl.col("value_after") - pl.col("value_before")).alias("value_change"),
                (
                    (pl.col("value_after") - pl.col("value_before"))
                    / pl.col("value_before")
                    * 100
                ).alias("pct_change"),
            ]
        )

        if len(effects) == 0:
            return self._empty_result()

        summary = (
            effects.group_by(["drug", "lab_test"])
            .agg(
                [
                    pl.count().alias("n_episodes"),
                    pl.col("value_before").mean().alias("avg_value_before"),
                    pl.col("value_after").mean().alias("avg_value_after"),
                    pl.col("value_change").mean().alias("avg_change"),
                    pl.col("pct_change").mean().alias("avg_pct_change"),
                    pl.col("value_change").std().alias("std_change"),
                ]
            )
            .to_dicts()[0]
        )

        positive_effects = len(effects.filter(pl.col("value_change") < 0))
        negative_effects = len(effects.filter(pl.col("value_change") >= 0))

        return {
            "drug": summary["drug"],
            "lab_test": summary["lab_test"],
            "n_episodes": summary["n_episodes"],
            "avg_value_before": round(summary["avg_value_before"], 2),
            "avg_value_after": round(summary["avg_value_after"], 2),
            "avg_change": round(summary["avg_change"], 2),
            "avg_pct_change": round(summary["avg_pct_change"], 2),
            "std_change": round(summary["std_change"], 2),
            "positive_outcomes": positive_effects,
            "negative_outcomes": negative_effects,
            "success_rate": round(positive_effects / (positive_effects + negative_effects) * 100, 1),
        }

    def _empty_result(self) -> Mapping[str, Any]:
        return {
            "drug": None,
            "lab_test": None,
            "n_episodes": 0,
            "avg_value_before": 0,
            "avg_value_after": 0,
            "avg_change": 0,
            "avg_pct_change": 0,
            "std_change": 0,
            "positive_outcomes": 0,
            "negative_outcomes": 0,
            "success_rate": 0,
        }

    def generate_data_report(
        self,
        drug: str,
        lab_test: str,
        sex: str | None = None,
        age_range: tuple[int, int] | None = None,
    ) -> str:
        stats = self.analyze_treatment_effect(drug, lab_test, sex, age_range)

        if stats["n_episodes"] == 0:
            return "Insufficient data available for analysis."

        demographics = []
        if sex:
            demographics.append(f"Sex: {sex}")
        if age_range:
            demographics.append(f"Age range: {age_range[0]}-{age_range[1]} years")
        demo_str = ", ".join(demographics) if demographics else "All patients"

        report = f"""Historical Data Report: {stats['drug']} Effect on {stats['lab_test']}

Patient Demographics: {demo_str}

Sample Size: {stats['n_episodes']} treatment episodes

Treatment Effects:
- Average {stats['lab_test']} before treatment: {stats['avg_value_before']}
- Average {stats['lab_test']} after treatment: {stats['avg_value_after']}
- Average change: {stats['avg_change']} ({stats['avg_pct_change']:.1f}%)
- Standard deviation of change: {stats['std_change']}

Outcomes:
- Positive outcomes (value decreased): {stats['positive_outcomes']} episodes ({stats['success_rate']}%)
- Negative outcomes (value increased): {stats['negative_outcomes']} episodes ({100 - stats['success_rate']:.1f}%)"""

        return report
