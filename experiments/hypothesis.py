from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from interface.client import OpenRouterChatClient, SGLangChatClient


class HypothesisGenerator:
    def __init__(
        self,
        data_dir: str | Path,
        client: OpenRouterChatClient | SGLangChatClient,
        model: str = "anthropic/claude-3.5-sonnet",
    ):
        self.data_dir = Path(data_dir)
        self.client = client
        self.model = model
        self.patients = pl.read_parquet(self.data_dir / "patients.parquet")
        self.medications = pl.read_parquet(self.data_dir / "medications.parquet")
        self.labs = pl.read_parquet(self.data_dir / "labs.parquet")

    def get_patient_info(self, patient_id: int) -> Mapping[str, Any] | None:
        patient = self.patients.filter(pl.col("subject_id") == patient_id)
        if len(patient) == 0:
            return None
        return {
            "patient_id": patient_id,
            "sex": patient["sex"][0],
            "age": patient["anchor_age"][0],
        }

    def get_patient_medications(self, patient_id: int) -> pl.DataFrame:
        return (
            self.medications.filter(pl.col("subject_id") == patient_id)
            .filter(pl.col("drug").is_not_null())
            .select(["drug", "charttime"])
            .unique(subset=["drug"])
        )

    def get_patient_labs(self, patient_id: int, limit: int = 5) -> pl.DataFrame:
        return (
            self.labs.filter(pl.col("subject_id") == patient_id)
            .select(["lab_test", "valuenum", "valueuom"])
            .sort("lab_test")
            .head(limit)
        )

    def generate_hypothesis(
        self, patient_id: int, drug: str, lab_test: str
    ) -> str | None:
        patient_info = self.get_patient_info(patient_id)
        if not patient_info:
            return None

        patient_labs = self.get_patient_labs(patient_id)
        lab_summary = "\n".join(
            [
                f"  - {row['lab_test']}: {row['valuenum']} {row['valueuom']}"
                for row in patient_labs.iter_rows(named=True)
            ]
        )

        prompt = f"""You are a medical expert tasked with generating a clinical hypothesis.

Patient Information:
- Patient ID: {patient_info['patient_id']}
- Sex: {patient_info['sex']}
- Age: {patient_info['age']} years old

Recent Lab Values:
{lab_summary}

Question: Should this patient receive {drug} to manage their {lab_test} levels?

Generate a clear, concise clinical hypothesis about whether this treatment would be beneficial for this patient. Consider the patient's demographics and current lab values. Format your response as a single paragraph hypothesis statement."""

        messages = [{"role": "user", "content": prompt}]
        response = self.client.create_completion(
            model=self.model, messages=messages, temperature=0.7
        )
        return response

    def get_drug_lab_pairs(self, min_episodes: int = 10) -> list[tuple[str, str]]:
        drug_lab_effects = (
            self.medications.join(
                self.labs, on=["subject_id", "hadm_id"], how="inner", suffix="_lab"
            )
            .group_by(["drug", "lab_test"])
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") >= min_episodes)
            .sort("count", descending=True)
        )

        return [
            (row["drug"], row["lab_test"])
            for row in drug_lab_effects.iter_rows(named=True)
        ]

    def select_patients_for_drug(
        self, drug: str, n_patients: int = 5
    ) -> list[int]:
        patient_ids = (
            self.medications.filter(pl.col("drug").str.contains(f"(?i){drug}"))
            .select("subject_id")
            .unique()
            .head(n_patients)["subject_id"]
            .to_list()
        )
        return patient_ids
