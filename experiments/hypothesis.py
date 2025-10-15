from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import yaml

from interface.client import OpenRouterChatClient, SGLangChatClient


class HypothesisGenerator:
    def __init__(
        self,
        data_dir: str | Path,
        client: OpenRouterChatClient | SGLangChatClient,
        model: str = "anthropic/claude-3.5-sonnet",
        prompt_path: str | Path = "prompts/experiments/hypothesis_generator.yaml",
    ):
        self.data_dir = Path(data_dir)
        self.client = client
        self.model = model
        self.patients = pl.read_parquet(self.data_dir / "patients.parquet")
        self.medications = pl.read_parquet(self.data_dir / "medications.parquet")
        self.labs = pl.read_parquet(self.data_dir / "labs.parquet")

        # Load prompt template from YAML file
        with open(prompt_path, "r") as f:
            prompt_config = yaml.safe_load(f)
            self.prompt_template = prompt_config["prompt"]

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

        prompt = self.prompt_template.format(
            patient_id=patient_info["patient_id"],
            sex=patient_info["sex"],
            age=patient_info["age"],
            lab_summary=lab_summary,
            drug=drug,
            lab_test=lab_test,
        )

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
            .filter(pl.col("drug").is_not_null() & pl.col("lab_test").is_not_null())
            .group_by(["drug", "lab_test"])
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") >= min_episodes)
            .sort("count", descending=True)
        )

        return [
            (row["drug"], row["lab_test"])
            for row in drug_lab_effects.iter_rows(named=True)
        ]

    def select_patients_for_drug(self, drug: str, n_patients: int = 5) -> list[int]:
        patient_ids = (
            self.medications.filter(pl.col("drug").str.contains(f"(?i){drug}"))
            .select("subject_id")
            .unique()
            .head(n_patients)["subject_id"]
            .to_list()
        )
        return patient_ids
