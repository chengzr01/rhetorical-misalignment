from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from interface.client import OpenRouterChatClient, SGLangChatClient, NvidiaChatClient


def setup_client(backend: str) -> OpenRouterChatClient | SGLangChatClient | NvidiaChatClient:
    """Setup the LLM client based on backend choice."""
    if backend == "nvidia":
        return NvidiaChatClient()
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=30000, base_url="http://127.0.0.1")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def fetch_abstract_from_ncbi(pmid: str, email: Optional[str] = None,
                              api_key: Optional[str] = None,
                              sleep_sec: float = 0.34) -> Optional[str]:
    """Fetch abstract text for a given PMID from NCBI."""
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }

    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    try:
        r = requests.get(efetch_url, params=params, timeout=60)
        time.sleep(sleep_sec)

        if r.status_code != 200:
            return None

        # Parse XML to extract abstract
        root = ET.fromstring(r.text)
        article = root.find(".//MedlineCitation/Article")

        if article is None:
            return None

        # Extract abstract text
        abstract_texts = []
        for abst in article.findall("Abstract/AbstractText"):
            text = "".join(abst.itertext()).strip()
            if text:
                label = abst.attrib.get("Label") if isinstance(abst.attrib, dict) else None
                if label:
                    text = f"{label}: {text}"
                abstract_texts.append(text)

        if abstract_texts:
            return "\n\n".join(abstract_texts)

        return None

    except Exception as e:
        print(f"Error fetching abstract for PMID {pmid}: {e}")
        return None


def parse_age(age_str: str) -> str:
    """Parse age from the dataset format."""
    try:
        age_list = eval(age_str) if isinstance(age_str, str) else age_str
        if isinstance(age_list, list) and len(age_list) > 0:
            value, unit = age_list[0]
            return f"{value} {unit}s"
        return "Unknown"
    except:
        return "Unknown"


def parse_relevant_articles(articles_str: str) -> list[tuple[str, int]]:
    """Parse relevant articles dict and return sorted list of (PMID, score)."""
    try:
        articles = eval(articles_str) if isinstance(articles_str, str) else articles_str
        if isinstance(articles, dict):
            # Sort by score descending
            sorted_articles = sorted(articles.items(), key=lambda x: x[1], reverse=True)
            return sorted_articles
        return []
    except:
        return []


def load_prompt_template(template_path: str | Path) -> str:
    """Load prompt template from YAML file."""
    with open(template_path, "r") as f:
        template_config = yaml.safe_load(f)
        return template_config["prompt"]


def generate_hypothesis(
    client: OpenRouterChatClient | SGLangChatClient | NvidiaChatClient,
    model: str,
    prompt_template: str,
    patient_summary: str,
    age: str,
    gender: str,
    paper_abstract: str,
    temperature: float = 0.7,
) -> Optional[str]:
    """Generate treatment hypothesis using LLM."""

    prompt = prompt_template.format(
        patient_summary=patient_summary,
        age=age,
        gender=gender,
        paper_abstract=paper_abstract,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.create_completion(
            model=model, messages=messages, temperature=temperature
        )
        return response
    except Exception as e:
        print(f"Error generating hypothesis: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hypotheses from PMC patient dataset"
    )
    parser.add_argument("--server", type=str, default="nvidia",
                       help="LLM backend: nvidia, openrouter, or sglang")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-v3.1",
                       help="Model name to use for hypothesis generation")
    parser.add_argument("--pmc-csv", type=str,
                       default="datasets/PMC/raw/PMC-Patients.csv",
                       help="Path to PMC-Patients.csv")
    parser.add_argument("--output", type=str,
                       default="experiments/input/hypothesis_pmc.json",
                       help="Output path for hypothesis JSON")
    parser.add_argument("--n-hypotheses", type=int, default=100,
                       help="Number of hypotheses to generate")
    parser.add_argument("--email", type=str,
                       default=os.getenv("NCBI_EMAIL", "ziruic4@illinois.edu"),
                       help="Email for NCBI API")
    parser.add_argument("--api-key", type=str,
                       default=os.getenv("NCBI_API_KEY"),
                       help="NCBI API key for higher rate limits")
    parser.add_argument("--sleep", type=float, default=0.34,
                       help="Sleep seconds between NCBI requests")
    parser.add_argument("--prompt-template", type=str,
                       default="prompts/experiments/pmc_hypothesis_generator.yaml",
                       help="Path to prompt template YAML")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for patient sampling")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Setup LLM client
    print(f"Setting up {args.server} client with model {args.model}")
    client = setup_client(args.server)

    # Load prompt template
    print(f"Loading prompt template from {args.prompt_template}")
    prompt_template = load_prompt_template(args.prompt_template)

    # Load PMC patient dataset
    print(f"Loading PMC patient dataset from {args.pmc_csv}")
    df = pd.read_csv(args.pmc_csv)
    print(f"Total patients in dataset: {len(df)}")

    # Sample patients
    if len(df) > args.n_hypotheses:
        df_sample = df.sample(n=args.n_hypotheses, random_state=args.random_seed)
    else:
        df_sample = df

    print(f"Processing {len(df_sample)} patients")

    hypotheses = []
    successful = 0

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        try:
            patient_id = row["patient_id"]
            patient_uid = row["patient_uid"]
            patient_summary = row["patient"]
            age = parse_age(row["age"])
            gender = row["gender"]

            # Get relevant articles
            relevant_articles = parse_relevant_articles(row["relevant_articles"])

            if not relevant_articles:
                print(f"No relevant articles found for patient {patient_id}")
                continue

            print(f"\nProcessing patient {patient_id} (UID: {patient_uid})")
            print(f"Age: {age}, Gender: {gender}")
            print(f"Found {len(relevant_articles)} relevant articles")

            # Try to fetch abstracts for multiple papers (or at least one)
            paper_abstracts = []
            pmids_fetched = []

            for pmid, score in relevant_articles[:5]:  # Try up to 5 most relevant
                print(f"Fetching abstract for PMID: {pmid} (score: {score})")

                abstract = fetch_abstract_from_ncbi(
                    pmid,
                    email=args.email,
                    api_key=args.api_key,
                    sleep_sec=args.sleep
                )

                if abstract:
                    paper_abstracts.append(abstract)
                    pmids_fetched.append(pmid)
                    print(f"  Successfully fetched (length: {len(abstract)} chars)")
                else:
                    print(f"  Failed to fetch abstract")

                # If we have at least one abstract, we can proceed
                # Continue trying to get more if available
                if len(paper_abstracts) >= 3:  # Limit to 3 abstracts per patient
                    break

            if not paper_abstracts:
                print(f"Failed to fetch any abstracts for patient {patient_id}")
                continue

            # Combine multiple abstracts if available
            combined_abstract = "\n\n---\n\n".join([
                f"Paper {i+1} (PMID: {pmid}):\n{abstract}"
                for i, (pmid, abstract) in enumerate(zip(pmids_fetched, paper_abstracts))
            ])

            print(f"Total abstracts fetched: {len(paper_abstracts)}")
            print(f"Combined abstract length: {len(combined_abstract)} characters")

            # Generate hypothesis
            print("Generating hypothesis...")
            hypothesis = generate_hypothesis(
                client=client,
                model=args.model,
                prompt_template=prompt_template,
                patient_summary=patient_summary,
                age=age,
                gender=gender,
                paper_abstract=combined_abstract,
            )

            if not hypothesis:
                print("Failed to generate hypothesis")
                continue

            print(f"Generated hypothesis: {hypothesis[:100]}...")

            # Create hypothesis entry (no text field, no agent/principal configs)
            hypotheses.append({
                "patient_id": patient_id,
                "patient_uid": patient_uid,
                "age": age,
                "gender": gender,
                "patient_summary": patient_summary,
                "pmids": pmids_fetched,
                "paper_abstracts": paper_abstracts,
                "hypothesis": hypothesis,
            })

            successful += 1
            print(f"Successfully processed {successful}/{args.n_hypotheses} patients")

            if successful >= args.n_hypotheses:
                break

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(hypotheses, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Successfully generated {len(hypotheses)} hypotheses")
    print(f"Saved to {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
