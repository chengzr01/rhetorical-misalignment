#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given one or more PMIDs, this script:
  1) Fetches article metadata and the abstract via NCBI E-utilities (EFetch, retmode=xml).
  2) Saves the abstract to {PMID}.abstract.txt (and basic metadata to {PMID}.meta.json).
  3) Converts PMID -> PMCID using the PMC idconv service.
  4) If a PMCID exists and the article is in PubMed Central, downloads the PDF from PMC to {PMCID}.pdf.
  5) Writes a CSV log with outcomes for each PMID.

Usage:
  python download.py --pmids 37252113 37252114 --outdir processed/papers --email ziruic4@illinois.edu
  python download.py --pmid-file pmids.txt --outdir processed/papers --api-key <NCBI_API_KEY>

Notes:
  - Set an API key for higher rate limits: --api-key or environment variable NCBI_API_KEY
  - Provide an email and a tool name per NCBI policy: --email, --tool or env NCBI_EMAIL, NCBI_TOOL
  - Be considerate with rate limiting (the script sleeps between requests).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import requests

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

def ncbi_params(email: Optional[str], tool: Optional[str], api_key: Optional[str]) -> Dict[str, str]:
    p = {}
    if email:
        p["email"] = email
    if tool:
        p["tool"] = tool
    if api_key:
        p["api_key"] = api_key
    return p

def fetch_pubmed_xml(pmid: str, email: Optional[str], tool: Optional[str], api_key: Optional[str], sleep_sec: float) -> Optional[str]:
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    params.update(ncbi_params(email, tool, api_key))
    r = requests.get(EFETCH_URL, params=params, timeout=60)
    time.sleep(sleep_sec)
    if r.status_code == 200:
        return r.text
    return None

def parse_abstract_from_pubmed_xml(xml_text: str) -> Dict[str, Optional[str]]:
    """
    Returns dict with keys: title, abstract (joined paragraphs), journal, year
    """
    out = {"title": None, "abstract": None, "journal": None, "year": None}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    ns = {}  # default namespace not used

    # Find the first MedlineCitation -> Article
    article = root.find(".//MedlineCitation/Article", ns)
    if article is None:
        return out

    # Title
    atitle = article.findtext("ArticleTitle")
    if atitle:
        out["title"] = "".join(atitle.split()) if "\n" in atitle else atitle

    # Journal & year
    journal_title = article.findtext("Journal/Title")
    out["journal"] = journal_title or None
    pub_year = article.findtext("Journal/JournalIssue/PubDate/Year")
    if not pub_year:
        # Some records may use MedlineDate like "2020 Jan-Feb"
        pub_year = article.findtext("Journal/JournalIssue/PubDate/MedlineDate")
        if pub_year and len(pub_year) >= 4 and pub_year[:4].isdigit():
            pub_year = pub_year[:4]
    out["year"] = pub_year or None

    # Abstract can have multiple sections
    abstract_texts = []
    for abst in article.findall("Abstract/AbstractText"):
        # Abst.text may be None if it has nested tags; join all inner text
        text = "".join(abst.itertext()).strip()
        if text:
            # If the AbstractText has a Label attribute, prepend it.
            label = abst.attrib.get("Label") if isinstance(abst.attrib, dict) else None
            if label:
                text = f"{label}: {text}"
            abstract_texts.append(text)

    if abstract_texts:
        out["abstract"] = "\n\n".join(abstract_texts)

    return out

def pmid_to_pmcid(pmid: str, email: Optional[str], tool: Optional[str], api_key: Optional[str], sleep_sec: float) -> Optional[str]:
    params = {"ids": pmid, "format": "json"}
    params.update(ncbi_params(email, tool, api_key))
    r = requests.get(IDCONV_URL, params=params, timeout=60)
    time.sleep(sleep_sec)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
        recs = data.get("records", [])
        if recs:
            pmcid = recs[0].get("pmcid")
            return pmcid
    except Exception:
        return None
    return None

def download_pmc_pdf(pmcid: str, outdir: Path, sleep_sec: float) -> Optional[Path]:
    # Generic PMC pdf URL (will 302 to the exact filename)
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    r = requests.get(pdf_url, allow_redirects=True, timeout=120)
    time.sleep(sleep_sec)
    if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
        # Derive a decent filename
        fname = f"{pmcid}.pdf"
        fpath = outdir / fname
        with open(fpath, "wb") as f:
            f.write(r.content)
        return fpath
    # Sometimes the content-type isn't set yet still returns a PDF on redirect; try anyway
    if r.status_code == 200 and r.content and len(r.content) > 1024:
        fname = f"{pmcid}.pdf"
        fpath = outdir / fname
        with open(fpath, "wb") as f:
            f.write(r.content)
        return fpath
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", nargs="+", help="List of PMIDs")
    ap.add_argument("--pmid-file", help="Path to a file with one PMID per line")
    ap.add_argument("--outdir", default="downloads", help="Output directory")
    ap.add_argument("--email", default=os.getenv("NCBI_EMAIL"), help="Contact email for NCBI")
    ap.add_argument("--tool", default=os.getenv("NCBI_TOOL", "pmid_downloader"), help="Tool name for NCBI")
    ap.add_argument("--api-key", default=os.getenv("NCBI_API_KEY"), help="NCBI API key for higher rate limits")
    ap.add_argument("--sleep", type=float, default=0.34, help="Sleep seconds between requests (increase if no API key)")
    args = ap.parse_args()

    # Collect PMIDs
    pmids: List[str] = []
    if args.pmids:
        pmids.extend([str(x).strip() for x in args.pmids if str(x).strip()])
    if args.pmid_file:
        with open(args.pmid_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    pmids.append(s)
    pmids = list(dict.fromkeys(pmids))  # de-duplicate, preserve order
    if not pmids:
        print("No PMIDs provided. Use --pmids or --pmid-file.", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Log CSV
    log_path = outdir / "download_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        w = csv.writer(logf)
        w.writerow(["pmid","pmcid","abstract_saved","pdf_saved","title","journal","year","note"])

        for pmid in pmids:
            note = ""
            title = journal = year = ""
            abstract_saved = pdf_saved = False
            pmcid = None

            # 1) Fetch XML & parse abstract
            try:
                xml_text = fetch_pubmed_xml(pmid, args.email, args.tool, args.api_key, args.sleep)
                if xml_text:
                    meta = parse_abstract_from_pubmed_xml(xml_text)
                    title = meta.get("title") or ""
                    journal = meta.get("journal") or ""
                    year = meta.get("year") or ""

                    abstract = meta.get("abstract")
                    if abstract:
                        # Save abstract
                        abs_path = outdir / f"{pmid}.abstract.txt"
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(abstract)
                        abstract_saved = True

                    # Save meta JSON
                    meta_path = outdir / f"{pmid}.meta.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                else:
                    note += "EFetch failed; "
            except Exception as e:
                note += f"EFetch error: {e}; "

            # 2) PMID -> PMCID
            try:
                pmcid = pmid_to_pmcid(pmid, args.email, args.tool, args.api_key, args.sleep)
            except Exception as e:
                note += f"idconv error: {e}; "

            # 3) Download PDF if PMCID is available
            if pmcid:
                try:
                    pdf_path = download_pmc_pdf(pmcid, outdir, args.sleep)
                    if pdf_path:
                        pdf_saved = True
                    else:
                        note += "PMC PDF not available; "
                except Exception as e:
                    note += f"PDF error: {e}; "
            else:
                note += "No PMCID (not in PMC or restricted); "

            w.writerow([pmid, pmcid or "", abstract_saved, pdf_saved, title, journal, year, note.strip()])

    print(f"Done. Log written to: {log_path}")
    print(f"Outputs saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()