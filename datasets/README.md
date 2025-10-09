# Minimum MIMIC-IV Treatment Effectiveness Explorer

Two simple scripts to explore AI-assisted decision-making with MIMIC-IV data.

## Tables Used
- `hosp.patients` - Patient demographics
- `hosp.admissions` - Hospital admissions
- `hosp.emar` - Medication administrations
- `hosp.emar_detail` - Medication details
- `hosp.labevents` - Lab test results
- `hosp.d_labitems` - Lab test definitions

## Quick Start

### 1. Load the data
```bash
python load.py /projects/bdhh/haopeng/physionet.org ./output
```

This creates 4 parquet files:
- `patients.parquet` - Patient info
- `admissions.parquet` - Admission records
- `medications.parquet` - Drugs given (with patient demographics)
- `labs.parquet` - Lab test results (with patient demographics)

### 2. Explore treatment effectiveness

Show top drugs and lab tests:
```bash
python explore.py ./output
```

Analyze specific drug effects on lab values:
```bash
# Example: How does insulin affect glucose?
python explore.py ./output insulin glucose

# Example: How does heparin affect platelets?
python explore.py ./output heparin platelet
```

This shows:
- Number of episodes where we have before/after lab measurements
- Average change in lab values
- Time windows (hours before/after medication)

## Dependencies
```bash
pip install polars pyarrow
```

## What the scripts do

**simple_load.py**:
- Loads 6 CSV tables
- Joins medications with details to get drug names
- Joins labs with definitions to get test names
- Adds patient demographics
- Saves as parquet files

**explore_treatments.py**:
- Loads parquet files
- For each medication, finds lab values before and after
- Calculates change in lab values
- Summarizes effectiveness by drug and lab test

---

# Advanced: Full Pipeline (Optional)

If you need more complex analytics, there's a full pipeline available with dimension/fact tables.
See `pipeline.py` and related modules for details.
