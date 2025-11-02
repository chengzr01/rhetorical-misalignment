# Persuasive Misalignment

LLM-driven principal-agent framework for clinical decision-making.

## Setup

```bash
pip install polars pyarrow openai pyyaml
export NVIDIA_API_KEY="your-key-here"  # For NVIDIA backend (default)
# OR
export OPENROUTER_API_KEY="your-key-here"  # For OpenRouter backend
```
## Dataset

+ Dataset: `/projects/bdhh/haopeng/physionet.org`
+ Tables: `hosp.patients`, `hosp.admissions`, `hosp.emar`, `hosp.emar_detail`, `hosp.labevents`, `hosp.d_labitems`
+ Quota: `/work/hdd/bdhh/physionet.org`

## Usage

+ Channel: `ssh -N -f -R 30001:localhost:30000 gpua000`