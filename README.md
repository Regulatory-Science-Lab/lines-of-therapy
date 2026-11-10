# Lines-of-Therapy Algorithm Transportability

This repository contains code supporting a study evaluating the cross-jurisdictional transportability of two algorithms for determining lines of systemic therapy in oncology: a rule-based cyclic algorithm and a BERT-based model.

The algorithms were evaluated using real-world data from two datasets, across tumour-specific and pooled cohorts.

# Input Data Structure

The analysis requires harmonized, medication-level datasets containing patient identifiers, tumour classifications, treatment characteristics, clinical dates, and reference lines of therapy. Required variables are specified in config/config.yaml. Missing values are allowed where a feature is unavailable and are encoded as missing tokens. Preprocessing is performed separately before this analysis.

# Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── config/
│   └── config.yaml
├── python/ – BERT training, application, and evaluation
│   ├── __init__.py
│   ├── common.py
│   ├── cohorts.py
│   ├── pipeline.py
│   ├── train_bert.py
│   ├── apply_bert.py
│   ├── evaluate.py
│   └── README.md
├── tables/ – Generation of manuscript tables
│   └── generate.py
└── R/ – Cyclic algorithm and evaluation
    ├── cyclic_algorithm.R
    ├── run_cyclic.R
    ├── evaluate_cyclic.R
    └── README.md
```

# Usage

Python 3.10–3.12 is recommended. 

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Update config/config.yaml with the appropriate input and output paths, then run:

```bash
python -m python.pipeline --config config/config.yaml
```

# Data Availability

Patient-level data are not included in this repository due to privacy and data-access restrictions.