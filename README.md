# 🎓 Intern Performance & Risk Prediction System
> **Internee.pk Technical Project** | Production-Grade Machine Learning & Predictive Intervention Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Model-XGBoost%20%7C%20Random%20Forest-orange.svg)
![Deployment](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

---

## 📌 Executive Summary
Managing large cohorts of virtual interns requires early identification of candidate needs. The **Intern Performance & Risk Prediction System** analyzes operational telemetry (task turnaround, attendance, mentor feedback, collaboration) to accurately forecast continuous performance (0–100) and triage candidates into clear operational tiers before final grading.

---

## 🏗️ System Architecture & Workflow

```text
├── data/
│   └── intern_data.csv               # Synthetic telemetry dataset (600 records)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb     # Exploratory analysis, feature impact & benchmarking
├── models/
│   └── best_model.pkl                # Serialized XGBoost pipeline & metadata
├── app.py                            # Interactive Streamlit triage web dashboard
├── data_generator.py                 # Domain-constrained data synthesis engine
├── train.py                          # Automated training & artifact serialization pipeline
├── requirements.txt                  # Environment dependencies
└── README.md                         # Comprehensive documentation