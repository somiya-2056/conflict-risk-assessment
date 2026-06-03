# 🛡️ AI-Based Geopolitical Conflict Risk Assessment System
### DRDO Internship Project | India (Base Country) vs Pakistan & China

---

## 📌 Project Overview

This project develops an **AI-powered conflict risk assessment framework** to estimate regional escalation probabilities involving **India, Pakistan, and China** using historical conflict, economic, military, and geographical indicators.

> ⚠️ This is a **research/academic project** built on open-source data. All predictions are for strategic analysis purposes only.

---

## 🎯 Objectives

- Predict conflict risk level (LOW / MEDIUM / HIGH) between India-Pakistan and India-China
- Analyze key risk drivers using machine learning
- Visualize historical military and economic trends (1947–2024)
- Estimate war probability, winning probability, and diplomatic compromise chances

---

## 📊 Data Sources

| Dataset | Source | What it contains |
|---------|--------|-----------------|
| Military Expenditure | [SIPRI](https://sipri.org/databases/milex) | Defence budgets 1949–2025 |
| GDP & Trade Data | [World Bank](https://data.worldbank.org) | Economic indicators |
| Conflict Events | [UCDP GED v25](https://ucdp.uu.se/downloads) | Historical conflict events 1989–2024 |
| Historical Dataset | Custom built | 1947–2024 India-Pak & India-China scenarios |

---

## 🤖 ML Models Used

### 1. Random Forest Classifier
- 200 decision trees
- Trained on 1947–2024 historical conflict data
- Features: Military ratio, Nuclear weapons, Border tensions, Diplomatic relations, etc.

### 2. Gradient Boosting Classifier  
- 150 estimators
- Learns from previous errors iteratively
- Combined with Random Forest for ensemble prediction

### Why these models?
- Work well with small datasets (57–150 rows)
- Interpretable — provide feature importance
- No need for large compute resources
- Suitable for defence research prototypes

---

## 🗂️ Project Structure

```
DRDO-Conflict-Risk-Assessment/
│
├── dashboard.py                    # Main Streamlit dashboard
├── model.py                        # ML model training script
├── test.py                         # Dummy testing & validation
│
├── data/
│   ├── ml_dataset.csv              # Training dataset
│   ├── ml_dataset_v2.csv           # Updated dataset with UCDP data
│   ├── historical_dataset_1947_2024.csv  # Full historical dataset
│   ├── sipri_full.csv              # SIPRI military data 1949–2025
│   ├── combined_data.csv           # Combined World Bank + SIPRI
│   └── expanded_dataset.csv        # Feature-rich dataset
│
├── models/
│   └── conflict_model.pkl          # Saved trained model
│
└── outputs/
    ├── model_analysis.png          # Feature importance + risk trend graph
    └── feature_importance.png      # Feature importance chart
```

---

## ⚙️ Features Used in Model

| Feature | Description |
|---------|-------------|
| Military_Ratio | India's military power vs enemy |
| Nuclear_Weapons | Enemy nuclear arsenal count |
| Historical_Wars | Number of past wars |
| Kashmir_Tension | LOC tension score (0–10) |
| LAC_Tension | India-China border tension (0–10) |
| Cyber_Attack_Risk | Cyber threat level (0–10) |
| Trade_Dependency | Economic interdependence (0–10) |
| Diplomatic_Relations | Quality of relations (0–10) |
| Land_Attack_Risk | Ground attack probability (0–10) |
| Naval_Attack_Risk | Maritime threat level (0–10) |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Random Forest Train Accuracy | ~97% |
| Random Forest Test Accuracy | ~85% |
| Cross Validation (5-fold) | ~80% avg |
| Dummy Test Score | 5–6/7 scenarios |

> **Limitation:** Dataset is relatively small (open-source data). Classified intelligence data would significantly improve accuracy.

---

## 🖥️ Dashboard Features

- **Historical graphs** — Military expenditure & GDP (1947–2025)
- **Scenario inputs** — Adjust 10 geopolitical indicators via sliders
- **Conflict prediction** — War probability, full-scale war risk
- **Win probability** — India vs enemy if conflict occurs
- **Attack zone analysis** — Land, Naval, Cyber, Air risk
- **Future forecast** — 2025–2030 war probability trend
- **Strategic analysis** — Key flashpoints, compromise chances

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn plotly openpyxl
```

### 2. Run the dashboard
```bash
streamlit run dashboard.py
```

### 3. Train the model (optional)
```bash
python model.py
```

### 4. Run dummy tests
```bash
python test.py
```

---

## 🔬 Key Findings

1. **Land Attack Risk** and **Diplomatic Relations** are the top conflict risk drivers
2. **India-Pakistan** risk consistently higher than India-China (4 wars vs 1)
3. **Nuclear deterrence** prevents full-scale war despite high tension
4. **China's military spending** has grown 7x since 2005 — widening the gap with India
5. **Trade interdependence** with China acts as a partial conflict deterrent

---

## ⚠️ Limitations

- Open-source datasets only — no classified intelligence data
- Small dataset (150 rows) — may cause slight overfitting
- China conflict data limited in UCDP (only 39 events vs India's 17,997)
- Predictions are probabilistic estimates, not definitive forecasts

---

## 👩‍💻 Author

**Somiya Aggarwal**  
DRDO Internship Project  
Data Sources: SIPRI · World Bank · UCDP  
For academic/research purposes only