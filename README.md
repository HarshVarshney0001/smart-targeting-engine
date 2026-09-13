# 🎯 Smart Targeting Engine

**Causal AI for Marketing — Uplift Modeling on the Hillstrom E-Mail Dataset**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![CausalML](https://img.shields.io/badge/CausalML-Uplift%20Modeling-green)

**🔗 Live Demo:** [https://smart-targeting-engine.streamlit.app/](https://smart-targeting-engine.streamlit.app/)

---

## 📌 Problem Statement

Most marketing ML projects answer the wrong question: *"Which customers will convert?"*

This project answers the question that actually matters for a marketing budget: **"Which customers will convert *because* of the marketing, and not otherwise?"**

A standard classification model can't tell the difference between:
- A customer who buys **because** they got an email (marketing worked)
- A customer who was going to buy **anyway** (marketing budget wasted)
- A customer who is put off by the email and buys **less** because of it (marketing actively hurt)

**Uplift modeling** (causal inference) solves this by estimating the *individual treatment effect* of a campaign on each customer, not just their raw probability of converting. This project builds a complete, deployable uplift-modeling pipeline — from raw data to a live, interactive targeting tool.

---

## 🧠 What Is Uplift Modeling? (4-Segment Framework)

Every customer falls into one of four behavioral segments, based on how they'd behave *with* vs *without* the marketing email:

| Segment | Behavior | Action |
|---|---|---|
| **Persuadables** | Converts *only if* targeted | ✅ Target — this is where marketing budget creates real value |
| **Sure Things** | Converts *regardless* of targeting | ⚪ Skip — targeting them wastes budget |
| **Lost Causes** | Never converts, targeted or not | ⚪ Skip — no value either way |
| **Sleeping Dogs** | Converts *less* if targeted (gets annoyed) | ⛔ Avoid — targeting actively hurts |

A classification model can only tell you "will convert or not." An uplift model tells you "does *targeting* change that outcome" — which is the number that actually drives ROI.

---

## 🏗️ Project Architecture

```mermaid
graph LR
A[Hillstrom Dataset<br/>42,613 customers] --> B[Preprocessing &<br/>Feature Engineering]
B --> C[T-Learner & X-Learner<br/>Training - CausalML]
C --> D[Qini Evaluation<br/>5-fold Cross-Validation]
D --> E[Best Model Selected:<br/>T-Learner]
E --> F[Customer Segmentation<br/>4-Quadrant Uplift Logic]
F --> G[Budget-Constrained<br/>Optimization]
G --> H[Streamlit UI<br/>Single + Bulk Prediction]
H --> I[Deployed on<br/>Streamlit Community Cloud]
```

---

## 📂 Repository Structure

```
smart-targeting-engine/
│
├── data/
│   └── hillstrom.csv                  # Raw dataset (42,613 customers)
│
├── models/
│   ├── t_learner.pkl                  # Final uplift model (best: Qini = 2.109)
│   ├── x_learner.pkl                  # Alternate uplift model (Qini = 0.175)
│   ├── baseline_control_model.pkl     # Control-only conversion probability model
│   ├── propensity_model.pkl           # Propensity score model (for X-Learner)
│   ├── qini_scores.pkl                # Saved Qini evaluation results
│   ├── qini_curve.png                 # Qini curve plot
│   ├── segmentation_results.csv       # Full 42,613-customer segmentation output
│   ├── segment_distribution.png       # Segment distribution plot
│   ├── budget_scenarios.csv           # Budget-scenario simulation table
│   ├── budget_optimization_curve.png  # Budget optimization curve
│   └── optimized_targeting_list.csv   # Final recommended targeting list
│
├── RF01.py                # Step 1-2: EDA, preprocessing, baseline RF pipeline
├── uplift_models.py        # Step 3: T-Learner & X-Learner training (CausalML)
├── qini_evaluation.py      # Step 4: 5-fold CV Qini evaluation, model selection
├── segmentation.py         # Step 5: 4-quadrant customer segmentation
├── budget_optimization.py  # Step 6: Variable-cost, budget-constrained targeting
├── app.py                  # Step 7-8: Streamlit UI (single + bulk prediction)
│
├── requirements.txt         # Python dependencies (pinned versions)
├── packages.txt              # System-level dependencies (for CausalML build)
└── .gitignore
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Causal ML | CausalML (T-Learner, X-Learner meta-learners) |
| ML Models | scikit-learn (Random Forest Classifier/Regressor) |
| Data | pandas, numpy |
| Serialization | joblib |
| UI | Streamlit |
| Visualization | matplotlib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

## 🔍 Methodology — Step by Step

### Step 1-2: Setup & Baseline
Environment setup, EDA on the Hillstrom dataset, and a simple Random Forest classifier as a structural baseline (no causal logic yet — just to validate the pipeline shape).

### Step 3: Causal Learners (T-Learner + X-Learner)
Built two meta-learner architectures using CausalML:
- **T-Learner**: trains two separate Random Forests (treatment group, control group) and takes the difference in predicted probabilities as the uplift estimate.
- **X-Learner**: a more advanced meta-learner that also uses a propensity model to weight treatment/control imputations — theoretically more robust on imbalanced treatment/control splits.

**Key technical detail:** CausalML's `BaseXClassifier` does not support a `propensity_learner` constructor argument in the installed version — propensity scores had to be computed separately with a standalone `LogisticRegression` model and passed explicitly via the `p=` parameter in both `.fit()` and `.predict()`.

### Step 4: Evaluation (Qini Coefficient)
Accuracy is meaningless for uplift models — a model can perfectly predict *conversion* while being useless at predicting *incremental* conversion. Instead, this project uses the **Qini coefficient**, computed via **5-fold cross-validated out-of-fold predictions** (a single train/test split was unreliable given how few conversion events exist in the dataset).

**Result: T-Learner (Qini = 2.109) outperformed X-Learner (Qini = 0.175)** — T-Learner was selected as the production model.

### Step 5: Segmentation
Customers are split into the 4 segments above using **uplift score** (from the T-Learner) and **baseline conversion probability** (from a control-only Random Forest). 

**A meaningful design fix was made here:** the first version prioritized baseline probability over uplift when assigning "Sure Things," which let some genuinely high-uplift customers get miscategorized. This was corrected to an uplift-priority quadrant assignment, which is more consistent with the uplift-modeling literature (uplift sign/magnitude is the primary determinant of Persuadable vs. Sure Thing, not baseline alone).

**Final segmentation (42,613 customers):**

| Segment | Count | % | Avg. Uplift |
|---|---|---|---|
| Persuadables | 19,648 | 46.1% | 0.0122 |
| Lost Causes | 11,385 | 26.7% | 0.0037 |
| Sure Things | 8,263 | 19.4% | 0.0034 |
| Sleeping Dogs | 3,317 | 7.8% | -0.0025 |

### Step 6: Budget-Constrained Optimization
Customers are ranked by **uplift-per-dollar-cost**, using a documented, illustrative variable-cost assumption based on each customer's historical channel (Phone = $0.50, Web = $0.15, Multichannel = $0.10 — Phone is assumed costliest to re-engage). This produces a natural break-even point and a set of budget-scenario simulations:

| Budget | Customers Targeted | Cost | Expected Incremental Conversions |
|---|---|---|---|
| 10% | 9,483 | $1,263 | 134.6 |
| 25% | 19,025 | $3,158 | 214.0 |
| 50% | 27,575 | $6,315 | 274.8 |
| 75% | 34,540 | $9,473 | 302.2 |
| **Break-even** | **39,296 (92.2%)** | **$11,624** | **309.6** |
| 100% | 42,613 | $12,631 | 301.5 |

Note the diminishing marginal returns (avg. uplift per targeted customer drops from 0.0142 → 0.0071 as budget grows) — and the fact that total incremental conversions *decline slightly* past the break-even point, since remaining customers include Sleeping Dogs.

### Step 7-8: Streamlit UI + Polish
A two-tab interactive app:
- **Single Customer Check** — a form that returns uplift score, segment, cost, recommendation, and estimated ROI, with a plain-language explanation of every number.
- **Bulk CSV Upload** — batch-scores an uploaded customer list, with a segment-distribution chart, color-coded results table, and a downloadable results CSV.

Includes input validation (invalid/missing values are cleaned or defaulted with visible warnings rather than crashing), loading spinners, and an embedded Qini curve for model-performance transparency.

---

## ⚠️ Key Design Decisions & Limitations (Honest Notes)

- **No data leakage:** `visit`, `spend`, `conversion`, `segment`, and `treatment` are strictly excluded from the feature set — verified before every model change.
- **Cost assumptions are illustrative, not real financial data.** The Hillstrom dataset has no real per-customer treatment cost; the channel-based cost tiers used in Step 6 are a documented, synthetic assumption meant to demonstrate variable-cost optimization, not actual campaign economics.
- **The "channel" column is not the treatment channel.** All treatment in this dataset is email-based; `channel` reflects the customer's *historical purchase channel* (Phone/Web/Multichannel), reused here only as a cost proxy.
- **ROI figures shown in the app are estimates**, derived from historical average order value among converters — not guaranteed revenue.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/HarshVarshney0001/smart-targeting-engine.git
cd smart-targeting-engine

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

To retrain the full pipeline from scratch, run the scripts in order:
```bash
python RF01.py
python uplift_models.py
python qini_evaluation.py
python segmentation.py
python budget_optimization.py
```

---

## 🔮 Future Improvements

- **Docker containerization** — package the full app + dependencies into a container for one-command reproducibility on any machine.
- **CI/CD via GitHub Actions** — automated tests on push (model loads correctly, prediction output shape is valid, no data leakage regressions).
- **Real cost data integration** — replace the illustrative channel-based cost assumption with actual campaign cost data, if available.
- **Additional meta-learners** — R-Learner or DR-Learner for comparison against T/X-Learner.

---

## 📸 Screenshots

*(Add screenshots of the Streamlit app here — single customer prediction, bulk upload results, and the Qini curve.)*

---

## 👤 Author

**GitHub:** [@HarshVarshney0001](https://github.com/HarshVarshney0001)