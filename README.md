# 🩺 GlucoCheck — Diabetes Risk Screening Without a Blood Test

GlucoCheck is a machine-learning web app that estimates a person's risk of **prediabetes or diabetes** from 21 simple health and lifestyle questions — no lab test required. High-risk users are encouraged to get a proper blood test.

> ⚠️ **Disclaimer:** GlucoCheck is a student project and a **screening tool only**. It does not diagnose diabetes and is not a substitute for medical advice. Only a blood test (fasting blood sugar or HbA1c) can confirm diabetes.

---

## 💡 The problem

- About **589 million adults** worldwide live with diabetes (IDF Diabetes Atlas, 11th ed., 2025).
- **42.8% of them — about 252 million people — don't know they have it.**
- Undiagnosed diabetes leads to late treatment and serious complications: heart disease, kidney failure, nerve damage, and vision loss.
- Diagnosis needs a clinic visit and a blood test, which many people never get.

**Our idea:** a free, 2-minute online questionnaire that flags people at high risk so they know to get tested.

---

## 📊 Dataset

**Diabetes Health Indicators Dataset** — from the U.S. CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.
Source: [Kaggle — alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

We use `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`:

| | |
|---|---|
| Rows | 70,692 people |
| Features | 21 (health conditions, lifestyle, demographics) |
| Target | `Diabetes_binary` — 0 = no diabetes, 1 = prediabetes/diabetes |
| Class balance | 50% / 50% |
| Missing values | None |

**Features:** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income.

---

## 🤖 Models

**Type of task:** supervised learning — binary classification.

All models are trained and tested on the **same 80/20 stratified split** (`random_state=42`).

| Model | How it works |
|---|---|
| **Logistic Regression** | Weighs each feature, sums them, and passes the total through a sigmoid to get a probability. Simple, fast, interpretable baseline. |
| **Random Forest** | Many decision trees, each trained on random samples of rows and features; final answer is the majority vote. Reduces overfitting and captures interactions. |
| **XGBoost** | Gradient-boosted trees built one after another, each correcting the previous trees' errors. |
| **LightGBM** | Gradient boosting with a faster, leaf-wise tree-growing strategy. |

The best model (by Recall) is then tuned with randomized search and retrained on all data.

---

## 📈 Results (test set, 14,139 people)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **XGBoost (tuned) ⭐** | **0.7536** | 0.7322 | **0.7997** | **0.7644** | 0.8306 |
| XGBoost | 0.7534 | 0.7326 | 0.7981 | 0.7640 | 0.8292 |
| LightGBM | 0.7525 | 0.7314 | 0.7980 | 0.7632 | 0.8308 |
| Random Forest | 0.7489 | 0.7292 | 0.7918 | 0.7592 | 0.8279 |
| Logistic Regression | 0.7458 | 0.7372 | 0.7639 | 0.7503 | 0.8232 |

**Why Recall matters most:** for a screening tool, missing a real diabetic person (false negative) is worse than a false alarm, which only leads to a cheap follow-up blood test.

**XGBoost confusion matrix (before tuning):**

| | Predicted: No | Predicted: Diabetes |
|---|---|---|
| **Actual: No** | 5,011 (TN) | 2,059 (FP) |
| **Actual: Diabetes** | 1,427 (FN) | 5,642 (TP) |

The model catches **about 8 in 10** people with diabetes or prediabetes.

---

## 🌐 Web app

- Friendly 4-section questionnaire (BMI calculated from height and weight)
- Risk percentage with Low / Moderate / High level (≥ 60% = High, 40–59% = Moderate)
- Personalised precautions based on the user's answers
- Clear medical disclaimer; answers are not stored

**Stack:** Python · scikit-learn · XGBoost · LightGBM · Flask · HTML/CSS/JavaScript

---

## 🚀 How to run

```bash
# 1. Clone
git clone https://github.com/pasthananv/GlucoCheck.git
cd GlucoCheck

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install libraries
pip install -r requirements.txt

# 4. (Optional) Retrain the model
#    Put the CSV in a folder named data/, or let the script download it with kagglehub
python train_model.py

# 5. Start the web app (opens your browser automatically)
python app.py
```

> Mac users: if XGBoost or LightGBM shows a `libomp.dylib` error, run `brew install libomp`.

---

## 📁 Project structure

```
GlucoCheck/
├── app.py                  # Flask backend: loads the model, serves the page, /predict API
├── index.html              # Landing page and questionnaire
├── train_model.py          # Trains, compares, tunes and saves the models
├── diabetes_model.pkl      # Saved best model (XGBoost)
├── model_results.json      # All evaluation metrics
├── diabetes_pycaret.ipynb  # PyCaret version: EDA + model comparison (Google Colab)
└── requirements.txt
```

---

## 🔧 Limitations & future improvements

- **US survey data from 2015** — self-reported answers; may not fully represent Thai or Asian populations (who develop diabetes at lower BMI).
- **Correlation, not causation** — e.g. having had a cholesterol check slightly raises predicted risk, because people with health problems get checked more often.
- **Improvements:** lower the decision threshold to raise recall; train on the full 253k-row dataset; add data such as family history, waist size or lab values; use SHAP to explain each prediction; validate on Thai health data; add a Thai-language version.

---

## 📚 References

- International Diabetes Federation. *IDF Diabetes Atlas*, 11th edition (2025). https://diabetesatlas.org
- CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015.
- Teboul, A. *Diabetes Health Indicators Dataset*, Kaggle.
