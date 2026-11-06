# EasyVisa | Visa Application Outcome Prediction

**Machine Learning · Classification · Ensemble Methods · Model Evaluation**

## Project overview

This portfolio project explores historical visa application outcomes and develops classification models to investigate patterns associated with certification and denial. The goal is to evaluate whether predictive models can support better understanding and review workflows, **not** to automate immigration decisions.

## Business / analytical question

Which attributes are associated with application outcomes, and how do ensemble classifiers compare when evaluated on a held-out validation set?

## Dataset

- **25,480 records**, **12 original columns** (`EasyVisa.csv`)
- Includes applicant education and experience, employer information, wage attributes, employment region, and recorded case status
- Target: `case_status` (Certified / Denied)
- `case_id` is an identifier, not a predictive feature

## Workflow

1. Audit the data and explore outcome patterns across features.
2. Address negative employee counts and normalize wage units for analysis.
3. Prepare features and split data into training, validation, and test sets.
4. Compare models trained on original, oversampled, and undersampled data.
5. Tune **Random Forest, AdaBoost, Gradient Boosting, and XGBoost**.
6. Compare validation metrics, review performance for **both classes**, and assess the selected model on the held-out test set.
7. Examine feature importance, limitations, and possible operational implications.

## Results reported in the original notebook

The notebook selects **tuned XGBoost** based on validation-set comparisons. Its written analysis reports **validation F1 ≈ 0.827** for XGBoost and discusses the tradeoff between recognizing Certified versus Denied applications. These are **not independently reproduced performance claims**: run the notebook from start to finish in a compatible Python environment to verify them.

The analysis highlights education, job experience, wage payment unit, and geography as associated variables. **Associations and feature importance are not causal effects.** Group-level differences should not be used as a basis for discriminatory screening or automatic eligibility decisions.

## Run locally

Keep these files in one folder:

```text
EasyVisa_Ensemble_Classification.ipynb
EasyVisa.csv
README.md
requirements.txt
```

Install dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open `EasyVisa_Ensemble_Classification.ipynb` and choose **Restart Kernel and Run All**. Model tuning can take time. The dataset is loaded with `pd.read_csv("EasyVisa.csv")` and does not require Google Drive.

## Limitations and responsible use

- The notebook was adapted for local paths; **its full end-to-end execution has not been independently verified** in this package.
- Results depend on preprocessing, random seeds, library versions, and the model tuning procedure.
- Predictive correlations do not establish eligibility or cause of outcomes.
- This is an educational analytics exercise, not legal or immigration advice. Any real-world use would require rigorous bias, privacy, governance, and regulatory review.

## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Matplotlib, Seaborn, Jupyter.

## Author

Roblyn Houston · Senior Data Analyst | Business Intelligence & Analytics
