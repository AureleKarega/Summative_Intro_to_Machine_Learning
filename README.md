# Credit Card Default Prediction: Traditional ML vs Deep Learning

A binary classification project that predicts whether a credit card client will default on next month's payment. It compares tuned scikit-learn models against Keras neural networks built with `tf.data` pipelines.

**Links**
- Report: [add link or file name]
- Google Drive folder (raw dataset + trained models): [add link]
- Presentation video: [add link]

## Problem and dataset
- **Task:** classify clients as default / no default (about 22% positive class).
- **Data:** UCI Machine Learning Repository, *Default of Credit Card Clients* (Yeh & Lien, 2009). 30,000 clients, 23 features (credit limit, demographics, 6 months of repayment status, bill amounts, and payment amounts).
- **The dataset is not stored in this repository.** Download it from the UCI repository or from the shared Google Drive folder, then place `default of credit card clients.csv` in the same folder as the notebook.

## Repository contents
| File | Purpose |
|---|---|
| `credit_default_classification.ipynb` | Full pipeline: cleaning, feature engineering, experiments, evaluation, model saving |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps data and model files out of the repo |

## Methods in brief
1. Cleaning and a stratified 80/20 train/test split (preprocessors fit on training data only).
2. Feature engineering: delay counts, credit utilization, payment ratios, bill/payment trends.
3. Traditional ML: Logistic Regression, Random Forest, Gradient Boosting, SVM, tuned with randomized search and cross-validation.
4. Deep learning: Keras MLPs (Functional API) fed by `tf.data`, with class weights, dropout, batch normalization, and early stopping.
5. Evaluation: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrices, ROC/PR curves, learning curves, permutation importance.

## How to run
**Local**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter notebook credit_default_classification.ipynb
```
**Google Colab:** open the notebook from GitHub, upload the CSV via the file panel, then *Runtime > Run all*.

Set `QUICK = True` in the setup cell for a fast test run. The full run takes roughly 10-30 minutes depending on hardware. Results are seeded, but small differences between machines and library versions are expected.

## Results
*Replace with the numbers from your own final run.*

| Model | Test ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Saved models
The best traditional model (`.joblib`) and best deep learning model (`.keras`, `.h5`, and SavedModel) are in the Google Drive folder, together with the preprocessor the Keras model needs. Load the scikit-learn model with the same scikit-learn version listed in `requirements.txt`.

```python
import joblib
model = joblib.load("best_ml_model.joblib")        # full pipeline: preprocessing + classifier
proba = model.predict_proba(new_clients_df)[:, 1]
```

## Limitations
Data from a single country and period (Taiwan, 2005), no temporal split, sensitive attributes (sex, age, marital status) in the features, undocumented category codes, and a moderate class imbalance. See the report for details.

## Reference
Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473-2480.
