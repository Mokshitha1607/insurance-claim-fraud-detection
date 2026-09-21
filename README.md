# Insurance Claim Fraud Detection
# Insurance Claim Fraud Detection

A machine learning-based system for detecting potentially fraudulent insurance claims using Python, Scikit-learn, and SMOTE.

## Project Overview

Insurance fraud can cause significant financial losses for insurance companies. This project builds a machine learning classification system that analyzes insurance claim information and predicts whether a claim is likely to be fraudulent.

The project follows an end-to-end machine learning workflow:

**Data → EDA → Preprocessing → Feature Engineering → SMOTE → Model Training → Evaluation → Prediction**

## Dataset

- Dataset: Health Insurance Claims Data for Fraud Detection
- Records: 20,100
- Features: 30
- Duplicate records removed: 28
- Final records used: 20,072
- Fraudulent claims after duplicate removal: 4,983
- Genuine claims after duplicate removal: 15,089

The dataset contains information about patients, providers, claims, procedures, admission types, claim amounts, previous claims, and other claim-related attributes.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Google Colab

## Machine Learning Models

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

SMOTE was applied to the training data to address class imbalance.

## Data Preprocessing

The preprocessing workflow included:

- Duplicate record removal
- Missing-value checking
- Date conversion
- Feature engineering
- Removal of unnecessary identifier/date columns
- One-Hot Encoding for categorical variables
- Train-test splitting
- SMOTE for minority-class oversampling
- Feature scaling for Logistic Regression

## Feature Engineering

The following features were created from the original date fields:

- Days Between Service and Claim
- Days Until Policy Expiration
- Claim Month
- Claim Year

## Exploratory Data Analysis

EDA was performed to investigate relationships between fraud and:

- Claim amount
- Length of hospital stay
- Previous patient claims
- Previous provider claims
- Admission type
- Provider type
- Fraudulent vs genuine claim distribution

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 73.03% | 47.10% | 70.11% | 56.35% | 79.64% | 60.28% |
| Random Forest | 85.58% | 93.72% | 44.93% | 60.75% | 86.61% | 78.86% |
| Gradient Boosting | 85.78% | 85.26% | 51.65% | 64.33% | 87.26% | 78.63% |

Gradient Boosting produced the highest F1-score and ROC-AUC among the baseline models on the held-out test set.

## Hyperparameter Tuning

Random Forest hyperparameters were tuned using RandomizedSearchCV with 3-fold cross-validation.

Best parameters:

- Number of estimators: 100
- Maximum depth: 10
- Minimum samples split: 2
- Minimum samples leaf: 1

Best cross-validation F1-score: **0.8129**

On the untouched test set, the tuned Random Forest achieved:

- Accuracy: 84.91%
- Precision: 89.34%
- Recall: 44.53%
- F1-Score: 59.44%
- ROC-AUC: 85.64%
- PR-AUC: 75.65%

The tuned model was compared against the baseline models using the untouched test set rather than selecting a model only from the cross-validation score.

## New Claim Prediction

The trained model was saved and tested on a new synthetic insurance claim.

Example result:

- Prediction: Genuine Claim
- Fraud Probability: 14.35%

The saved model can be loaded and used to process new claims using the same preprocessing steps applied during training.

## Saved Model Files

- `preprocessor.pkl`
- `fraud_detection_model.pkl`

## Project Structure

```text
Insurance-Claim-Fraud-Detection/
│
├── Insurance_Claim_Fraud_Detection.ipynb
├── README.md
├── preprocessor.pkl
└── fraud_detection_model.pkl
