# Module 2 - Analytics Pipeline

Loading the titanic dataset (only once!), checking it out, cleaning it,
making some charts to understand who survived, then building actual ML
models to predict survival + a bonus regression task.

## My files
- `01_eda.py` - part A, loads the data, saves a raw copy to titanic.csv,
  explores it, cleans missing values, makes charts
- `02_modeling.py` - part B, reads that same titanic.csv (raw, not the
  cleaned version - see note below for why), trains/tests 3 models,
  compares imbalance handling, tunes hyperparameters, does the bonus
  regression task, saves the best model

## why 02_modeling.py doesn't use the cleaned data from 01_eda.py
this confused me at first too. if I used the version 01_eda.py already
cleaned, the missing ages would've been filled in using stats from the
WHOLE dataset (train + test all mixed together) before I even split it.
that means info from the test set leaks into what the model learns from -
called data leakage, and it makes your evaluation results look better
than they'd actually be in real life. so 02_modeling.py deliberately re-
does its own cleaning, but INSIDE a pipeline that only learns from
X_train, and just applies the same thing to X_test without re-learning
anything from it.

## other decisions I made
- missing value rule: <5% missing = drop rows, 5-30% = fill in with
  median/mode, >30% = column too messy (this only hit "deck" which was
  way missing, so I made "Missing" its own category instead of just
  dropping the whole column, since maybe deck being unrecorded means
  something on its own)
- correlation heatmap only uses the 6 columns they asked for (skipping
  adult_male and alone since those are literally just calculated FROM
  the other columns)
- used f1 score to decide which classifier is "best" since it balances
  precision and recall together
- saved the WHOLE pipeline (cleaning steps + model) with joblib, not just
  the bare model, so it can be reused directly on new raw data later

## how to run it
```
pip install seaborn scikit-learn imbalanced-learn joblib matplotlib pandas
python 01_eda.py
python 02_modeling.py
```
(01_eda.py needs internet the first time so seaborn can download+cache
the dataset)

## REAL RUN RESULTS

### Part A - EDA
- loaded 891 rows, 15 columns
- missing values: deck 77.22% (made "Missing" its own category), age
  19.87% (filled with median = 28.0), embarked 0.22% (dropped 2 rows),
  embark_town 0.22% (0 extra rows dropped - same 2 rows as embarked)
- shape after cleaning: 889 rows
- outliers (IQR rule): age had 65, fare had 114 (fare bounds were -26.8
  to 65.7, so anything pricier than ~66 counted)
- fare: mean=32.10, median=14.45, mode=8.05 -> right-skewed, matches the
  "few expensive tickets pull the average up" theory
- survival rate: female 0.740, male 0.189
- survival rate by class: 1st=0.626, 2nd=0.473, 3rd=0.242
- survival rate by sex+class: female/1st=0.967, female/2nd=0.921,
  female/3rd=0.5, male/1st=0.369, male/2nd=0.157, male/3rd=0.135 - so
  even 3rd class women survived at 50%, way better than 1st class men
- top 2 correlations: pclass vs fare = -0.548 (higher class number =
  cheaper fare, makes sense), sibsp vs parch = 0.415 (people traveling
  with siblings/spouses also tend to have parents/kids along)
- z-score check confirmed ~0 mean and ~1 std after standardizing age/fare

### Part B - Modeling
- class balance: 61.6% did not survive, 38.4% survived
- Logistic Regression: accuracy=0.804, precision=0.793, recall=0.667,
  f1=0.724, auc=0.843
- Decision Tree: accuracy=0.788, precision=0.844, recall=0.551,
  f1=0.667, auc=0.821
- Random Forest: accuracy=0.793, precision=0.767, recall=0.667,
  f1=0.713, auc=0.829
- imbalance comparison (all using random forest): baseline f1=0.713,
  class_weight_balanced f1=0.726, smote f1=0.737 - SMOTE won, matches
  the theory that both techniques trade some precision for better recall
- GridSearchCV best params: max_depth=8, max_features='sqrt',
  n_estimators=200, with an OOB score of 0.822
- regression (predicting fare): MAE=21.78, RMSE=58.86, R2=0.331,
  Adjusted R2=0.296 - residual spread was 13.36 at low predictions vs
  93.63 at high predictions, clearly heteroscedastic
- final pick: Logistic Regression, since it had the best f1 (0.724) of
  the 3 classifiers, with auc=0.843
- saved the Logistic Regression pipeline to model_pipeline.joblib,
  reload check ran fine (predicted [0, 0, 0] on 3 sample rows - worth
  double checking those 3 passengers were actually likely non-survivors,
  but the reload/predict mechanics themselves worked correctly)
