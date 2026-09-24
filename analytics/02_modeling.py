# 02_modeling.py
# Part B - building actual ML models to predict who survived.
#
# IMPORTANT THING I LEARNED: this script does NOT use the cleaned data
# from 01_eda.py. It reads the same RAW titanic.csv again and does its
# OWN cleaning, but this time inside a pipeline that's fit only on the
# training data. If I used the already-cleaned version, the missing ages
# would've been filled in using the median of the WHOLE dataset (train +
# test together) before even splitting - which means info from the test
# set would leak into training. That's called data leakage and it makes
# your evaluation numbers lie to you. So yes this looks like I forgot to
# use the cleaned csv, but it's actually on purpose.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# STEP 1: load the RAW data (same file 01_eda.py saved, not the cleaned one)
# ---------------------------------------------------------
df = pd.read_csv("titanic.csv")
df = df.dropna(subset=["survived"])  # cant use a row if we dont know the answer

feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[feature_cols]
y = df["survived"].astype(int)

print("survival class balance:")
print(y.value_counts(normalize=True))
print("using a STRATIFIED split because these classes arent 50/50 - a normal")
print("random split could accidentally put way more survivors in test vs train")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ---------------------------------------------------------
# STEP 2: build the preprocessing pipeline
# this handles missing values + encoding + scaling, but ONLY learns
# its fill values/scale from X_train, then just applies (not re-learns)
# the same thing to X_test
# ---------------------------------------------------------
def make_preprocessor():
    # writing this as a function because I need a FRESH unfitted one for
    # a couple different comparisons later, cant reuse the same fitted one
    numeric_cols = ["age", "sibsp", "parch", "fare"]
    categorical_cols = ["pclass", "sex", "embarked"]

    numeric_part = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_part = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_part, numeric_cols),
        ("cat", categorical_part, categorical_cols),
    ])


# ---------------------------------------------------------
# STEP 3: train the 3 classifiers and check how good they are
# ---------------------------------------------------------
def check_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n{name}:")
    print("confusion matrix:\n", cm)
    print(f"accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} auc={auc:.3f}")

    # roc curve chart
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{name} (auc={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC - " + name)
    ax.legend()
    fig.savefig(f"charts/roc_{name.replace(' ', '_').lower()}.png")
    plt.close(fig)

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


all_results = []

print("\n--- training Logistic Regression ---")
log_reg = Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
log_reg.fit(X_train, y_train)
all_results.append(check_model("Logistic Regression", log_reg, X_test, y_test))

print("\n--- training Decision Tree ---")
tree_model = Pipeline([("prep", make_preprocessor()), ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))])
tree_model.fit(X_train, y_train)
all_results.append(check_model("Decision Tree", tree_model, X_test, y_test))

# drawing the actual tree so we can see what it's doing
encoded_cat_names = list(
    tree_model.named_steps["prep"].named_transformers_["cat"]
    .named_steps["encode"].get_feature_names_out(["pclass", "sex", "embarked"])
)
all_feature_names = ["age", "sibsp", "parch", "fare"] + encoded_cat_names

fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(tree_model.named_steps["clf"], feature_names=all_feature_names,
          class_names=["did not survive", "survived"], filled=True, fontsize=7, ax=ax)
fig.savefig("charts/decision_tree.png", dpi=150)
plt.close(fig)

print("\n--- training Random Forest ---")
forest_model = Pipeline([("prep", make_preprocessor()), ("clf", RandomForestClassifier(random_state=42))])
forest_model.fit(X_train, y_train)
all_results.append(check_model("Random Forest", forest_model, X_test, y_test))

results_table = pd.DataFrame(all_results)
print("\n=== comparing all 3 classifiers ===")
print(results_table.to_string(index=False))


# ---------------------------------------------------------
# STEP 4: dealing with class imbalance - trying 3 different ways
# ---------------------------------------------------------
print("\n=== imbalance comparison (using random forest for all 3) ===")

def get_prf(model, X_test, y_test):
    preds = model.predict(X_test)
    return precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds)

imbalance_rows = []

# (a) just the normal baseline, nothing special
baseline_model = Pipeline([("prep", make_preprocessor()), ("clf", RandomForestClassifier(random_state=42))])
baseline_model.fit(X_train, y_train)
imbalance_rows.append(("baseline", *get_prf(baseline_model, X_test, y_test)))

# (b) telling the model to pay more attention to the minority class
weighted_model = Pipeline([("prep", make_preprocessor()),
                            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
weighted_model.fit(X_train, y_train)
imbalance_rows.append(("class_weight_balanced", *get_prf(weighted_model, X_test, y_test)))

# (c) SMOTE - this creates fake extra examples of the minority class.
# IMPORTANT: only doing this to the TRAINING data, never touching X_test,
# otherwise we'd be testing on fake data which isn't fair
prep_for_smote = make_preprocessor()
X_train_ready = prep_for_smote.fit_transform(X_train)
X_test_ready = prep_for_smote.transform(X_test)

X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train_ready, y_train)
smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_smote, y_train_smote)
smote_preds = smote_model.predict(X_test_ready)
imbalance_rows.append(("smote", precision_score(y_test, smote_preds), recall_score(y_test, smote_preds), f1_score(y_test, smote_preds)))

imbalance_table = pd.DataFrame(imbalance_rows, columns=["strategy", "precision", "recall", "f1"])
print(imbalance_table.to_string(index=False))

best_strategy = imbalance_table.loc[imbalance_table["f1"].idxmax(), "strategy"]
print(f"\nmy conclusion: '{best_strategy}' had the best f1 score. class_weight and smote both")
print("seem to trade away a bit of precision to get better recall than baseline, which makes")
print("sense - they're both specifically trying to catch more of the minority class")


# ---------------------------------------------------------
# STEP 5: hyperparameter tuning with GridSearchCV
# ---------------------------------------------------------
print("\n=== tuning the random forest with GridSearchCV ===")

prep_for_tuning = make_preprocessor()
X_train_tuned_ready = prep_for_tuning.fit_transform(X_train)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 8, None],
    "max_features": ["sqrt", "log2"],
}

# note: oob_score only works if you pass oob_score=True when you CREATE
# the model, cant turn it on after - learned this the hard way
rf_for_tuning = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
grid_search = GridSearchCV(rf_for_tuning, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train_tuned_ready, y_train)

print("best params found:", grid_search.best_params_)
print("oob score of the best one:", grid_search.best_estimator_.oob_score_)


# ---------------------------------------------------------
# STEP 6: bonus regression task - predicting FARE this time, not survival
# ---------------------------------------------------------
print("\n=== bonus task: predicting fare with linear regression ===")

reg_feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
reg_df = df.dropna(subset=["fare"] + reg_feature_cols)

X_reg = reg_df[reg_feature_cols]
y_reg = reg_df["fare"]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
     ["age", "sibsp", "parch"]),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]),
     ["pclass", "sex", "embarked", "survived"]),
])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_pipeline = Pipeline([("prep", reg_preprocessor), ("reg", LinearRegression())])
reg_pipeline.fit(X_reg_train, y_reg_train)
fare_predictions = reg_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, fare_predictions)
rmse = mean_squared_error(y_reg_test, fare_predictions) ** 0.5
r2 = r2_score(y_reg_test, fare_predictions)
n_rows = X_reg_test.shape[0]
n_features = X_reg_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_features - 1)

print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} Adjusted R2={adjusted_r2:.3f}")

residuals = y_reg_test - fare_predictions
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(fare_predictions, residuals, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("predicted fare")
ax.set_ylabel("residual (actual - predicted)")
ax.set_title("residual plot")
fig.savefig("charts/regression_residuals.png")
plt.close(fig)

# checking if residuals spread out more at higher predictions
# (that would mean heteroscedasticity - fancy word for "not constant spread")
avg_pred = fare_predictions.mean()
low_group_std = residuals[fare_predictions < avg_pred].std()
high_group_std = residuals[fare_predictions >= avg_pred].std()
print(f"residual spread - low predictions: {low_group_std:.2f}, high predictions: {high_group_std:.2f}")
if high_group_std > 1.5 * low_group_std or low_group_std > 1.5 * high_group_std:
    print("this looks heteroscedastic - the spread is pretty different between low and high predictions")
else:
    print("spread looks fairly constant, probably fine (homoscedastic)")


# ---------------------------------------------------------
# STEP 7: put it all together - final comparison + pick the best one
# ---------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
print("\nclassifiers:")
print(results_table.to_string(index=False))
print(f"\nregression: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} Adjusted_R2={adjusted_r2:.3f}")

best_row = results_table.loc[results_table["f1"].idxmax()]
print(f"\nmy pick to deploy: {best_row['model']} - it had the best f1 ({best_row['f1']:.3f}) out of")
print(f"the 3, with precision {best_row['precision']:.3f} and recall {best_row['recall']:.3f}, and an")
print(f"auc of {best_row['auc']:.3f} which seems solid for telling the two classes apart")

# save the WHOLE pipeline (preprocessing steps + the model together), not
# just the bare model - that way we can feed it raw new data later and it
# handles all the cleaning/encoding/scaling itself
model_lookup = {"Logistic Regression": log_reg, "Decision Tree": tree_model, "Random Forest": forest_model}
best_pipeline = model_lookup[best_row["model"]]
joblib.dump(best_pipeline, "model_pipeline.joblib")
print(f"\nsaved the {best_row['model']} pipeline to model_pipeline.joblib")

# quick sanity check that loading it back actually works
reloaded_pipeline = joblib.load("model_pipeline.joblib")
sample_rows = X_test.iloc[:3]
print("reload check, predictions on 3 raw rows:", reloaded_pipeline.predict(sample_rows).tolist())
