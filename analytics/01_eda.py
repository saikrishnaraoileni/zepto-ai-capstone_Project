# 01_eda.py
# Part A of the analytics module - loading the titanic dataset, checking
# it out, cleaning up missing values, and making some charts to understand
# who survived and why.

import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # so it just saves the plots as files instead of
                        # trying to pop up a window (which doesn't work
                        # well when running from a script)
import matplotlib.pyplot as plt

# make a folder to dump all our chart images into
if not os.path.exists("charts"):
    os.makedirs("charts")

# ---------------------------------------------------------
# STEP 1: load the data (only doing this ONCE like the assignment says)
# ---------------------------------------------------------
print("loading titanic dataset...")
df = sns.load_dataset("titanic")

# saving this raw version right away so even if seaborn cant reach the
# internet later, we still have a local copy to work from
df.to_csv("titanic.csv", index=False)
print("saved raw copy to titanic.csv, shape is", df.shape)

# ---------------------------------------------------------
# STEP 2: look at the data first before touching anything
# ---------------------------------------------------------
print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe(include="all"))

print("\nshape:", df.shape)

# check how much is missing in each column
missing_percent = (df.isna().mean() * 100).round(2)
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
print("\n--- columns with missing values ---")
print(missing_percent)

# ---------------------------------------------------------
# STEP 3: clean up the missing values
# using the rule: <5% missing = just drop those rows
#                 5-30% missing = fill in with median/mode
#                 >30% missing = column too messy, either drop it or make
#                                 "missing" its own category
# ---------------------------------------------------------
# making a copy so i dont mess up the original df, going to work on
# df_clean from here for the rest of this script
df_clean = df.copy()

for col in missing_percent.index:
    pct = missing_percent[col]

    if pct < 5:
        before = len(df_clean)
        df_clean = df_clean[df_clean[col].notna()]
        print(f"{col}: only {pct}% missing, dropped {before - len(df_clean)} rows")

    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_value)
            print(f"{col}: {pct}% missing, filled with median = {fill_value}")
        else:
            fill_value = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(fill_value)
            print(f"{col}: {pct}% missing, filled with mode = {fill_value}")

    else:
        # deck is like 77% missing which is way too much to just guess a
        # value for every row, so instead I'm keeping "missing" as its own
        # category - seems like the fact its missing might mean something
        # (like maybe cheaper tickets didn't get a cabin logged?)
        if col == "deck":
            df_clean[col] = df_clean[col].astype(str)
            df_clean.loc[df_clean[col] == "nan", col] = "Missing"
            print(f"{col}: {pct}% missing, way too much to fill in, made 'Missing' its own category instead")
        else:
            df_clean = df_clean.drop(columns=[col])
            print(f"{col}: {pct}% missing, just dropped the whole column")

print("\nshape after cleaning:", df_clean.shape)

# ---------------------------------------------------------
# STEP 4: univariate analysis - looking at age and fare by themselves
# ---------------------------------------------------------
print("\n--- checking age and fare for outliers ---")

for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df_clean[col], kde=True, ax=axes[0])
    axes[0].set_title(col + " histogram")
    sns.boxplot(x=df_clean[col], ax=axes[1])
    axes[1].set_title(col + " boxplot")
    fig.savefig(f"charts/univariate_{col}.png")
    plt.close(fig)

    # IQR rule for outliers - anything outside Q1-1.5*IQR to Q3+1.5*IQR
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    print(f"{col}: {outlier_count} outliers (using IQR bounds {lower_bound:.1f} to {upper_bound:.1f})")

# checking if fare is skewed
fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"\nfare -> mean={fare_mean:.2f} median={fare_median:.2f} mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    print("this looks right-skewed (mean > median > mode) - probably because a few super expensive tickets pull the average up")
else:
    print("doesn't look like the typical right-skew pattern, would need to look at the histogram")

# ---------------------------------------------------------
# STEP 5: bivariate - survival rate broken down different ways
# ---------------------------------------------------------
print("\n--- survival rates ---")

# doing this with boolean masking like the assignment wants
survived_female = df_clean.loc[df_clean["sex"] == "female", "survived"].mean()
survived_male = df_clean.loc[df_clean["sex"] == "male", "survived"].mean()
print("survival rate - female:", round(survived_female, 3), " male:", round(survived_male, 3))

for pclass_value in sorted(df_clean["pclass"].unique()):
    rate = df_clean.loc[df_clean["pclass"] == pclass_value, "survived"].mean()
    print("survival rate - class", pclass_value, ":", round(rate, 3))

print("\nsurvival rate by sex AND class combined:")
for sex_value in ["female", "male"]:
    for pclass_value in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == sex_value) & (df_clean["pclass"] == pclass_value)
        rate = df_clean.loc[mask, "survived"].mean()
        print(f"  {sex_value}, class {pclass_value}: {round(rate, 3)}")

# correlation heatmap - only the 6 columns the assignment asks for
# (leaving out adult_male and alone since those are just calculated FROM
# sex/age and sibsp/parch, not really their own independent thing)
corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_columns].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("correlation heatmap")
fig.savefig("charts/correlation_heatmap.png")
plt.close(fig)

# find the top 2 strongest correlations (not counting the diagonal which is always 1)
pairs_checked = []
for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        col_a = corr_columns[i]
        col_b = corr_columns[j]
        value = corr_matrix.loc[col_a, col_b]
        pairs_checked.append((col_a, col_b, value))

pairs_checked.sort(key=lambda pair: abs(pair[2]), reverse=True)
print("\ntop 2 strongest correlations:")
print(pairs_checked[0])
print(pairs_checked[1])

# ---------------------------------------------------------
# STEP 6: multivariate charts - trying to tell a "story" with these
# ---------------------------------------------------------
print("\nmaking the story charts...")

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", ax=ax)
ax.set_title("survival by class and sex")
fig.savefig("charts/story_class_sex.png")
plt.close(fig)
print("chart 1: women survived way more than men in every class, class barely closes the gap")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df_clean, x="survived", y="age", ax=ax)
ax.set_title("age vs survival")
fig.savefig("charts/story_age_survival.png")
plt.close(fig)
print("chart 2: survivors are a little younger on average, matches the whole 'women and children first' thing")

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df_clean, x="fare", y="age", hue="survived", style="pclass", ax=ax)
ax.set_title("fare vs age colored by survival")
fig.savefig("charts/story_fare_age.png")
plt.close(fig)
print("chart 3: the higher fare (mostly 1st class) dots lean toward survived no matter the age")

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df_clean, x="embarked", y="survived", ax=ax)
ax.set_title("survival by embark port")
fig.savefig("charts/story_embarked.png")
plt.close(fig)
print("chart 4: people who got on at Cherbourg (C) survived more, probably cause more of them were 1st class")

# ---------------------------------------------------------
# STEP 7: standardization check (just for exploring, NOT used for modeling)
# ---------------------------------------------------------
print("\n--- z-score check on age and fare (just curious, not used later) ---")
print("before:")
print(df_clean[["age", "fare"]].agg(["mean", "std"]))

z_scored = df_clean[["age", "fare"]].apply(lambda col: (col - col.mean()) / col.std())
print("after z-score (should be ~0 mean, ~1 std):")
print(z_scored.agg(["mean", "std"]))

print("\ndone with the EDA part! titanic.csv on disk is still the RAW version -")
print("02_modeling.py will read that same raw file and do its OWN cleaning")
print("inside the model pipeline (has to be done that way to avoid leaking")
print("test data info into training - more on that in 02_modeling.py)")
