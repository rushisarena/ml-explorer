"""Generate sample datasets for the ML Explorer."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris, fetch_california_housing

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DIR.mkdir(exist_ok=True)

# --- Iris (Classification) ---
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
iris_df.to_csv(SAMPLE_DIR / "iris.csv", index=False)
print(f"Created iris.csv ({len(iris_df)} rows)")

# --- Titanic (Classification) ---
np.random.seed(42)
n = 891
titanic_df = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Survived": np.random.choice([0, 1], n, p=[0.616, 0.384]),
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
    "Age": np.clip(np.random.normal(30, 14, n), 0.5, 80).round(1),
    "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.05, 0.02, 0.02]),
    "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.09, 0.02]),
    "Fare": np.clip(np.random.exponential(32, n), 0, 512).round(2),
    "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
})
# Make survival correlate with class and sex
mask_female = titanic_df["Sex"] == "female"
mask_first = titanic_df["Pclass"] == 1
titanic_df.loc[mask_female, "Survived"] = np.random.choice([0, 1], mask_female.sum(), p=[0.25, 0.75])
titanic_df.loc[mask_first & mask_female, "Survived"] = np.random.choice([0, 1], (mask_first & mask_female).sum(), p=[0.05, 0.95])
# Add some missing values
missing_age = np.random.choice(n, 177, replace=False)
titanic_df.loc[missing_age, "Age"] = np.nan
titanic_df.to_csv(SAMPLE_DIR / "titanic.csv", index=False)
print(f"Created titanic.csv ({len(titanic_df)} rows)")

# --- Housing (Regression) ---
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df["MedianHouseValue"] = housing.target
# Subsample to 2000 for speed
housing_df = housing_df.sample(2000, random_state=42).reset_index(drop=True)
housing_df.to_csv(SAMPLE_DIR / "housing.csv", index=False)
print(f"Created housing.csv ({len(housing_df)} rows)")

print("✅ All sample datasets created!")
