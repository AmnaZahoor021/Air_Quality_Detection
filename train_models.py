import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "AQI"

FEATURES = [
    "CO",
    "NO",
    "NO2",
    "O3",
    "SO2",
    "PM2.5",
    "PM10",
    "NH3"
]

CITY_FILES = {
    "Faisalabad": "historical_air_pollution_all_faisalabad.csv",
    "Islamabad": "historical_air_pollution_all_islamabad.csv",
    "Karachi": "historical_air_pollution_all_karachi.csv",
    "Lahore": "historical_air_pollution_all_lahore.csv",
    "Larkana": "historical_air_pollution_all_larkana.csv",
    "Multan": "historical_air_pollution_all_multan.csv",
    "Peshawar": "historical_air_pollution_all_peshawar.csv",
    "Quetta": "historical_air_pollution_all_quetta.csv",
    "Rawalpindi": "historical_air_pollution_all_rawalpindi.csv"
}


# --------------------------------------------------
# 2. READ AND COMBINE THE NINE CITY DATASETS
# --------------------------------------------------

all_cities = []
total_invalid_aqi = 0

for city, file_name in CITY_FILES.items():

    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Keep only the required columns.
    df = df[["Timestamp", TARGET] + FEATURES].copy()

    # Convert timestamp into datetime format.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # Convert pollutant measurements and AQI into numeric values.
    for col in [TARGET] + FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Treat -9999 as a missing or invalid measurement.
    df = df.replace(-9999, np.nan)

    # Remove exact duplicate records.
    df = df.drop_duplicates()

    # Remove rows without a valid timestamp or AQI.
    df = df.dropna(
        subset=["Timestamp", TARGET]
    )

    # Keep only valid, whole-number AQI labels from 1 to 5.
    aqi_values = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid_aqi = (
        aqi_values.between(1, 5)
        & np.isclose(
            aqi_values,
            np.round(aqi_values),
            rtol=0,
            atol=1e-6
        )
    )

    total_invalid_aqi += int((~valid_aqi).sum())

    df = df.loc[valid_aqi].copy()

    df[TARGET] = np.round(
        pd.to_numeric(df[TARGET])
    ).astype(int)

    # Add city name as an input feature.
    df["City"] = city

    all_cities.append(df)


# Combine all nine city datasets.
data = pd.concat(
    all_cities,
    ignore_index=True
)

# Sort records chronologically within each city.
data = data.sort_values(
    ["City", "Timestamp"]
).reset_index(drop=True)

print("Combined dataset shape:", data.shape)
print("Invalid AQI records excluded:", total_invalid_aqi)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())
print("\nAQI classes:", sorted(data[TARGET].unique()))


# --------------------------------------------------
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

train_parts = []
test_parts = []

for city in CITY_FILES:

    city_data = data[
        data["City"] == city
    ].sort_values("Timestamp")

    split_index = int(len(city_data) * 0.80)

    train_parts.append(
        city_data.iloc[:split_index]
    )

    test_parts.append(
        city_data.iloc[split_index:]
    )


train_data = pd.concat(
    train_parts,
    ignore_index=True
)

test_data = pd.concat(
    test_parts,
    ignore_index=True
)

X_train = train_data[["City"] + FEATURES]
y_train = train_data[TARGET]

X_test = test_data[["City"] + FEATURES]
y_test = test_data[TARGET]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining AQI class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting AQI class distribution:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# 4. PREPROCESSING PIPELINES
# --------------------------------------------------

numeric_features = FEATURES
categorical_features = ["City"]

# Logistic Regression needs scaling.
logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Decision Tree and Random Forest do not need scaling.
tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 5. DEFINE THE THREE MACHINE LEARNING MODELS
# --------------------------------------------------

candidates = {

    "Logistic Regression": Pipeline([
        ("preprocessor", logistic_preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced"
            )
        )
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", tree_preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=14,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", tree_preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
        )
    ])
}


# --------------------------------------------------
# 6. TRAIN AND EVALUATE ALL THREE MODELS
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in candidates.items():

    print("\nTraining:", model_name)

    # Train using training data only.
    model.fit(X_train, y_train)

    # Predict using test data.
    y_pred = model.predict(X_test)

    trained_models[model_name] = model

    # Calculate evaluation metrics.
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision (macro)": precision,
        "Recall (macro)": recall,
        "F1 (macro)": f1_macro,
        "F1 (weighted)": f1_weighted
    })

    # Save confusion matrix.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4, 5]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[1, 2, 3, 4, 5],
        columns=[1, 2, 3, 4, 5]
    )

    cm_path = os.path.join(
        MODEL_DIR,
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# --------------------------------------------------
# 7. COMPARE MODELS AND SELECT THE BEST MODEL
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 (macro)",
    ascending=False
).reset_index(drop=True)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_f1 = results_df.iloc[0]["F1 (macro)"]

print("\nBest Model:", best_model_name)
print("Best Macro F1-Score:", best_f1)


# --------------------------------------------------
# 8. SAVE THE BEST MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "best_air_quality_classifier.joblib"
)

joblib.dump(
    best_model,
    model_path,
    compress=3
)

print("\nBest model saved at:")
print(model_path)


# --------------------------------------------------
# 9. SAVE MODEL INFORMATION
# --------------------------------------------------

metadata = {
    "project": "AI Air Quality Detection",
    "target": TARGET,
    "features": FEATURES,
    "categorical_features": categorical_features,
    "cities": list(CITY_FILES.keys()),
    "classes": [1, 2, 3, 4, 5],
    "best_model": best_model_name,
    "best_macro_f1": float(best_f1),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "invalid_aqi_records_excluded": total_invalid_aqi,
    "train_test_split": "80/20 chronological split per city",
    "missing_value_handling": (
        "Median imputation inside training pipelines"
    ),
    "target_validation": (
        "Only whole-number AQI values from 1 to 5 were retained. "
        "Invalid and out-of-range values were excluded."
    )
}

metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

with open(metadata_path, "w") as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nModel metadata saved at:")
print(metadata_path)

print("\nAll models trained and saved successfully!")
