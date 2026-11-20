# AI Air Quality Detection — Nine-City Classification Project

## 1. Project Overview

This project uses machine learning to predict air quality categories based on pollutant measurements collected from nine cities in Pakistan.

The system provides a Streamlit dashboard where users can either select a city to check its latest available historical air-quality data or manually enter pollutant measurements to obtain a predicted AQI category.

## 2. Project Structure

- `Air_Quality_Detection_Classification.ipynb`: Jupyter Notebook covering data preparation, exploratory data analysis, model training, evaluation, comparison, and model selection.
- `train_models.py`: Reproducible machine-learning model training script.
- `app.py`: Streamlit application with Simple City Check and Manual Pollutant Input.
- `dataset/`: Contains CSV datasets for nine Pakistani cities.
- `models/`: Contains the selected trained model, model comparison results, confusion matrices, and model metadata.
- `mainpage_bg.png`, `sidebar_bg.png`: Dashboard background image assets.

## 3. Dataset

The project uses historical air-quality records from nine Pakistani cities:

1. Faisalabad
2. Islamabad
3. Karachi
4. Lahore
5. Larkana
6. Multan
7. Peshawar
8. Quetta
9. Rawalpindi

### Input Features

The model uses eight pollutant measurements:

- CO
- NO
- NO2
- O3
- SO2
- PM2.5
- PM10
- NH3

The city name is also included as a categorical input feature.

### Target Variable

The target variable is `AQI`. Only valid whole-number AQI values from 1 to 5 are retained as the five project-defined classes.

## 4. Data Preprocessing

The following preprocessing steps are performed:

- Combined the nine city datasets.
- Converted timestamps and numeric columns into appropriate data types.
- Treated `-9999` as a missing or invalid value.
- Removed duplicate records.
- Removed records without a valid timestamp or AQI value.
- Retained only whole-number AQI values within the range of 1–5.
- Excluded invalid, fractional, and out-of-range AQI records.
- Applied median imputation inside the model pipelines using training data only.
- Applied one-hot encoding to the city feature.
- Applied standard scaling for Logistic Regression.

Missing pollutant values are handled during model preprocessing to prevent information leakage from the test data.

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to examine:

- AQI class distribution.
- Air-quality records across cities.
- Pollutant distributions.
- Relationships between pollutant measurements and AQI.
- Median PM2.5 values across AQI classes.
- Correlations between pollutant features and AQI.

The cleaned data contains 309,560 records. Class 5 has the highest number of records, while Class 1 has the lowest. Lahore has the highest number of cleaned records (34,424). Karachi, Larkana, and Quetta have the lowest counts (34,376 each). PM10 has the strongest positive correlation with AQI (approximately 0.566).

## 6. Model Training and Evaluation

Three machine-learning classifiers were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The cleaned dataset contains **309,560 records**.

A chronological 80/20 train-test split was performed separately for each city:

- Training records: **247,644**
- Testing records: **61,916**

### Model Comparison

The models were evaluated using Accuracy, Macro Precision, Macro Recall, Macro F1-score, and Weighted F1-score.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.744493 | 0.745788 | 0.804858 | 0.758891 | 0.736115 |
| Decision Tree | 0.764277 | 0.811935 | 0.832029 | 0.808272 | 0.760848 |
| Random Forest | 0.906825 | 0.919397 | 0.939669 | 0.926836 | 0.906988 |

### Best Model Selection

Random Forest was selected as the best-performing model because it achieved the highest accuracy and macro F1-score among the three evaluated classifiers.

Its evaluation results are:

- Accuracy: 90.68%
- Macro Precision: 91.94%
- Macro Recall: 93.97%
- Macro F1-score: 92.68%
- Weighted F1-score: 90.70%

Random Forest combines predictions from multiple decision trees, helping it capture complex relationships in the dataset and improve overall classification performance.

## 7. Application Integration

The selected Random Forest pipeline is integrated into a Streamlit dashboard.

The application provides two prediction modes:

### Simple City Check

Users select one of the nine cities. The application uses the latest available historical record from that city's dataset to predict its AQI category.

### Manual Pollutant Input

Users select a city and manually enter values for the eight pollutant features. The application predicts the corresponding AQI category.

The dashboard also displays the predicted category and relevant air-quality guidance.

## 8. Installation and Execution

### Prerequisites

- Python 3.11 or later.
- Git installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmnaZahoor021/Air_Quality_Detection.git
```

Replace `YOUR_GITHUB_REPOSITORY_LINK` with your actual GitHub repository URL.

### Step 2: Navigate to the Project Directory

```bash
cd Air_Quality_Detection
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 4: Activate the Environment

For Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\\Scripts\\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

The trained model is already included in the `models/` folder.

```bash
streamlit run app.py
```

### Optional: Retrain the Models

To reproduce model training and regenerate the saved model files, execute:

```bash
python train_models.py
```

## 9. Important Limitations

- The application uses historical dataset records and does not provide live sensor monitoring.
- AQI categories 1–5 are project-defined classes, not official regulatory AQI thresholds.
- The category names displayed in the dashboard are user-facing labels for these dataset classes.
- Predictions are estimates based on the supplied dataset and are not medical advice or official monitoring-station readings.
- Model performance on new sensors, locations, or future periods may differ from the reported test results.

## 10. Dataset Source

The project uses nine city CSV files supplied for this project.

**Dataset Hosting Page:**

https://huggingface.co/datasets/mk12rule/pakistan_air_quality_dataset

This page is provided as the dataset hosting reference. The original upstream data provider and collection methodology should be verified separately if required for formal reporting.
