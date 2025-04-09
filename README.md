
```markdown
# Customer Churn Prediction using Apache Spark MLlib

This project demonstrates a complete machine learning pipeline to predict customer churn using Apache Spark's MLlib. The pipeline includes data preprocessing, model training, feature selection, and hyperparameter tuning with cross-validation.

## Dataset

The dataset used is `customer_churn.csv`, containing customer information and churn status. Key features include demographics, service usage, and billing information.

---

##  Tasks Overview

The assignment is divided into four tasks:

1. **Data Preprocessing and Feature Engineering**
2. **Train and Evaluate a Logistic Regression Model**
3. **Feature Selection using Chi-Square Test**
4. **Hyperparameter Tuning and Model Comparison**

---

##  How to Run the Project

### Prerequisites

- Python 3.x
- Apache Spark with PySpark
- Dataset: `customer_churn.csv` in the working directory

Install PySpark using pip:

```bash
pip install pyspark
```

### Run the Script

```bash
python churn_prediction.py
```

---

##  Task 1: Data Preprocessing and Feature Engineering

###  Objective
Prepare raw data for machine learning by handling missing values, encoding categorical variables, and assembling features.

###  Steps
- Fill missing values in `TotalCharges` with `0`.
- Convert `gender`, `PhoneService`, and `InternetService` to numeric indices using `StringIndexer`.
- One-hot encode the indexed categorical columns using `OneHotEncoder`.
- Combine all features (categorical and numerical) using `VectorAssembler` into a single `features` vector.
- Label column `Churn` is indexed to `ChurnIndexed`.

###  Sample Output
```text
+--------------------+------------+
|            features|ChurnIndexed|
+--------------------+------------+
|[1.0,0.0,1.0,0.0,...|         0.0|
|[0.0,1.0,0.0,1.0,...|         1.0|
+--------------------+------------+
```

---

##  Task 2: Train and Evaluate a Logistic Regression Model

###  Objective
Train a logistic regression model and evaluate its performance.

###  Steps
- Split the data (80% training, 20% testing).
- Train a `LogisticRegression` model on the training data.
- Evaluate predictions on the test set using `BinaryClassificationEvaluator` with the AUC metric.

###  Sample Output
```text
AUC (Area Under ROC Curve): 0.84612

+-----+----------+--------------------+
|Churn|prediction|        probability |
+-----+----------+--------------------+
|  No |       0.0|[0.918, 0.082]      |
| Yes|       1.0|[0.379, 0.621]       |
+-----+----------+--------------------+
```

---

##  Task 3: Feature Selection using Chi-Square Test

###  Objective
Select the top 5 most relevant features using the Chi-Square test.

###  Steps
- Use `ChiSqSelector` to identify the top 5 predictive features from the assembled `features` vector.
- Output includes the label and the selected features vector.

###  Sample Output
```text
+-----+--------------------+
|Churn|    selected_features|
+-----+--------------------+
| Yes |[0.0, 1.0, 0.0, ...]|
| No  |[1.0, 0.0, 0.0, ...]|
+-----+--------------------+
```

---

##  Task 4: Hyperparameter Tuning and Model Comparison

###  Objective
Use cross-validation to tune hyperparameters and compare performance of different classifiers.

###  Steps
- Define and tune models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosted Trees
- Use `ParamGridBuilder` to define parameter combinations.
- Apply 3-fold cross-validation using `CrossValidator`.
- Evaluate models on test data using AUC.

###  Sample Output
```text
AUC for Logistic Regression: 0.847
AUC for Decision Tree: 0.781
AUC for Random Forest: 0.862
AUC for Gradient Boosting: 0.879
```

---

##  Project Structure

```
.
├── churn_prediction.py         # Main script
├── customer_churn.csv          # Dataset
└── README.md                   # Documentation
```

---

##  Key Libraries Used

- `pyspark.sql.SparkSession`
- `pyspark.ml.feature`: `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `ChiSqSelector`
- `pyspark.ml.classification`: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GBTClassifier`
- `pyspark.ml.evaluation`: `BinaryClassificationEvaluator`
- `pyspark.ml.tuning`: `ParamGridBuilder`, `CrossValidator`

---
