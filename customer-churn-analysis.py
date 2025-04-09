from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import udf, col
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import ArrayType, DoubleType, StringType

# Function to extract vector values from the OneHotEncoder output
def extract_vector_values(vector):
    return vector.toArray().tolist()

# Register the UDF
extract_values_udf = udf(extract_vector_values, ArrayType(DoubleType()))

# Function to convert array of doubles to a comma-separated string
def array_to_string(array):
    return ",".join([str(i) for i in array])

# Register the UDF
array_to_string_udf = udf(array_to_string, StringType())

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Fill missing values
    df = df.fillna({'TotalCharges': 0})

    # Encode categorical variables using StringIndexer
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
    phone_service_indexer = StringIndexer(inputCol="PhoneService", outputCol="phone_service_index")
    internet_service_indexer = StringIndexer(inputCol="InternetService", outputCol="internet_service_index")
    churn_indexer = StringIndexer(inputCol="Churn", outputCol="ChurnIndexed")
    
    # Apply One-Hot Encoding to the indexed columns
    gender_encoder = OneHotEncoder(inputCol="gender_index", outputCol="gender_onehot")
    phone_service_encoder = OneHotEncoder(inputCol="phone_service_index", outputCol="phone_service_onehot")
    internet_service_encoder = OneHotEncoder(inputCol="internet_service_index", outputCol="internet_service_onehot")

    # Combine all features into a single feature vector
    assembler = VectorAssembler(inputCols=["gender_onehot", "phone_service_onehot", "internet_service_onehot", 
                                           "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
                                outputCol="features")

    # Create a pipeline to apply all transformations
    pipeline = Pipeline(stages=[gender_indexer, phone_service_indexer, internet_service_indexer, churn_indexer,
                                 gender_encoder, phone_service_encoder, internet_service_encoder, assembler])
    
    preprocessed_df = pipeline.fit(df).transform(df)
    
    preprocessed_df.select("features", "ChurnIndexed").show()
    
    return preprocessed_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    # Split data into training and testing sets (80% training, 20% testing)
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Train the Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndexed")
    lr_model = lr.fit(train_data)
    
    # Make predictions on the test data
    predictions = lr_model.transform(test_data)
    
    # Evaluate the model using AUC (Area Under the ROC Curve)
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndexed", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    
    # Print the AUC score
    print(f"AUC (Area Under ROC Curve): {auc}")

    # Show a few predictions
    predictions.select("Churn", "prediction", "probability").show(5)

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    # Feature selection using Chi-Square Test
    chi_selector = ChiSqSelector(featuresCol="features", labelCol="ChurnIndexed", outputCol="selected_features", numTopFeatures=5)
    selected_df = chi_selector.fit(df).transform(df)

    # Show selected features
    selected_df.select("Churn", "selected_features").show(5)

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # Split data into training and test sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Define models
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndexed")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="ChurnIndexed")
    rf = RandomForestClassifier(featuresCol="features", labelCol="ChurnIndexed")
    gbt = GBTClassifier(featuresCol="features", labelCol="ChurnIndexed")

    # Define hyperparameter grids
    param_grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.5, 0.7]).build()
    param_grid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).addGrid(dt.maxBins, [32, 64]).build()
    param_grid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()
    param_grid_gbt = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20]).addGrid(gbt.maxDepth, [5, 10]).build()

    # Define evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndexed", metricName="areaUnderROC")

    # Perform cross-validation for each model
    cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=param_grid_lr, evaluator=evaluator, numFolds=3)
    cv_dt = CrossValidator(estimator=dt, estimatorParamMaps=param_grid_dt, evaluator=evaluator, numFolds=3)
    cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=param_grid_rf, evaluator=evaluator, numFolds=3)
    cv_gbt = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid_gbt, evaluator=evaluator, numFolds=3)

    # Fit models with cross-validation
    cv_lr_model = cv_lr.fit(train_data)
    cv_dt_model = cv_dt.fit(train_data)
    cv_rf_model = cv_rf.fit(train_data)
    cv_gbt_model = cv_gbt.fit(train_data)

    # Get the best models
    best_lr_model = cv_lr_model.bestModel
    best_dt_model = cv_dt_model.bestModel
    best_rf_model = cv_rf_model.bestModel
    best_gbt_model = cv_gbt_model.bestModel

    # Make predictions on the test data
    lr_predictions = best_lr_model.transform(test_data)
    dt_predictions = best_dt_model.transform(test_data)
    rf_predictions = best_rf_model.transform(test_data)
    gbt_predictions = best_gbt_model.transform(test_data)

    # Evaluate models
    auc_lr = evaluator.evaluate(lr_predictions)
    auc_dt = evaluator.evaluate(dt_predictions)
    auc_rf = evaluator.evaluate(rf_predictions)
    auc_gbt = evaluator.evaluate(gbt_predictions)

    # Print the AUC scores for each model
    print(f"AUC for Logistic Regression: {auc_lr}")
    print(f"AUC for Decision Tree: {auc_dt}")
    print(f"AUC for Random Forest: {auc_rf}")
    print(f"AUC for Gradient Boosting: {auc_gbt}")
    
    # Save the output of Task 4 (cross-validation results)


# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
