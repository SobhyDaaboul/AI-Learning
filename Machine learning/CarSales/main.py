import pandas as pd

car_sales = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended-missing-data.csv")

print(car_sales.head())

# Get information about the car sales DataFrame
print(car_sales.info())

# Find number of missing values in each column
print(car_sales.isna().sum())

# Find the datatypes of each column of car_sales
print(car_sales.dtypes)

# Remove rows with no labels (NaN's in the Price column)
car_sales.dropna(subset=['Price'],inplace=True)

# Building a pipeline
# Since our car_sales data has missing numerical values as well as the data isn't all numerical, 
# we'll have to fix these things before we can fit a machine learning model on it.

# Import Pipeline from sklearn's pipeline module
from sklearn.pipeline import Pipeline

# Import ColumnTransformer from sklearn's compose module
from sklearn.compose import ColumnTransformer

# Import SimpleImputer from sklearn's impute module
from sklearn.impute import SimpleImputer

# Import OneHotEncoder from sklearn's preprocessing module
from sklearn.preprocessing import OneHotEncoder

# Import train_test_split from sklearn's model_selection module
from sklearn.model_selection import train_test_split

#Define different categorical features 
categorical_feature=["Make","Colour"]

# Create categorical transformer Pipeline
categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
    ("onehotencoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
])

# Define Doors features
doors_feature=['Doors']
doors_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant",fill_value=4)),
])

# Define numeric features (only the Odometer (KM) column)
odometer_feature=['Odometer (KM)']
odometer_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median"))
])

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor=ColumnTransformer(
    transformers=[
        ("cat",categorical_transformer,categorical_feature),
        ("door",doors_transformer,doors_feature),
        ("odometer",odometer_transformer,odometer_feature)
    ]
)

# Import Ridge from sklearn's linear_model module
from sklearn.linear_model import Ridge

# Import SVR from sklearn's svm module
from sklearn.svm import SVR

# Import RandomForestRegressor from sklearn's ensemble module
from sklearn.ensemble import RandomForestRegressor


# Create dictionary of model instances, there should be 4 total key, value pairs in the form {"model_name": model_instance}.

regression_model={
    'ridge':Ridge(),
    'SVR_Linear':SVR(kernel="linear"),
    'SVR_rbf':SVR(kernel="rbf"),
    'RandomForestRegressor':RandomForestRegressor()
}
result_regression={}


X=car_sales.drop("Price",axis=1)
y=car_sales["Price"]


# Use train_test_split to split the car_sales_X and car_sales_y data into training and test sets.
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# Check the shapes of the training and test datasets
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

 
# Loop through the items in the regression_models dictionary
for model_name, model in regression_model.items():
# Create a model Pipeline with a preprocessor step and model step
    model_pipeline=Pipeline(steps=[('preprocessor',preprocessor),
                                   ('model',model)])
    # Fit the model Pipeline to the car sales training data
    print(f"fiting{model_name}...")
    model_pipeline.fit(X_train,y_train)
    #Score the model Pipeline on the test data appending the model_name to the results dictionar
    result_regression[model_name]=model_pipeline.score(X_test,y_test)

print(result_regression)
print("-------------------------------------")

import matplotlib.pyplot as plt

models = list(result_regression.keys())
scores = list(result_regression.values())

plt.figure(figsize=(8, 5))
plt.bar(models, scores)
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Import mean_absolute_error,r2_score,mean_squared_error from sklearn's metrics module
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

ridge_pipeline =Pipeline(steps=[("preprocessor",preprocessor),
                                 ("model", Ridge())])


ridge_pipeline.fit(X_train,y_train)

y_preds = ridge_pipeline.predict(X_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()


type(y_preds)
print(y_preds[:50])

mse=mean_squared_error(y_test,y_preds)
print(mse)

mae=mean_absolute_error(y_test,y_preds)
print(mae)

r2score=r2_score(y_test,y_preds)
print(r2score)


