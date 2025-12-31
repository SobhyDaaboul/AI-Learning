import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# load the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# Info about the data 
train_dataset.info()
test_dataset.info()

print("---------------------------------")

# First 5 rows in the dataset 
print(train_dataset.head())
print(test_dataset.head())

#check if dataset is balanced 
print(train_dataset["Survived"].value_counts())

#find number of missing values 
print(train_dataset.isna().sum())
print(test_dataset.isna().sum())


# Visualize relationships 
plt.figure()
sns.countplot(data=train_dataset, x="Sex", hue="Survived")
plt.title("Survival Count by Sex")
plt.show()

plt.figure()
sns.countplot(data=train_dataset, x="Pclass", hue="Survived")
plt.title("survival count by class")
plt.show()

plt.figure()
sns.histplot(data=train_dataset, x="Age", hue="Survived", bins=30, kde=True)
plt.title("survival count by Age")
plt.show()

plt.figure()
sns.countplot(data=train_dataset, x="Embarked", hue="Survived")
plt.title("survival count by Embarked")
plt.show()


# Data Preprocessing 
train=train_dataset.drop(["Cabin","Ticket"], axis=1)
test=test_dataset.drop(["Cabin","Ticket"],axis=1)

X=train.drop("Survived",axis=1)
y=train['Survived']

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)


# handle missing values 
numeric_features = ["Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

num_imputer = SimpleImputer(strategy="median")

X_train[numeric_features] = num_imputer.fit_transform(X_train[numeric_features])
X_test[numeric_features] = num_imputer.transform(X_test[numeric_features])

cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])

print(X_train.isna().sum())
print(X_test.isna().sum())

#Encoding 
categorical_features = ["Sex", "Embarked"]
numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# model building 
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Feature Importance
feature_names = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    rf_pipeline.named_steps["model"]
    .feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Feature Importances - Random Forest")
plt.show()


