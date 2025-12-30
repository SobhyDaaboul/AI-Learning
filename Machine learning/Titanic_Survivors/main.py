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
sex_encoder = LabelEncoder()
X_train["Sex"] = sex_encoder.fit_transform(X_train["Sex"])
X_test["Sex"] = sex_encoder.transform(X_test["Sex"])

embarked_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
embarked_encoded_train = embarked_encoder.fit_transform(
    X_train[["Embarked"]]
)
embarked_encoded_test = embarked_encoder.transform(
    X_test[["Embarked"]]
)
embarked_cols = embarked_encoder.get_feature_names_out(["Embarked"])
embarked_train_df = pd.DataFrame(
    embarked_encoded_train,
    columns=embarked_cols,
    index=X_train.index
)
embarked_test_df = pd.DataFrame(
    embarked_encoded_test,
    columns=embarked_cols,
    index=X_test.index
)
X_train = X_train.drop("Embarked", axis=1)
X_test = X_test.drop("Embarked", axis=1)
X_train = pd.concat([X_train, embarked_train_df], axis=1)
X_test = pd.concat([X_test, embarked_test_df], axis=1)
print(X_train.dtypes)
print(X_test.dtypes)







# categorical_features=["Sex","Embarked"]

# categorical_transformer=Pipeline(steps=[
#     ("imputer",SimpleImputer(strategy="most_frequent"))
#     ("oneHotEncoder",OneHotEncoder())
# ])
# numeric_features=["Age","Fare"]




