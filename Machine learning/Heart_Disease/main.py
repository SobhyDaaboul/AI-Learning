import pandas as pd
# Step 1: Getting the Data Ready
heart_disease=pd.read_csv("heart.csv")
print(heart_disease.head())

X=heart_disease.drop("target",axis=1)
y=heart_disease['target']


from sklearn.model_selection import train_test_split
# Splitting the data into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y)
#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# Step 2: Choosing Model and Training it
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()


# Step 3: Fitting the model to the training data
clf.fit(X_train,y_train)


# Step 4: Making Predictions
y_preds=clf.predict(X_test)
print(y_preds)

# Step 5: Evaluating the Model

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))