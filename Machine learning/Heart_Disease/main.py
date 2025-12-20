import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
print("----------------------------------------")

# Step 4: Making Predictions
y_preds=clf.predict(X_test)
print(y_preds)

print("----------------------------------------")
# Step 5: Evaluating the Model

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))

print("----------------------------------------")
#Trying with different models

from sklearn.svm import LinearSVC ,SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

models ={
    "Linear SVC": LinearSVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVC": SVC(),
    "Random Forest": RandomForestClassifier()
}

results={}

for model_name, model in models.items():
    model.fit(X_train,y_train)
    results[model_name]=model.score(X_test,y_test)

print(results)

print("----------------------------------------")

# Dataframe of results 
results_df=pd.DataFrame(results.values(),
                        results.keys(),
                        columns=["Accuracy"]
                        )
print(results_df)
print("----------------------------------------")

#Creating bar plot for results
results_df.plot.bar()


#Different hyperparameters for logistic regression
log_reg_grid={
    'C':np.logspace(-4,4,20),
    'solver':['liblinear']
}
np.random.seed(42)

#Hyperparameter tuning with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV  

rs_log_reg=RandomizedSearchCV(estimator=LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)
#fit instance of RandomizedSearchCV
rs_log_reg.fit(X_train,y_train)
print(rs_log_reg.best_params_)
print(rs_log_reg.score(X_test,y_test))
print("----------------------------------------")

#instnace of logreg with best hyperparameters
clf=LogisticRegression(**rs_log_reg.best_params_)
print(clf.fit(X_train,y_train))


#importing confusion matrix and classification report
from sklearn.metrics import confusion_matrix,classification_report
#import precision_score,recall_score,f1_score
from sklearn.metrics import precision_score,recall_score,f1_score
#import plot roc curve
from sklearn.metrics import RocCurveDisplay

#making predictions
y_preds=clf.predict(X_test)
#confusion matrix
print(confusion_matrix(y_test,y_preds))
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_preds),annot=True,cbar=False)
plt.ylabel("Predicted label")   
plt.xlabel("True label")
plt.title("Confusion Matrix")
plt.show()
#classification report
print(classification_report(y_test,y_preds))
#precision score
print(precision_score(y_test,y_preds))
#recall score
print(recall_score(y_test,y_preds))
#f1 score   
print(f1_score(y_test,y_preds))


#Exporting and importing the trained model
from joblib import dump,load

dump(clf,"heart_disease_model.joblib")

#for use it 
loaded_model=load("heart_disease_model.joblib")
print(loaded_model.score(X_test,y_test))