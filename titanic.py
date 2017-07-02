import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold


df = pd.read_csv("train.csv")


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
    

avg_age = np.mean(df["Age"])
df["Age"].fillna(avg_age, inplace = True)


def train_model(model, predictors, outcome, data):
    model.fit(data[predictors], data[outcome])
    predictions = model.predict(data[predictors])
    score = accuracy_score(data[outcome],predictions)
    print("Accuracy : %s" % "{0:.3%}".format(score))
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
model = RandomForestClassifier()
predictors = ["Pclass", "Age", "Sex"]
outcome = "Survived"
train_model(model, predictors, outcome, df)

def test_model(model, predictors, outcome, data):
    predictions = model.predict(data[predictors])
    final = pd.DataFrame({"PassengerId":data["PassengerId"], "Survived":predictions})
    final.to_csv("predictions.csv")
    
df1 = pd.read_csv("test.csv")
df1["Sex"] = le.fit_transform(df1["Sex"])
avg_age1 = np.mean(df1["Age"])
df1["Age"].fillna(avg_age1, inplace = True)

test_model(model, predictors, outcome, df1)
