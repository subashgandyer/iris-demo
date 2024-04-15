import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

def train():
    df = pd.read_csv("Iris.csv")
    X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y=df[['Species']]

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)

    model_filename = "My_KNN_Model.model"

    joblib.dump(model, model_filename)

    loaded_model = joblib.load(model_filename)
    predict1 = loaded_model.predict(X_test)
    # assert(metrics.accuracy_score(predict1, y_test), 0.9666666666666667)
    return metrics.accuracy_score(predict1, y_test)

def test_train():
    assert train() == 0.9666666666666667
