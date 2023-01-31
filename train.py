import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import tree

def train(dataset: pd.DataFrame):
    # Prepare dataset
    X = dataset.drop(labels=["is_happy"], axis="columns")
    y = dataset["is_happy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # We use a pipeline to scale the input before training or inference
    clf = tree.DecisionTreeClassifier()
    # Train model
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # decision boundary
    # print(clf._final_estimator.coef_)
    # print(clf._final_estimator.intercept_)

    return clf
