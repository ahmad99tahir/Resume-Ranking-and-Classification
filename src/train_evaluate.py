from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train):
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    return clf

from sklearn import metrics

def evaluate_model(clf, X_test, y_test):
    # Predict using the trained model
    prediction = clf.predict(X_test)
    
    # Print classification report
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))


