"""
Classic machine learning algorithm implementations: KNN, Naive Bayes, SVM, Decision Tree.
Corresponds to Section 2.2 and Section 4.4 (Figures 4.5 ~ 4.8) of the paper.
"""
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def _run_sklearn_model(model, name, X_train, y_train, X_test, y_test):
    """Generic sklearn training and evaluation routine."""
    t1 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    t2 = time.time()
    acc = accuracy_score(y_test, pred)
    elapsed = t2 - t1
    print(f'[{name}] Test accuracy: {acc:.4f}, time spent: {elapsed:.2f}s')
    return {
        'model': model,
        'accuracy': acc,
        'time': elapsed,
        'pred': pred,
    }


def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Gaussian Naive Bayes - paper Section 2.2.3."""
    return _run_sklearn_model(
        GaussianNB(), 'NaiveBayes',
        X_train, y_train, X_test, y_test
    )


def train_decision_tree(X_train, y_train, X_test, y_test, **kwargs):
    """Decision tree - paper Section 2.2.4."""
    return _run_sklearn_model(
        DecisionTreeClassifier(random_state=0, **kwargs), 'DecisionTree',
        X_train, y_train, X_test, y_test
    )


def train_svm(X_train, y_train, X_test, y_test, **kwargs):
    """Support Vector Machine - paper Section 2.2.5.

    Note: SVM training is extremely slow on large datasets; a subset can be used during testing.
    """
    return _run_sklearn_model(
        SVC(**kwargs), 'SVM',
        X_train, y_train, X_test, y_test
    )


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    """K-Nearest Neighbors - paper Section 2.2.2."""
    return _run_sklearn_model(
        KNeighborsClassifier(n_neighbors=n_neighbors), 'KNN',
        X_train, y_train, X_test, y_test
    )
