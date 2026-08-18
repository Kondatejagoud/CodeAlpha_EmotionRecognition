import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_svm_model(C: float = 1.0, kernel: str = 'rbf') -> Pipeline:
    """
    Returns a pipeline with a Standard Scaler and a Support Vector Classifier (SVM).
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(C=C, kernel=kernel, probability=True, random_state=42))
    ])
    return model

def save_sklearn_model(model: Pipeline, filepath: str):
    """
    Saves the scikit-learn model pipeline to disk.
    """
    joblib.dump(model, filepath)

def load_sklearn_model(filepath: str) -> Pipeline:
    """
    Loads the scikit-learn model pipeline from disk.
    """
    return joblib.load(filepath)
