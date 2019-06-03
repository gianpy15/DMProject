import joblib

def save(model, path):
    joblib.dump(model, path)

def load(path):
    return joblib.load(path) 
