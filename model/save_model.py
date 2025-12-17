import joblib
from model.train import model, rf_model


joblib.dump(model, "model/logistic_model.pkl")
joblib.dump(rf_model, "model/rf_model.pkl")

print("Models saved successfully")
