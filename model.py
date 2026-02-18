import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


df = pd.read_csv("asl_angles.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=15, gamma="scale", probability=True)),
    ]
)

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val accuracy: {scores.mean():.4f}")

model.fit(X, y)
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
