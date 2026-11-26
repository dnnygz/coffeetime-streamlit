from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "clientes_historicos.csv")
X = df.drop(columns=["cliente_id", "regresa_30_dias"])
y = df["regresa_30_dias"]
categoricas = ["canal_compra", "producto_favorito"]
pre = ColumnTransformer(
    [("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas)],
    remainder="passthrough"
)
model = Pipeline([
    ("preprocesamiento", pre),
    ("clasificador", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, digits=4, zero_division=0))
print("Matriz de confusion (filas: real; columnas: prediccion; orden: 0, 1):")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
joblib.dump(model, BASE / "model.joblib", compress=3)
print("Modelo guardado: model.joblib")
