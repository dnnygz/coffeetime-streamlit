# CoffeeTime Café — Predicción de retorno
**Autora:** Gomez Rodriguez Dayana Kety

Actividad2_GomezRodriguezDayanaKety_Deployment

Aplicación de clasificación binaria para estimar el retorno de clientes en 30 días.
Dataset: 1500 registros históricos, 10 variables predictoras; se excluye cliente_id.
Modelo: Random Forest de 200 árboles, OneHotEncoder para categorías, división estratificada 80/20, semilla 42.

## Ejecución local (Python 3.13)
```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
El modelo ya está entrenado; el segundo comando solo es necesario para reentrenarlo.

## Evaluación (300 registros de prueba)
Accuracy: 68.67%.

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 (no regresa) | 60.00% | 36.45% | 45.35% |
| 1 (regresa) | 71.06% | 86.53% | 78.04% |

Matriz de confusión: TN=39, FP=68, FN=26, TP=167.
El modelo identifica bien a quienes regresan, pero detecta solo 36.45% de quienes no regresan; esto limita su utilidad para campañas de recuperación.

## Despliegue
Crear repositorio público `coffeetime-streamlit`. Subir app.py, model.joblib, requirements.txt, README.md y opcionalmente train_model.py. No publicar el CSV salvo que se confirme que es sintético o anonimizado.
En https://share.streamlit.io/ seleccionar el repositorio, rama main, archivo app.py y Python 3.13 en configuración avanzada. Desplegar y probar la URL pública.
