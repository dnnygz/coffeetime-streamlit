# CoffeeTime Café - Predicción de retorno

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

La aplicación fue desplegada en Streamlit Community Cloud mediante la integración con un repositorio público de GitHub, permitiendo su acceso desde cualquier dispositivo con conexión a internet, sin necesidad de instalar Python o ejecutar el código localmente.

**Aplicación web:** [CoffeeTime Café - Predicción de retorno de clientes](https://coffeetime-dayana-gomez.streamlit.app/)

La aplicación permite ingresar las características de un cliente y obtener, mediante el modelo Random Forest previamente entrenado:

- Predicción de retorno durante los próximos 30 días.
- Probabilidad estimada de retorno.
- Interpretación del resultado de la clasificación.

### Configuración del despliegue

| Parámetro | Configuración |
|---|---|
| Repositorio | coffeetime-streamlit |
| Rama | main |
| Archivo principal | app.py |
| Modelo entrenado | model.joblib |
| Plataforma | Streamlit Community Cloud |
| Estado | Desplegado |

## Análisis de correlación y limitaciones del modelo

Se analizaron las correlaciones entre las variables numéricas y la variable objetivo `regresa_30_dias`, obteniendo los siguientes resultados:

| Variable | Correlación absoluta |
|---|---:|
| Días desde la última visita | 0.231 |
| Visitas en los últimos 90 días | 0.188 |
| Frecuencia de compras | 0.166 |
| Descuentos utilizados | 0.071 |
| Edad | 0.055 |
| Calificación promedio | 0.055 |
| Gasto promedio | 0.042 |
| Programa de fidelidad | 0.011 |

Los resultados muestran una correlación lineal débil entre las variables individuales y la variable objetivo. Las características con mayor asociación corresponden al comportamiento de compra y la recurrencia de los clientes.

El modelo Random Forest obtuvo un Accuracy del 68.67 %. Las bajas correlaciones sugieren una capacidad predictiva individual limitada de las variables disponibles, lo que podría contribuir a explicar por qué no se alcanza un Accuracy cercano al 90 %.

Aunque Random Forest permite capturar relaciones no lineales e interacciones entre variables, los resultados obtenidos evidencian que existe margen de mejora mediante la incorporación de nuevas características, ingeniería de variables y optimización de hiperparámetros.
