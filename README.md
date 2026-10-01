# Estimador del valor de vivienda en California

Modelo predictivo que estima el **valor mediano de las viviendas de un distrito censal de California** a partir de su ubicación, el ingreso de los hogares y las características de las viviendas. Incluye una aplicación web hecha con Streamlit para usar el modelo.

**Práctica 3 · Minería de Datos en Python · Universidad Pontificia Bolivariana**
Daniel Cardona González · Juan José Tamayo Ospina

**Aplicación en línea:** _agregar aquí el enlace de Streamlit Community Cloud_

## Resultados

El modelo final es un **XGBoost hiperparametrizado con GridSearch**. Estas son sus métricas en el 30% de los datos que se reservó desde el inicio y no se usó para entrenar ni para elegir hiperparámetros:

| Métrica | Valor |
|---|---|
| R² | 0,855 |
| MAE (error absoluto medio) | 28.338 USD |
| RMSE | 43.577 USD |
| MAPE (error relativo medio) | 16,0% |

Comparación de los modelos con validación cruzada de 5 particiones, con la configuración inicial de cada técnica:

| Modelo | R² entrenamiento | R² validación | RMSE validación (USD) | Diagnóstico |
|---|---|---|---|---|
| XGBoost | 1,000 | 0,822 | 48.761 | Overfitting fuerte |
| Random Forest | 0,937 | 0,808 | 50.603 | Overfitting moderado |
| Árbol de regresión | 0,817 | 0,718 | 61.356 | Overfitting leve |
| Red neuronal (objetivo estandarizado) | 0,730 | 0,712 | 61.955 | Underfitting leve |
| KNN | 0,804 | 0,695 | 63.839 | Overfitting leve |
| SVM (objetivo estandarizado) | 0,691 | 0,684 | 64.971 | Underfitting leve |
| Línea base (predice la media) | 0,000 | 0,000 | 115.737 | Referencia |

Después de hiperparametrizar XGBoost, el RMSE de validación bajó de 48.761 a 45.396 USD y el R² de entrenamiento pasó de 1,000 a 0,946, es decir, el modelo dejó de memorizar los datos.

## Metodología

1. **Preparación de los datos:** imputación de nulos, codificación de `ocean_proximity` con variables dummies y análisis de correlaciones.
2. **Selección de factores:** se midió la relevancia de 14 variables candidatas con correlación, información mutua e importancia por permutación, y se validaron varios conjuntos con validación cruzada. Se eligieron 9 variables: longitud, latitud, antigüedad, ingreso, personas por vivienda, cuartos por vivienda y las tres dummies de proximidad al océano.
3. **Modelos con validación cruzada:** árbol de regresión, KNN, red neuronal, SVM, Random Forest y XGBoost, evaluados con K-fold de 5 particiones y comparando entrenamiento contra validación para diagnosticar overfitting y underfitting.
4. **Hiperparametrización:** GridSearchCV sobre XGBoost con 48 combinaciones (240 entrenamientos). Mejores hiperparámetros: `max_depth=6`, `learning_rate=0.05`, `n_estimators=600`, `min_child_weight=1`, `subsample=0.8`.
5. **Modelo final:** XGBoost con esos hiperparámetros, reentrenado con el 100% de los datos y guardado en formato JSON de XGBoost.
6. **Despliegue:** aplicación web con Streamlit.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Practica3_MineriaDeDatos_DanielCardona_JuanJoseTamayo.ipynb` | Notebook de modelos: preparación, selección de factores, validación cruzada, GridSearch y guardado del modelo final |
| `Despliegue_Streamlit_DanielCardona_JuanJoseTamayo.ipynb` | Notebook de despliegue: prueba del modelo y ejecución de la aplicación en Google Colab |
| `app.py` | Aplicación Streamlit |
| `requirements.txt` | Librerías necesarias |
| `.streamlit/config.toml` | Tema de la aplicación |
| `modelo_xgb_final.json` | Modelo final entrenado |
| `modelo_info.json` | Variables del modelo, hiperparámetros, métricas e importancia de las variables |
| `housing.csv` | Base de datos |
| `pantallazo_despliegue.png` | Pantallazo de la aplicación |

## Qué hace la aplicación

- Recibe los datos de un distrito en su escala original: ubicación (por ciudad de referencia o por coordenadas), ingreso mediano, antigüedad de las viviendas, número de cuartos, población y hogares. La proximidad al océano se detecta según los distritos vecinos o se elige manualmente.
- Muestra el valor estimado con su rango de error típico, el percentil frente a toda California y la comparación con los 10 distritos reales más cercanos.
- Advierte cuando los datos ingresados son poco comunes o cuando la estimación se acerca al tope del censo.
- **Mapa:** todos los distritos de California coloreados por su valor real.
- **Análisis:** cómo cambia el valor al variar el ingreso o la antigüedad, y la distribución de valores en California.
- **Sobre el modelo:** métricas, importancia de las variables, hiperparámetros y limitaciones.
- **Predicción por lotes:** estima muchos distritos a la vez desde un archivo CSV.

## Datos

[California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices): 20.640 distritos censales de California con datos del censo de 1990.

| Variable | Descripción |
|---|---|
| `longitude`, `latitude` | Ubicación del distrito |
| `housing_median_age` | Antigüedad mediana de las viviendas (años) |
| `total_rooms`, `total_bedrooms` | Total de cuartos y de dormitorios del bloque |
| `population`, `households` | Población y número de hogares del bloque |
| `median_income` | Ingreso mediano de los hogares (decenas de miles de USD) |
| `ocean_proximity` | Proximidad al océano |
| `median_house_value` | Valor mediano de la vivienda en USD (variable objetivo) |

## Limitaciones

- Los datos son del censo de 1990, así que los valores no corresponden a precios actuales.
- El censo registró como 500.001 USD todos los valores superiores. En esos distritos el error promedio del modelo es de unos 58.700 USD, frente a unos 26.900 en el resto.
- La información está agregada por bloque censal: el modelo estima el valor típico de un sector, no el de una vivienda en particular.
