# Databricks notebook source
# MAGIC %md
# MAGIC ![image info](https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/images/banner_1.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Taller: Construcción e implementación de árboles de decisión y métodos de ensamblaje
# MAGIC 
# MAGIC En este taller podrá poner en práctica los sus conocimientos sobre construcción e implementación de árboles de decisión y métodos de ensamblajes. El taller está constituido por 9 puntos, 5 relacionados con árboles de decisión (parte A) y 4 con métodos de ensamblaje (parte B).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte A - Árboles de decisión
# MAGIC 
# MAGIC En esta parte del taller se usará el conjunto de datos de Capital Bikeshare de Kaggle, donde cada observación representa el alquiler de una bicicleta durante una hora y día determinado. Para más detalles puede visitar los siguientes enlaces: [datos](https://github.com/justmarkham/DAT8/blob/master/data/bikeshare.csv), [dicccionario de datos](https://www.kaggle.com/c/bike-sharing-demand/data).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datos prestamo de bicicletas

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# Importación de librerías
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz

# COMMAND ----------

# Lectura de la información de archivo .csv
bikes = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/bikeshare.csv', index_col='datetime', parse_dates=True)
# Renombrar variable "count" a "total"
bikes.rename(columns={'count':'total'}, inplace=True)
# Crear la hora como una variable 
bikes['hour'] = bikes.index.hour
# Visualización
bikes.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 1 - Análisis descriptivo
# MAGIC 
# MAGIC Ejecute las celdas 1.1 y 1.2. A partir de los resultados realice un análisis descriptivo sobre las variables hour y workingday, escriba sus inferencias sobre los datos. Para complementar su análisis puede usar métricas como máximo, mínimo, percentiles entre otros.

# COMMAND ----------

# Celda 1.1
bikes.groupby('workingday').total.mean()

# COMMAND ----------

# Celda 1.2
bikes.groupby('hour').total.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 2 - Análisis de gráficos
# MAGIC 
# MAGIC Primero ejecute la celda 2.1 y asegúrese de comprender el código y el resultado. Luego, en cada una de celdas 2.2 y 2.3 escriba un código que genere una gráfica de las rentas promedio por hora cuando la variable "workingday" es igual a 0 e igual a 1, respectivamente. Analice y escriba sus hallazgos.

# COMMAND ----------

# Celda 2.1 - rentas promedio para cada valor de la variable "hour"
bikes.groupby('hour').total.mean().plot()

# COMMAND ----------

# Celda 2.2 - "workingday"=0 escriba su código y hallazgos 
1+1

# COMMAND ----------

# Celda 2.3 - "workingday"=1 escriba su código y hallazgos 


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 3 - Regresión lineal
# MAGIC En la celda 3 ajuste un modelo de regresión lineal a todo el conjunto de datos, utilizando "total" como variable de respuesta y "hour" y "workingday" como las únicas variables predictoras. Luego, imprima los coeficientes e interprételos. ¿Cuáles son las limitaciones de la regresión lineal en este caso?

# COMMAND ----------

# Celda 3


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 4 - Árbol de decisión manual
# MAGIC En la celda 4 cree un árbol de decisiones para pronosticar la variable "total" iterando **manualmente** sobre las variables "hour" y  "workingday". El árbol debe tener al menos 6 nodos finales.

# COMMAND ----------

# Celda 4


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 5 - Árbol de decisión con librería
# MAGIC En la celda 5 entrene un árbol de decisiones con la **librería sklearn**, usando las variables predictoras "hour" y "workingday" y calibre los parámetros que considere conveniente para obtener un mejor desempeño. Comente el desempeño del modelo con alguna métrica de desempeño de modelos de clasificación y compare desempeño con el modelo del punto 3.

# COMMAND ----------

# Celda 5


# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte B - Métodos de ensamblajes
# MAGIC En esta parte del taller se usará el conjunto de datos de Popularidad de Noticias Online. El objetivo es predecir la cantidad de reacciones en redes sociales (popularidad) de la notica. Para más detalles puede visitar el sigueinte enlace: [datos](https://archive.ics.uci.edu/ml/datasets/online+news+popularity).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datos popularidad de noticias

# COMMAND ----------

# Lectura de la información de archivo .csv
df = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/mashable.csv', index_col=0)
df.head()

# COMMAND ----------

# Definición variable de interes y variables predictoras
X = df.drop(['url', 'Popular'], axis=1)
y = df['Popular']
y.mean()

# COMMAND ----------

# División de la muestra en set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 6 - Árbol de decisión y regresión logística
# MAGIC En la celda 6 construya un árbol de decisión y una regresión logística. Para el árbol calibre al menos un parámetro y evalúe el desempeño de cada modelo usando las métricas de Accuracy y F1-Score.

# COMMAND ----------

# Celda 6


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 7 - Votación Mayoritaria
# MAGIC En la celda 7 elabore un esamble con la metodología de **Votación mayoritaria** compuesto por 300 muestras bagged para cada uno de los siguientes escenarios:
# MAGIC 
# MAGIC -100 árboles de decisión donde max_depth = None\
# MAGIC -100 árboles de decisión donde max_depth = 2\
# MAGIC -100 regresiones logísticas
# MAGIC 
# MAGIC Evalúe los modelos utilizando las métricas de Accuracy y F1-Score.

# COMMAND ----------

# Celda 7


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 8 - Votación Ponderada
# MAGIC En la celda 8 elabore un ensamble con la metodología de **Votación ponderada** compuesto por 300 muestras bagged para los mismos tres escenarios del punto 7. Evalúe los modelos utilizando las métricas de Accuracy y F1-Score

# COMMAND ----------

# Celda 8


# COMMAND ----------

# MAGIC %md
# MAGIC ### Punto 9 - Comparación y análisis de resultados
# MAGIC En la celda 9 comente sobre los resultados obtenidos con las metodologías usadas en los puntos 7 y 8, compare los resultados y enuncie posibles ventajas o desventajas de cada una de ellas.

# COMMAND ----------

# Celda 9
