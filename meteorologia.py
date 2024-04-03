
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("datosMeteorologia.csv")
df = df.fillna(0)


# Crear una lista de las columnas predictoras
predictors = ['temperatura_media','temperatura_minima','humedad','presion_atmosferica','velocidad_viento','racha_viento','precipitacion','media_precipitacion','punto_rocio','sensacion_termica',]

# Crear un DataFrame con las columnas predictoras
X = df[predictors]
print(X)
# Crear un DataFrame con la variable objetivo (temperatura_maxima)
y = df['temperatura_maxima']
print(y)
# Crear y entrenar el modelo de regresion lineal multiple
model = LinearRegression()
model.fit(X, y)

pickle.dump(model,open('model.pkl','wb'))