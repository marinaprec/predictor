    
import streamlit as st
import pickle
import numpy as np
import pandas as pd

model=pickle.load(open('model.pkl','rb'))


def predict(velocidad_viento,precipitacion,humedad):
    input=np.array([[velocidad_viento,precipitacion,humedad]]).astype(np.float64)
    pred=model.predict(input)
    return float(pred)

def main():
    
    st.title("Predictor temperatura máxima")
    velocidad_viento = st.text_input("velocidad_viento")
    precipitacion = st.text_input("precipitacion")
    humedad = st.text_input("humedad")

    

    if st.button("Predicción"):
        output=predict(velocidad_viento,precipitacion,humedad)
        st.success('La temperatura_maxima es {}'.format(output))


if __name__=='__main__':
    main()

