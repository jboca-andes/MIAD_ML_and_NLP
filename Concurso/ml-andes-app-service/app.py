from flask import Flask, request, json
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

def preparar(entrada):
    original = pd.read_csv('assets/complemento.csv')
    original['tipo'] = 'Train'
    entrada['tipo'] = 'Test'
    fin = pd.concat([original, entrada])
    fin = pd.get_dummies(fin, drop_first = True)
    fin = fin[fin.tipo_Train==0]
    fin = fin.drop(['tipo_Train', 'Price'], axis=1)
    to_pred = fin.to_numpy()
    return to_pred

@app.route('/xgb_andes',  methods = ['POST'])
def xgb_andes():
    salida = {
    }
    try:
        model = pickle.load(open('assets/modelo_andes.pkl', 'rb'))
    except Exception as ex:
        salida["Mensaje"] = "Ocurrió un error al cargar el modelo"
        salida["Descripcion"] = str(ex)
        salida = json.dumps(salida)
        codigo = 400
    try:
        data = pd.DataFrame(request.get_json())
    except Exception as ex:
        salida["Mensaje"] = "El request no está bien formado, revise el JSON en el body"
        salida["Descripcion"] = str(ex)
        salida = json.dumps(salida)
        codigo = 400
    try:
        datos_prediccion = preparar(data)
        salida = pd.DataFrame(model.predict(datos_prediccion), columns=["Prediccion"]).to_json()
        codigo = 200
    except Exception as ex:
        salida["Mensaje"] = "Ocurrió un error al realizar la predicción"
        salida["Descripcion"] = str(ex)
        salida = json.dumps(salida)
        codigo = 400
    
    response = app.response_class(
        response=salida,
        status=codigo,
        mimetype='application/json'
    )

    return response


if __name__ == '__main__': 
    app.run()