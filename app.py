from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación
with open('modelo.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)


@app.route('/')
def home():
    return jsonify({
        "message": "API del modelo de IA funcionando",
        "status": "active"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()

        # Convertir a formato que entienda tu modelo
        # Ajusta según cómo sea tu modelo
        features = np.array(data['features']).reshape(1, -1)

        # Hacer predicción
        prediction = modelo.predict(features)
        probability = modelo.predict_proba(features)

        return jsonify({
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
