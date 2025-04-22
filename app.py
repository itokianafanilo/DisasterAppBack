import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, jsonify
from tensorflow.keras.metrics import MeanSquaredError
import joblib

# Votre clé API WeatherAPI
API_KEY = "b9170461dc884ee0a72160526252603"

# Fonction pour obtenir les prévisions météorologiques pour Antsirabe avec WeatherAPI
def get_weather_forecast():
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q=Antsirabe&days=3"
    response = requests.get(url)
    data = response.json()
    
    # Extraire les données météo nécessaires pour les 3 prochains jours
    forecast_data = []
    for day in data['forecast']['forecastday']:
        day_data = day['day']
        forecast_data.append({
            'QV2M': day_data['avghumidity'],            # Humidité
            'RH2M': day_data['avgtemp_c'],              # Température moyenne
            'PRECTOTCORR': day_data['totalprecip_mm'],  # Précipitations totales
            'PS': 1020,                                 # Pression atmosphérique (valeur par défaut)
            'WS50M': day_data['maxwind_kph'],           # Vitesse du vent max
            'WS10M': day_data['maxwind_kph'],           # Même vitesse pour simplifier
            'TS': day_data['mintemp_c'],                # Température minimale
            'CLRSKY_SFC_SW_DWN': day_data['condition']['code']  # Code de condition météo pour les nuages
        })
    
    df = pd.DataFrame(forecast_data)
    print("\nDonnées météo reçues :", df)  # Log pour vérification
    return df

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('innondation.html')

@app.route('/predict', methods=['GET'])
def predict_innondation():
    # Charger le modèle
    model = tf.keras.models.load_model('ml_model/model.h5', custom_objects={'mse': MeanSquaredError()})

    # Charger le scaler entraîné
    scaler = joblib.load('ml_model/scaler.pkl')

    # Obtenir les prévisions météorologiques pour Antsirabe
    forecast_df = get_weather_forecast()

    # Vérifier que les colonnes attendues sont bien présentes
    expected_features = ['QV2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS50M', 'WS10M', 'TS', 'CLRSKY_SFC_SW_DWN']
    if not all(col in forecast_df.columns for col in expected_features):
        return jsonify({'error': 'Les données météo ne contiennent pas toutes les colonnes attendues'})

    # Normalisation des données avec le scaler entraîné
    X_scaled = scaler.transform(forecast_df)

    # Reshape pour le modèle LSTM (1 sample, nombre de jours, nombre de features)
    X_scaled = np.reshape(X_scaled, (1, X_scaled.shape[0], X_scaled.shape[1]))

    # Faire des prédictions
    predictions = model.predict(X_scaled)

    # Transformation en score de risque
    temperature_predite = predictions[0][0]  # Extraction de la température prédite
    humidite_actuelle = forecast_df['QV2M'].mean()  # Moyenne humidité

    # Seuils basés sur l'analyse des données historiques
    seuil_temperature = 22  # Ex: Température moyenne des jours sans inondation
    seuil_humidite = 80     # Ex: Seuil d’humidité pour considérer un risque

    # Calcul d’un score de risque pondéré
    risque = max(0, min(1, (humidite_actuelle / seuil_humidite) * (seuil_temperature / max(temperature_predite, 1))))

    # Convertir en pourcentage
    risque = round(risque * 100, 2)

    return jsonify({'probabilite_risque': risque})

if __name__ == '__main__':
    app.run(debug=True)
