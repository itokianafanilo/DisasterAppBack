import requests
import numpy as np
#import tensorflow as tf
import joblib
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template


# Configuration de l'API WeatherAPI
API_KEY = "b9170461dc884ee0a72160526252603"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

# Fonction pour récupérer les données météo
def get_weather_data(city):
    params = {"key": API_KEY, "q": city}
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    return None

#city = request.args.get("city", "Antananarivo")
weather_data = get_weather_data("Antananarivo")
print(weather_data)