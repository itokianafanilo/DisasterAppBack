import requests
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from geopy.geocoders import Nominatim
import overpy

from models import db

# === Initialisation Flask ===
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/disaster_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# === Import modèles ===
#from models import DisasterEvent, UserPreference, Notification, Tip
from models import *

# === Chargement des modèles ML ===
model_flood = tf.keras.models.load_model("ml_model/model_Flood.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
model_drought = tf.keras.models.load_model("ml_model/model_Drought.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
model_fire = tf.keras.models.load_model("ml_model/model_Fire.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler = joblib.load("ml_model/scaler.pkl")

API_KEY = "b9170461dc884ee0a72160526252603"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"

def get_weather_forecast(city, days=7):
    params = {"key": API_KEY, "q": city, "days": days}
    r = requests.get(WEATHER_API_URL, params=params)
    return r.json() if r.status_code == 200 else None

def estimate_gwetroot(humidity, precip, pressure):
    return min(1, max(0, (humidity / 100) * (precip + 1) / (pressure / 100)))

def get_coordinates(city):
    loc = Nominatim(user_agent="risk_app").geocode(city)
    return (loc.latitude, loc.longitude) if loc else (None, None)

def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    return r.json()["results"][0]["elevation"] if r.status_code == 200 else 0

def has_nearby_water(lat, lon, radius=1000):
    query = f"""
    [out:json];(way(around:{radius},{lat},{lon})[water];relation(around:{radius},{lat},{lon})[water];);out body;"""
    return len(overpy.Overpass().query(query).ways) > 0

def has_nearby_forest(lat, lon, radius=2000):
    query = f"""
    [out:json];(way(around:{radius},{lat},{lon})[landuse=forest];relation(around:{radius},{lat},{lon})[landuse=forest];);out body;"""
    return len(overpy.Overpass().query(query).ways) > 0

def preprocess_entry(date, daily_data, year, doy):
    temp = daily_data["avgtemp_c"]
    humidity = daily_data["avghumidity"]
    precip = daily_data["totalprecip_mm"]
    pressure = daily_data.get("pressure_mb", 1013) / 10
    wind = daily_data["maxwind_kph"] / 3.6
    cloud = daily_data.get("cloud", 50)
    gwetroot = estimate_gwetroot(humidity, precip, pressure)
    features = np.array([[year, doy, temp, temp, temp, humidity, precip, pressure, wind, wind, gwetroot, cloud]])
    return np.tile(scaler.transform(features), (1, 24, 1))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    city = request.args.get("city", "Antananarivo")
    forecast_data = get_weather_forecast(city, days=10)
    if not forecast_data: return jsonify({"error": "Erreur API météo"}), 500

    lat, lon = get_coordinates(city)
    elevation = get_elevation(lat, lon)
    near_water = has_nearby_water(lat, lon)
    near_forest = has_nearby_forest(lat, lon)

    today = datetime.strptime(forecast_data["location"]["localtime"], "%Y-%m-%d %H:%M")
    predictions = {}

    for offset, label in zip([0, 1, 7, 30, 365], ["Aujourdhui", "Dans_1_jour", "Dans_7_jours", "Dans_1_mois", "Dans_1_an"]):
        date = today + timedelta(days=offset)
        doy = date.timetuple().tm_yday
        year = date.year
        day_data = forecast_data["forecast"]["forecastday"][min(offset, 9)]["day"]
        X_input = preprocess_entry(date, day_data, year, doy)

        flood_risk = model_flood.predict(X_input)[0][0]
        drought_risk = model_drought.predict(X_input)[0][0]
        fire_risk = model_fire.predict(X_input)[0][0]

        if label in ["Aujourdhui", "Dans_1_jour"]:
            if elevation < 50 or near_water: flood_risk = min(1.0, flood_risk + 0.2)
        if near_forest: fire_risk = min(1.0, fire_risk + 0.2)

        predictions[label] = {"Innondation": float(flood_risk*100), "Secheresse": float(drought_risk*100), "Incendie_forestiere": float(fire_risk*100)}

    return jsonify({"Ville": city, "elevation": elevation, "cours_eau_pres": near_water, "foret_pres": near_forest, "predictions": predictions})

@app.route("/api/disasters", methods=["GET"])
def get_disasters():
    disasters = DisasterEvent.query.all()
    return jsonify([d.to_dict() for d in disasters])

@app.route("/api/disasters/<int:id>", methods=["GET"])
def get_disaster_detail(id):
    disaster = DisasterEvent.query.get_or_404(id)
    return jsonify(disaster.to_dict())

@app.route("/api/user/preferences", methods=["GET", "POST"])
def user_preferences():
    if request.method == "POST":
        data = request.json
        user_id = data.get("user_id")
        pref = UserPreference.query.get(user_id) or UserPreference(user_id=user_id)
        pref.disaster_types = data.get("disaster_types")
        pref.alert_frequency = data.get("alert_frequency")
        pref.location = data.get("location")
        db.session.add(pref)
        db.session.commit()
        return jsonify({"message": "Préférences enregistrées"})
    else:
        user_id = request.args.get("user_id")
        pref = UserPreference.query.get(user_id)
        return jsonify(pref.to_dict() if pref else {})

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    user_id = request.args.get("user_id")
    notifications = Notification.query.filter_by(user_id=user_id).all()
    return jsonify([n.to_dict() for n in notifications])

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    city = request.args.get("city", "Antananarivo")
    forecast = get_weather_forecast(city, 7)
    return jsonify(forecast)

@app.route("/api/tips", methods=["GET"])
def get_tips():
    dtype = request.args.get("type")
    tips = Tip.query.filter_by(disaster_type=dtype).all()
    return jsonify([t.to_dict() for t in tips])

@app.route("/api/offline-data", methods=["GET"])
def get_offline_data():
    disasters = [d.to_dict() for d in DisasterEvent.query.all()]
    tips = [t.to_dict() for t in Tip.query.all()]
    return jsonify({"disasters": disasters, "tips": tips})

if __name__ == "__main__":
    app.run(debug=True)

@app.cli.command("init-db")
def init_db():
    with app.app_context():
        db.create_all()
        print("Base de données créée avec succès")



"""
import requests
import numpy as np
import tensorflow as tf
import joblib
import json
from datetime import datetime

from datetime import timedelta
from flask import Flask, request, jsonify, render_template

from geopy.geocoders import Nominatim
import folium
import overpy
import rasterio
from rasterio.plot import show
import osmnx as ox

# === Chargement des modèles ===
model_flood = tf.keras.models.load_model("ml_model/model_Flood.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
model_drought = tf.keras.models.load_model("ml_model/model_Drought.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
model_fire = tf.keras.models.load_model("ml_model/model_Fire.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# === Scaler ===
scaler = joblib.load("ml_model/scaler.pkl")

# === API météo ===
API_KEY = "b9170461dc884ee0a72160526252603"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"

# === Flask ===
app = Flask(__name__)

# === Fonction utilitaires ===
def get_weather_forecast(city, days=7):
    params = {
        "key": API_KEY,
        "q": city,
        "days": days
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def estimate_gwetroot(humidity, precip, pressure):
    return min(1, max(0, (humidity / 100) * (precip + 1) / (pressure / 100)))

def get_coordinates(city):
    geolocator = Nominatim(user_agent="risk_app")
    location = geolocator.geocode(city)
    return location.latitude, location.longitude

def get_elevation(lat, lon):
    # Utilisation de l'API Open-Elevation
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()["results"][0]["elevation"]
    return 0

def has_nearby_water(lat, lon, radius=1000):
    api = overpy.Overpass()
    query = f""" """
    [out:json];
    (
      way(around:{radius},{lat},{lon})[water];
      relation(around:{radius},{lat},{lon})[water];
    );
    out body;
    """ """
    result = api.query(query)
    return len(result.ways) > 0 or len(result.relations) > 0

def has_nearby_forest(lat, lon, radius=2000):
    api = overpy.Overpass()
    query = f""" """
    [out:json];
    (
      way(around:{radius},{lat},{lon})[landuse=forest];
      relation(around:{radius},{lat},{lon})[landuse=forest];
    );
    out body;
    """ """
    result = api.query(query)
    return len(result.ways) > 0 or len(result.relations) > 0

def preprocess_entry(date, daily_data, year, doy):
    temp = daily_data["avgtemp_c"]
    humidity = daily_data["avghumidity"]
    precip = daily_data["totalprecip_mm"]
    pressure = daily_data.get("pressure_mb", 1013) / 10  # fallback
    wind = daily_data["maxwind_kph"] / 3.6
    cloud = daily_data.get("cloud", 50)  # fallback

    gwetroot = estimate_gwetroot(humidity, precip, pressure)

    features = np.array([[year, doy, temp, temp, temp, humidity, precip,
                          pressure, wind, wind, gwetroot, cloud]])
    features_scaled = scaler.transform(features)
    X_input = np.tile(features_scaled, (1, 24, 1))
    return X_input

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    city = request.args.get("city", "Antananarivo")
    forecast_data = get_weather_forecast(city, days=10)
    if not forecast_data:
        return jsonify({"error": "Erreur lors de l'obtention des données météo."}), 500

    lat, lon = get_coordinates(city)
    elevation = get_elevation(lat, lon)
    near_water = has_nearby_water(lat, lon)
    near_forest = has_nearby_forest(lat, lon)

    today = datetime.strptime(forecast_data["location"]["localtime"], "%Y-%m-%d %H:%M")
    results = {
        "city": city,
        "elevation": elevation,
        "near_water": near_water,
        "near_forest": near_forest,
        "predictions": {
            "today": {},
            "1_day": {},
            "7_days": {},
            "30_days": {},
            "365_days": {}
        }
    }

    drought_model_inputs = {}

    for offset, label in zip([0, 1, 7, 30, 365], ["today", "1_day", "7_days", "30_days", "365_days"]):
        date = today + timedelta(days=offset)
        doy = date.timetuple().tm_yday
        year = date.year

        if offset <= 9:
            day_data = forecast_data["forecast"]["forecastday"][min(offset, 9)]["day"]
        else:
            day_data = forecast_data["forecast"]["forecastday"][-1]["day"]  # réutilise le dernier dispo

        X_input = preprocess_entry(date, day_data, year, doy)

        flood_risk = model_flood.predict(X_input)[0][0]
        drought_risk = model_drought.predict(X_input)[0][0]
        fire_risk = model_fire.predict(X_input)[0][0]

        # Ajustement selon l'élévation et présence de cours d'eau
        if label in ["today", "1_day"]:
            if elevation < 50 or near_water:
                flood_risk = min(1.0, flood_risk + 0.2)

        if near_forest:
            fire_risk = min(1.0, fire_risk + 0.2)

        results["predictions"][label] = {
            "flood": float(flood_risk),
            "drought": float(drought_risk),
            "fire": float(fire_risk)
        }

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
"""