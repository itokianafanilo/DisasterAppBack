import requests
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from geopy.geocoders import Nominatim
import overpy
import logging

from models import db

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Initialisation Flask ===
app = Flask(__name__)
# Configuration CORS pour permettre les requêtes depuis n'importe quelle origine
CORS(app, resources={r"/*": {"origins": "*"}})

# Dans main.py, modifiez la configuration de la base de données
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/disaster_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# === Import modèles ===
from models import Disaster, SafetyTip, UserPreferences

# === Chargement des modèles ML ===
try:
    model_flood = tf.keras.models.load_model("ml_model/model_Flood.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    model_drought = tf.keras.models.load_model("ml_model/model_Drought.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    model_fire = tf.keras.models.load_model("ml_model/model_Fire.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load("ml_model/scaler.pkl")
    logger.info("Modèles ML chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles ML: {str(e)}")
    # Créer des modèles fictifs pour le développement si les modèles ne peuvent pas être chargés
    class DummyModel:
        def predict(self, X):
            return np.array([[np.random.random()]])
    
    model_flood = DummyModel()
    model_drought = DummyModel()
    model_fire = DummyModel()
    scaler = None
    logger.warning("Utilisation de modèles fictifs pour le développement")

# === Configuration WeatherAPI.com ===
WEATHERAPI_KEY = "b9170461dc884ee0a72160526252603"  # Remplacez par votre clé API
WEATHERAPI_URL = "https://api.weatherapi.com/v1/forecast.json"

def get_weather_forecast_weatherapi(city, days=5):
    """
    Récupère les prévisions météo depuis WeatherAPI.com.
    WeatherAPI.com fournit des prévisions sur 14 jours dans le plan payant, 
    mais seulement 3 jours dans le plan gratuit.
    """
    logger.info(f"Récupération des prévisions météo pour {city} sur {days} jours")
    
    params = {
        "key": WEATHERAPI_KEY,
        "q": city,
        "days": min(days, 3),  # Maximum 3 jours dans le plan gratuit
        "aqi": "no",
        "alerts": "no"
    }
    
    try:
        response = requests.get(WEATHERAPI_URL, params=params)
        logger.info(f"Statut de la réponse WeatherAPI: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Données météo reçues pour {city}")
            return data
        else:
            logger.error(f"Erreur WeatherAPI: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception lors de la requête WeatherAPI: {str(e)}")
        return None

def get_mock_weather_forecast(city, days=7):
    """
    Génère des données météo fictives pour le développement.
    À utiliser en cas d'échec des API météo.
    """
    logger.info(f"Génération de données météo fictives pour {city} sur {days} jours")
    
    mock_data = {
        "location": {
            "name": city,
            "region": "Ile-de-France",
            "country": "France",
            "lat": 48.85,
            "lon": 2.35,
            "localtime": datetime.now().strftime("%Y-%m-%d %H:%M")
        },
        "forecast": {
            "forecastday": []
        }
    }
    
    # Générer des données pour chaque jour
    for i in range(days):
        day_data = {
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "day": {
                "avgtemp_c": 20 + i % 5,
                "avghumidity": 70 - i % 20,
                "totalprecip_mm": 5 - i % 5,
                "maxwind_kph": 15 + i % 10,
                "pressure_mb": 1013 + i % 10,
                "cloud": 50 - i % 30
            }
        }
        mock_data["forecast"]["forecastday"].append(day_data)
    
    return mock_data

def estimate_gwetroot(humidity, precip, pressure):
    return min(1, max(0, (humidity / 100) * (precip + 1) / (pressure / 100)))

def get_coordinates(city):
    try:
        loc = Nominatim(user_agent="risk_app").geocode(city)
        if loc:
            return (loc.latitude, loc.longitude)
        logger.warning(f"Impossible de trouver les coordonnées pour {city}")
        return (0, 0)  # Valeurs par défaut
    except Exception as e:
        logger.error(f"Erreur lors de la géolocalisation: {str(e)}")
        return (0, 0)  # Valeurs par défaut

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()["results"][0]["elevation"]
        logger.warning(f"Impossible d'obtenir l'élévation pour {lat},{lon}")
        return 0  # Valeur par défaut
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'élévation: {str(e)}")
        return 0  # Valeur par défaut

def has_nearby_water(lat, lon, radius=1000):
    try:
        query = f"""
        [out:json];(way(around:{radius},{lat},{lon})[water];relation(around:{radius},{lat},{lon})[water];);out body;"""
        result = overpy.Overpass().query(query)
        return len(result.ways) > 0
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des cours d'eau: {str(e)}")
        return False  # Valeur par défaut

def has_nearby_forest(lat, lon, radius=2000):
    try:
        query = f"""
        [out:json];(way(around:{radius},{lat},{lon})[landuse=forest];relation(around:{radius},{lat},{lon})[landuse=forest];);out body;"""
        result = overpy.Overpass().query(query)
        return len(result.ways) > 0
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des forêts: {str(e)}")
        return False  # Valeur par défaut

def preprocess_entry(date, daily_data, year, doy):
    """
    Prétraite les données météo pour les modèles ML.
    Gère les erreurs et les valeurs manquantes.
    """
    try:
        temp = daily_data.get("avgtemp_c", 20)
        humidity = daily_data.get("avghumidity", 70)
        precip = daily_data.get("totalprecip_mm", 0)
        pressure = daily_data.get("pressure_mb", 1013) / 10
        wind = daily_data.get("maxwind_kph", 15) / 3.6
        cloud = daily_data.get("cloud", 50)
        
        gwetroot = estimate_gwetroot(humidity, precip, pressure)
        
        features = np.array([[year, doy, temp, temp, temp, humidity, precip, pressure, wind, wind, gwetroot, cloud]])
        
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            # Si le scaler n'est pas disponible, normaliser manuellement
            features_scaled = features / np.array([[2023, 365, 50, 50, 50, 100, 100, 120, 30, 30, 1, 100]])
        
        return np.tile(features_scaled, (1, 24, 1))
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données: {str(e)}")
        # Retourner des valeurs par défaut en cas d'erreur
        default_features = np.zeros((1, 24, 12))
        return default_features

@app.route("/api/health", methods=["GET"])
def health_check():
    """Point de terminaison pour vérifier que l'API est en ligne"""
    return jsonify({"status": "ok", "message": "API is running"})

@app.route("/api/predict", methods=["GET"])
def predict():
    city = request.args.get("city", "Paris")
    logger.info(f"Requête de prédiction reçue pour la ville: {city}")
    
    try:
        # Essayer d'abord avec WeatherAPI
        forecast_data = get_weather_forecast_weatherapi(city, days=3)
        
        # Si WeatherAPI échoue, utiliser des données fictives
        if not forecast_data:
            logger.warning(f"Utilisation de données météo fictives pour {city}")
            forecast_data = get_mock_weather_forecast(city, days=7)
        
        lat, lon = get_coordinates(city)
        elevation = get_elevation(lat, lon)
        near_water = has_nearby_water(lat, lon)
        near_forest = has_nearby_forest(lat, lon)
        
        today = datetime.now()
        if "location" in forecast_data and "localtime" in forecast_data["location"]:
            try:
                today = datetime.strptime(forecast_data["location"]["localtime"], "%Y-%m-%d %H:%M")
            except:
                logger.warning("Format de date incorrect, utilisation de la date actuelle")
        
        predictions = {}
        available_days = len(forecast_data["forecast"]["forecastday"])
        logger.info(f"Nombre de jours disponibles dans les prévisions: {available_days}")
        
        # Utiliser le dernier jour disponible pour les prévisions à long terme
        last_available_day_data = forecast_data["forecast"]["forecastday"][-1]["day"]
        
        for offset, label in zip([0, 1, 7, 30, 365], ["Aujourdhui", "Dans_1_jour", "Dans_7_jours", "Dans_1_mois", "Dans_1_an"]):
            date = today + timedelta(days=offset)
            doy = date.timetuple().tm_yday
            year = date.year
            
            # Utiliser les données du dernier jour disponible pour les jours au-delà de ce que l'API fournit
            if offset < available_days:
                day_data = forecast_data["forecast"]["forecastday"][offset]["day"]
            else:
                day_data = last_available_day_data
            
            X_input = preprocess_entry(date, day_data, year, doy)
            
            try:
                flood_risk = float(model_flood.predict(X_input)[0][0])
                drought_risk = float(model_drought.predict(X_input)[0][0])
                fire_risk = float(model_fire.predict(X_input)[0][0])
                
                # Ajustements basés sur l'environnement
                if label in ["Aujourdhui", "Dans_1_jour"]:
                    if elevation < 50 or near_water:
                        flood_risk = min(1.0, flood_risk + 0.2)
                if near_forest:
                    fire_risk = min(1.0, fire_risk + 0.2)
                
                predictions[label] = {
                    "Innondation": float(flood_risk*100),
                    "Secheresse": float(drought_risk*100),
                    "Incendie_forestiere": float(fire_risk*100)
                }
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour {label}: {str(e)}")
                # Valeurs par défaut en cas d'erreur
                predictions[label] = {
                    "Innondation": 30.0,
                    "Secheresse": 30.0,
                    "Incendie_forestiere": 30.0
                }
        
        return jsonify({
            "Ville": city, 
            "elevation": elevation, 
            "cours_eau_pres": near_water, 
            "foret_pres": near_forest, 
            "predictions": predictions
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({
            "error": "Une erreur s'est produite lors de la prédiction",
            "message": str(e)
        }), 500

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    try:
        city = request.args.get("city", "Paris")
        forecast = get_weather_forecast_weatherapi(city, 3)
        if not forecast:
            forecast = get_mock_weather_forecast(city, 7)
        return jsonify(forecast)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prévisions météo: {str(e)}")
        return jsonify({"error": "Impossible d'obtenir les prévisions météo"}), 500

# Les autres routes restent inchangées...
@app.route("/api/disasters", methods=["GET"])
def get_disasters():
    try:
        disasters = Disaster.query.all()
        return jsonify([d.to_dict() for d in disasters])
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des catastrophes: {str(e)}")
        # Retourner des données fictives en cas d'erreur
        mock_disasters = [
            {
                "id": 1,
                "type": "flood",
                "description": "Inondation majeure suite aux fortes pluies",
                "latitude": 48.8566,
                "longitude": 2.3522,
                "severity": 3,
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": 2,
                "type": "fire",
                "description": "Incendie de forêt en progression",
                "latitude": 43.2965,
                "longitude": 5.3698,
                "severity": 2,
                "timestamp": datetime.now().isoformat()
            }
        ]
        return jsonify(mock_disasters)

@app.route("/api/disasters/<int:id>", methods=["GET"])
def get_disaster_detail(id):
    try:
        disaster = Disaster.query.get_or_404(id)
        return jsonify(disaster.to_dict())
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du détail de la catastrophe {id}: {str(e)}")
        return jsonify({"error": "Catastrophe non trouvée"}), 404

@app.route("/api/user/preferences", methods=["GET", "POST"])
def user_preferences():
    try:
        if request.method == "POST":
            data = request.json
            user_id = data.get("user_id")
            pref = UserPreferences.query.filter_by(user_id=user_id).first()
            
            if not pref:
                pref = UserPreferences(user_id=user_id)
            
            pref.types = data.get("types", pref.types)
            pref.min_severity = data.get("min_severity", pref.min_severity)
            
            db.session.add(pref)
            db.session.commit()
            return jsonify({"message": "Préférences enregistrées", "preferences": pref.to_dict()})
        else:
            user_id = request.args.get("user_id")
            pref = UserPreferences.query.filter_by(user_id=user_id).first()
            return jsonify(pref.to_dict() if pref else {"message": "Aucune préférence trouvée"})
    except Exception as e:
        logger.error(f"Erreur lors de la gestion des préférences utilisateur: {str(e)}")
        return jsonify({"error": "Erreur lors de la gestion des préférences utilisateur"}), 500

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    try:
        user_id = request.args.get("user_id")
        
        # Exemple de données de notification (à remplacer par des données réelles de la base de données)
        notifications = [
            {
                "id": "1",
                "title": "Alerte inondation",
                "details": "Risque d'inondation élevé dans votre région",
                "severity": "high",
                "type": "flood",
                "date": datetime.now().isoformat(),
                "read": False
            },
            {
                "id": "2",
                "title": "Alerte sécheresse",
                "details": "Restrictions d'eau en vigueur dans votre région",
                "severity": "medium",
                "type": "drought",
                "date": (datetime.now() - timedelta(days=1)).isoformat(),
                "read": True
            }
        ]
        
        return jsonify(notifications)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des notifications: {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération des notifications"}), 500

@app.route("/api/tips", methods=["GET"])
def get_tips():
    try:
        dtype = request.args.get("type")
        
        # Exemple de conseils de sécurité (à remplacer par des données réelles de la base de données)
        all_tips = {
            "flood": [
                {"id": 1, "tip": "Déplacez-vous vers un terrain plus élevé immédiatement"},
                {"id": 2, "tip": "Ne traversez jamais une zone inondée à pied ou en voiture"},
                {"id": 3, "tip": "Préparez un kit d'urgence avec de l'eau, de la nourriture et des médicaments"}
            ],
            "fire": [
                {"id": 4, "tip": "Évacuez immédiatement si les autorités vous le demandent"},
                {"id": 5, "tip": "Couvrez votre nez et votre bouche avec un tissu humide"},
                {"id": 6, "tip": "Fermez toutes les fenêtres et portes pour empêcher la fumée d'entrer"}
            ],
            "drought": [
                {"id": 7, "tip": "Limitez votre consommation d'eau"},
                {"id": 8, "tip": "Évitez d'arroser les jardins pendant les heures chaudes"},
                {"id": 9, "tip": "Récupérez l'eau de pluie pour un usage non potable"}
            ]
        }
        
        if dtype and dtype in all_tips:
            return jsonify(all_tips[dtype])
        return jsonify([tip for tips in all_tips.values() for tip in tips])
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des conseils: {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération des conseils"}), 500

@app.route("/api/offline-data", methods=["GET"])
def get_offline_data():
    try:
        disasters = [d.to_dict() for d in Disaster.query.all()]
        tips = [t.to_dict() for t in SafetyTip.query.all()]
        
        # Si la base de données est vide, utiliser des données fictives
        if not disasters:
            disasters = [
                {
                    "id": 1,
                    "type": "flood",
                    "description": "Inondation majeure suite aux fortes pluies",
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "severity": 3,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "id": 2,
                    "type": "fire",
                    "description": "Incendie de forêt en progression",
                    "latitude": 43.2965,
                    "longitude": 5.3698,
                    "severity": 2,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        if not tips:
            tips = [
                {"type": "flood", "tip": "Déplacez-vous vers un terrain plus élevé immédiatement"},
                {"type": "fire", "tip": "Évacuez immédiatement si les autorités vous le demandent"},
                {"type": "drought", "tip": "Limitez votre consommation d'eau"}
            ]
        
        return jsonify({"disasters": disasters, "tips": tips})
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données hors ligne: {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération des données hors ligne"}), 500

@app.route("/api/regions", methods=["GET"])
def get_regions():
    """Renvoie la liste des régions disponibles pour les cartes hors ligne"""
    try:
        regions = [
            {
                "id": "1",
                "name": "Paris et Île-de-France",
                "size": "45 MB",
                "image": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34"
            },
            {
                "id": "2",
                "name": "Lyon et Rhône-Alpes",
                "size": "38 MB",
                "image": "https://images.unsplash.com/photo-1524396309943-e03f5249f002"
            },
            {
                "id": "3",
                "name": "Marseille et PACA",
                "size": "42 MB",
                "image": "https://images.unsplash.com/photo-1589708532758-dfa3c4ff81b1"
            },
            {
                "id": "4",
                "name": "Bordeaux et Nouvelle-Aquitaine",
                "size": "36 MB",
                "image": "https://images.unsplash.com/photo-1589708532758-dfa3c4ff81b1"
            },
            {
                "id": "5",
                "name": "Lille et Hauts-de-France",
                "size": "30 MB",
                "image": "https://images.unsplash.com/photo-1589708532758-dfa3c4ff81b1"
            }
        ]
        return jsonify(regions)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des régions: {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération des régions"}), 500

# Initialisation de la base de données au démarrage
with app.app_context():
    try:
        db.create_all()
        logger.info("Base de données initialisée au démarrage de l'application")
        
        # Vérifier si des données existent déjà
        if Disaster.query.count() == 0:
            # Ajouter des catastrophes de test
            disasters = [
                Disaster(
                    type="flood",
                    description="Inondation majeure suite aux fortes pluies",
                    latitude=48.8566,
                    longitude=2.3522,
                    severity=3
                ),
                Disaster(
                    type="fire",
                    description="Incendie de forêt en progression",
                    latitude=43.2965,
                    longitude=5.3698,
                    severity=2
                ),
                Disaster(
                    type="drought",
                    description="Sécheresse sévère affectant les cultures",
                    latitude=44.8378,
                    longitude=0.5792,
                    severity=2
                )
            ]
            
            for disaster in disasters:
                db.session.add(disaster)
            
            logger.info(f"{len(disasters)} catastrophes ajoutées")
        
        # Ajouter des conseils de sécurité s'il n'y en a pas
        if SafetyTip.query.count() == 0:
            tips = [
                SafetyTip(type="flood", tip="Déplacez-vous vers un terrain plus élevé immédiatement"),
                SafetyTip(type="flood", tip="Ne traversez jamais une zone inondée à pied ou en voiture"),
                SafetyTip(type="flood", tip="Préparez un kit d'urgence avec de l'eau, de la nourriture et des médicaments"),
                SafetyTip(type="fire", tip="Évacuez immédiatement si les autorités vous le demandent"),
                SafetyTip(type="fire", tip="Couvrez votre nez et votre bouche avec un tissu humide"),
                SafetyTip(type="fire", tip="Fermez toutes les fenêtres et portes pour empêcher la fumée d'entrer"),
                SafetyTip(type="drought", tip="Limitez votre consommation d'eau"),
                SafetyTip(type="drought", tip="Évitez d'arroser les jardins pendant les heures chaudes"),
                SafetyTip(type="drought", tip="Récupérez l'eau de pluie pour un usage non potable")
            ]
            
            for tip in tips:
                db.session.add(tip)
            
            logger.info(f"{len(tips)} conseils de sécurité ajoutés")
        
        db.session.commit()
        logger.info("Données de test ajoutées avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")

if __name__ == "__main__":
    # Démarrer l'application
    app.run(debug=True, host='0.0.0.0')


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