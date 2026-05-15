# DisasterAppBack

Backend Flask pour un projet de prédiction des risques de catastrophes naturelles.

Ce dépôt contient l'API utilisée par le frontend `DisasterAppFront`. Le backend récupère des données météo, prépare les variables nécessaires aux modèles de machine learning, puis renvoie des scores de risque pour plusieurs types de catastrophes.

## Objectif du projet

Le projet vise à estimer les risques de catastrophes naturelles à partir de données météorologiques et géographiques.

Les risques actuellement pris en compte sont :

- inondation ;
- sécheresse ;
- incendie forestier.

L'API peut aussi exposer des données utiles au frontend, comme les événements enregistrés, les notifications, les préférences utilisateur, les conseils de sécurité et des données utilisables hors ligne.

## Structure du projet

```text
DisasterAppBack/
|-- main.py                 # API Flask principale
|-- app.py                  # Ancienne version / démo centrée sur l'inondation
|-- models.py               # Modèles SQLAlchemy
|-- data_clearance.py       # Préparation et fusion de données CSV
|-- data_test.py            # Petit script de test des données
|-- api_test.py             # Petit script de test WeatherAPI
|-- data/                   # Données météo CSV
|-- ml_model/               # Modèles entraînés et scaler
|   |-- model_Flood.h5
|   |-- model_Drought.h5
|   |-- model_Fire.h5
|   |-- scaler.pkl
|   `-- train_model.py
|-- static/                 # Fichiers statiques
`-- templates/              # Pages HTML de test
```

## Fonctionnement général

Le fichier principal est `main.py`.

Au démarrage, il :

1. initialise une application Flask ;
2. active CORS pour permettre au frontend d'appeler l'API ;
3. configure une base de données MySQL ;
4. charge trois modèles TensorFlow/Keras déjà entraînés ;
5. charge le scaler utilisé pour normaliser les données ;
6. expose des routes API pour les prédictions et les données applicatives.

Pour une prédiction, l'API récupère les prévisions météo d'une ville via WeatherAPI, géolocalise la ville, récupère son altitude, vérifie la présence de cours d'eau ou de forêt à proximité, puis calcule des scores de risque.

## Prérequis

- Python 3.10 ou plus récent recommandé
- MySQL
- Une base de données nommée `disaster_db`
- Les modèles `.h5` présents dans `ml_model/`
- Une clé WeatherAPI

Le projet utilise notamment :

- Flask
- Flask-CORS
- Flask-SQLAlchemy
- PyMySQL
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- joblib
- requests
- geopy
- overpy

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install flask flask-cors flask-sqlalchemy pymysql tensorflow numpy pandas scikit-learn joblib requests geopy overpy
```

## Configuration

Dans `main.py`, la base MySQL est actuellement configurée comme ceci :

```python
mysql+pymysql://root:@localhost/disaster_db
```

Il faut donc avoir une base locale `disaster_db`, ou modifier cette URL selon votre configuration MySQL.

La clé WeatherAPI est actuellement écrite directement dans le code. Pour un projet en production, il est préférable de la placer dans une variable d'environnement.

## Lancement

Lancer l'API principale :

```bash
python main.py
```

Par défaut, Flask démarre sur :

```text
http://127.0.0.1:5000
```

Initialiser les tables SQLAlchemy si nécessaire :

```bash
flask --app main init-db
```

## Routes principales

### Prédiction

```http
GET /predict?city=Antananarivo
```

Retourne les risques estimés pour la ville demandée.

Exemple de réponse :

```json
{
  "Ville": "Antananarivo",
  "elevation": 1276,
  "cours_eau_pres": false,
  "foret_pres": true,
  "predictions": {
    "Aujourdhui": {
      "Innondation": 12.5,
      "Secheresse": 30.1,
      "Incendie_forestiere": 42.8
    }
  }
}
```

### Données météo brutes

```http
GET /api/forecast?city=Antananarivo
```

Retourne les prévisions météo utilisées comme base de calcul.

### Données applicatives

```http
GET /api/disasters
GET /api/disasters/<id>
GET /api/notifications?user_id=<id>
GET /api/tips?type=<type>
GET /api/offline-data
```

### Préférences utilisateur

```http
GET /api/user/preferences?user_id=<id>
POST /api/user/preferences
```

## Entraînement des modèles

Le script d'entraînement se trouve dans :

```text
ml_model/train_model.py
```

Il utilise les données CSV du dossier `data/`, prépare les risques cibles, entraîne trois modèles LSTM séparés, puis sauvegarde :

- `model_Flood.h5`
- `model_Drought.h5`
- `model_Fire.h5`
- `scaler.pkl`

Pour le lancer depuis le dossier `ml_model/` :

```bash
python train_model.py
```

## Lien avec le frontend

Le frontend est dans un dépôt séparé :

```text
DisasterAppFront
```

Ce backend doit être lancé avant le frontend pour que les appels API fonctionnent correctement. Grâce à CORS, le frontend peut appeler l'API Flask pendant le développement local.

## Notes importantes

- `main.py` est le point d'entrée principal du backend.
- `app.py` semble être une ancienne version ou une page de test centrée sur les inondations.
- Aucun fichier `requirements.txt` n'est présent pour le moment.
- La configuration sensible, comme la clé WeatherAPI et l'accès MySQL, devrait idéalement être déplacée dans des variables d'environnement.
- Les modèles SQLAlchemy et certaines routes doivent rester synchronisés si de nouvelles entités sont ajoutées.
