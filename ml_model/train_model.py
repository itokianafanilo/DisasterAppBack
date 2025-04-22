import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# 1. Charger les données
data = pd.read_csv('../data/toliara_hourly_completed.csv')

# 2. Gestion des valeurs manquantes (-999)
data.replace(-999, np.nan, inplace=True)
data.ffill(inplace=True)  # Correction du warning

### 3. Calcul des seuils de risques :
# a. Inondation
data['FloodRisk'] = (data['PRECTOTCORR'] > 50).astype(int) + \
                    (data['GWETROOT'] > 0.8).astype(int) + \
                    (data['PS'] < 100.5).astype(int) + \
                    (data['WS10M'] > 8.33).astype(int) + \
                    (data['CLOUD_AMT'] > 60).astype(int)

# b. Sécheresse
data['DroughtRisk'] = (data['PRECTOTCORR'] < 5).astype(int) + \
                      (data['RH2M'] < 30).astype(int) + \
                      (data['GWETROOT'] < 0.3).astype(int) + \
                      (data['T2M_MAX'] > 35).astype(int)

# c. Incendie
data['FireRisk'] = (data['T2M_MAX'] > 40).astype(int) + \
                   (data['RH2M'] < 25).astype(int) + \
                   (data['GWETROOT'] < 0.2).astype(int) + \
                   (data['WS10M'] > 11.1).astype(int) + \
                   (data['CLOUD_AMT'] < 20).astype(int) + \
                   (data['PRECTOTCORR'] < 2).astype(int)

# Normalisation des scores de risque (0 à 1)
data['FloodRisk'] /= 5  
data['DroughtRisk'] /= 4  
data['FireRisk'] /= 6  

### 4. Sélection des caractéristiques et des cibles
features = ['YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
            'PRECTOTCORR', 'PS', 'WS2M', 'WS10M', 'GWETROOT', 'CLOUD_AMT']

# Décaler `y` pour prévoir la catastrophe du lendemain
data['FloodRisk_next'] = data['FloodRisk'].shift(-24)
data['DroughtRisk_next'] = data['DroughtRisk'].shift(-24)
data['FireRisk_next'] = data['FireRisk'].shift(-24)

# Supprimer les lignes avec NaN après le décalage
data.dropna(subset=['FloodRisk_next', 'DroughtRisk_next', 'FireRisk_next'], inplace=True)

# Extraction des features et cibles après suppression des NaN
X = data[features].values

# Prendre une seule valeur par jour après le décalage
data_daily = data.iloc[::24]  # Sélectionne une seule ligne toutes les 24 heures
y_flood = data_daily['FloodRisk_next'].values
y_drought = data_daily['DroughtRisk_next'].values
y_fire = data_daily['FireRisk_next'].values

### 5. Normalisation des données
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Sauvegarder le scaler pour utilisation future
joblib.dump(scaler, 'scaler.pkl')

# Ajuster la taille pour être un multiple de 24
taille_multiple_24 = (X_scaled.shape[0] // 24) * 24
X_scaled = X_scaled[:taille_multiple_24]
y_flood = y_flood[:taille_multiple_24]
y_drought = y_drought[:taille_multiple_24]
y_fire = y_fire[:taille_multiple_24]

# Reformater X pour inclure une séquence temporelle de 24 heures
X_reshaped = X_scaled.reshape(-1, 24, len(features))  # (jours, 24h, nombre de features)

# Vérification finale
assert X_reshaped.shape[0] == len(y_flood), f"Erreur: X et y ne sont pas alignés ! {X_reshaped.shape[0]} != {len(y_flood)}"

# 6. Séparer les données en entraînement et test (sans mélanger les dates)
split_idx = int(0.8 * len(X_reshaped))

X_train, X_test = X_reshaped[:split_idx], X_reshaped[split_idx:]
y_flood_train, y_flood_test = y_flood[:split_idx], y_flood[split_idx:]
y_drought_train, y_drought_test = y_drought[:split_idx], y_drought[split_idx:]
y_fire_train, y_fire_test = y_fire[:split_idx], y_fire[split_idx:]

### 7. Fonction pour créer un modèle LSTM
def create_lstm_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(24, len(features))),
        tf.keras.layers.Dropout(0.2),  # Évite l'overfitting

        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),  # Couche Dense
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sortie probabiliste (0-1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# 8. Entraîner trois modèles distincts
models = {}

for y_train, y_test, risk_name in zip(
    [y_flood_train, y_drought_train, y_fire_train],
    [y_flood_test, y_drought_test, y_fire_test],
    ['Flood', 'Drought', 'Fire']
):
    print(f"Training model for {risk_name}...")
    
    model = create_lstm_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Sauvegarde du modèle
    model.save(f'model_{risk_name}.h5')
    models[risk_name] = model

# 9. Évaluation des modèles
for risk_name, model, y_test in zip(['Flood', 'Drought', 'Fire'], models.values(), [y_flood_test, y_drought_test, y_fire_test]):
    loss, mse = model.evaluate(X_test, y_test)
    print(f"{risk_name} Model - Test Loss: {loss}, MSE: {mse}")
