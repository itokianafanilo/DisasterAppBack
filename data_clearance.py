import pandas as pd
import numpy as np

data_toliara = pd.read_csv('data/toliara_data.csv', sep=",")  #Données contenant CLOUD_AMT
data_toliara_without = pd.read_csv('data/toliara_hourly_2014_2024_without_cloud_rootzone.csv', sep=",")  #Données horaires à compléter
data_toliara_root = pd.read_csv('data/toliara_daily_root_tm.csv', sep=",")  #Données journalières

#Filtrer les données de 2014 à 2024 dans data_toliara
df_filtre = data_toliara[(data_toliara["YEAR"] >= 2014) & (data_toliara["YEAR"] <= 2024)]
df_cloud = df_filtre[["YEAR", "DOY", "CLOUD_AMT"]]

#Filtrer les données journalières dans data_toliara_root
df_root_filtre = data_toliara_root[(data_toliara_root["YEAR"] >= 2014) & (data_toliara_root["YEAR"] <= 2024)]
df_root_filtre = df_root_filtre[["YEAR", "DOY", "T2M_MIN", "T2M_MAX", "GWETROOT"]]

#Fusionner les données journalières avec les données horaires (en utilisant YEAR et DOY)
data_toliara_without = data_toliara_without.merge(df_cloud, on=["YEAR", "DOY"], how="left")
data_toliara_without = data_toliara_without.merge(df_root_filtre, on=["YEAR", "DOY"], how="left")

#Sauvegarder le fichier complété
data_toliara_without.to_csv("data/toliara_hourly_completed.csv", index=False)

print("Fait")
