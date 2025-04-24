# init_db.py
from main import app, db
from models import Disaster, SafetyTip, UserPreferences
from datetime import datetime

def init_database():
    with app.app_context():
        # Créer les tables
        db.create_all()
        print("Tables créées avec succès")
        
        # Vérifier si des données existent déjà
        if Disaster.query.count() == 0:
            # Ajouter des catastrophes
            disasters = [
                Disaster(
                    type="flood",
                    description="Inondation majeure suite aux fortes pluies",
                    latitude=48.8566,
                    longitude=2.3522,
                    severity=3,
                    timestamp=datetime.now()
                ),
                Disaster(
                    type="fire",
                    description="Incendie de forêt en progression",
                    latitude=43.2965,
                    longitude=5.3698,
                    severity=2,
                    timestamp=datetime.now()
                ),
                Disaster(
                    type="drought",
                    description="Sécheresse sévère affectant les cultures",
                    latitude=44.8378,
                    longitude=0.5792,
                    severity=2,
                    timestamp=datetime.now()
                )
            ]
            
            for disaster in disasters:
                db.session.add(disaster)
            print(f"{len(disasters)} catastrophes ajoutées")
        
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
            print(f"{len(tips)} conseils de sécurité ajoutés")
        
        # Ajouter un utilisateur de test avec des préférences
        if UserPreferences.query.count() == 0:
            test_user_pref = UserPreferences(
                user_id=1,
                types="flood,fire,drought",
                min_severity=1
            )
            
            db.session.add(test_user_pref)
            print("Préférences utilisateur de test ajoutées")
        
        # Valider les changements
        db.session.commit()
        print("Base de données initialisée avec succès")

if __name__ == "__main__":
    init_database()