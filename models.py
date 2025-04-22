from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Disaster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    severity = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }

class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, unique=True)
    types = db.Column(db.String(200), default="flood,fire,drought")
    min_severity = db.Column(db.Integer, default=1)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "types": self.types,
            "min_severity": self.min_severity
        }

    def update_from_dict(self, data):
        self.types = data.get("types", self.types)
        self.min_severity = data.get("min_severity", self.min_severity)

class SafetyTip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    tip = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            "type": self.type,
            "tip": self.tip
        }

class OfflineData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(100), nullable=False)
    data = db.Column(db.Text, nullable=False)  # JSON string of cached data

    def to_dict(self):
        return {
            "region": self.region,
            "data": self.data
        }
