# traffic_sign_recognition/models.py

from django.db import models

class Prediction(models.Model):
    image = models.ImageField(upload_to='predictions/')
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()

    def __str__(self):
        return f"Prediction for {self.image.name}: {self.predicted_class} (Confidence: {self.confidence})"

class XAIImage(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='xai_images/')
    explanation_method = models.CharField(max_length=100, blank=True)
    features_used = models.CharField(max_length=100, blank=True)
    explanation = models.TextField(blank=True)

    def __str__(self):
        return f"XAI Image for {self.prediction.image.name}"