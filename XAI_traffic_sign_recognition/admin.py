from django.contrib import admin
from .models import Prediction, XAIImage


admin.site.register(Prediction)
admin.site.register(XAIImage)

