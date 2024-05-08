from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/<int:prediction_id>/', views.results, name='results'),
]
