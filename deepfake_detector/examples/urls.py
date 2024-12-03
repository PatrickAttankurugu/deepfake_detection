from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_video, name='analyze_video'),
] 