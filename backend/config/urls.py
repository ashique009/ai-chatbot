"""
URL configuration for chatpartner project.
"""
from django.contrib import admin
from django.urls import path, include
from .views import health_check

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/health/', health_check, name='health-check'),
    path('api/health-check/', health_check, name='health-check-alias'),
    path('api/', include('chat.urls')),
    path('api/proactive/', include('proactive_messaging.urls')),
    path('api/instagram/', include('instagram.urls')),
]
