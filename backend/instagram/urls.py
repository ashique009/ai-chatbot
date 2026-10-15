"""
URL routing for Meta's Instagram Messaging API integration.
"""
from django.urls import path
from .views import instagram_webhook_view, instagram_status_view

urlpatterns = [
    # Official Meta Webhook (GET for verification handshake, POST for event receiver)
    path('webhook/', instagram_webhook_view, name='instagram-webhook'),

    # Sanitized status & metrics diagnostic (Authenticated)
    path('status/', instagram_status_view, name='instagram-status'),
]
