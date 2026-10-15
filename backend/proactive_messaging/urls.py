"""
URL routing for proactive messaging: preferences, inbox, and scheduled check-ins.
"""
from django.urls import path
from .views import (
    companion_preferences_view,
    inbox_messages_view,
    inbox_message_detail_view,
    inbox_mark_all_read_view,
    scheduled_messages_view,
    scheduled_message_detail_view,
    generate_daily_schedule_view,
    push_vapid_public_key_view,
    push_subscribe_view,
    push_unsubscribe_view,
    push_subscriptions_list_view,
    push_subscription_delete_view,
    push_send_test_notification_view,
)

urlpatterns = [
    # Preferences & Quiet Hours Configuration
    path('preferences/', companion_preferences_view, name='proactive-preferences'),

    # In-App Companion Inbox
    path('inbox/', inbox_messages_view, name='proactive-inbox-list'),
    path('inbox/<uuid:message_id>/', inbox_message_detail_view, name='proactive-inbox-detail'),
    path('inbox/read-all/', inbox_mark_all_read_view, name='proactive-inbox-read-all'),

    # Scheduled Check-Ins
    path('scheduled/', scheduled_messages_view, name='proactive-scheduled-list'),
    path('scheduled/<uuid:message_id>/', scheduled_message_detail_view, name='proactive-scheduled-detail'),
    path('schedule-daily/', generate_daily_schedule_view, name='proactive-schedule-daily'),

    # Browser Web Push Notifications
    path('push/vapid-public-key/', push_vapid_public_key_view, name='proactive-push-vapid-public-key'),
    path('push/subscribe/', push_subscribe_view, name='proactive-push-subscribe'),
    path('push/unsubscribe/', push_unsubscribe_view, name='proactive-push-unsubscribe'),
    path('push/subscriptions/', push_subscriptions_list_view, name='proactive-push-subscriptions-list'),
    path('push/subscriptions/<uuid:subscription_id>/', push_subscription_delete_view, name='proactive-push-subscription-delete'),
    path('push/test/', push_send_test_notification_view, name='proactive-push-test'),
]
