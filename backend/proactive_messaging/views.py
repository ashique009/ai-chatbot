"""
Authenticated REST API views for proactive companion messaging:
- Companion Preferences & Settings (quiet hours, daily limits, language, tone)
- In-App Inbox (proactive check-ins delivered to the companion chat)
- Scheduled Messages (listing, scheduling custom check-ins, cancellation)
- Automated daily schedule generation
"""
import datetime
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from django.conf import settings
from proactive_messaging.models import (
    ScheduledMessage,
    CompanionPreference,
    InAppMessage,
    PushDeviceSubscription,
)
from proactive_messaging.services.delivery.push import PushNotificationDeliveryHandler
from proactive_messaging.services.exceptions import PushDeliveryError, PushProviderNotConfiguredError
from proactive_messaging.services.scheduler import (
    create_daily_checkin_schedule,
    schedule_single_checkin,
)


def _serialize_preference(pref: CompanionPreference) -> dict:
    return {
        "companion_name": pref.companion_name,
        "user_nickname": pref.user_nickname,
        "preferred_language": pref.preferred_language,
        "companion_tone": pref.companion_tone,
        "relationship_style": pref.relationship_style,
        "proactive_messaging_enabled": pref.proactive_messaging_enabled,
        "frequency": pref.frequency,
        "quiet_hours_enabled": pref.quiet_hours_enabled,
        "quiet_hours_start": pref.quiet_hours_start.strftime("%H:%M"),
        "quiet_hours_end": pref.quiet_hours_end.strftime("%H:%M"),
        "timezone": pref.timezone,
        "max_daily_messages": pref.max_daily_messages,
        "min_minutes_between_messages": pref.min_minutes_between_messages,
        "updated_at": pref.updated_at.isoformat() if pref.updated_at else None,
    }


def _serialize_in_app_message(msg: InAppMessage) -> dict:
    return {
        "id": str(msg.id),
        "sender_name": msg.sender_name,
        "content": msg.content,
        "message_type": msg.message_type,
        "is_read": msg.is_read,
        "created_at": msg.created_at.isoformat(),
        "scheduled_message_id": str(msg.scheduled_message_id) if msg.scheduled_message_id else None,
    }


def _serialize_scheduled_message(msg: ScheduledMessage) -> dict:
    return {
        "id": str(msg.id),
        "message_type": msg.message_type,
        "message_type_display": msg.get_message_type_display(),
        "scheduled_time": msg.scheduled_time.isoformat(),
        "status": msg.status,
        "channel": msg.channel,
        "is_ai_generated": msg.is_ai_generated,
        "content": msg.content,
        "attempts": msg.attempts,
        "max_attempts": msg.max_attempts,
        "error_message": msg.error_message,
    }


def _serialize_push_subscription(sub: PushDeviceSubscription) -> dict:
    return {
        "id": str(sub.id),
        "device_type": sub.device_type,
        "endpoint": sub.endpoint,
        "is_active": sub.is_active,
        "created_at": sub.created_at.isoformat() if sub.created_at else None,
        "updated_at": sub.updated_at.isoformat() if sub.updated_at else None,
    }



# ============================================================================
# Companion Preferences API
# ============================================================================

@api_view(['GET', 'PATCH', 'PUT'])
@permission_classes([IsAuthenticated])
def companion_preferences_view(request):
    """
    GET: Retrieve the companion preferences for the authenticated user.
    PATCH / PUT: Update proactive messaging toggles, quiet hours, limits, and tone.
    """
    pref, _ = CompanionPreference.objects.get_or_create(user=request.user)

    if request.method == 'GET':
        return Response(_serialize_preference(pref), status=status.HTTP_200_OK)

    data = request.data or {}

    # Validate and apply scalar fields
    if "companion_name" in data and isinstance(data["companion_name"], str):
        pref.companion_name = data["companion_name"].strip() or "Ami"

    if "user_nickname" in data and isinstance(data["user_nickname"], str):
        pref.user_nickname = data["user_nickname"].strip()

    if "preferred_language" in data:
        lang = data["preferred_language"].lower().strip()
        if lang in [CompanionPreference.LanguagePreference.MANGLISH, CompanionPreference.LanguagePreference.ENGLISH]:
            pref.preferred_language = lang

    if "companion_tone" in data:
        tone = data["companion_tone"].lower().strip()
        if tone in [c[0] for c in CompanionPreference.CompanionTone.choices]:
            pref.companion_tone = tone

    if "relationship_style" in data and isinstance(data["relationship_style"], str):
        pref.relationship_style = data["relationship_style"].strip() or "loving partner"

    if "frequency" in data:
        freq = str(data["frequency"]).lower().strip()
        if freq in [c[0] for c in CompanionPreference.FrequencyChoice.choices]:
            pref.frequency = freq
            if freq == CompanionPreference.FrequencyChoice.DISABLED:
                pref.proactive_messaging_enabled = False
            else:
                pref.proactive_messaging_enabled = True
        else:
            return Response(
                {"error": f"Invalid frequency '{freq}'. Choose from: disabled, daily, few_times_week."},
                status=status.HTTP_400_BAD_REQUEST
            )

    if "proactive_messaging_enabled" in data:
        enabled = bool(data["proactive_messaging_enabled"])
        pref.proactive_messaging_enabled = enabled
        if not enabled:
            pref.frequency = CompanionPreference.FrequencyChoice.DISABLED
        elif pref.frequency == CompanionPreference.FrequencyChoice.DISABLED:
            pref.frequency = CompanionPreference.FrequencyChoice.DAILY

    if "quiet_hours_enabled" in data:
        pref.quiet_hours_enabled = bool(data["quiet_hours_enabled"])

    # Quiet hours times
    if "quiet_hours_start" in data:
        try:
            val = str(data["quiet_hours_start"]).strip()
            parts = [int(p) for p in val.split(":")[:2]]
            pref.quiet_hours_start = datetime.time(parts[0], parts[1])
        except (ValueError, IndexError):
            return Response(
                {"error": "Invalid quiet_hours_start format. Use 'HH:MM'."},
                status=status.HTTP_400_BAD_REQUEST
            )

    if "quiet_hours_end" in data:
        try:
            val = str(data["quiet_hours_end"]).strip()
            parts = [int(p) for p in val.split(":")[:2]]
            pref.quiet_hours_end = datetime.time(parts[0], parts[1])
        except (ValueError, IndexError):
            return Response(
                {"error": "Invalid quiet_hours_end format. Use 'HH:MM'."},
                status=status.HTTP_400_BAD_REQUEST
            )

    if "timezone" in data and isinstance(data["timezone"], str):
        tz_name = data["timezone"].strip()
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(tz_name)
            pref.timezone = tz_name
        except Exception:
            return Response(
                {"error": f"Invalid timezone '{tz_name}'."},
                status=status.HTTP_400_BAD_REQUEST
            )

    # Frequency limits
    if "max_daily_messages" in data:
        try:
            val = int(data["max_daily_messages"])
            if val < 1 or val > 24:
                return Response(
                    {"error": "max_daily_messages must be between 1 and 24."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            pref.max_daily_messages = val
        except (ValueError, TypeError):
            return Response(
                {"error": "max_daily_messages must be an integer."},
                status=status.HTTP_400_BAD_REQUEST
            )

    if "min_minutes_between_messages" in data:
        try:
            val = int(data["min_minutes_between_messages"])
            if val < 15 or val > 1440:
                return Response(
                    {"error": "min_minutes_between_messages must be between 15 and 1440 minutes."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            pref.min_minutes_between_messages = val
        except (ValueError, TypeError):
            return Response(
                {"error": "min_minutes_between_messages must be an integer."},
                status=status.HTTP_400_BAD_REQUEST
            )

    pref.save()
    return Response(_serialize_preference(pref), status=status.HTTP_200_OK)


# ============================================================================
# In-App Inbox APIs
# ============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def inbox_messages_view(request):
    """
    GET: Returns delivered proactive messages for the authenticated user's companion chat inbox.
    """
    msgs = InAppMessage.objects.filter(user=request.user).order_by('-created_at')[:50]
    unread_count = InAppMessage.objects.filter(user=request.user, is_read=False).count()

    return Response({
        "messages": [_serialize_in_app_message(m) for m in msgs],
        "unread_count": unread_count,
        "total_count": msgs.count(),
    }, status=status.HTTP_200_OK)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def inbox_message_detail_view(request, message_id):
    """
    GET: Retrieve specific in-app message.
    PATCH: Mark as read or unread.
    DELETE: Remove the in-app message.
    """
    msg = get_object_or_404(InAppMessage, id=message_id, user=request.user)

    if request.method == 'GET':
        return Response(_serialize_in_app_message(msg), status=status.HTTP_200_OK)

    if request.method == 'PATCH':
        if 'is_read' in request.data:
            msg.is_read = bool(request.data['is_read'])
            msg.save(update_fields=['is_read'])
        return Response(_serialize_in_app_message(msg), status=status.HTTP_200_OK)

    if request.method == 'DELETE':
        msg.delete()
        return Response({"deleted": True}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def inbox_mark_all_read_view(request):
    """
    POST: Marks all unread in-app messages as read for the authenticated user.
    """
    updated = InAppMessage.objects.filter(user=request.user, is_read=False).update(is_read=True)
    return Response({"marked_read": updated}, status=status.HTTP_200_OK)


# ============================================================================
# Scheduled Messages & Scheduler APIs
# ============================================================================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def scheduled_messages_view(request):
    """
    GET: List scheduled messages for the authenticated user.
    POST: Schedule a personalized check-in message.
    """
    if request.method == 'GET':
        msgs = ScheduledMessage.objects.filter(user=request.user).order_by('scheduled_time')[:50]
        return Response([_serialize_scheduled_message(m) for m in msgs], status=status.HTTP_200_OK)

    data = request.data or {}
    message_type = data.get("message_type", ScheduledMessage.MessageType.CASUAL_CHAT)
    content = data.get("content", "")
    channel = data.get("channel", None)
    ai_context = data.get("ai_prompt_context", {})

    scheduled_time_str = data.get("scheduled_time")
    scheduled_time = None
    if scheduled_time_str:
        try:
            scheduled_time = datetime.datetime.fromisoformat(scheduled_time_str)
        except Exception:
            return Response(
                {"error": "Invalid ISO format for scheduled_time."},
                status=status.HTTP_400_BAD_REQUEST
            )

    msg = schedule_single_checkin(
        user=request.user,
        message_type=message_type,
        scheduled_time=scheduled_time,
        content=content,
        channel=channel,
        ai_prompt_context=ai_context
    )

    return Response(_serialize_scheduled_message(msg), status=status.HTTP_201_CREATED)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def scheduled_message_detail_view(request, message_id):
    """
    GET: Details of a scheduled message.
    DELETE: Cancels a scheduled message.
    """
    msg = get_object_or_404(ScheduledMessage, id=message_id, user=request.user)

    if request.method == 'GET':
        return Response(_serialize_scheduled_message(msg), status=status.HTTP_200_OK)

    if request.method == 'DELETE':
        msg.cancel("Cancelled by user request.")
        return Response({"cancelled": True, "id": str(msg.id)}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_daily_schedule_view(request):
    """
    POST: Automatically schedules natural daily check-ins for the authenticated user,
    respecting quiet hours, frequency limits, and idempotency.
    """
    target_date = None
    target_date_str = request.data.get("target_date") if request.data else None
    if target_date_str:
        try:
            target_date = datetime.date.fromisoformat(target_date_str)
        except Exception:
            return Response(
                {"error": "Invalid date format. Use 'YYYY-MM-DD'."},
                status=status.HTTP_400_BAD_REQUEST
            )

    items = create_daily_checkin_schedule(request.user, target_date=target_date)
    return Response({
        "scheduled_count": len(items),
        "messages": [_serialize_scheduled_message(m) for m in items]
    }, status=status.HTTP_200_OK)


# ============================================================================
# Web Push & Device Subscription APIs
# ============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def push_vapid_public_key_view(request):
    """
    GET: Returns the VAPID public key and push enabled status.
    Never exposes the VAPID private key.
    """
    public_key = getattr(settings, 'VAPID_PUBLIC_KEY', '')
    enabled = getattr(settings, 'PUSH_NOTIFICATIONS_ENABLED', False)
    return Response({
        "public_key": public_key,
        "enabled": enabled,
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def push_subscribe_view(request):
    """
    POST: Register or update the authenticated user's push device subscription.
    Accepts standard Web Push subscription object:
    {
        "endpoint": "https://...",
        "keys": {
            "p256dh": "...",
            "auth": "..."
        },
        "device_type": "web"
    }
    """
    data = request.data or {}
    endpoint = data.get("endpoint")
    if not endpoint or not isinstance(endpoint, str) or not endpoint.strip():
        return Response(
            {"error": "Subscription 'endpoint' is required."},
            status=status.HTTP_400_BAD_REQUEST
        )

    endpoint = endpoint.strip()
    keys = data.get("keys", {})
    p256dh = keys.get("p256dh") or data.get("p256dh", "")
    auth = keys.get("auth") or data.get("auth", "")
    device_type = data.get("device_type", PushDeviceSubscription.DeviceType.WEB)

    # Deactivate any duplicate subscriptions using this endpoint from other users (e.g. shared device)
    PushDeviceSubscription.objects.filter(endpoint=endpoint).exclude(user=request.user).update(is_active=False)

    sub, created = PushDeviceSubscription.objects.update_or_create(
        user=request.user,
        endpoint=endpoint,
        defaults={
            "p256dh_key": p256dh,
            "auth_key": auth,
            "device_type": device_type,
            "is_active": True,
        }
    )

    return Response(_serialize_push_subscription(sub), status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def push_unsubscribe_view(request):
    """
    POST: Unsubscribe or deactivate push notifications.
    If endpoint is provided, deactivates that specific subscription.
    If no endpoint is provided, deactivates all active subscriptions for the user.
    """
    data = request.data or {}
    endpoint = data.get("endpoint")

    if endpoint and isinstance(endpoint, str) and endpoint.strip():
        updated = PushDeviceSubscription.objects.filter(
            user=request.user,
            endpoint=endpoint.strip()
        ).update(is_active=False)
    else:
        updated = PushDeviceSubscription.objects.filter(
            user=request.user,
            is_active=True
        ).update(is_active=False)

    return Response({"unsubscribed": True, "count": updated}, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def push_subscriptions_list_view(request):
    """
    GET: List all active push device subscriptions for the authenticated user.
    """
    subs = PushDeviceSubscription.objects.filter(user=request.user, is_active=True).order_by('-created_at')
    return Response([_serialize_push_subscription(s) for s in subs], status=status.HTTP_200_OK)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def push_subscription_delete_view(request, subscription_id):
    """
    DELETE: Permanently delete a specific push device subscription belonging to the user.
    """
    sub = get_object_or_404(PushDeviceSubscription, id=subscription_id, user=request.user)
    sub.delete()
    return Response({"deleted": True, "id": str(subscription_id)}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def push_send_test_notification_view(request):
    """
    POST: Sends a test push notification to the authenticated user's registered active devices.
    """
    data = request.data or {}
    companion_name = "Ami"
    if hasattr(request.user, 'companion_preference') and request.user.companion_preference:
        companion_name = request.user.companion_preference.companion_name or "Ami"

    content = data.get("message") or f"Hey {request.user.username}! This is a test notification from {companion_name} 💕"
    title = data.get("title") or f"{companion_name} Check-in"

    handler = PushNotificationDeliveryHandler()
    if not handler.is_configured():
        return Response(
            {
                "error": "Push notification provider is not configured.",
                "configured": False,
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        result = handler.deliver(
            scheduled_message=None,
            content=content,
            user=request.user,
            title=title,
        )
        return Response({
            "success": True,
            "message": "Push notification dispatched successfully.",
            "details": result,
        }, status=status.HTTP_200_OK)
    except PushDeliveryError as exc:
        return Response(
            {"error": str(exc), "success": False},
            status=status.HTTP_400_BAD_REQUEST
        )

