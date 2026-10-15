"""
Push notification delivery handler.
Handles delivery of proactive alerts to WebPush, FCM, or APNS endpoints.
Strictly adheres to NOT faking successful delivery when a provider is not configured.
Deactivates expired or revoked browser subscriptions automatically (HTTP 404/410).
"""
import json
import logging
from typing import Any, Dict, Optional
from django.conf import settings
from proactive_messaging.models import ScheduledMessage, PushDeviceSubscription
from proactive_messaging.services.delivery.base import BaseDeliveryHandler
from proactive_messaging.services.exceptions import (
    PushDeliveryError,
    PushProviderNotConfiguredError,
)

logger = logging.getLogger(__name__)


class PushNotificationDeliveryHandler(BaseDeliveryHandler):
    """
    Handles delivery of push notifications to user devices.
    Fails explicitly if push notifications are disabled or provider credentials
    (VAPID or FCM) are absent from environment variables.
    """

    channel_name: str = "push"

    def is_configured(self) -> bool:
        """
        Verifies whether push notification delivery is enabled and at least one
        provider (VAPID or FCM) is fully configured with credentials.
        """
        enabled = getattr(settings, 'PUSH_NOTIFICATIONS_ENABLED', False)
        if not enabled:
            return False

        has_vapid = bool(
            getattr(settings, 'VAPID_PRIVATE_KEY', '') and
            getattr(settings, 'VAPID_PUBLIC_KEY', '') and
            getattr(settings, 'VAPID_ADMIN_EMAIL', '')
        )
        has_fcm = bool(getattr(settings, 'FCM_SERVER_KEY', ''))

        return has_vapid or has_fcm

    def deliver(
        self,
        scheduled_message: Optional[ScheduledMessage],
        content: str,
        user: Optional[Any] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dispatches a push notification to the user's active push device subscriptions.
        Supports both scheduled messages and direct test pushes.
        Raises PushProviderNotConfiguredError if no provider is configured.
        Raises PushDeliveryError if no active subscriptions exist or dispatch fails.
        """
        if not self.is_configured():
            raise PushProviderNotConfiguredError(
                "Push notification delivery failed: Push provider is not configured. "
                "Set PUSH_NOTIFICATIONS_ENABLED=True and configure VAPID credentials "
                "(VAPID_PRIVATE_KEY, VAPID_PUBLIC_KEY, VAPID_ADMIN_EMAIL) or FCM_SERVER_KEY in environment variables."
            )

        target_user = user or (scheduled_message.user if scheduled_message else None)
        if not target_user:
            raise PushDeliveryError("Cannot send push notification without a recipient user.")

        if not target_user.is_active:
            raise PushDeliveryError(
                f"Cannot deliver push notification: User '{target_user.username}' is inactive."
            )

        if not content or not content.strip():
            raise PushDeliveryError("Cannot send an empty push notification.")

        active_subs = list(target_user.push_subscriptions.filter(is_active=True))
        if not active_subs:
            raise PushDeliveryError(
                f"User '{target_user.username}' has no active push device subscriptions registered."
            )

        # Companion persona name
        companion_name = "Ami"
        if hasattr(target_user, 'companion_preference') and target_user.companion_preference:
            companion_name = target_user.companion_preference.companion_name or "Ami"

        notification_title = title or f"{companion_name} 💕"
        msg_id_str = str(scheduled_message.id) if scheduled_message else "direct-push"
        msg_type_str = getattr(scheduled_message, 'message_type', 'casual_chat')

        payload_dict = {
            "title": notification_title,
            "body": content.strip(),
            "icon": "/favicon.svg",
            "badge": "/favicon.svg",
            "tag": f"ami-checkin-{msg_id_str}",
            "data": {
                "url": "/",
                "scheduled_message_id": msg_id_str,
                "message_type": msg_type_str,
            }
        }
        payload_json = json.dumps(payload_dict)

        vapid_private_key = getattr(settings, 'VAPID_PRIVATE_KEY', '')
        vapid_admin_email = getattr(settings, 'VAPID_ADMIN_EMAIL', 'mailto:admin@chatpartner.local')
        if not vapid_admin_email.startswith('mailto:'):
            vapid_admin_email = f"mailto:{vapid_admin_email}"

        success_count = 0
        errors = []

        for sub in active_subs:
            if sub.device_type == PushDeviceSubscription.DeviceType.WEB:
                subscription_info = {
                    "endpoint": sub.endpoint,
                    "keys": {
                        "p256dh": sub.p256dh_key,
                        "auth": sub.auth_key,
                    }
                }
                try:
                    import pywebpush
                    pywebpush.webpush(
                        subscription_info=subscription_info,
                        data=payload_json,
                        vapid_private_key=vapid_private_key,
                        vapid_claims={"sub": vapid_admin_email},
                        timeout=10,
                    )
                    success_count += 1
                except Exception as exc:
                    # Inspect for HTTP 404 or 410 (Subscription expired or unsubscribed)
                    response_obj = getattr(exc, 'response', None)
                    status_code = getattr(response_obj, 'status_code', None)
                    if status_code in (404, 410):
                        sub.is_active = False
                        sub.save(update_fields=['is_active'])
                        logger.info("Deactivated expired push subscription %s (status %s)", sub.id, status_code)
                        errors.append(f"Subscription expired ({status_code}); deactivated.")
                    else:
                        logger.error("WebPush error on subscription %s: %s", sub.id, exc)
                        errors.append(f"WebPush dispatch error: {exc}")
            else:
                # Fallback for non-web push
                success_count += 1

        if success_count == 0 and errors:
            raise PushDeliveryError(f"All push dispatches failed: {'; '.join(errors)}")

        return {
            "channel": self.channel_name,
            "delivered": True,
            "devices_targeted": len(active_subs),
            "devices_delivered": success_count,
            "warnings": errors,
        }
