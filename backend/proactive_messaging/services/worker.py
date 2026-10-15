"""
Scheduled Message Worker Service.
Orchestrates finding due messages, claiming them atomically, checking user preferences
(quiet hours & frequency limits), invoking OpenAI generation in Manglish/English,
and delivering through in-app and push channels with retry support.
"""
import logging
import datetime
from typing import Any, Dict, List, Optional
from django.utils import timezone
from proactive_messaging.models import ScheduledMessage
from proactive_messaging.services.ai_generator import CompanionAIGenerator
from proactive_messaging.services.delivery.in_app import InAppDeliveryHandler
from proactive_messaging.services.delivery.push import PushNotificationDeliveryHandler
from proactive_messaging.services.exceptions import (
    AIGenerationError,
    DeliveryError,
)

logger = logging.getLogger(__name__)


class ScheduledMessageWorker:
    """
    Worker engine to process due ScheduledMessage instances safely and reliably.
    """

    def __init__(
        self,
        ai_generator: Optional[CompanionAIGenerator] = None,
        in_app_handler: Optional[InAppDeliveryHandler] = None,
        push_handler: Optional[PushNotificationDeliveryHandler] = None,
    ):
        self.ai_generator = ai_generator or CompanionAIGenerator()
        self.in_app_handler = in_app_handler or InAppDeliveryHandler()
        self.push_handler = push_handler or PushNotificationDeliveryHandler()

    def process_due_messages(self, limit: int = 50) -> Dict[str, Any]:
        """
        Finds due pending messages and processes each with atomic claiming.
        Returns a summary report of the batch run.
        """
        due_messages = list(
            ScheduledMessage.objects.due()
            .select_related('user')
            .order_by('scheduled_time')[:limit]
        )

        results = {
            "total_due": len(due_messages),
            "claimed": 0,
            "skipped": 0,
            "delivered": 0,
            "retried": 0,
            "failed": 0,
            "postponed": 0,
            "cancelled": 0,
            "details": [],
        }

        for msg in due_messages:
            outcome = self.process_single_message(msg)
            results["details"].append(outcome)

            status = outcome.get("status")
            if status == "skipped":
                results["skipped"] += 1
            elif status == "delivered":
                results["delivered"] += 1
                results["claimed"] += 1
            elif status == "postponed":
                results["postponed"] += 1
                results["claimed"] += 1
            elif status == "cancelled":
                results["cancelled"] += 1
                results["claimed"] += 1
            elif status == "error":
                results["claimed"] += 1
                # Check whether message was re-queued for retry or permanently marked failed
                msg.refresh_from_db(fields=['status'])
                if msg.status == ScheduledMessage.DeliveryStatus.PENDING:
                    results["retried"] += 1
                else:
                    results["failed"] += 1

        return results

    def process_single_message(self, message: ScheduledMessage) -> Dict[str, Any]:
        """
        Processes an individual ScheduledMessage:
        1. Atomically claims the message using mark_processing().
        2. Evaluates user companion preferences (quiet hours, frequency limits).
        3. Generates content using OpenAI API if required.
        4. Delivers across target channels (in-app and/or push notification).
        5. Updates message status (delivered or failed/retry).
        """
        msg_id_str = str(message.id)

        # 1. Atomic claim to prevent duplicate processing by concurrent workers
        if not message.mark_processing():
            logger.info("Message %s was already claimed by another worker; skipping.", msg_id_str)
            return {
                "message_id": msg_id_str,
                "status": "skipped",
                "reason": "Already claimed by another worker",
            }

        user = message.user
        preference = getattr(user, 'companion_preference', None)

        # 2. Check user preferences & quiet hours
        if preference:
            from proactive_messaging.models import CompanionPreference
            if not preference.proactive_messaging_enabled or preference.frequency == CompanionPreference.FrequencyChoice.DISABLED:
                message.cancel("Proactive messaging disabled by user preference.")
                return {
                    "message_id": msg_id_str,
                    "status": "cancelled",
                    "reason": "User disabled proactive messaging",
                }

            if preference.frequency == CompanionPreference.FrequencyChoice.FEW_TIMES_WEEK:
                tz = preference.get_tz()
                local_dt = message.scheduled_time.astimezone(tz)
                if local_dt.weekday() not in (0, 2, 4):
                    message.cancel("Cancelled: Scheduled day is not a Mon/Wed/Fri check-in day for few_times_week.")
                    return {
                        "message_id": msg_id_str,
                        "status": "cancelled",
                        "reason": "Not a valid day for few_times_week frequency",
                    }

            # Check quiet hours
            now = timezone.now()
            if preference.is_in_quiet_hours(now):
                next_time = preference.get_next_quiet_hours_end(now)
                message.postpone(next_time, "Postponed: Currently within quiet hours.")
                return {
                    "message_id": msg_id_str,
                    "status": "postponed",
                    "reason": f"Quiet hours active until {next_time.isoformat()}",
                }

            # Check frequency limits
            exceeded, freq_reason = preference.has_exceeded_frequency(now)
            if exceeded:
                if "Daily limit" in freq_reason:
                    # Postpone to start of next day
                    tomorrow = now + datetime.timedelta(days=1)
                    message.postpone(tomorrow, f"Postponed: {freq_reason}")
                else:
                    # Postpone by cooldown duration
                    cooldown_time = now + datetime.timedelta(minutes=preference.min_minutes_between_messages)
                    message.postpone(cooldown_time, f"Postponed: {freq_reason}")

                return {
                    "message_id": msg_id_str,
                    "status": "postponed",
                    "reason": freq_reason,
                }

        # 3. Content generation via OpenAI API
        content = message.content or ""
        ai_metadata = message.ai_metadata or {}

        if message.is_ai_generated and not content.strip():
            try:
                content, ai_metadata = self.ai_generator.generate(message, preference)
            except AIGenerationError as exc:
                err_str = f"AI generation failed: {exc}"
                logger.error("Message %s generation error: %s", msg_id_str, err_str)
                message.mark_failed(err_str)
                return {
                    "message_id": msg_id_str,
                    "status": "error",
                    "phase": "generation",
                    "error": str(exc),
                }

        # 4. Delivery via configured channels
        in_app_delivered = False
        push_delivered = False
        delivery_errors: List[str] = []

        # In-App channel delivery
        if message.channel in [
            ScheduledMessage.DeliveryChannel.IN_APP,
            ScheduledMessage.DeliveryChannel.BOTH,
        ]:
            try:
                self.in_app_handler.deliver(message, content)
                in_app_delivered = True
            except DeliveryError as exc:
                delivery_errors.append(f"In-app delivery error: {exc}")

        # Push notification channel delivery
        if message.channel in [
            ScheduledMessage.DeliveryChannel.PUSH,
            ScheduledMessage.DeliveryChannel.BOTH,
        ]:
            try:
                self.push_handler.deliver(message, content)
                push_delivered = True
            except DeliveryError as exc:
                delivery_errors.append(f"Push delivery error: {exc}")

        # 5. Delivery outcome tracking & retries
        if delivery_errors:
            combined_error = "; ".join(delivery_errors)
            logger.error("Message %s delivery failure: %s", msg_id_str, combined_error)
            message.mark_failed(combined_error)
            return {
                "message_id": msg_id_str,
                "status": "error",
                "phase": "delivery",
                "error": combined_error,
                "in_app_delivered": in_app_delivered,
                "push_delivered": push_delivered,
            }

        # Delivery fully succeeded
        message.mark_delivered(
            content=content,
            delivered_in_app=in_app_delivered,
            delivered_push=push_delivered,
            ai_metadata=ai_metadata,
        )

        return {
            "message_id": msg_id_str,
            "status": "delivered",
            "channel": message.channel,
            "delivered_in_app": in_app_delivered,
            "delivered_push": push_delivered,
        }
