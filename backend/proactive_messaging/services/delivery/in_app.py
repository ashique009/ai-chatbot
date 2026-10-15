"""
In-app message delivery handler.
Stores delivered messages in the InAppMessage model for display in the React companion chat.
"""
from typing import Any, Dict
from django.conf import settings
from proactive_messaging.models import ScheduledMessage, InAppMessage
from proactive_messaging.services.delivery.base import BaseDeliveryHandler
from proactive_messaging.services.exceptions import (
    InAppDeliveryError,
    InAppProviderNotConfiguredError,
)


class InAppDeliveryHandler(BaseDeliveryHandler):
    """
    Handles delivery of proactive messages to the user's in-app chat inbox.
    Refuses delivery if the in-app messaging feature is disabled in settings.
    """

    channel_name: str = "in_app"

    def is_configured(self) -> bool:
        """Returns True if in-app messaging is enabled in configuration."""
        return getattr(settings, 'IN_APP_MESSAGING_ENABLED', True)

    def deliver(self, scheduled_message: ScheduledMessage, content: str) -> Dict[str, Any]:
        """
        Stores the message into the user's InAppMessage feed.
        """
        if not self.is_configured():
            raise InAppProviderNotConfiguredError(
                "In-app messaging delivery is disabled in settings (IN_APP_MESSAGING_ENABLED=False)."
            )

        user = scheduled_message.user
        if not user.is_active:
            raise InAppDeliveryError(
                f"Cannot deliver in-app message: User '{user.username}' is inactive."
            )

        if not content or not content.strip():
            raise InAppDeliveryError("Cannot deliver an empty in-app message.")

        # Determine sender persona name
        sender_name = "AI Companion"
        if hasattr(user, 'companion_preference') and user.companion_preference:
            sender_name = user.companion_preference.companion_name or "AI Companion"

        try:
            from django.db import transaction
            from chat.models import Conversation, ChatMessage

            with transaction.atomic():
                in_app_msg = InAppMessage.objects.filter(scheduled_message=scheduled_message).first()
                if not in_app_msg:
                    in_app_msg = InAppMessage.objects.create(
                        user=user,
                        scheduled_message=scheduled_message,
                        sender_name=sender_name,
                        content=content.strip(),
                        message_type=scheduled_message.message_type,
                        is_read=False,
                    )

                # Persist proactive message to the user's active Conversation as ChatMessage
                conversation = Conversation.objects.filter(user=user).order_by('-updated_at').first()
                if not conversation:
                    conversation = Conversation.objects.create(
                        user=user,
                        title="Chat with Ami"
                    )

                msg_id_str = str(scheduled_message.id)
                # Uniqueness protection: ensure exactly one chat message per scheduled message
                chat_msg = ChatMessage.objects.filter(
                    conversation__user=user,
                    metadata__scheduled_message_id=msg_id_str
                ).first()

                if not chat_msg:
                    chat_msg = ChatMessage.objects.create(
                        conversation=conversation,
                        sender=ChatMessage.Sender.COMPANION,
                        content=content.strip(),
                        metadata={
                            "scheduled_message_id": msg_id_str,
                            "message_type": scheduled_message.message_type,
                            "delivery_channel": self.channel_name,
                            "is_proactive": True,
                        }
                    )
                    conversation.save(update_fields=['updated_at'])

            return {
                "channel": self.channel_name,
                "delivered": True,
                "in_app_message_id": str(in_app_msg.id),
                "chat_message_id": str(chat_msg.id) if chat_msg else None,
                "conversation_id": str(conversation.id),
                "created_at": in_app_msg.created_at.isoformat(),
            }
        except Exception as exc:
            raise InAppDeliveryError(f"Database error delivering in-app message: {exc}") from exc
