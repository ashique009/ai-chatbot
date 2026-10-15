"""
Base interface and result contracts for delivery handlers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from proactive_messaging.models import ScheduledMessage


class BaseDeliveryHandler(ABC):
    """
    Abstract base delivery handler.
    Subclasses must implement `deliver` and must NOT fake successful delivery
    when provider credentials or infrastructure are absent.
    """

    channel_name: str = "base"

    @abstractmethod
    def deliver(self, scheduled_message: ScheduledMessage, content: str) -> Dict[str, Any]:
        """
        Delivers the message through this specific channel.

        :param scheduled_message: The ScheduledMessage instance.
        :param content: The message content to deliver.
        :return: Dict containing delivery details.
        :raises DeliveryError: If delivery fails or provider is not configured.
        """
        raise NotImplementedError
