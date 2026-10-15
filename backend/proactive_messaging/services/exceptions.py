"""
Custom exceptions for the proactive messaging system.
"""


class ProactiveMessagingError(Exception):
    """Base exception for all proactive messaging operations."""
    pass


class AIGenerationError(ProactiveMessagingError):
    """Raised when message generation via OpenAI fails."""
    pass


class OpenAIConfigurationError(AIGenerationError):
    """Raised when OpenAI API key or required settings are missing."""
    pass


class DeliveryError(ProactiveMessagingError):
    """Base exception for message delivery failures."""
    pass


class InAppDeliveryError(DeliveryError):
    """Raised when in-app message creation or delivery fails."""
    pass


class InAppProviderNotConfiguredError(DeliveryError):
    """Raised when in-app messaging is disabled or unconfigured."""
    pass


class PushDeliveryError(DeliveryError):
    """Raised when push notification delivery fails."""
    pass


class PushProviderNotConfiguredError(PushDeliveryError):
    """Raised when push notification provider credentials or service are not configured."""
    pass
