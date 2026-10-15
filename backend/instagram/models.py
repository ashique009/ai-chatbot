"""
Database models for Meta's Official Instagram Messaging API integration.
Enforces Meta's 24-hour messaging window policy and tracks Instagram customer interactions.
"""
import uuid
from datetime import timedelta
from django.db import models
from django.conf import settings
from django.utils import timezone


class InstagramCustomer(models.Model):
    """
    Represents an Instagram user who has interacted with Ami's Instagram Professional Account.
    Tracks Instagram-Scoped IDs (IGSID) and the 24-hour standard messaging window.
    
    Platform Policy Requirement:
    Instagram messages can only be sent within 24 hours of the user's latest interaction
    (the 24-hour standard messaging window). Unsolicited first DMs are strictly prohibited.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='instagram_customers',
        help_text="Optional link to authenticated companion user."
    )
    igsid = models.CharField(
        max_length=128,
        unique=True,
        db_index=True,
        help_text="Instagram-Scoped ID (IGSID) of the customer."
    )
    username = models.CharField(
        max_length=128,
        blank=True,
        default='',
        help_text="Instagram handle if resolved from Meta Graph API."
    )
    last_interaction_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Timestamp of the most recent user interaction; starts Meta's 24-hour messaging window."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this customer is currently active for Instagram messaging."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['igsid', 'is_active'], name='idx_ig_cust_igsid_active'),
            models.Index(fields=['last_interaction_at'], name='idx_ig_cust_last_interact'),
        ]
        verbose_name = 'Instagram Customer'
        verbose_name_plural = 'Instagram Customers'

    def __str__(self):
        label = self.username or f"IGSID:{self.igsid[:8]}..."
        return f"{label} ({'Active 24h' if self.is_within_24h_window() else 'Window Closed'})"

    def is_within_24h_window(self, check_dt=None) -> bool:
        """
        Verifies if the current time is within Meta's 24-hour messaging window
        initiated by the user's latest interaction.
        """
        if not self.last_interaction_at or not self.is_active:
            return False
        now = check_dt or timezone.now()
        return now - self.last_interaction_at <= timedelta(hours=24)

    def can_receive_message(self, check_dt=None) -> tuple:
        """
        Returns (can_receive: bool, reason: str).
        Strictly enforces Meta platform messaging rules:
        - Must be active
        - Must have had a user-initiated interaction
        - Must be within the 24-hour window
        """
        if not self.is_active:
            return False, "Instagram customer is marked inactive."

        if not self.last_interaction_at:
            return (
                False,
                "No prior user interaction recorded. Meta requires the user to initiate "
                "the conversation first. Unsolicited outbound DMs are prohibited."
            )

        now = check_dt or timezone.now()
        elapsed = now - self.last_interaction_at
        if elapsed > timedelta(hours=24):
            hours_ago = int(elapsed.total_seconds() / 3600)
            return (
                False,
                f"Meta standard messaging window expired ({hours_ago}h since last user message). "
                "Ami must wait for the user to message again."
            )

        return True, "Within Meta's 24-hour messaging window."

    def record_interaction(self, interaction_dt=None):
        """
        Updates the customer's last interaction timestamp to renew the 24-hour messaging window.
        """
        self.last_interaction_at = interaction_dt or timezone.now()
        self.save(update_fields=['last_interaction_at', 'updated_at'])


class InstagramWebhookEventLog(models.Model):
    """
    Audit log of received Instagram webhook events.
    Records delivery metadata, event type, and sender IGSID.
    Strictly DOES NOT store private message contents or tokens.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entry_id = models.CharField(
        max_length=128,
        blank=True,
        default='',
        help_text="Instagram Professional Account ID receiving the event."
    )
    sender_igsid = models.CharField(
        max_length=128,
        blank=True,
        default='',
        db_index=True,
        help_text="Instagram-Scoped ID of the user triggering the event."
    )
    event_type = models.CharField(
        max_length=64,
        default='message',
        help_text="Type of webhook event (e.g. message, delivery, read)."
    )
    status = models.CharField(
        max_length=32,
        default='received',
        help_text="Processing status."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['sender_igsid', 'created_at'], name='idx_ig_evt_sender_created'),
        ]
        verbose_name = 'Instagram Webhook Event Log'
        verbose_name_plural = 'Instagram Webhook Event Logs'

    def __str__(self):
        return f"IG Event {self.event_type} from {self.sender_igsid[:8]}... at {self.created_at}"
