import uuid
import datetime
from zoneinfo import ZoneInfo
from django.db import models, transaction
from django.conf import settings
from django.utils import timezone



class ScheduledMessageQuerySet(models.QuerySet):
    """Custom queryset providing helper methods for querying scheduled messages."""

    def due(self):
        """Returns messages that are pending and whose scheduled time has arrived."""
        return self.filter(
            status=ScheduledMessage.DeliveryStatus.PENDING,
            scheduled_time__lte=timezone.now()
        )

    def pending(self):
        """Returns all pending messages."""
        return self.filter(status=ScheduledMessage.DeliveryStatus.PENDING)

    def processing(self):
        """Returns messages currently being handled by a background worker."""
        return self.filter(status=ScheduledMessage.DeliveryStatus.PROCESSING)

    def delivered(self):
        """Returns successfully delivered messages."""
        return self.filter(status=ScheduledMessage.DeliveryStatus.DELIVERED)

    def failed(self):
        """Returns failed messages."""
        return self.filter(status=ScheduledMessage.DeliveryStatus.FAILED)


class ScheduledMessageManager(models.Manager):
    """Manager for ScheduledMessage providing direct access to custom queryset filters."""

    def get_queryset(self):
        return ScheduledMessageQuerySet(self.model, using=self._db)

    def due(self):
        return self.get_queryset().due()

    def pending(self):
        return self.get_queryset().pending()

    def processing(self):
        return self.get_queryset().processing()

    def delivered(self):
        return self.get_queryset().delivered()

    def failed(self):
        return self.get_queryset().failed()


class ScheduledMessage(models.Model):
    """
    Represents an AI companion's proactive message scheduled for a user.
    Supports frequent personalized check-ins (good morning, casual chats, evening messages),
    OpenAI-driven message generation, and dual delivery via in-app chat & push notifications.
    Includes concurrency guards and idempotency to prevent duplicate deliveries.
    """

    class MessageType(models.TextChoices):
        GOOD_MORNING = 'good_morning', 'Good Morning Check-in'
        CASUAL_CHAT = 'casual_chat', 'Casual Conversation'
        EVENING_CHECKIN = 'evening_checkin', 'Evening Message'
        AFFIRMATION = 'affirmation', 'Daily Affirmation'
        REMINDER = 'reminder', 'Personalized Reminder'
        CUSTOM = 'custom', 'Custom Check-in'

    class DeliveryStatus(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        DELIVERED = 'delivered', 'Delivered'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'

    class DeliveryChannel(models.TextChoices):
        IN_APP = 'in_app', 'In-App Chat'
        PUSH = 'push', 'Push Notification'
        BOTH = 'both', 'Both In-App & Push Notification'

    # Identity and Recipient
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for the scheduled message."
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='scheduled_messages',
        help_text="The user receiving this proactive companion message."
    )

    # Message Type and Scheduled Time
    message_type = models.CharField(
        max_length=50,
        choices=MessageType.choices,
        default=MessageType.CASUAL_CHAT,
        db_index=True,
        help_text="Type of check-in (e.g. good morning, casual conversation, evening message)."
    )
    scheduled_time = models.DateTimeField(
        db_index=True,
        help_text="Timestamp when the message is scheduled to be triggered."
    )

    # Status, Concurrency Guard & Duplicate Delivery Prevention
    status = models.CharField(
        max_length=20,
        choices=DeliveryStatus.choices,
        default=DeliveryStatus.PENDING,
        db_index=True,
        help_text="Current delivery status."
    )
    idempotency_key = models.CharField(
        max_length=255,
        unique=True,
        null=True,
        blank=True,
        db_index=True,
        help_text="Unique key ensuring no duplicate message is scheduled or processed."
    )
    attempts = models.PositiveIntegerField(
        default=0,
        help_text="Number of delivery attempts made."
    )
    max_attempts = models.PositiveIntegerField(
        default=3,
        help_text="Maximum number of retry attempts before marking as failed."
    )
    error_message = models.TextField(
        blank=True,
        null=True,
        help_text="Error details if delivery or generation encounters a failure."
    )

    # Delivery Channels & Channel Delivery Statuses
    channel = models.CharField(
        max_length=20,
        choices=DeliveryChannel.choices,
        default=DeliveryChannel.BOTH,
        help_text="Target delivery channel(s): in-app chat, push notifications, or both."
    )
    delivered_in_app = models.BooleanField(
        default=False,
        help_text="True if delivered successfully to in-app chat."
    )
    delivered_push = models.BooleanField(
        default=False,
        help_text="True if delivered successfully via push notification."
    )

    # AI Generation via OpenAI API
    is_ai_generated = models.BooleanField(
        default=True,
        help_text="Whether this message content should be dynamically generated using OpenAI."
    )
    content = models.TextField(
        blank=True,
        default='',
        help_text="The message body text (either pre-set or generated by OpenAI)."
    )
    ai_prompt_context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Context payload for OpenAI (e.g. companion persona, memory highlights, tone, topic)."
    )
    ai_model = models.CharField(
        max_length=64,
        blank=True,
        default='gpt-4o-mini',
        help_text="OpenAI model identifier targeted or utilized for generation."
    )
    ai_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Metadata returned by OpenAI API (tokens used, response ID, finish reason, etc.)."
    )

    # Audit Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Timestamp when this scheduled message was created."
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp when this scheduled message was last updated."
    )
    delivered_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when message delivery was successfully completed."
    )

    objects = ScheduledMessageManager()

    class Meta:
        ordering = ['scheduled_time']
        indexes = [
            models.Index(fields=['status', 'scheduled_time'], name='idx_msg_status_sched_time'),
            models.Index(fields=['user', 'status'], name='idx_msg_user_status'),
            models.Index(fields=['message_type', 'status'], name='idx_msg_type_status'),
        ]
        verbose_name = 'Scheduled Message'
        verbose_name_plural = 'Scheduled Messages'

    def __str__(self):
        return f"{self.get_message_type_display()} for {self.user} at {self.scheduled_time} [{self.status}]"

    def is_due(self) -> bool:
        """Checks if the message is pending and its scheduled time is now or in the past."""
        return self.status == self.DeliveryStatus.PENDING and self.scheduled_time <= timezone.now()

    def mark_processing(self) -> bool:
        """
        Atomically transitions the message status from PENDING to PROCESSING.
        Prevents duplicate delivery across concurrent background jobs or workers.
        Returns True if acquired successfully, False otherwise.
        """
        with transaction.atomic():
            rows_updated = ScheduledMessage.objects.filter(
                pk=self.pk,
                status=self.DeliveryStatus.PENDING
            ).update(
                status=self.DeliveryStatus.PROCESSING,
                attempts=models.F('attempts') + 1,
                updated_at=timezone.now()
            )
            if rows_updated:
                self.refresh_from_db(fields=['status', 'attempts', 'updated_at'])
                return True
            return False

    def mark_delivered(self, content: str = None, delivered_in_app: bool = None, delivered_push: bool = None, ai_metadata: dict = None):
        """
        Marks the message as successfully delivered and records delivery metadata.
        """
        self.status = self.DeliveryStatus.DELIVERED
        self.delivered_at = timezone.now()
        self.error_message = None
        update_fields = ['status', 'delivered_at', 'error_message', 'updated_at']

        if content is not None:
            self.content = content
            update_fields.append('content')
        if delivered_in_app is not None:
            self.delivered_in_app = delivered_in_app
            update_fields.append('delivered_in_app')
        if delivered_push is not None:
            self.delivered_push = delivered_push
            update_fields.append('delivered_push')
        if ai_metadata is not None:
            self.ai_metadata = ai_metadata
            update_fields.append('ai_metadata')

        self.save(update_fields=update_fields)

    def mark_failed(self, error_message: str):
        """
        Marks an attempt as failed. If retry threshold is exceeded, marks as FAILED;
        otherwise returns to PENDING so it can be retried by the next run.
        """
        self.error_message = error_message
        if self.attempts >= self.max_attempts:
            self.status = self.DeliveryStatus.FAILED
        else:
            self.status = self.DeliveryStatus.PENDING
        self.save(update_fields=['status', 'error_message', 'updated_at'])

    def postpone(self, new_time, reason: str = None):
        """
        Postpones a scheduled message to a future scheduled_time and resets/keeps it PENDING.
        Used when quiet hours or frequency limits temporarily hold delivery.
        """
        self.scheduled_time = new_time
        self.status = self.DeliveryStatus.PENDING
        if reason:
            self.error_message = reason
        self.save(update_fields=['scheduled_time', 'status', 'error_message', 'updated_at'])

    def cancel(self, reason: str = None):
        """Marks message as cancelled."""
        self.status = self.DeliveryStatus.CANCELLED
        if reason:
            self.error_message = reason
        self.save(update_fields=['status', 'error_message', 'updated_at'])


class CompanionPreference(models.Model):
    """
    Stores a user's AI companion personality, language, quiet hours,
    and message frequency limits.
    """
    class LanguagePreference(models.TextChoices):
        MANGLISH = 'manglish', 'Manglish (Malayalam + English)'
        ENGLISH = 'english', 'English'

    class CompanionTone(models.TextChoices):
        CARING = 'caring', 'Caring & Warm'
        PLAYFUL = 'playful', 'Playful & Teasing'
        CALM = 'calm', 'Calm & Supportive'
        ENTHUSIASTIC = 'enthusiastic', 'Enthusiastic'

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='companion_preference',
        help_text="User associated with these companion preferences."
    )
    companion_name = models.CharField(
        max_length=100,
        default='Ami',
        help_text="Name of the AI companion."
    )
    user_nickname = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="How the companion addresses the user (e.g. da, dear, or first name)."
    )
    preferred_language = models.CharField(
        max_length=20,
        choices=LanguagePreference.choices,
        default=LanguagePreference.MANGLISH,
        help_text="Preferred language style: Manglish or English."
    )
    companion_tone = models.CharField(
        max_length=50,
        choices=CompanionTone.choices,
        default=CompanionTone.CARING,
        help_text="Personality tone of the AI companion."
    )
    relationship_style = models.CharField(
        max_length=100,
        default='loving partner',
        help_text="Nature of the dynamic (e.g. 'loving partner', 'caring companion')."
    )
    class FrequencyChoice(models.TextChoices):
        DISABLED = 'disabled', 'Disabled'
        DAILY = 'daily', 'Daily'
        FEW_TIMES_WEEK = 'few_times_week', 'Few Times a Week'

    frequency = models.CharField(
        max_length=20,
        choices=FrequencyChoice.choices,
        default=FrequencyChoice.DAILY,
        db_index=True,
        help_text="Frequency of proactive check-ins: disabled, daily (1/day), or few_times_week (Mon/Wed/Fri)."
    )
    proactive_messaging_enabled = models.BooleanField(
        default=True,
        help_text="Master toggle for proactive messaging."
    )

    # Quiet Hours
    quiet_hours_enabled = models.BooleanField(
        default=True,
        help_text="Whether quiet hours are enforced to prevent nighttime messages."
    )
    quiet_hours_start = models.TimeField(
        default='22:00',
        help_text="Local start time of quiet hours (default: 10:00 PM)."
    )
    quiet_hours_end = models.TimeField(
        default='07:00',
        help_text="Local end time of quiet hours (default: 7:00 AM)."
    )
    timezone = models.CharField(
        max_length=64,
        default='Asia/Kolkata',
        help_text="User's IANA timezone name (e.g. Asia/Kolkata, America/New_York)."
    )

    # Frequency Limits
    max_daily_messages = models.PositiveIntegerField(
        default=5,
        help_text="Maximum proactive messages sent per day."
    )
    min_minutes_between_messages = models.PositiveIntegerField(
        default=120,
        help_text="Minimum cooldown minutes between proactive messages."
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Companion Preference'
        verbose_name_plural = 'Companion Preferences'

    def __str__(self):
        return f"{self.user.username}'s Companion Preference ({self.companion_name})"

    def save(self, *args, **kwargs):
        # Synchronize frequency and proactive_messaging_enabled
        if self.frequency == self.FrequencyChoice.DISABLED:
            self.proactive_messaging_enabled = False
        elif not self.proactive_messaging_enabled and self.frequency != self.FrequencyChoice.DISABLED:
            self.frequency = self.FrequencyChoice.DISABLED
        elif self.proactive_messaging_enabled and self.frequency == self.FrequencyChoice.DISABLED:
            self.frequency = self.FrequencyChoice.DAILY

        update_fields = kwargs.get('update_fields')
        if update_fields is not None:
            update_fields = set(update_fields)
            update_fields.add('frequency')
            update_fields.add('proactive_messaging_enabled')
            kwargs['update_fields'] = list(update_fields)

        super().save(*args, **kwargs)

    def get_tz(self):
        try:
            return ZoneInfo(self.timezone)
        except Exception:
            return ZoneInfo('UTC')

    def is_in_quiet_hours(self, check_dt=None) -> bool:
        """Determines whether the given datetime falls within user's quiet hours."""
        if not self.quiet_hours_enabled:
            return False
        if check_dt is None:
            check_dt = timezone.now()

        local_dt = check_dt.astimezone(self.get_tz())
        local_time = local_dt.time()
        start = self.quiet_hours_start
        end = self.quiet_hours_end

        if start > end:
            # Overnights (e.g. 22:00 to 07:00)
            return local_time >= start or local_time < end
        else:
            return start <= local_time < end

    def get_next_quiet_hours_end(self, check_dt=None) -> datetime.datetime:
        """Returns the next datetime when quiet hours end, in UTC."""
        if check_dt is None:
            check_dt = timezone.now()

        tz = self.get_tz()
        local_dt = check_dt.astimezone(tz)
        local_time = local_dt.time()
        start = self.quiet_hours_start
        end = self.quiet_hours_end

        if start > end:
            if local_time >= start:
                target_date = local_dt.date() + datetime.timedelta(days=1)
            else:
                target_date = local_dt.date()
        else:
            if local_time < end:
                target_date = local_dt.date()
            else:
                target_date = local_dt.date() + datetime.timedelta(days=1)

        end_local = datetime.datetime.combine(target_date, end, tzinfo=tz)
        return end_local.astimezone(datetime.timezone.utc)

    def has_exceeded_frequency(self, check_dt=None) -> tuple:
        """
        Returns (is_exceeded, reason).
        Checks:
        1. Daily count of delivered proactive messages.
        2. Cooldown since the most recently delivered message.
        """
        if check_dt is None:
            check_dt = timezone.now()

        tz = self.get_tz()
        local_dt = check_dt.astimezone(tz)
        start_of_day = datetime.datetime.combine(local_dt.date(), datetime.time.min, tzinfo=tz)
        start_of_day_utc = start_of_day.astimezone(datetime.timezone.utc)

        delivered_today_count = ScheduledMessage.objects.filter(
            user=self.user,
            status=ScheduledMessage.DeliveryStatus.DELIVERED,
            delivered_at__gte=start_of_day_utc
        ).count()

        if delivered_today_count >= self.max_daily_messages:
            return True, f"Daily limit of {self.max_daily_messages} proactive messages reached."

        last_delivered = ScheduledMessage.objects.filter(
            user=self.user,
            status=ScheduledMessage.DeliveryStatus.DELIVERED,
            delivered_at__isnull=False
        ).order_by('-delivered_at').first()

        if last_delivered and last_delivered.delivered_at:
            elapsed = check_dt - last_delivered.delivered_at
            cooldown = datetime.timedelta(minutes=self.min_minutes_between_messages)
            if elapsed < cooldown:
                remaining_mins = max(1, int((cooldown - elapsed).total_seconds() / 60))
                return True, f"Frequency cooldown active ({remaining_mins}m remaining)."

        return False, ""


class InAppMessage(models.Model):
    """
    Stores delivered in-app messages ready to be rendered in the React companion chat UI.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='in_app_messages'
    )
    scheduled_message = models.ForeignKey(
        ScheduledMessage,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='in_app_deliveries'
    )
    sender_name = models.CharField(max_length=100, default='AI Companion')
    content = models.TextField()
    message_type = models.CharField(max_length=50, default='casual_chat')
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_read', '-created_at'], name='idx_inapp_user_read'),
        ]

    def __str__(self):
        return f"In-app message to {self.user.username}: {self.content[:30]}..."


class PushDeviceSubscription(models.Model):
    """
    Stores user push notification subscriptions (WebPush / FCM / APNS).
    """
    class DeviceType(models.TextChoices):
        WEB = 'web', 'Web Push'
        FCM = 'fcm', 'Firebase Cloud Messaging'
        APNS = 'apns', 'Apple APNS'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='push_subscriptions'
    )
    device_type = models.CharField(
        max_length=20,
        choices=DeviceType.choices,
        default=DeviceType.WEB
    )
    endpoint = models.TextField(help_text="Push subscription endpoint or device token")
    p256dh_key = models.CharField(max_length=255, blank=True, default='')
    auth_key = models.CharField(max_length=255, blank=True, default='')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'is_active'], name='idx_push_user_active'),
        ]

    def __str__(self):
        return f"{self.device_type} subscription for {self.user.username} ({'active' if self.is_active else 'inactive'})"

