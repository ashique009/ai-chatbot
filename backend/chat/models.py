import os
import uuid
from django.db import models
from django.conf import settings


def voice_message_upload_path(instance, filename):
    """
    Store audio files partitioned by user ID for user isolation and clean storage.
    """
    user_id = 'common'
    try:
        if instance.conversation_id and hasattr(instance, 'conversation') and instance.conversation and instance.conversation.user_id:
            user_id = str(instance.conversation.user_id)
    except Exception:
        pass
    ext = os.path.splitext(filename)[1].lower() or '.mp3'
    return f"voice_messages/{user_id}/{instance.id}{ext}"


def get_default_companion_user():
    """
    Returns a default companion user when explicit authentication is not active.
    Ensures seamless development and single-user companion operation.
    """
    from django.contrib.auth import get_user_model
    User = get_user_model()
    user, _ = User.objects.get_or_create(
        username='companion_owner',
        defaults={'email': 'owner@chatpartner.local', 'first_name': 'Partner'}
    )
    return user


class Conversation(models.Model):
    """
    Represents an ongoing chat thread with the AI companion.
    Preserves a stable UUID so chats can be resumed across browser sessions.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Stable unique identifier for the conversation."
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='conversations',
        default=get_default_companion_user,
        help_text="The companion owner owning this conversation."
    )
    class Mood(models.TextChoices):
        CALM = 'calm', 'Calm'
        PLAYFUL = 'playful', 'Playful'
        HAPPY = 'happy', 'Happy'
        ANNOYED = 'annoyed', 'Annoyed'
        ANGRY = 'angry', 'Angry'
        HURT = 'hurt', 'Hurt'
        SAD = 'sad', 'Sad'
        SOFTENING = 'softening', 'Softening'

    title = models.CharField(
        max_length=255,
        default="New Conversation",
        help_text="Contextual title summarizing the conversation."
    )
    current_mood = models.CharField(
        max_length=20,
        choices=Mood.choices,
        default=Mood.CALM,
        db_index=True,
        help_text="Ami's current persistent emotional state in this conversation."
    )
    mood_context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Temporary unresolved emotional context (trigger, grievance notes, turn counts)."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', '-updated_at'], name='idx_conv_user_updated'),
        ]

    def __str__(self):
        return f"{self.title} ({self.id})"

    @property
    def message_count(self) -> int:
        return self.messages.count()


class ChatMessage(models.Model):
    """
    Individual chat messages persisted in PostgreSQL.
    """
    class Sender(models.TextChoices):
        USER = 'user', 'User'
        COMPANION = 'companion', 'Companion'

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    sender = models.CharField(
        max_length=20,
        choices=Sender.choices,
        default=Sender.USER
    )
    content = models.TextField(help_text="The raw text of the chat message.")
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional AI generation metadata (model, tokens, latency, etc.)."
    )
    class MessageType(models.TextChoices):
        TEXT = 'text', 'Text'
        VOICE = 'voice', 'Voice'

    message_type = models.CharField(
        max_length=20,
        choices=MessageType.choices,
        default=MessageType.TEXT,
        db_index=True
    )
    audio_file = models.FileField(
        upload_to=voice_message_upload_path,
        null=True,
        blank=True,
        help_text="Optional audio attachment for voice messages."
    )
    audio_duration = models.FloatField(
        null=True,
        blank=True,
        help_text="Audio duration in seconds."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at'], name='idx_msg_conv_created'),
        ]

    def __str__(self):
        return f"[{self.sender}|{self.message_type}] {self.content[:40]}..."

    @property
    def audio_url(self):
        if self.audio_file:
            return f"/api/messages/{self.id}/audio/"
        return None


class CompanionMemory(models.Model):
    """
    Long-term memory storing user-approved preferences, facts, and personal details.
    Retrieved dynamically to give Ami lasting personal context while respecting user consent.
    """
    class Category(models.TextChoices):
        PREFERENCE = 'preference', 'Preference & Habit'
        PERSONAL = 'personal', 'Personal Detail'
        WORK = 'work', 'Work & Studies'
        RELATIONSHIP = 'relationship', 'People & Relationships'
        GOAL = 'goal', 'Goals & Projects'
        OTHER = 'other', 'Other'

    class Importance(models.IntegerChoices):
        NORMAL = 1, 'Normal'
        IMPORTANT = 2, 'Important'
        CORE = 3, 'Core Personality'

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='companion_memories',
        default=get_default_companion_user
    )
    category = models.CharField(
        max_length=50,
        choices=Category.choices,
        default=Category.PERSONAL,
        db_index=True
    )
    content = models.TextField(
        help_text="Key fact or preference (e.g. 'Likes black tea with no sugar', 'Birthday is on Nov 12')."
    )
    importance = models.IntegerField(
        choices=Importance.choices,
        default=Importance.NORMAL,
        db_index=True,
        help_text="Priority level used when ranking memories for prompt inclusion."
    )
    is_approved = models.BooleanField(
        default=True,
        db_index=True,
        help_text="User consent flag: only approved memories are injected into conversational context."
    )
    source_message = models.ForeignKey(
        ChatMessage,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='derived_memories',
        help_text="Message from which this memory was extracted, if applicable."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-importance', '-created_at']
        indexes = [
            models.Index(fields=['user', 'is_approved', '-importance'], name='idx_mem_user_appr_imp'),
        ]
        verbose_name = 'Companion Memory'
        verbose_name_plural = 'Companion Memories'

    def __str__(self):
        return f"[{self.get_category_display()}] {self.content[:40]}... (Approved={self.is_approved})"
