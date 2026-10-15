import datetime
from datetime import timedelta
from unittest.mock import patch, MagicMock
from io import StringIO
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import IntegrityError
from django.conf import settings
from django.core.management import call_command
from rest_framework.test import APIClient
from rest_framework import status

from .models import (
    ScheduledMessage,
    CompanionPreference,
    InAppMessage,
    PushDeviceSubscription,
)
from .services.worker import ScheduledMessageWorker
from .services.ai_generator import CompanionAIGenerator
from .services.delivery.in_app import InAppDeliveryHandler
from .services.delivery.push import PushNotificationDeliveryHandler
from .services.exceptions import (
    AIGenerationError,
    OpenAIConfigurationError,
    PushProviderNotConfiguredError,
    PushDeliveryError,
    InAppProviderNotConfiguredError,
)

User = get_user_model()


class ScheduledMessageModelTest(TestCase):
    """Unit tests for ScheduledMessage database model, querysets, and status transitions."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='companion_owner',
            email='user@example.com',
            password='testpassword123'
        )
        self.now = timezone.now()

    def test_create_scheduled_message_types(self):
        """Test creating scheduled messages with different companion check-in types."""
        # Good morning check-in
        morning_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now + timedelta(hours=8),
            channel=ScheduledMessage.DeliveryChannel.BOTH,
            ai_prompt_context={'mood': 'energetic', 'topic': 'morning motivation'}
        )
        self.assertEqual(morning_msg.message_type, ScheduledMessage.MessageType.GOOD_MORNING)
        self.assertEqual(morning_msg.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertEqual(morning_msg.channel, ScheduledMessage.DeliveryChannel.BOTH)
        self.assertTrue(morning_msg.is_ai_generated)
        self.assertIsNotNone(morning_msg.id)
        self.assertEqual(morning_msg.ai_model, 'gpt-4o-mini')

        # Casual conversation
        casual_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now + timedelta(hours=2),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            ai_prompt_context={'interest': 'books', 'tone': 'playful'}
        )
        self.assertEqual(casual_msg.message_type, ScheduledMessage.MessageType.CASUAL_CHAT)
        self.assertEqual(casual_msg.channel, ScheduledMessage.DeliveryChannel.IN_APP)

        # Evening message
        evening_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=self.now + timedelta(hours=12),
            channel=ScheduledMessage.DeliveryChannel.PUSH,
            ai_prompt_context={'topic': 'evening unwind'}
        )
        self.assertEqual(evening_msg.message_type, ScheduledMessage.MessageType.EVENING_CHECKIN)
        self.assertEqual(evening_msg.channel, ScheduledMessage.DeliveryChannel.PUSH)

    def test_idempotency_prevents_duplicate_scheduling(self):
        """Test that duplicate messages with identical idempotency_key are rejected."""
        idempotency_key = 'unique-morning-checkin-2026-10-15'
        ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now + timedelta(hours=1),
            idempotency_key=idempotency_key
        )

        with self.assertRaises(IntegrityError):
            ScheduledMessage.objects.create(
                user=self.user,
                message_type=ScheduledMessage.MessageType.GOOD_MORNING,
                scheduled_time=self.now + timedelta(hours=1),
                idempotency_key=idempotency_key
            )

    def test_is_due_and_due_queryset(self):
        """Test identification of due messages for periodic scheduler execution."""
        due_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=5),
            status=ScheduledMessage.DeliveryStatus.PENDING
        )
        future_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=self.now + timedelta(hours=2),
            status=ScheduledMessage.DeliveryStatus.PENDING
        )
        delivered_msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now - timedelta(hours=1),
            status=ScheduledMessage.DeliveryStatus.DELIVERED
        )

        self.assertTrue(due_msg.is_due())
        self.assertFalse(future_msg.is_due())
        self.assertFalse(delivered_msg.is_due())

        due_messages = ScheduledMessage.objects.due()
        self.assertIn(due_msg, due_messages)
        self.assertNotIn(future_msg, due_messages)
        self.assertNotIn(delivered_msg, due_messages)

    def test_mark_processing_prevents_duplicate_processing(self):
        """Test atomic state transition to prevent race conditions during message pickup."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=1),
            status=ScheduledMessage.DeliveryStatus.PENDING
        )

        # First worker acquires
        acquired = msg.mark_processing()
        self.assertTrue(acquired)
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.PROCESSING)
        self.assertEqual(msg.attempts, 1)

        # Second worker attempts to acquire the same message
        re_acquired = msg.mark_processing()
        self.assertFalse(re_acquired)

    def test_mark_delivered(self):
        """Test successful delivery marks timestamps, content, and channel flags."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now - timedelta(minutes=2),
            status=ScheduledMessage.DeliveryStatus.PROCESSING
        )
        ai_generated_text = "Good morning! Hope you have a wonderful day ahead!"
        ai_meta = {'prompt_tokens': 25, 'completion_tokens': 12, 'model': 'gpt-4o-mini'}

        msg.mark_delivered(
            content=ai_generated_text,
            delivered_in_app=True,
            delivered_push=True,
            ai_metadata=ai_meta
        )

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.DELIVERED)
        self.assertEqual(msg.content, ai_generated_text)
        self.assertTrue(msg.delivered_in_app)
        self.assertTrue(msg.delivered_push)
        self.assertIsNotNone(msg.delivered_at)
        self.assertIsNone(msg.error_message)
        self.assertEqual(msg.ai_metadata, ai_meta)

    def test_mark_failed_and_retry_threshold(self):
        """Test failure tracking and threshold retry escalation."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=self.now - timedelta(minutes=5),
            status=ScheduledMessage.DeliveryStatus.PENDING,
            max_attempts=2
        )

        # Attempt 1 failure: should return to PENDING for retry
        msg.mark_processing()
        msg.mark_failed("OpenAI API rate limit exceeded")
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertEqual(msg.error_message, "OpenAI API rate limit exceeded")

        # Attempt 2 failure: reaches max_attempts (2), should transition to FAILED
        msg.mark_processing()
        msg.mark_failed("Network unreachable")
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.FAILED)
        self.assertEqual(msg.error_message, "Network unreachable")

    def test_environment_configuration(self):
        """Verify that OpenAI and DB configuration settings are loaded properly."""
        self.assertTrue(hasattr(settings, 'OPENAI_API_KEY'))
        self.assertTrue(hasattr(settings, 'OPENAI_MODEL'))
        self.assertEqual(settings.OPENAI_MODEL, 'gpt-4o-mini')
        self.assertEqual(settings.DATABASES['default']['ENGINE'], 'django.db.backends.postgresql')


class ScheduledMessageWorkerTest(TestCase):
    """
    Comprehensive tests for the scheduled worker engine:
    - Due-message processing & AI generation
    - Concurrent workers and atomic claiming
    - OpenAI failures & retry thresholds
    - In-app & push delivery handlers without faking success
    - Quiet hours & frequency limit respect
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username='ashique_user',
            email='ashique@example.com',
            password='secretpassword'
        )
        self.preference = CompanionPreference.objects.create(
            user=self.user,
            companion_name='Ami',
            user_nickname='da',
            preferred_language=CompanionPreference.LanguagePreference.MANGLISH,
            companion_tone=CompanionPreference.CompanionTone.CARING,
            proactive_messaging_enabled=True,
            quiet_hours_enabled=False,  # disabled for general tests
            max_daily_messages=5,
            min_minutes_between_messages=60
        )
        self.now = timezone.now()

    @patch('proactive_messaging.services.ai_generator.CompanionAIGenerator.generate')
    def test_due_message_processing_success(self, mock_generate):
        """Tests that a due message is claimed, generated in Manglish, and delivered to in-app inbox."""
        manglish_content = "Morning da! Kazhicho? Have a productive day ahead tto!"
        ai_meta = {'model': 'gpt-4o-mini', 'tokens': 42}
        mock_generate.return_value = (manglish_content, ai_meta)

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            is_ai_generated=True,
            ai_prompt_context={'mood': 'focused'}
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages(limit=10)

        self.assertEqual(result['total_due'], 1)
        self.assertEqual(result['delivered'], 1)
        self.assertEqual(result['failed'], 0)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.DELIVERED)
        self.assertEqual(msg.content, manglish_content)
        self.assertTrue(msg.delivered_in_app)
        self.assertFalse(msg.delivered_push)
        self.assertIsNotNone(msg.delivered_at)

        # Verify message exists in InAppMessage model
        in_app_deliveries = InAppMessage.objects.filter(scheduled_message=msg)
        self.assertEqual(in_app_deliveries.count(), 1)
        in_app_item = in_app_deliveries.first()
        self.assertEqual(in_app_item.content, manglish_content)
        self.assertEqual(in_app_item.sender_name, 'Ami')

    def test_concurrent_workers_prevents_duplicate_processing(self):
        """Tests that two concurrent workers do not double-process the same due message."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=2),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Pre-crafted message"
        )

        worker1 = ScheduledMessageWorker()
        worker2 = ScheduledMessageWorker()

        # Worker 1 acquires the atomic claim
        claimed = msg.mark_processing()
        self.assertTrue(claimed)

        # Worker 2 attempts to process the same message
        outcome2 = worker2.process_single_message(msg)
        self.assertEqual(outcome2['status'], 'skipped')
        self.assertIn('Already claimed', outcome2['reason'])

    @patch('proactive_messaging.services.ai_generator.CompanionAIGenerator.generate')
    def test_openai_failure_retries_under_max_attempts(self, mock_generate):
        """Tests that an OpenAI generation error re-queues the message for retry when attempts < max_attempts."""
        mock_generate.side_effect = AIGenerationError("OpenAI API rate limit exceeded (HTTP 429)")

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            attempts=0,
            max_attempts=3
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['retried'], 1)
        self.assertEqual(result['failed'], 0)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertEqual(msg.attempts, 1)
        self.assertIn("rate limit exceeded", msg.error_message)

    @patch('proactive_messaging.services.ai_generator.CompanionAIGenerator.generate')
    def test_openai_failure_marks_failed_when_max_attempts_reached(self, mock_generate):
        """Tests that reaching max_attempts marks the message as permanently FAILED."""
        mock_generate.side_effect = AIGenerationError("Authentication failed: invalid API key")

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            attempts=2,  # Already attempted twice
            max_attempts=3
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['retried'], 0)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.FAILED)
        self.assertEqual(msg.attempts, 3)
        self.assertIn("Authentication failed", msg.error_message)

    @override_settings(PUSH_NOTIFICATIONS_ENABLED=False)
    def test_push_delivery_fails_when_provider_not_configured(self):
        """Tests that delivery strictly refuses to fake success when push provider is not configured."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=self.now - timedelta(minutes=3),
            channel=ScheduledMessage.DeliveryChannel.PUSH,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Good night da!",
            max_attempts=1
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['failed'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.FAILED)
        self.assertFalse(msg.delivered_push)
        self.assertIn("Push provider is not configured", msg.error_message)

    @override_settings(PUSH_NOTIFICATIONS_ENABLED=False)
    def test_both_channels_fails_if_push_fails(self):
        """Tests that when BOTH channels are requested and push fails, failure is tracked and not faked."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now - timedelta(minutes=3),
            channel=ScheduledMessage.DeliveryChannel.BOTH,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Rise and shine da!",
            max_attempts=1
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['failed'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.FAILED)
        self.assertFalse(msg.delivered_push)
        self.assertIn("Push provider is not configured", msg.error_message)

    @override_settings(IN_APP_MESSAGING_ENABLED=False)
    def test_in_app_delivery_fails_when_in_app_disabled(self):
        """Tests that in-app delivery handler refuses delivery if feature is disabled in settings."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=1),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Hello dear",
            max_attempts=1
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['failed'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.FAILED)
        self.assertIn("In-app messaging delivery is disabled", msg.error_message)

    def test_quiet_hours_postpones_due_message(self):
        """Tests that messages due during quiet hours are postponed rather than sent."""
        # Enable quiet hours and simulate that current time is within quiet hours
        self.preference.quiet_hours_enabled = True
        self.preference.save()

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=self.now - timedelta(minutes=10),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Night check-in"
        )

        with patch.object(CompanionPreference, 'is_in_quiet_hours', return_value=True):
            worker = ScheduledMessageWorker()
            result = worker.process_due_messages()

        self.assertEqual(result['postponed'], 1)
        self.assertEqual(result['delivered'], 0)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertIn("Currently within quiet hours", msg.error_message)

    def test_frequency_limit_postpones_due_message(self):
        """Tests that messages exceeding daily limit or cooldown are postponed."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=self.now - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Casual check"
        )

        with patch.object(CompanionPreference, 'has_exceeded_frequency', return_value=(True, "Daily limit of 5 proactive messages reached.")):
            worker = ScheduledMessageWorker()
            result = worker.process_due_messages()

        self.assertEqual(result['postponed'], 1)
        self.assertEqual(result['delivered'], 0)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertIn("Daily limit", msg.error_message)

    def test_user_disabled_proactive_messaging_cancels_due_message(self):
        """Tests that a due message is cancelled if user disabled proactive messaging."""
        self.preference.proactive_messaging_enabled = False
        self.preference.save()

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=self.now - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Morning!"
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['cancelled'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.CANCELLED)
        self.assertIn("disabled by user preference", msg.error_message)

    def test_management_command_execution(self):
        """Tests executing the process_due_messages management command via call_command."""
        out = StringIO()
        call_command('process_due_messages', '--once', stdout=out)
        output = out.getvalue()
        self.assertIn("Proactive messaging worker started", output)


class CompanionAIGeneratorTest(TestCase):
    """Unit tests for CompanionAIGenerator prompt crafting, language support, and error cases."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='manglish_fan',
            email='manglish@example.com',
            password='testpassword'
        )
        self.generator = CompanionAIGenerator()

    @override_settings(AI_PROVIDER='openai', OPENAI_API_KEY='')
    def test_missing_api_key_raises_configuration_error(self):
        """Tests that attempting generation without an API key raises OpenAIConfigurationError."""
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now()
        )
        with self.assertRaises(OpenAIConfigurationError):
            self.generator.generate(msg)

    def test_manglish_system_prompt_structure(self):
        """Tests that Manglish instructions are embedded in the prompt when language is Manglish."""
        system_prompt = self.generator._build_system_prompt(
            companion_name='Ami',
            user_nickname='da',
            language=CompanionPreference.LanguagePreference.MANGLISH,
            tone='caring',
            relationship_style='loving partner'
        )
        self.assertIn("Manglish", system_prompt)
        self.assertIn("Kazhicho da?", system_prompt)
        self.assertIn("Ami", system_prompt)
        self.assertIn("da", system_prompt)

    def test_english_system_prompt_structure(self):
        """Tests that English instructions are embedded when language is English."""
        system_prompt = self.generator._build_system_prompt(
            companion_name='Sophia',
            user_nickname='dear',
            language=CompanionPreference.LanguagePreference.ENGLISH,
            tone='playful',
            relationship_style='loving partner'
        )
        self.assertIn("English", system_prompt)
        self.assertIn("Sophia", system_prompt)
        self.assertIn("dear", system_prompt)

    def test_generate_with_mock_client(self):
        """Tests successful generation workflow with a mock OpenAI response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Good morning da! Had breakfast?"))
        ]
        mock_response.model = 'gpt-4o-mini'
        mock_response.id = 'chatcmpl-test-123'
        mock_response.usage = MagicMock(prompt_tokens=30, completion_tokens=10, total_tokens=40)
        mock_client.chat.completions.create.return_value = mock_response

        generator = CompanionAIGenerator(client=mock_client)
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now()
        )
        content, meta = generator.generate(msg)

        self.assertEqual(content, "Good morning da! Had breakfast?")
        self.assertEqual(meta['model'], 'gpt-4o-mini')
        self.assertEqual(meta['completion_id'], 'chatcmpl-test-123')
        self.assertEqual(meta['total_tokens'], 40)


class PersonalizationAndContextIntegrationTests(TestCase):
    """
    Tests ensuring Ami's proactive messages incorporate the authenticated user's
    approved long-term memories and recent chat history.
    """

    def setUp(self):
        from chat.models import Conversation, ChatMessage, CompanionMemory

        self.user = User.objects.create_user(
            username='ashique_partner',
            email='ashique@test.local',
            password='Password123!'
        )
        self.preference = CompanionPreference.objects.create(
            user=self.user,
            companion_name='Ami',
            user_nickname='da',
            preferred_language=CompanionPreference.LanguagePreference.MANGLISH,
            companion_tone=CompanionPreference.CompanionTone.CARING,
        )

        # Approved memory
        self.approved_memory = CompanionMemory.objects.create(
            user=self.user,
            category=CompanionMemory.Category.PREFERENCE,
            content="Loves black tea with cardamom and no sugar",
            importance=CompanionMemory.Importance.IMPORTANT,
            is_approved=True
        )

        # Unapproved memory (must NOT be used)
        self.unapproved_memory = CompanionMemory.objects.create(
            user=self.user,
            category=CompanionMemory.Category.PERSONAL,
            content="Secret bank password or unapproved ephemeral note",
            importance=CompanionMemory.Importance.NORMAL,
            is_approved=False
        )

        # Recent conversation turns
        self.conversation = Conversation.objects.create(
            user=self.user,
            title="Work evening chat"
        )
        ChatMessage.objects.create(
            conversation=self.conversation,
            sender=ChatMessage.Sender.USER,
            content="Had a super exhausting sprint demo today at office"
        )
        ChatMessage.objects.create(
            conversation=self.conversation,
            sender=ChatMessage.Sender.COMPANION,
            content="Aww da, you worked so hard! Rest well tonight tto."
        )

    def test_personalization_includes_approved_memories_and_recent_chats(self):
        """Tests that generate() pulls approved memories & recent chats into the OpenAI prompts."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Good morning da! Hope you rested well after yesterday's sprint demo."))
        ]
        mock_response.model = 'gpt-4o-mini'
        mock_response.id = 'chatcmpl-personal-1'
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=15, total_tokens=65)
        mock_client.chat.completions.create.return_value = mock_response

        generator = CompanionAIGenerator(client=mock_client)
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now()
        )

        content, meta = generator.generate(msg)
        self.assertIn("sprint demo", content)

        # Verify the prompt sent to OpenAI contained the approved memory and conversation context
        called_args = mock_client.chat.completions.create.call_args[1]
        sent_messages = called_args['messages']
        system_text = sent_messages[0]['content']
        user_text = sent_messages[1]['content']

        # Approved memory present
        self.assertIn("black tea with cardamom", system_text)
        # Unapproved memory absent
        self.assertNotIn("Secret bank password", system_text)
        # Recent chat context present
        self.assertIn("exhausting sprint demo", user_text)


class SchedulerAndDuplicatePreventionTests(TestCase):
    """
    Tests for timezone-aware scheduling, daily check-in generation,
    quiet hours filtering, and idempotency.
    """

    def setUp(self):
        from proactive_messaging.services.scheduler import create_daily_checkin_schedule, schedule_single_checkin
        self.create_daily_checkin_schedule = create_daily_checkin_schedule
        self.schedule_single_checkin = schedule_single_checkin

        self.user = User.objects.create_user(
            username='scheduler_user',
            password='Password123!'
        )
        self.pref = CompanionPreference.objects.create(
            user=self.user,
            timezone='Asia/Kolkata',
            quiet_hours_enabled=True,
            quiet_hours_start=datetime.time(22, 0),
            quiet_hours_end=datetime.time(7, 0),
            max_daily_messages=3,
            min_minutes_between_messages=60
        )

    def test_daily_schedule_creates_sensible_slots(self):
        """Tests that create_daily_checkin_schedule creates morning, casual, and evening slots."""
        import datetime as dt
        today = dt.date(2026, 10, 15)
        items = self.create_daily_checkin_schedule(self.user, target_date=today)

        self.assertGreaterEqual(len(items), 1)
        self.assertLessEqual(len(items), 3)

        # Every scheduled item must belong to the user
        for item in items:
            self.assertEqual(item.user, self.user)
            self.assertEqual(item.status, ScheduledMessage.DeliveryStatus.PENDING)
            self.assertEqual(item.channel, ScheduledMessage.DeliveryChannel.IN_APP)

    def test_daily_schedule_is_idempotent(self):
        """Calling create_daily_checkin_schedule multiple times for same date does not duplicate."""
        import datetime as dt
        today = dt.date(2026, 10, 16)

        run1 = self.create_daily_checkin_schedule(self.user, target_date=today)
        run2 = self.create_daily_checkin_schedule(self.user, target_date=today)

        self.assertEqual(len(run1), len(run2))
        run1_ids = {m.id for m in run1}
        run2_ids = {m.id for m in run2}
        self.assertEqual(run1_ids, run2_ids)

        # Count total in DB for this date
        count = ScheduledMessage.objects.filter(user=self.user).count()
        self.assertEqual(count, len(run1))

    def test_schedule_disabled_user_returns_empty(self):
        """If proactive messaging is disabled, no daily check-ins are created."""
        self.pref.proactive_messaging_enabled = False
        self.pref.save()

        import datetime as dt
        items = self.create_daily_checkin_schedule(self.user, target_date=dt.date(2026, 10, 17))
        self.assertEqual(len(items), 0)


class ProactiveApiAndUserIsolationTests(TestCase):
    """
    Tests for authenticated API endpoints, CSRF / session authorization,
    cross-user isolation, and in-app inbox management.
    """

    def setUp(self):
        self.user_a = User.objects.create_user(
            username='user_alice',
            password='AlicePassword123!'
        )
        self.user_b = User.objects.create_user(
            username='user_bob',
            password='BobPassword123!'
        )

        self.pref_a = CompanionPreference.objects.create(
            user=self.user_a,
            companion_name='Ami',
            user_nickname='Allie',
            preferred_language=CompanionPreference.LanguagePreference.MANGLISH,
            max_daily_messages=4,
        )

        self.pref_b = CompanionPreference.objects.create(
            user=self.user_b,
            companion_name='Maya',
            user_nickname='Bobby',
            preferred_language=CompanionPreference.LanguagePreference.ENGLISH,
            max_daily_messages=2,
        )

        # User A inbox message
        self.inbox_msg_a = InAppMessage.objects.create(
            user=self.user_a,
            sender_name='Ami',
            content='Hey Allie! Kazhicho?',
            message_type='casual_chat'
        )

        # User B inbox message
        self.inbox_msg_b = InAppMessage.objects.create(
            user=self.user_b,
            sender_name='Maya',
            content='Hi Bobby! How was your run?',
            message_type='casual_chat'
        )

    def test_unauthenticated_requests_rejected(self):
        """Unauthenticated requests to proactive endpoints return 401 or 403."""
        endpoints = [
            ('/api/proactive/preferences/', 'get'),
            ('/api/proactive/inbox/', 'get'),
            (f'/api/proactive/inbox/{self.inbox_msg_a.id}/', 'get'),
            ('/api/proactive/inbox/read-all/', 'post'),
            ('/api/proactive/scheduled/', 'get'),
            ('/api/proactive/schedule-daily/', 'post'),
        ]
        for url, method in endpoints:
            func = getattr(self.client, method)
            res = func(url)
            self.assertIn(res.status_code, [401, 403], f"Expected 401/403 for {url}, got {res.status_code}")

    def test_preferences_get_and_patch(self):
        """Authenticated user can fetch and update their own preferences."""
        self.client.force_login(self.user_a)

        # GET
        res = self.client.get('/api/proactive/preferences/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['companion_name'], 'Ami')
        self.assertEqual(data['user_nickname'], 'Allie')

        # PATCH
        patch_res = self.client.patch(
            '/api/proactive/preferences/',
            data={
                'user_nickname': 'Kutta',
                'quiet_hours_start': '23:00',
                'quiet_hours_end': '06:30',
                'max_daily_messages': 6,
                'min_minutes_between_messages': 90,
            },
            content_type='application/json'
        )
        self.assertEqual(patch_res.status_code, 200)
        updated = patch_res.json()
        self.assertEqual(updated['user_nickname'], 'Kutta')
        self.assertEqual(updated['quiet_hours_start'], '23:00')
        self.assertEqual(updated['quiet_hours_end'], '06:30')
        self.assertEqual(updated['max_daily_messages'], 6)

        self.pref_a.refresh_from_db()
        self.assertEqual(self.pref_a.user_nickname, 'Kutta')

    def test_cross_user_isolation_in_inbox(self):
        """User A cannot view, modify, or delete User B's in-app inbox messages."""
        self.client.force_login(self.user_a)

        # User A views their inbox: should only see their own message
        res = self.client.get('/api/proactive/inbox/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        msg_ids = [m['id'] for m in data['messages']]
        self.assertIn(str(self.inbox_msg_a.id), msg_ids)
        self.assertNotIn(str(self.inbox_msg_b.id), msg_ids)

        # User A attempts to view User B's message directly -> 404
        direct_res = self.client.get(f'/api/proactive/inbox/{self.inbox_msg_b.id}/')
        self.assertEqual(direct_res.status_code, 404)

        # User A attempts to delete User B's message directly -> 404
        del_res = self.client.delete(f'/api/proactive/inbox/{self.inbox_msg_b.id}/')
        self.assertEqual(del_res.status_code, 404)
        self.assertTrue(InAppMessage.objects.filter(id=self.inbox_msg_b.id).exists())

    def test_inbox_mark_all_read(self):
        """User A can mark all their unread messages as read."""
        self.client.force_login(self.user_a)
        self.assertFalse(self.inbox_msg_a.is_read)

        res = self.client.post('/api/proactive/inbox/read-all/')
        self.assertEqual(res.status_code, 200)

        self.inbox_msg_a.refresh_from_db()
        self.assertTrue(self.inbox_msg_a.is_read)

        # User B's message remains untouched
        self.inbox_msg_b.refresh_from_db()
        self.assertFalse(self.inbox_msg_b.is_read)

    def test_cross_user_scheduled_message_isolation(self):
        """User A cannot view or cancel User B's scheduled messages."""
        msg_b = ScheduledMessage.objects.create(
            user=self.user_b,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=timezone.now() + timedelta(hours=3),
            channel=ScheduledMessage.DeliveryChannel.IN_APP
        )

        self.client.force_login(self.user_a)

        # User A list: does not include msg_b
        list_res = self.client.get('/api/proactive/scheduled/')
        self.assertEqual(list_res.status_code, 200)
        b_ids = [m['id'] for m in list_res.json()]
        self.assertNotIn(str(msg_b.id), b_ids)

        # User A tries to delete msg_b -> 404
        cancel_res = self.client.delete(f'/api/proactive/scheduled/{msg_b.id}/')
        self.assertEqual(cancel_res.status_code, 404)

        msg_b.refresh_from_db()
        self.assertEqual(msg_b.status, ScheduledMessage.DeliveryStatus.PENDING)


class WebPushNotificationTests(TestCase):
    """
    Comprehensive tests for Web Push notifications:
    - VAPID public key API (authenticated, no private key exposure)
    - Browser device subscription registration & listing
    - Ownership isolation across users
    - Subscription deactivation (unsubscribe & delete)
    - WebPush delivery via mock pywebpush
    - Automatic deactivation of expired subscriptions (HTTP 410 / 404)
    - Test push notification endpoint
    - Worker dual-channel (in-app + push) delivery without duplicates
    """

    def setUp(self):
        self.user_a = User.objects.create_user(username='alice', password='password123')
        self.user_b = User.objects.create_user(username='bob', password='password123')

        CompanionPreference.objects.create(
            user=self.user_a,
            companion_name='Ami',
            proactive_messaging_enabled=True,
            quiet_hours_enabled=False,
        )

        self.mock_vapid_settings = {
            'PUSH_NOTIFICATIONS_ENABLED': True,
            'VAPID_PUBLIC_KEY': 'test-public-key-12345',
            'VAPID_PRIVATE_KEY': 'test-private-key-67890',
            'VAPID_ADMIN_EMAIL': 'mailto:admin@example.com',
        }

    def test_vapid_public_key_endpoint_authenticated(self):
        """Authenticated user receives VAPID public key; private key is never exposed."""
        # Unauthenticated request fails with 401/403
        anon_res = self.client.get('/api/proactive/push/vapid-public-key/')
        self.assertIn(anon_res.status_code, (401, 403))

        # Authenticated request succeeds
        self.client.force_login(self.user_a)
        with override_settings(**self.mock_vapid_settings):
            res = self.client.get('/api/proactive/push/vapid-public-key/')
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data['public_key'], 'test-public-key-12345')
            self.assertTrue(data['enabled'])
            self.assertNotIn('private_key', data)
            self.assertNotIn('test-private-key-67890', str(data))

    def test_push_subscribe_and_list(self):
        """User can subscribe browser device and list their active subscriptions."""
        self.client.force_login(self.user_a)

        payload = {
            "endpoint": "https://fcm.googleapis.com/fcm/send/test-endpoint-1",
            "keys": {
                "p256dh": "BNcRdreALRF8...",
                "auth": "tBHItJI5svb...",
            },
            "device_type": "web"
        }

        res = self.client.post('/api/proactive/push/subscribe/', data=payload, content_type='application/json')
        self.assertEqual(res.status_code, 201)
        sub_id = res.json()['id']

        # Verify DB entry
        sub = PushDeviceSubscription.objects.get(id=sub_id)
        self.assertEqual(sub.user, self.user_a)
        self.assertEqual(sub.endpoint, payload['endpoint'])
        self.assertEqual(sub.p256dh_key, payload['keys']['p256dh'])
        self.assertEqual(sub.auth_key, payload['keys']['auth'])
        self.assertTrue(sub.is_active)

        # List active subscriptions
        list_res = self.client.get('/api/proactive/push/subscriptions/')
        self.assertEqual(list_res.status_code, 200)
        self.assertEqual(len(list_res.json()), 1)
        self.assertEqual(list_res.json()[0]['id'], sub_id)

    def test_push_subscription_cross_user_isolation(self):
        """User A cannot access or delete User B's push device subscriptions."""
        sub_b = PushDeviceSubscription.objects.create(
            user=self.user_b,
            endpoint="https://fcm.googleapis.com/fcm/send/user-b-device",
            p256dh_key="key-b",
            auth_key="auth-b",
            is_active=True
        )

        self.client.force_login(self.user_a)

        # User A list does not show User B's device
        list_res = self.client.get('/api/proactive/push/subscriptions/')
        self.assertEqual(list_res.status_code, 200)
        ids = [s['id'] for s in list_res.json()]
        self.assertNotIn(str(sub_b.id), ids)

        # User A attempts to delete User B's subscription -> 404
        del_res = self.client.delete(f'/api/proactive/push/subscriptions/{sub_b.id}/')
        self.assertEqual(del_res.status_code, 404)
        self.assertTrue(PushDeviceSubscription.objects.filter(id=sub_b.id).exists())

    def test_push_unsubscribe_deactivates_subscription(self):
        """Unsubscribe endpoint deactivates subscription without breaking in-app messaging."""
        sub = PushDeviceSubscription.objects.create(
            user=self.user_a,
            endpoint="https://fcm.googleapis.com/fcm/send/device-to-unsubscribe",
            p256dh_key="p256",
            auth_key="auth",
            is_active=True
        )

        self.client.force_login(self.user_a)
        unsub_res = self.client.post(
            '/api/proactive/push/unsubscribe/',
            data={"endpoint": sub.endpoint},
            content_type='application/json'
        )
        self.assertEqual(unsub_res.status_code, 200)
        self.assertTrue(unsub_res.json()['unsubscribed'])

        sub.refresh_from_db()
        self.assertFalse(sub.is_active)

    @patch('pywebpush.webpush')
    def test_push_delivery_handler_success(self, mock_webpush):
        """Test successful WebPush delivery via PushNotificationDeliveryHandler."""
        sub = PushDeviceSubscription.objects.create(
            user=self.user_a,
            endpoint="https://updates.push.services.mozilla.com/wpush/v2/test-token",
            p256dh_key="valid-p256dh",
            auth_key="valid-auth",
            is_active=True
        )

        handler = PushNotificationDeliveryHandler()
        with override_settings(**self.mock_vapid_settings):
            result = handler.deliver(
                scheduled_message=None,
                content="Hey Alice, drink some water! 💧",
                user=self.user_a,
                title="Ami Reminder"
            )

            self.assertTrue(result['delivered'])
            self.assertEqual(result['devices_delivered'], 1)
            mock_webpush.assert_called_once()
            call_kwargs = mock_webpush.call_args[1]
            self.assertEqual(call_kwargs['vapid_private_key'], 'test-private-key-67890')
            self.assertEqual(call_kwargs['vapid_claims'], {'sub': 'mailto:admin@example.com'})

    @patch('pywebpush.webpush')
    def test_push_delivery_handler_deactivates_expired_subscription(self, mock_webpush):
        """When push service returns HTTP 410/404 Gone, the expired subscription is marked inactive."""
        sub = PushDeviceSubscription.objects.create(
            user=self.user_a,
            endpoint="https://fcm.googleapis.com/fcm/send/expired-device-token",
            p256dh_key="key",
            auth_key="auth",
            is_active=True
        )

        # Simulate WebPushException with HTTP 410 response
        mock_response = MagicMock()
        mock_response.status_code = 410
        mock_exception = Exception("Push Service Error: 410 Gone")
        mock_exception.response = mock_response
        mock_webpush.side_effect = mock_exception

        handler = PushNotificationDeliveryHandler()
        with override_settings(**self.mock_vapid_settings):
            with self.assertRaises(PushDeliveryError):
                handler.deliver(
                    scheduled_message=None,
                    content="Hello!",
                    user=self.user_a
                )

        sub.refresh_from_db()
        self.assertFalse(sub.is_active)

    @patch('pywebpush.webpush')
    def test_push_send_test_notification_api(self, mock_webpush):
        """Authenticated user can send a test push notification to their registered device."""
        PushDeviceSubscription.objects.create(
            user=self.user_a,
            endpoint="https://fcm.googleapis.com/fcm/send/test-phone",
            p256dh_key="key",
            auth_key="auth",
            is_active=True
        )

        self.client.force_login(self.user_a)
        with override_settings(**self.mock_vapid_settings):
            res = self.client.post(
                '/api/proactive/push/test/',
                data={"message": "Test ping from Ami"},
                content_type='application/json'
            )
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()['success'])
            mock_webpush.assert_called_once()

    @patch('pywebpush.webpush')
    def test_worker_dual_channel_delivery_no_in_app_duplicates(self, mock_webpush):
        """
        Worker processes scheduled message with channel=BOTH:
        Both in-app message and push notification succeed.
        Retrying does not duplicate the in-app message.
        """
        PushDeviceSubscription.objects.create(
            user=self.user_a,
            endpoint="https://fcm.googleapis.com/fcm/send/active-device",
            p256dh_key="key",
            auth_key="auth",
            is_active=True
        )

        msg = ScheduledMessage.objects.create(
            user=self.user_a,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now() - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.BOTH,
            content="Good morning Alice!",
            status=ScheduledMessage.DeliveryStatus.PENDING,
        )

        worker = ScheduledMessageWorker()
        with override_settings(**self.mock_vapid_settings):
            result = worker.process_due_messages()

            self.assertEqual(result['delivered'], 1)
            msg.refresh_from_db()
            self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.DELIVERED)
            self.assertTrue(msg.delivered_in_app)
            self.assertTrue(msg.delivered_push)

            # Exactly 1 InAppMessage created
            in_app_count = InAppMessage.objects.filter(scheduled_message=msg).count()
            self.assertEqual(in_app_count, 1)

            # Running worker again should not process or deliver duplicates
            result_again = worker.process_due_messages()
            self.assertEqual(result_again['delivered'], 0)
            self.assertEqual(InAppMessage.objects.filter(scheduled_message=msg).count(), 1)


class ProactiveMessagingPhase3CompletionTests(TestCase):
    """
    Comprehensive test suite for Phase 3 completion requirements:
    1. Frequency preferences (disabled, daily, few_times_week)
    2. Scheduler enforcement of frequency & timezone boundaries
    3. Queued message cancellation upon disabled frequency or non-checkin days
    4. Gemini-compatible AI generator routing with mocked client
    5. Safe ChatMessage thread persistence, idempotency, and user isolation
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='phase3_user', password='password123')
        self.user_other = User.objects.create_user(username='other_user', password='password123')
        self.client.force_authenticate(user=self.user)
        self.pref, _ = CompanionPreference.objects.get_or_create(user=self.user)
        self.pref.timezone = 'Asia/Kolkata'
        self.pref.save()
        self.pref_url = '/api/proactive/preferences/'

    def test_frequency_preference_update_and_serialization(self):
        """Tests updating frequency via API view and verifying response payload."""
        # Update to few_times_week
        res = self.client.patch(self.pref_url, data={'frequency': 'few_times_week'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.json()['frequency'], 'few_times_week')
        self.assertTrue(res.json()['proactive_messaging_enabled'])

        # Update to disabled
        res_dis = self.client.patch(self.pref_url, data={'frequency': 'disabled'}, format='json')
        self.assertEqual(res_dis.status_code, status.HTTP_200_OK)
        self.assertEqual(res_dis.json()['frequency'], 'disabled')
        self.assertFalse(res_dis.json()['proactive_messaging_enabled'])

        # Reject invalid frequency choice
        res_bad = self.client.patch(self.pref_url, data={'frequency': 'every_hour'}, format='json')
        self.assertEqual(res_bad.status_code, status.HTTP_400_BAD_REQUEST)

    def test_frequency_daily_schedules_one_checkin_per_day(self):
        """Tests that daily frequency schedules exactly one check-in for the target day."""
        from proactive_messaging.services.scheduler import create_daily_checkin_schedule
        self.pref.frequency = CompanionPreference.FrequencyChoice.DAILY
        self.pref.save()

        # Choose a target date
        import datetime as dt
        target_date = dt.date(2026, 10, 15)  # Thursday
        items = create_daily_checkin_schedule(self.user, target_date=target_date)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].user, self.user)
        self.assertEqual(items[0].status, ScheduledMessage.DeliveryStatus.PENDING)

        # Calling again on same day is idempotent
        run2 = create_daily_checkin_schedule(self.user, target_date=target_date)
        self.assertEqual(len(run2), 1)
        self.assertEqual(run2[0].id, items[0].id)

    def test_frequency_few_times_week_schedules_on_mon_wed_fri_only(self):
        """Tests few_times_week schedules on Mon, Wed, Fri and returns empty on other days."""
        from proactive_messaging.services.scheduler import create_daily_checkin_schedule
        self.pref.frequency = CompanionPreference.FrequencyChoice.FEW_TIMES_WEEK
        self.pref.save()

        import datetime as dt
        # 2026-10-19 is Monday (weekday 0)
        # 2026-10-20 is Tuesday (weekday 1)
        # 2026-10-21 is Wednesday (weekday 2)
        # 2026-10-22 is Thursday (weekday 3)
        # 2026-10-23 is Friday (weekday 4)
        # 2026-10-24 is Saturday (weekday 5)
        mon = dt.date(2026, 10, 19)
        tue = dt.date(2026, 10, 20)
        wed = dt.date(2026, 10, 21)
        thu = dt.date(2026, 10, 22)
        fri = dt.date(2026, 10, 23)
        sat = dt.date(2026, 10, 24)

        items_mon = create_daily_checkin_schedule(self.user, target_date=mon)
        self.assertEqual(len(items_mon), 1)

        items_tue = create_daily_checkin_schedule(self.user, target_date=tue)
        self.assertEqual(len(items_tue), 0)

        items_wed = create_daily_checkin_schedule(self.user, target_date=wed)
        self.assertEqual(len(items_wed), 1)

        items_thu = create_daily_checkin_schedule(self.user, target_date=thu)
        self.assertEqual(len(items_thu), 0)

        items_fri = create_daily_checkin_schedule(self.user, target_date=fri)
        self.assertEqual(len(items_fri), 1)

        items_sat = create_daily_checkin_schedule(self.user, target_date=sat)
        self.assertEqual(len(items_sat), 0)

    def test_frequency_disabled_cancels_due_messages(self):
        """Tests that when frequency is disabled, due messages are cancelled by the worker."""
        self.pref.frequency = CompanionPreference.FrequencyChoice.DISABLED
        self.pref.save()

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now() - timedelta(minutes=5),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Morning da!"
        )

        worker = ScheduledMessageWorker()
        result = worker.process_due_messages()

        self.assertEqual(result['cancelled'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.CANCELLED)
        self.assertIn("disabled by user preference", msg.error_message)

    def test_frequency_few_times_week_cancels_queued_message_on_non_checkin_day(self):
        """Tests that a queued message on Tuesday is cancelled if user switched to few_times_week."""
        self.pref.frequency = CompanionPreference.FrequencyChoice.FEW_TIMES_WEEK
        self.pref.save()

        # Tuesday, Oct 20, 2026 at 09:00 IST = 03:30 UTC
        tue_dt_utc = datetime.datetime(2026, 10, 20, 3, 30, tzinfo=timezone.utc)

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=tue_dt_utc,
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Casual check"
        )

        worker = ScheduledMessageWorker()
        with patch('django.utils.timezone.now', return_value=tue_dt_utc + timedelta(minutes=1)):
            result = worker.process_due_messages()

        self.assertEqual(result['cancelled'], 1)
        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.CANCELLED)
        self.assertIn("few_times_week", msg.error_message)

    def test_gemini_generation_success_with_mock_client(self):
        """Tests CompanionAIGenerator with a mocked Gemini client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock(text="Morning da! Ready for today? Have breakfast tto! ☀️", thought=False)
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Morning da! Ready for today? Have breakfast tto! ☀️"
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=12, total_token_count=32)
        mock_client.models.generate_content.return_value = mock_response

        generator = CompanionAIGenerator(client=mock_client, provider='gemini')
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now()
        )

        content, meta = generator.generate(msg)
        self.assertEqual(content, "Morning da! Ready for today? Have breakfast tto! ☀️")
        self.assertEqual(meta['provider'], 'gemini')
        self.assertEqual(meta['total_tokens'], 32)
        mock_client.models.generate_content.assert_called_once()

    def test_gemini_generation_transient_retry_success(self):
        """Tests that a transient 429 error on Gemini is retried and succeeds."""
        mock_client = MagicMock()
        mock_response = MagicMock(text="Good evening da! Relaxing now? 😌", candidates=[], usage_metadata=None)

        # First call raises transient 429, second call succeeds
        mock_client.models.generate_content.side_effect = [
            Exception("429 ResourceExhausted: rate limit exceeded, retry in 1.0s"),
            mock_response
        ]

        generator = CompanionAIGenerator(client=mock_client, provider='gemini')
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.EVENING_CHECKIN,
            scheduled_time=timezone.now()
        )

        with patch('time.sleep', return_value=None):
            content, meta = generator.generate(msg)

        self.assertEqual(content, "Good evening da! Relaxing now? 😌")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_in_app_delivery_persists_to_active_conversation_chat_message(self):
        """Tests that InAppDeliveryHandler creates both InAppMessage and ChatMessage in the active conversation."""
        from chat.models import Conversation, ChatMessage
        from proactive_messaging.services.delivery.in_app import InAppDeliveryHandler

        conv = Conversation.objects.create(user=self.user, title="Primary Chat Thread")

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now() - timedelta(minutes=2),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Morning da! Tea kudicho?"
        )

        handler = InAppDeliveryHandler()
        result = handler.deliver(msg, msg.content)

        self.assertTrue(result['delivered'])
        self.assertIsNotNone(result['chat_message_id'])
        self.assertEqual(result['conversation_id'], str(conv.id))

        # Check InAppMessage
        in_app = InAppMessage.objects.get(id=result['in_app_message_id'])
        self.assertEqual(in_app.user, self.user)
        self.assertEqual(in_app.content, "Morning da! Tea kudicho?")

        # Check ChatMessage in conversation
        chat_msg = ChatMessage.objects.get(id=result['chat_message_id'])
        self.assertEqual(chat_msg.conversation, conv)
        self.assertEqual(chat_msg.sender, ChatMessage.Sender.COMPANION)
        self.assertEqual(chat_msg.content, "Morning da! Tea kudicho?")
        self.assertTrue(chat_msg.metadata.get('is_proactive'))
        self.assertEqual(chat_msg.metadata.get('scheduled_message_id'), str(msg.id))

    def test_in_app_delivery_idempotent_no_duplicate_chat_messages(self):
        """Tests that repeated delivery calls for the same scheduled message produce exactly one ChatMessage."""
        from chat.models import Conversation, ChatMessage
        from proactive_messaging.services.delivery.in_app import InAppDeliveryHandler

        conv = Conversation.objects.create(user=self.user, title="Active Thread")

        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=timezone.now() - timedelta(minutes=1),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Work-il busy aano da?"
        )

        handler = InAppDeliveryHandler()
        res1 = handler.deliver(msg, msg.content)
        res2 = handler.deliver(msg, msg.content)

        self.assertEqual(res1['chat_message_id'], res2['chat_message_id'])
        chat_msg_count = ChatMessage.objects.filter(
            conversation=conv,
            metadata__scheduled_message_id=str(msg.id)
        ).count()
        self.assertEqual(chat_msg_count, 1)

    def test_in_app_delivery_user_isolation(self):
        """Tests that user A's proactive delivery never writes to user B's conversation."""
        from chat.models import Conversation, ChatMessage
        from proactive_messaging.services.delivery.in_app import InAppDeliveryHandler

        conv_a = Conversation.objects.create(user=self.user, title="User A Thread")
        conv_b = Conversation.objects.create(user=self.user_other, title="User B Thread")

        msg_a = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.AFFIRMATION,
            scheduled_time=timezone.now() - timedelta(minutes=2),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="You are doing great da!"
        )

        handler = InAppDeliveryHandler()
        result = handler.deliver(msg_a, msg_a.content)

        # Message must appear in User A's conversation only
        self.assertEqual(result['conversation_id'], str(conv_a.id))
        self.assertEqual(conv_a.messages.count(), 1)
        self.assertEqual(conv_b.messages.count(), 0)

    def test_end_to_end_gemini_worker_generation_to_chat_message(self):
        """
        End-to-end integration test:
        1. User with active conversation has due pending AI-generated checkin.
        2. Worker claims message atomically and generates text with Gemini (mocked).
        3. Delivers via in-app handler creating ChatMessage in PostgreSQL.
        4. Chat API GET /api/chat/conversations/<id>/ returns the proactive message.
        5. Running worker again does not duplicate the message.
        """
        from chat.models import Conversation, ChatMessage
        from proactive_messaging.services.worker import ScheduledMessageWorker
        from proactive_messaging.services.ai_generator import CompanionAIGenerator

        self.pref.frequency = CompanionPreference.FrequencyChoice.DAILY
        self.pref.quiet_hours_enabled = False
        self.pref.save()

        conv = Conversation.objects.create(user=self.user, title="Ami Daily Chat")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="Night da!")

        due_time = timezone.now() - timedelta(minutes=1)
        msg = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=due_time,
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            is_ai_generated=True,
            idempotency_key=f"e2e_daily_{self.user.id}_morning"
        )

        mock_client = MagicMock()
        mock_response = MagicMock(text="Morning da! Ready for the day? ☀️", candidates=[], usage_metadata=None)
        mock_client.models.generate_content.return_value = mock_response

        generator = CompanionAIGenerator(client=mock_client, provider='gemini')
        worker = ScheduledMessageWorker(ai_generator=generator)

        # 1. Process due messages
        results = worker.process_due_messages()
        self.assertEqual(results['delivered'], 1)
        self.assertEqual(results['total_due'], 1)

        msg.refresh_from_db()
        self.assertEqual(msg.status, ScheduledMessage.DeliveryStatus.DELIVERED)
        self.assertEqual(msg.content, "Morning da! Ready for the day? ☀️")
        self.assertTrue(msg.delivered_in_app)

        # 2. Verify ChatMessage in PostgreSQL
        chat_msg = ChatMessage.objects.filter(conversation=conv, metadata__scheduled_message_id=str(msg.id)).first()
        self.assertIsNotNone(chat_msg)
        self.assertEqual(chat_msg.sender, ChatMessage.Sender.COMPANION)
        self.assertEqual(chat_msg.content, "Morning da! Ready for the day? ☀️")
        self.assertTrue(chat_msg.metadata.get('is_proactive'))

        # 3. Verify Chat API GET /api/conversations/<id>/ delivers it to UI
        res = self.client.get(f"/api/conversations/{conv.id}/")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv_payload = res.json()
        messages_list = conv_payload.get('messages', [])
        self.assertEqual(len(messages_list), 2)
        companion_texts = [m['content'] for m in messages_list if m['sender'] == 'companion']
        self.assertIn("Morning da! Ready for the day? ☀️", companion_texts)

        # 4. Idempotency verification: running worker again produces 0 new deliveries
        second_run = worker.process_due_messages()
        self.assertEqual(second_run['total_due'], 0)
        self.assertEqual(ChatMessage.objects.filter(conversation=conv).count(), 2)

    def test_end_to_end_frequency_and_cooldown_holding(self):
        """Verify that worker postpones messages when daily limit or cooldown is violated."""
        from proactive_messaging.services.worker import ScheduledMessageWorker
        self.pref.frequency = CompanionPreference.FrequencyChoice.DAILY
        self.pref.max_daily_messages = 1
        self.pref.quiet_hours_enabled = False
        self.pref.save()

        # Mark 1 message as already delivered today
        ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.GOOD_MORNING,
            scheduled_time=timezone.now() - timedelta(hours=2),
            delivered_at=timezone.now() - timedelta(hours=2),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.DELIVERED,
            content="Earlier check-in"
        )

        # Another message is due now
        msg2 = ScheduledMessage.objects.create(
            user=self.user,
            message_type=ScheduledMessage.MessageType.CASUAL_CHAT,
            scheduled_time=timezone.now() - timedelta(minutes=1),
            channel=ScheduledMessage.DeliveryChannel.IN_APP,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            content="Second check-in"
        )

        worker = ScheduledMessageWorker()
        results = worker.process_due_messages()

        self.assertEqual(results['postponed'], 1)
        self.assertEqual(results['delivered'], 0)

        msg2.refresh_from_db()
        self.assertEqual(msg2.status, ScheduledMessage.DeliveryStatus.PENDING)
        self.assertIn("Daily limit of 1 proactive messages reached", msg2.error_message)
        self.assertGreater(msg2.scheduled_time, timezone.now())





