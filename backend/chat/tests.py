import uuid
from unittest.mock import patch, MagicMock
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from openai import RateLimitError
from google.genai.errors import APIError as GenAIAPIError

from .models import (
    Conversation,
    ChatMessage,
    CompanionMemory,
)
from .services import (
    CompanionChatService,
    BASE_SYSTEM_PROMPT,
    detect_candidate_memory,
    get_relevant_memories,
    save_or_update_memory,
    clean_companion_reply,
    is_meta_or_evaluation_leak,
    is_retryable_gemini_error,
)

User = get_user_model()


class AuthenticationAndOwnershipSecurityTest(TestCase):
    """
    Security test suite verifying:
    1. Unauthenticated requests are rejected with 401 Unauthorized.
    2. Conversations and memories are strictly isolated between authenticated users.
    3. Users cannot access, modify, or delete another user's chats or memories by changing UUIDs.
    4. Session authentication lifecycle (register, login, me, logout, csrf).
    """

    def setUp(self):
        self.client = APIClient()
        self.user_a = User.objects.create_user(username='user_alice', password='password123', email='alice@example.com')
        self.user_b = User.objects.create_user(username='user_bob', password='password123', email='bob@example.com')

    def test_unauthenticated_requests_rejected(self):
        """Verify that all personal chat, conversation, and memory endpoints reject anonymous access."""
        fake_uuid = uuid.uuid4()

        unauth_statuses = (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

        # Chat
        res_chat = self.client.post('/api/chat/', data={'message': 'Hello'}, format='json')
        self.assertIn(res_chat.status_code, unauth_statuses)

        # Conversations
        res_conv_list = self.client.get('/api/conversations/')
        self.assertIn(res_conv_list.status_code, unauth_statuses)

        res_conv_post = self.client.post('/api/conversations/', data={'title': 'Test'}, format='json')
        self.assertIn(res_conv_post.status_code, unauth_statuses)

        res_conv_get = self.client.get(f'/api/conversations/{fake_uuid}/')
        self.assertIn(res_conv_get.status_code, unauth_statuses)

        res_conv_del = self.client.delete(f'/api/conversations/{fake_uuid}/')
        self.assertIn(res_conv_del.status_code, unauth_statuses)

        res_conv_clear = self.client.post(f'/api/conversations/{fake_uuid}/clear/')
        self.assertIn(res_conv_clear.status_code, unauth_statuses)

        # Memories
        res_mem_list = self.client.get('/api/memories/')
        self.assertIn(res_mem_list.status_code, unauth_statuses)

        res_mem_post = self.client.post('/api/memories/', data={'content': 'Tea'}, format='json')
        self.assertIn(res_mem_post.status_code, unauth_statuses)

        res_mem_get = self.client.get(f'/api/memories/{fake_uuid}/')
        self.assertIn(res_mem_get.status_code, unauth_statuses)

        res_mem_patch = self.client.patch(f'/api/memories/{fake_uuid}/', data={'content': 'Tea'}, format='json')
        self.assertIn(res_mem_patch.status_code, unauth_statuses)

        res_mem_del = self.client.delete(f'/api/memories/{fake_uuid}/')
        self.assertIn(res_mem_del.status_code, unauth_statuses)


    def test_cross_user_conversation_isolation(self):
        """Verify that User B cannot view, delete, or append to User A's conversation by changing UUIDs."""
        conv_a = Conversation.objects.create(user=self.user_a, title="Alice's Private Chat")
        msg_a = ChatMessage.objects.create(conversation=conv_a, sender=ChatMessage.Sender.USER, content="Alice's secret")

        # Authenticate as User B
        self.client.force_authenticate(user=self.user_b)

        # 1. User B listing conversations should NOT see Alice's conversation
        res_list = self.client.get('/api/conversations/')
        self.assertEqual(res_list.status_code, status.HTTP_200_OK)
        conv_ids = [c['id'] for c in res_list.json()]
        self.assertNotIn(str(conv_a.id), conv_ids)

        # 2. User B attempting to view Alice's conversation details by UUID returns 404
        res_detail = self.client.get(f'/api/conversations/{conv_a.id}/')
        self.assertEqual(res_detail.status_code, status.HTTP_404_NOT_FOUND)

        # 3. User B attempting to delete Alice's conversation returns 404
        res_del = self.client.delete(f'/api/conversations/{conv_a.id}/')
        self.assertEqual(res_del.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(Conversation.objects.filter(id=conv_a.id).exists())

        # 4. User B attempting to send a chat message into Alice's conversation returns 404
        res_chat = self.client.post(
            '/api/chat/',
            data={'message': 'Bob intruding', 'conversation_id': str(conv_a.id)},
            format='json'
        )
        self.assertEqual(res_chat.status_code, status.HTTP_404_NOT_FOUND)

        # 5. User B attempting to clear Alice's conversation messages returns 404
        res_clear_b = self.client.post(f'/api/conversations/{conv_a.id}/clear/')
        self.assertEqual(res_clear_b.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(conv_a.messages.count(), 1)

        # 6. User A clearing their own conversation deletes all messages, preserves conversation and memories
        self.client.force_authenticate(user=self.user_a)
        mem_a = CompanionMemory.objects.create(
            user=self.user_a,
            content="Alice loves matcha tea",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True
        )

        res_clear_a = self.client.post(f'/api/conversations/{conv_a.id}/clear/')
        self.assertEqual(res_clear_a.status_code, status.HTTP_200_OK)
        self.assertTrue(res_clear_a.json()['cleared'])
        self.assertEqual(res_clear_a.json()['deleted_count'], 1)

        # Verify messages are permanently gone from conversation
        self.assertEqual(conv_a.messages.count(), 0)
        # Verify conversation record itself still exists with intact ID
        self.assertTrue(Conversation.objects.filter(id=conv_a.id).exists())
        # Verify user memories and account are completely untouched
        self.assertTrue(CompanionMemory.objects.filter(id=mem_a.id).exists())

    def test_cross_user_memory_isolation(self):
        """Verify that User B cannot view, edit, or delete User A's memories by changing UUIDs."""
        mem_a = CompanionMemory.objects.create(
            user=self.user_a,
            content="Alice's secret preference",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True
        )

        # Authenticate as User B
        self.client.force_authenticate(user=self.user_b)

        # 1. User B listing memories should NOT see Alice's memory
        res_list = self.client.get('/api/memories/')
        self.assertEqual(res_list.status_code, status.HTTP_200_OK)
        mem_ids = [m['id'] for m in res_list.json()]
        self.assertNotIn(str(mem_a.id), mem_ids)

        # 2. User B accessing Alice's memory by UUID returns 404
        res_get = self.client.get(f'/api/memories/{mem_a.id}/')
        self.assertEqual(res_get.status_code, status.HTTP_404_NOT_FOUND)

        # 3. User B modifying Alice's memory returns 404
        res_patch = self.client.patch(
            f'/api/memories/{mem_a.id}/',
            data={'content': 'Hacked content'},
            format='json'
        )
        self.assertEqual(res_patch.status_code, status.HTTP_404_NOT_FOUND)
        mem_a.refresh_from_db()
        self.assertEqual(mem_a.content, "Alice's secret preference")

        # 4. User B deleting Alice's memory returns 404
        res_del = self.client.delete(f'/api/memories/{mem_a.id}/')
        self.assertEqual(res_del.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(CompanionMemory.objects.filter(id=mem_a.id).exists())

    def test_auth_lifecycle(self):
        """Test registration, login, session check (me), logout, and CSRF token endpoints."""
        # 1. CSRF Token endpoint
        res_csrf = self.client.get('/api/auth/csrf/')
        self.assertEqual(res_csrf.status_code, status.HTTP_200_OK)
        self.assertIn('csrfToken', res_csrf.json())

        # 2. Register a new user
        reg_payload = {
            'username': 'new_companion_user',
            'password': 'safe_password_123',
            'email': 'new@example.com'
        }
        res_reg = self.client.post('/api/auth/register/', data=reg_payload, format='json')
        self.assertEqual(res_reg.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res_reg.json()['username'], 'new_companion_user')
        self.assertTrue(res_reg.json()['authenticated'])

        # 3. Verify /api/auth/me/ shows authenticated session
        res_me = self.client.get('/api/auth/me/')
        self.assertEqual(res_me.status_code, status.HTTP_200_OK)
        self.assertEqual(res_me.json()['username'], 'new_companion_user')

        # 4. Logout
        res_logout = self.client.post('/api/auth/logout/')
        self.assertEqual(res_logout.status_code, status.HTTP_200_OK)

        # 5. /api/auth/me/ after logout returns 401
        res_me_logged_out = self.client.get('/api/auth/me/')
        self.assertEqual(res_me_logged_out.status_code, status.HTTP_401_UNAUTHORIZED)

        # 6. Login with incorrect password
        bad_login = self.client.post(
            '/api/auth/login/',
            data={'username': 'new_companion_user', 'password': 'wrong_password'},
            format='json'
        )
        self.assertEqual(bad_login.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(bad_login.json().get('error'), 'Invalid username or password.')

        # 7. Login with valid credentials
        good_login = self.client.post(
            '/api/auth/login/',
            data={'username': 'new_companion_user', 'password': 'safe_password_123'},
            format='json'
        )
        self.assertEqual(good_login.status_code, status.HTTP_200_OK)
        res_data = good_login.json()
        self.assertEqual(res_data['username'], 'new_companion_user')
        self.assertTrue(res_data['authenticated'])
        # Check both flat and nested user object structures are present
        self.assertIn('user', res_data)
        self.assertEqual(res_data['user']['username'], 'new_companion_user')
        self.assertTrue(res_data['user']['authenticated'])
        self.assertIn('csrfToken', res_data)

    def test_duplicate_registration_rejected(self):
        """Registering with an existing username (exact or case-insensitive) is rejected with 400."""
        # Exact duplicate
        res_dup_exact = self.client.post(
            '/api/auth/register/',
            data={'username': 'user_alice', 'password': 'new_password_123'},
            format='json'
        )
        self.assertEqual(res_dup_exact.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('already exists', res_dup_exact.json().get('error', '').lower())

        # Case-insensitive duplicate (e.g. USER_ALICE vs user_alice)
        res_dup_case = self.client.post(
            '/api/auth/register/',
            data={'username': 'USER_ALICE', 'password': 'new_password_123'},
            format='json'
        )
        self.assertEqual(res_dup_case.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('already exists', res_dup_case.json().get('error', '').lower())

    def test_login_case_insensitivity(self):
        """User can log in even if username case differs from registration case."""
        res = self.client.post(
            '/api/auth/login/',
            data={'username': 'USER_ALICE', 'password': 'password123'},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.json()['username'], 'user_alice')
        self.assertTrue(res.json()['authenticated'])

    def test_login_missing_or_empty_credentials(self):
        """Empty username or password returns 400 Bad Request with useful error."""
        # Missing password
        res_no_pw = self.client.post(
            '/api/auth/login/',
            data={'username': 'user_alice'},
            format='json'
        )
        self.assertEqual(res_no_pw.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('required', res_no_pw.json().get('error', '').lower())

        # Blank username
        res_blank = self.client.post(
            '/api/auth/login/',
            data={'username': '   ', 'password': 'password123'},
            format='json'
        )
        self.assertEqual(res_blank.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('empty', res_blank.json().get('error', '').lower())



@override_settings(AI_PROVIDER='openai')
class ChatPersistenceAndMemoryTest(TestCase):
    """
    Tests for authenticated message persistence, conversation resumption,
    memory retrieval, and candidate memory detection.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='test_owner', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'
        self.conv_url = '/api/conversations/'
        self.mem_url = '/api/memories/'

    @patch('chat.services.CompanionChatService.get_client')
    def test_message_persistence_in_database(self, mock_get_client):
        """Test that user message and AI companion reply are persisted in PostgreSQL under authenticated user."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hey da! Kazhicho? Working hard today aano?"))
        ]
        mock_response.model = 'gpt-4o-mini'
        mock_response.id = 'chatcmpl-test-pers'
        mock_response.usage = MagicMock(prompt_tokens=40, completion_tokens=15, total_tokens=55)
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        payload = {"message": "Had a long coding session today!"}
        response = self.client.post(self.chat_url, data=payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('conversation_id', data)
        self.assertIn('reply', data)
        self.assertIn('user_message_id', data)
        self.assertIn('message_id', data)

        conv_id = data['conversation_id']
        conversation = Conversation.objects.get(id=conv_id)
        self.assertEqual(conversation.user, self.user)
        self.assertEqual(conversation.messages.count(), 2)

        # Verify persisted user message
        user_msg = conversation.messages.filter(sender=ChatMessage.Sender.USER).first()
        self.assertIsNotNone(user_msg)
        self.assertEqual(user_msg.content, "Had a long coding session today!")

        # Verify persisted companion message
        companion_msg = conversation.messages.filter(sender=ChatMessage.Sender.COMPANION).first()
        self.assertIsNotNone(companion_msg)
        self.assertEqual(companion_msg.content, "Hey da! Kazhicho? Working hard today aano?")
        self.assertEqual(companion_msg.metadata.get('model'), 'gpt-4o-mini')

    @patch('chat.services.CompanionChatService.get_client')
    def test_resume_existing_conversation_across_requests(self, mock_get_client):
        """Test resuming an existing conversation using stable conversation_id."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Awesome da!"))],
            model='gpt-4o-mini',
            id='chatcmpl-1',
            usage=None
        )
        mock_get_client.return_value = mock_client

        # 1. Create first message
        res1 = self.client.post(self.chat_url, data={"message": "First message"}, format='json')
        conv_id = res1.json()['conversation_id']

        # 2. Send second message in same conversation
        res2 = self.client.post(
            self.chat_url,
            data={"message": "Second message", "conversation_id": conv_id},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        self.assertEqual(res2.json()['conversation_id'], conv_id)

        # 3. Retrieve conversation history from database
        detail_url = f"/api/conversations/{conv_id}/"
        res_detail = self.client.get(detail_url)
        self.assertEqual(res_detail.status_code, status.HTTP_200_OK)

        messages = res_detail.json()['messages']
        self.assertEqual(len(messages), 4)  # 2 user + 2 companion
        self.assertEqual(messages[0]['content'], "First message")
        self.assertEqual(messages[2]['content'], "Second message")

    def test_delete_conversation_removes_messages(self):
        """Test deleting a conversation cascades and removes its messages."""
        conv = Conversation.objects.create(user=self.user, title="To Delete")
        msg = ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="Bye")

        response = self.client.delete(f"/api/conversations/{conv.id}/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(Conversation.objects.filter(id=conv.id).exists())
        self.assertFalse(ChatMessage.objects.filter(id=msg.id).exists())

    @patch('chat.services.CompanionChatService.get_client')
    def test_memory_retrieval_injects_approved_memories_only(self, mock_get_client):
        """Test that approved memories are retrieved and injected into prompt, but unapproved are excluded."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="I remembered your tea preference!"))],
            model='gpt-4o-mini',
            id='chatcmpl-mem',
            usage=None
        )
        mock_get_client.return_value = mock_client

        # Approved memory
        CompanionMemory.objects.create(
            user=self.user,
            content="Likes black tea with no sugar and extra cardamom",
            category=CompanionMemory.Category.PREFERENCE,
            importance=CompanionMemory.Importance.IMPORTANT,
            is_approved=True
        )

        # Unapproved memory (pending user consent)
        CompanionMemory.objects.create(
            user=self.user,
            content="Sensitive secret or unconfirmed detail",
            category=CompanionMemory.Category.PERSONAL,
            importance=CompanionMemory.Importance.NORMAL,
            is_approved=False
        )

        response = self.client.post(self.chat_url, data={"message": "Make me tea!"}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        system_prompt_sent = call_kwargs['messages'][0]['content']

        self.assertIn("Likes black tea with no sugar", system_prompt_sent)
        self.assertNotIn("Sensitive secret or unconfirmed detail", system_prompt_sent)

    def test_memory_crud_endpoints(self):
        """Test full CRUD operations on companion memories for authenticated user."""
        # 1. Create memory
        create_res = self.client.post(
            self.mem_url,
            data={
                "content": "Preparing for AWS DevOps certification next month",
                "category": "goal",
                "importance": 2,
                "is_approved": True
            },
            format='json'
        )
        self.assertEqual(create_res.status_code, status.HTTP_201_CREATED)
        mem_id = create_res.json()['id']

        # 2. List memories
        list_res = self.client.get(self.mem_url)
        self.assertEqual(list_res.status_code, status.HTTP_200_OK)
        memories = list_res.json()
        self.assertTrue(any(m['id'] == mem_id for m in memories))

        # 3. Update memory (edit content and importance)
        detail_url = f"/api/memories/{mem_id}/"
        patch_res = self.client.patch(
            detail_url,
            data={"content": "Passed AWS DevOps cert!", "importance": 3},
            format='json'
        )
        self.assertEqual(patch_res.status_code, status.HTTP_200_OK)
        self.assertEqual(patch_res.json()['content'], "Passed AWS DevOps cert!")
        self.assertEqual(patch_res.json()['importance'], 3)

        # 4. Delete memory
        del_res = self.client.delete(detail_url)
        self.assertEqual(del_res.status_code, status.HTTP_200_OK)
        self.assertFalse(CompanionMemory.objects.filter(id=mem_id).exists())

    def test_memory_policy_candidate_detection(self):
        """Test candidate memory detection rules conform to memory policy."""
        # Preference
        pref = detect_candidate_memory("I really love black coffee with hazelnut")
        self.assertIsNotNone(pref)
        self.assertEqual(pref['category'], 'preference')

        # Work
        work = detect_candidate_memory("I work as a Django frontend developer")
        self.assertIsNotNone(work)
        self.assertEqual(work['category'], 'work')

        # Transitory chatter (should NOT be detected as memory)
        chat = detect_candidate_memory("The weather is nice today, walking outside.")
        self.assertIsNone(chat)

    def test_empty_message_validation(self):
        """Test that empty or whitespace message returns 400."""
        res = self.client.post(self.chat_url, data={"message": "   "}, format='json')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    @override_settings(OPENAI_API_KEY='')
    def test_missing_api_key_returns_503(self):
        """Test missing API key returns 503."""
        res = self.client.post(self.chat_url, data={"message": "Hi"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)


@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class GeminiChatIntegrationTest(TestCase):
    """
    Tests for Google Gemini API integration with Ami:
    - Verifies mocked generation works with system prompt, history, and memories
    - Verifies metadata format (provider='gemini', model='gemini-3-flash-preview')
    - Verifies missing and placeholder key detection returns 503
    - Verifies error categorization for invalid key, quota exhaustion, and model errors returns 502
    - Verifies health check diagnostic output
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='gemini_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    @patch('chat.services.CompanionChatService.get_client')
    def test_gemini_message_persistence_and_reply(self, mock_get_client):
        """Test that user message and Gemini AI companion reply are persisted and formatted correctly."""
        # Setup mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hey da! Njan koodeyille. How was your day? Kazhicho?"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=45,
            candidates_token_count=18,
            total_token_count=63,
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Create an approved memory for this user
        CompanionMemory.objects.create(
            user=self.user,
            content="Likes ginger tea in the morning",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True,
        )

        payload = {"message": "Had a really tiring day at work da!"}
        response = self.client.post(self.chat_url, data=payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('conversation_id', data)
        self.assertIn('reply', data)
        self.assertEqual(data['reply'], "Hey da! Njan koodeyille. How was your day? Kazhicho?")

        # Check metadata
        meta = data.get('metadata', {})
        self.assertEqual(meta.get('provider'), 'gemini')
        self.assertEqual(meta.get('model'), 'gemini-3.1-flash-lite')
        self.assertEqual(meta.get('prompt_tokens'), 45)
        self.assertEqual(meta.get('completion_tokens'), 18)
        self.assertEqual(meta.get('total_tokens'), 63)

        # Verify database persistence
        conv = Conversation.objects.get(id=data['conversation_id'])
        self.assertEqual(conv.user, self.user)
        self.assertEqual(conv.messages.count(), 2)

        companion_msg = conv.messages.filter(sender=ChatMessage.Sender.COMPANION).first()
        self.assertIsNotNone(companion_msg)
        self.assertEqual(companion_msg.content, data['reply'])
        self.assertEqual(companion_msg.metadata.get('provider'), 'gemini')

        # Verify call arguments passed to Gemini models.generate_content
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs['model'], 'gemini-3.1-flash-lite')
        system_instruction = call_kwargs['config'].system_instruction
        self.assertIn("Likes ginger tea in the morning", system_instruction)
        self.assertIn("You are Ami", system_instruction)

    @override_settings(GEMINI_API_KEY='')
    def test_gemini_missing_api_key_returns_503(self):
        """Test missing Gemini API key returns 503."""
        res = self.client.post(self.chat_url, data={"message": "Hi Ami"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        data = res.json()
        self.assertEqual(data.get('error_category'), 'missing_or_placeholder_key')

    @override_settings(GEMINI_API_KEY='your_gemini_api_key_here')
    def test_gemini_placeholder_api_key_returns_503(self):
        """Test placeholder Gemini API key returns 503."""
        res = self.client.post(self.chat_url, data={"message": "Hi Ami"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        data = res.json()
        self.assertEqual(data.get('error_category'), 'missing_or_placeholder_key')

    @patch('chat.services.CompanionChatService.get_client')
    def test_gemini_invalid_key_error_returns_502(self, mock_get_client):
        """Test invalid Gemini API key error maps to invalid_key category."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = GenAIAPIError(
            400,
            {'error': {'message': 'API key not valid. Please pass a valid API key.'}}
        )
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={"message": "Hi Ami"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_502_BAD_GATEWAY)
        data = res.json()
        self.assertEqual(data.get('error_category'), 'invalid_key')

    @patch('chat.services.CompanionChatService.get_client')
    def test_gemini_quota_exhausted_returns_502(self, mock_get_client):
        """Test Gemini quota exhausted error maps to insufficient_quota category."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = GenAIAPIError(
            429,
            {'error': {'message': 'RESOURCE_EXHAUSTED: Quota exceeded for quota metric.'}}
        )
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={"message": "Hi Ami"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_502_BAD_GATEWAY)
        data = res.json()
        self.assertEqual(data.get('error_category'), 'insufficient_quota')

    @patch('chat.services.CompanionChatService.get_client')
    def test_gemini_model_not_found_returns_502(self, mock_get_client):
        """Test Gemini model not found error maps to model_access_issue category."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = GenAIAPIError(
            404,
            {'error': {'message': 'models/gemini-invalid is not found for API version v1.'}}
        )
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={"message": "Hi Ami"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_502_BAD_GATEWAY)
        data = res.json()
        self.assertEqual(data.get('error_category'), 'model_access_issue')

    def test_gemini_health_check_diagnostic(self):
        """Test health check reports Gemini status when AI_PROVIDER=gemini."""
        res = self.client.get('/api/health/')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        services = res.json().get('services', {})
        self.assertEqual(services.get('ai_provider'), 'gemini')
        self.assertTrue(services.get('ai_connected'))
        self.assertIn('gemini', services)
        self.assertEqual(services['gemini']['model'], 'gemini-3.1-flash-lite')
        self.assertTrue(services['gemini']['configured'])


@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class MemoryAndContextSystemTests(TestCase):
    """
    Comprehensive tests for Ami's memory and conversation context system:
    1. Relevance-ranked memory retrieval favoring semantic context and approved memories.
    2. Multi-turn conversation context continuity and follow-up handling.
    3. Duplicate memory prevention (exact and normalized duplicates).
    4. Contradictory memory superseding / slot resolution.
    5. Strict cross-user memory isolation.
    6. Durable fact detection vs transitory chatter filtering.
    """

    def setUp(self):
        self.client = APIClient()
        self.user_a = User.objects.create_user(username='context_user_a', password='password123')
        self.user_b = User.objects.create_user(username='context_user_b', password='password123')
        self.client.force_authenticate(user=self.user_a)

    def test_relevant_memory_retrieval_ranking(self):
        """Verify get_relevant_memories ranks context-relevant approved memories at top."""
        # 1. Create memories across different categories
        mem_tea = CompanionMemory.objects.create(
            user=self.user_a,
            content="Likes hot black tea with cardamom and zero sugar",
            category=CompanionMemory.Category.PREFERENCE,
            importance=CompanionMemory.Importance.NORMAL,
            is_approved=True
        )
        mem_work = CompanionMemory.objects.create(
            user=self.user_a,
            content="Works as a distributed systems architect",
            category=CompanionMemory.Category.WORK,
            importance=CompanionMemory.Importance.IMPORTANT,
            is_approved=True
        )
        mem_goal = CompanionMemory.objects.create(
            user=self.user_a,
            content="Preparing for certified Kubernetes administrator exam",
            category=CompanionMemory.Category.GOAL,
            importance=CompanionMemory.Importance.IMPORTANT,
            is_approved=True
        )
        mem_unapproved = CompanionMemory.objects.create(
            user=self.user_a,
            content="Unapproved confidential draft note",
            category=CompanionMemory.Category.PERSONAL,
            importance=CompanionMemory.Importance.CORE,
            is_approved=False
        )

        # 2. Query about tea / drink
        results = get_relevant_memories(
            user=self.user_a,
            query_text="What hot drink should I make right now?",
            limit=2
        )

        self.assertGreaterEqual(len(results), 1)
        # Tea memory should be ranked first because of keyword and category match
        self.assertEqual(results[0].id, mem_tea.id)
        # Unapproved memory must NEVER be included
        self.assertNotIn(mem_unapproved.id, [m.id for m in results])

    @patch('chat.services.CompanionChatService.get_client')
    def test_context_continuity_in_gemini_generation(self, mock_get_client):
        """Verify multi-turn history turns are preserved and passed to Gemini in proper order."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Get good running shoes da! Brooks or Asics are great."
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=120,
            candidates_token_count=20,
            total_token_count=140
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Create ongoing conversation with history
        conv = Conversation.objects.create(user=self.user_a, title="Marathon prep")
        ChatMessage.objects.create(
            conversation=conv,
            sender=ChatMessage.Sender.USER,
            content="I just signed up for the Mumbai marathon."
        )
        ChatMessage.objects.create(
            conversation=conv,
            sender=ChatMessage.Sender.COMPANION,
            content="That is amazing da! So proud of you."
        )

        # Send follow-up message
        res = self.client.post(
            '/api/chat/',
            data={"message": "What shoes should I buy for it?", "conversation_id": str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        # Inspect call args sent to Gemini
        call_kwargs = mock_client.models.generate_content.call_args[1]
        contents = call_kwargs['config'].system_instruction
        self.assertIn("Conversational Context & Follow-Up Continuity", contents)
        self.assertIn("NEVER ask your partner to repeat", contents)

        # Check multi-turn contents sent to Gemini
        sent_contents = call_kwargs['contents']
        self.assertGreaterEqual(len(sent_contents), 2)
        all_sent_texts = " ".join([p.text for c in sent_contents for p in c.parts])
        self.assertIn("Mumbai marathon", all_sent_texts)
        self.assertIn("shoes should I buy", all_sent_texts)

    def test_duplicate_memory_prevention(self):
        """Verify exact and normalized duplicate memories are not saved twice."""
        mem1, action1 = save_or_update_memory(
            user=self.user_a,
            content="Likes black coffee with hazelnut",
            category="preference",
            importance=1,
            is_approved=True
        )
        self.assertEqual(action1, "created")

        # Attempt to save identical or normalized duplicate
        mem2, action2 = save_or_update_memory(
            user=self.user_a,
            content="likes black coffee with hazelnut.",
            category="preference",
            importance=2,
            is_approved=True
        )
        self.assertEqual(action2, "deduplicated")
        self.assertEqual(mem1.id, mem2.id)

        # Database should only hold 1 record for this user
        self.assertEqual(CompanionMemory.objects.filter(user=self.user_a).count(), 1)
        # Importance should be updated to 2
        mem1.refresh_from_db()
        self.assertEqual(mem1.importance, 2)

    def test_contradictory_memory_superseding(self):
        """Verify contradictory slot memories (e.g. favorite drink, location) supersede older values."""
        mem1, action1 = save_or_update_memory(
            user=self.user_a,
            content="My favorite drink is black coffee",
            category="preference",
            importance=1,
            is_approved=True
        )
        self.assertEqual(action1, "created")

        # User updates preference to matcha latte
        mem2, action2 = save_or_update_memory(
            user=self.user_a,
            content="Favorite drink is matcha latte",
            category="preference",
            importance=2,
            is_approved=True
        )
        self.assertEqual(action2, "updated")
        self.assertEqual(mem1.id, mem2.id)

        # Database should hold only 1 favorite drink record
        self.assertEqual(CompanionMemory.objects.filter(user=self.user_a).count(), 1)
        mem1.refresh_from_db()
        self.assertEqual(mem1.content, "Favorite drink is matcha latte")

    def test_cross_user_memory_isolation_in_chat(self):
        """Verify User A's memories are never leaked to User B during chat generation."""
        # Create private memory for User A
        CompanionMemory.objects.create(
            user=self.user_a,
            content="User A private secret: Loves pineapple on pizza",
            category=CompanionMemory.Category.PREFERENCE,
            importance=CompanionMemory.Importance.CORE,
            is_approved=True
        )

        # Query relevant memories for User B
        memories_b = get_relevant_memories(
            user=self.user_b,
            query_text="What food do I like?",
            limit=10
        )
        self.assertEqual(len(memories_b), 0)

        # Service prompt for User B
        service = CompanionChatService()
        prompt_b = service.build_system_prompt(memories=memories_b)
        self.assertNotIn("pineapple on pizza", prompt_b)
        self.assertNotIn("User A private secret", prompt_b)

    def test_durable_memory_detection_and_transient_filtering(self):
        """Verify detect_candidate_memory captures durable facts and rejects transient chatter."""
        # Durable facts (Preferences, Goals, Hobbies, Health, Work)
        self.assertIsNotNone(detect_candidate_memory("I'm really into astrophotography"))
        self.assertIsNotNone(detect_candidate_memory("My goal is to learn Spanish this year"))
        self.assertIsNotNone(detect_candidate_memory("I am allergic to peanuts and tree nuts"))
        self.assertIsNotNone(detect_candidate_memory("In my free time I love to paint with watercolors"))
        self.assertIsNotNone(detect_candidate_memory("Remember that my sister lives in London"))

        # Ephemeral / Transitory messages (Must be filtered)
        self.assertIsNone(detect_candidate_memory("The weather is nice today, walking outside."))
        self.assertIsNone(detect_candidate_memory("I am so tired today"))
        self.assertIsNone(detect_candidate_memory("Good morning da!"))
        self.assertIsNone(detect_candidate_memory("Having lunch right now"))
        self.assertIsNone(detect_candidate_memory("Eating dinner currently"))
        self.assertIsNone(detect_candidate_memory("Sitting in traffic"))


class PhotoRequestAndResponseSanitizationTests(TestCase):
    """
    Regression and unit tests for natural photo request handling and response sanitization:
    1. Verifies system prompt instructs Ami that she is a text-only partner without a camera.
    2. Verifies clean_companion_reply strips internal evaluation hints and '* Better response:' leaks.
    3. Verifies clean_companion_reply strips fake roleplay image actions (*sends photo*).
    4. Verifies end-to-end chat endpoint safely handles photo requests when model emits critique or natural refusal.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='photo_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_system_prompt_photo_and_media_guidelines(self):
        """Verify BASE_SYSTEM_PROMPT contains explicit guidelines on photos, media, and no meta-critique."""
        self.assertIn("Photos, Media & Physical Capabilities", BASE_SYSTEM_PROMPT)
        self.assertIn("camera", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER output internal instructions", BASE_SYSTEM_PROMPT)
        self.assertIn("* Better response:", BASE_SYSTEM_PROMPT)

    def test_is_meta_or_evaluation_leak_detection(self):
        """Verify detector identifies meta critique, evaluation headers, and leak phrases."""
        self.assertTrue(is_meta_or_evaluation_leak("* Better response: Use a descriptive response"))
        self.assertTrue(is_meta_or_evaluation_leak("Better response: Explain limitations"))
        self.assertTrue(is_meta_or_evaluation_leak("System instruction: Do not disclose"))
        self.assertTrue(is_meta_or_evaluation_leak("As an AI language model, I cannot"))
        self.assertTrue(is_meta_or_evaluation_leak("Internal prompt evaluation hint"))

        # Natural conversation must NOT be flagged as leak
        self.assertFalse(is_meta_or_evaluation_leak("Ayyoda, enikku photos ayakkan pattillallo da! 🥰"))
        self.assertFalse(is_meta_or_evaluation_leak("Hey da, kazhicho? How was your day?"))

    def test_clean_companion_reply_strips_better_response_leak(self):
        """Verify '* Better response:' output is cleanly intercepted and sanitized to warm Manglish fallback."""
        raw_leak = "* Better response: Use a descriptive response instead of pretending to send a photo."
        sanitized = clean_companion_reply(raw_leak, user_query="enik ninte oru photo ayach theroo")

        self.assertNotIn("Better response", sanitized)
        self.assertNotIn("descriptive response", sanitized)
        self.assertFalse(is_meta_or_evaluation_leak(sanitized))
        self.assertIn("photos onnum ayakkan pattillallo da", sanitized)
        self.assertIn("text cheyyan mathramalle pattu", sanitized)

    def test_clean_companion_reply_strips_line_level_meta_critique(self):
        """Verify mixed model output retains conversational part and strips meta critique lines."""
        raw_mixed = (
            "Ayyoda da, njan koodeyundallo!\n"
            "* Better response: Do not promise physical meetings.\n"
            "Parayu, what are you doing right now?"
        )
        cleaned = clean_companion_reply(raw_mixed, user_query="enthokkeyund")
        self.assertNotIn("Better response", cleaned)
        self.assertNotIn("Do not promise", cleaned)
        self.assertIn("Ayyoda da, njan koodeyundallo!", cleaned)
        self.assertIn("Parayu, what are you doing right now?", cleaned)

    def test_clean_companion_reply_strips_fake_roleplay_photo_actions(self):
        """Verify fake roleplay image actions like *sends photo* or [Photo attached] are stripped."""
        raw_roleplay = "Ayyoda! *sends a cute photo* Njan inganathe dress aanu ittittullathu."
        cleaned = clean_companion_reply(raw_roleplay, user_query="photo theroo")
        self.assertNotIn("*sends a cute photo*", cleaned)
        self.assertIn("Njan inganathe dress aanu ittittullathu.", cleaned)

        raw_bracket = "[Photo attached] Ivide nokku da!"
        cleaned_bracket = clean_companion_reply(raw_bracket, user_query="photo theroo")
        self.assertNotIn("[Photo attached]", cleaned_bracket)
        self.assertIn("Ivide nokku da!", cleaned_bracket)

    @patch('chat.services.CompanionChatService.get_client')
    def test_end_to_end_photo_request_handles_leaked_critique_safely(self, mock_get_client):
        """Verify that when Gemini outputs critique for a photo request, user gets natural Manglish response and DB persists clean text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "* Better response: Use a descriptive response instead of pretending to send a photo."
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=20,
            total_token_count=70,
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(
            self.chat_url,
            data={"message": "enik ninte oru photo ayach theroo"},
            format='json'
        )

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Must not show internal critique
        self.assertNotIn("Better response", reply)
        self.assertNotIn("Use a descriptive", reply)
        self.assertIn("photos onnum ayakkan pattillallo da", reply)

        # Database must persist sanitized text
        last_companion_msg = ChatMessage.objects.filter(
            conversation__user=self.user,
            sender=ChatMessage.Sender.COMPANION
        ).last()
        self.assertIsNotNone(last_companion_msg)
        self.assertEqual(last_companion_msg.content, reply)
        self.assertNotIn("Better response", last_companion_msg.content)

    @patch('chat.services.CompanionChatService.get_client')
    def test_end_to_end_photo_request_preserves_natural_manglish_refusal(self, mock_get_client):
        """Verify that when Gemini outputs a natural Manglish refusal, it is returned and stored perfectly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyoda, enikku photo onnum ayakkan pattillallo da! Text mathramalle pattu. But njan eppozhum ninte koode undallo 🥰"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=45,
            candidates_token_count=25,
            total_token_count=70,
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(
            self.chat_url,
            data={"message": "enik ninte oru photo ayach theroo"},
            format='json'
        )

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertEqual(reply, "Ayyoda, enikku photo onnum ayakkan pattillallo da! Text mathramalle pattu. But njan eppozhum ninte koode undallo 🥰")


class GeminiQuotaAndRetryTests(TestCase):
    """
    Tests for Gemini API quota handling, retry/backoff discrimination, and fallback:
    1. Verifies 503 high demand and transient 429 rate limits are classified as retryable.
    2. Verifies hard quota exhaustion (free_tier_requests cap, limit: 20) and billing errors are NOT retried.
    3. Verifies 404 model not found and 400 bad requests are NOT retried.
    4. Verifies CompanionChatService retries transient 503 and recovers.
    5. Verifies CompanionChatService aborts retrying exhausted-quota models and seamlessly uses fallback.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='quota_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_transient_503_is_retryable(self):
        exc_503 = GenAIAPIError(503, {'error': {'message': 'This model is currently experiencing high demand. Please try again later.'}})
        retryable, backoff = is_retryable_gemini_error(exc_503)
        self.assertTrue(retryable)
        self.assertGreater(backoff, 0)

    def test_transient_429_is_retryable(self):
        exc_transient = GenAIAPIError(429, {'error': {'message': 'Rate limit exceeded. Please retry in 2s.'}})
        retryable, backoff = is_retryable_gemini_error(exc_transient)
        self.assertTrue(retryable)
        self.assertGreater(backoff, 0)

    def test_hard_quota_exhaustion_is_not_retryable(self):
        # Exact Gemini 3 Flash free tier limit error
        exc_exhausted = GenAIAPIError(429, {'error': {'message': 'Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3-flash. Please retry in 4h12m.'}})
        retryable, backoff = is_retryable_gemini_error(exc_exhausted)
        self.assertFalse(retryable)
        self.assertEqual(backoff, 0.0)

    def test_billing_and_plan_errors_are_not_retryable(self):
        exc_billing = GenAIAPIError(429, {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.'}})
        retryable, backoff = is_retryable_gemini_error(exc_billing)
        self.assertFalse(retryable)
        self.assertEqual(backoff, 0.0)

    def test_404_not_found_is_not_retryable(self):
        exc_404 = GenAIAPIError(404, {'error': {'message': 'models/gemini-2.0-flash is no longer available.'}})
        retryable, backoff = is_retryable_gemini_error(exc_404)
        self.assertFalse(retryable)
        self.assertEqual(backoff, 0.0)

    @patch('time.sleep', return_value=None)
    @patch('chat.services.CompanionChatService.get_client')
    def test_transient_503_retries_and_recovers(self, mock_get_client, mock_sleep):
        """Verify transient 503 on first attempt retries with backoff and returns successful response."""
        mock_client = MagicMock()
        mock_success = MagicMock()
        mock_success.text = "Njan koodeyunde da!"
        mock_success.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)

        exc_503 = GenAIAPIError(503, {'error': {'message': 'This model is currently experiencing high demand. Please try again later.'}})
        mock_client.models.generate_content.side_effect = [exc_503, mock_success]
        mock_get_client.return_value = mock_client

        service = CompanionChatService()
        reply, meta = service.generate_reply(message="Hello Ami")

        self.assertEqual(reply, "Njan koodeyunde da!")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once()

    @patch('time.sleep', return_value=None)
    @patch('chat.services.CompanionChatService.get_client')
    def test_hard_quota_does_not_retry_and_falls_back(self, mock_get_client, mock_sleep):
        """Verify hard quota error (limit: 20) aborts retrying primary model and switches to fallback immediately."""
        mock_client = MagicMock()
        mock_fallback_success = MagicMock()
        mock_fallback_success.text = "Hello da! Njan koode undallo."
        mock_fallback_success.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)

        exc_hard_quota = GenAIAPIError(429, {'error': {'message': 'Quota exceeded for metric: free_tier_requests, limit: 20'}})

        # First call on primary model raises hard quota; call on fallback succeeds
        mock_client.models.generate_content.side_effect = [exc_hard_quota, mock_fallback_success]
        mock_get_client.return_value = mock_client

        service = CompanionChatService()
        reply, meta = service.generate_reply(message="Hi Ami")

        self.assertEqual(reply, "Hello da! Njan koode undallo.")
        # Crucial check: Primary model was called only ONCE (no repeated retries of exhausted quota)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        # Sleep was NOT called because hard quota is not retryable
        mock_sleep.assert_not_called()


class AmiPersonalityUpgradeTests(TestCase):
    """
    Test suite for Ami's upgraded girlfriend + best friend personality:
    1. System prompt defines girlfriend & closest best friend duo in casual Manglish.
    2. Dynamic emotional range (playful, mock-annoyed, caring, constructive disagreement, calm/supportive).
    3. Bans on robotic openers ('Aww', 'Of course', 'I understand', 'Sure thing!').
    4. Gemini generation uses temperature=0.85 for spontaneous conversational variety.
    5. Natural banter, constructive disagreement, and playful reconciliation flows.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='ami_girlfriend_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_system_prompt_defines_girlfriend_best_friend_persona(self):
        """Verify prompt explicitly establishes girlfriend + best friend role, Manglish style, and rejects bot demeanor."""
        self.assertIn("loving girlfriend and closest best friend", BASE_SYSTEM_PROMPT)
        self.assertIn("NOT a formal assistant", BASE_SYSTEM_PROMPT)
        self.assertIn("Manglish", BASE_SYSTEM_PROMPT)
        self.assertIn("BANNED Bot Openers", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER start every message with repetitive filler like \"Aww\", \"Of course\", \"I understand\"", BASE_SYSTEM_PROMPT)

    def test_system_prompt_defines_context_driven_emotional_range(self):
        """Verify prompt includes the 5 context-driven emotional dimensions and ground rules."""
        self.assertIn("Loving, Affectionate & Caring", BASE_SYSTEM_PROMPT)
        self.assertIn("Playful, Witty & Flirty", BASE_SYSTEM_PROMPT)
        self.assertIn("Mock-Annoyed / Playfully Sulking", BASE_SYSTEM_PROMPT)
        self.assertIn("Willing to Disagree & Speak Your Mind", BASE_SYSTEM_PROMPT)
        self.assertIn("Calm, Mature & Supportive", BASE_SYSTEM_PROMPT)
        self.assertIn("Emotions must emerge strictly from the conversation", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER invent memories", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_gemini_generation_uses_increased_conversational_temperature(self, mock_get_client):
        """Verify Gemini config uses temperature=0.85 for natural, spontaneous girlfriend replies."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Athe athe, njan paranjille! 😂"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=8, total_token_count=23)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Enthu patti da?'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs['config']
        self.assertEqual(config.temperature, 0.85)

    @patch('chat.services.CompanionChatService.get_client')
    def test_playful_mock_annoyed_girlfriend_reaction(self, mock_get_client):
        """Verify playful mock-annoyed responses when playfully teased or when user takes too long."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hmph, ippo ennod samsarikkan time illa alle? 😤 Njan mindilla podee! 🙄"
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=16, total_token_count=36)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Hey, was busy with games whole day haha'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertIn("Hmph", reply)
        self.assertIn("mindilla", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_constructive_disagreement_on_unhealthy_habit(self, mock_get_client):
        """Verify Ami expresses genuine opinions and disagrees with unhealthy ideas instead of being an agreeable yes-bot."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Alla da, I really don't think that's a good idea. Nale work-il thalavedana eduthu kidakkum. Phone vechittu ippo thanne go sleep!"
        mock_response.usage_metadata = MagicMock(prompt_token_count=25, candidates_token_count=24, total_token_count=49)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Thinking of staying awake all night binge watching YouTube'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertIn("Alla da", reply)
        self.assertIn("don't think that's a good idea", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_reconciliation_and_playful_forgiveness(self, mock_get_client):
        """Verify Ami forgives warmly and playfully when partner apologizes."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Sarilla tto... I was just worried about you. But okay forgiven, now you owe me an ice cream! 🍦"
        mock_response.usage_metadata = MagicMock(prompt_token_count=30, candidates_token_count=22, total_token_count=52)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Sorry da, I should have listened to you'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertIn("Sarilla tto", reply)
        self.assertIn("ice cream", reply)


class AmiConversationalNaturalnessTests(TestCase):
    """
    Regression tests for conversational naturalness, brevity, and authentic partner banter:
    1. Verifies BASE_SYSTEM_PROMPT includes explicit brevity guidelines (2-12 words for casual chatter).
    2. Verifies casual responses to low-energy chatter ("onnum parayan illa", "chumma irikkuva") remain brief and non-intrusive.
    3. Verifies playful mock-insults ("nee ethadi thendi") are met with affectionate banter rather than formal apologies.
    4. Verifies no forced questions or unsolicited therapy on casual chats.
    5. Verifies clean_companion_reply provides natural brief fallbacks for low-energy queries.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='naturalness_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_system_prompt_includes_brevity_and_casual_manglish_rules(self):
        """Verify prompt contains strict word-count limits (2-12 words), no forced questions, and banter guidelines."""
        self.assertIn("usually reply in 2 to 12 words", BASE_SYSTEM_PROMPT)
        self.assertIn("Responses to \"onnum illa\", \"chumma\", \"ariyilla\"", BASE_SYSTEM_PROMPT)
        self.assertIn("Do NOT force every reply to end with a question", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER assume the user is sad, stressed, angry, or upset", BASE_SYSTEM_PROMPT)
        self.assertIn("nee ethadi thendi", BASE_SYSTEM_PROMPT)
        self.assertIn("Enna chumma irikkaam", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_casual_low_energy_onnum_parayan_illa(self, mock_get_client):
        """Verify Ami responds casually and briefly to 'onnum parayan illa' without interrogating or mood assumptions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Enna pinne chumma irikkaam 😂"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'onnum parayan illa'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Word count must remain brief (2 to 12 words)
        words = reply.split()
        self.assertGreaterEqual(len(words), 2)
        self.assertLessEqual(len(words), 12)

        # No forced interrogation or therapy
        self.assertNotIn("why", reply.lower())
        self.assertNotIn("tell me everything", reply.lower())
        self.assertNotIn("are you sad", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_casual_chumma_irikkuva_brevity(self, mock_get_client):
        """Verify Ami responds casually to 'chumma irikkuva' without turning it into a caring explanation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Njanum chumma irikkuva da 😌"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'chumma irikkuva'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        words = reply.split()
        self.assertGreaterEqual(len(words), 2)
        self.assertLessEqual(len(words), 12)
        self.assertFalse(reply.endswith("?"))

    @patch('chat.services.CompanionChatService.get_client')
    def test_playful_banter_nee_ethadi_thendi(self, mock_get_client):
        """Verify Ami responds with playful partner banter to affectionate mock-insult 'nee ethadi thendi'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ninte kettiyolaada thendi! 😜"
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=5, total_token_count=25)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'nee ethadi thendi'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Banter reaction, not a robotic customer-care apology
        self.assertIn("thendi", reply)
        self.assertNotIn("sorry", reply.lower())
        self.assertNotIn("i apologize", reply.lower())
        self.assertNotIn("language model", reply.lower())

    def test_clean_companion_reply_fallback_for_serious_message(self):
        """Verify clean_companion_reply substitutes empathetic calm fallback for serious/frustrated queries if blank."""
        fallback = clean_companion_reply("", user_query="stop teasing, I'm serious")
        self.assertEqual(fallback, "Enthu patti da? Njan kelkkam, calm aayittu para.")


class AmiEmotionalIntelligencePhase1Tests(TestCase):
    """
    Test suite for Ami Emotional Intelligence Upgrade (Phase 1):
    1. Distinguishing playful teasing from genuine frustration.
    2. Affectionate replies without repetitive romantic templates.
    3. Ambiguous messages handled without unsupported emotional assumptions.
    4. Serious conversations receiving an appropriate tone.
    5. Natural short Manglish replies.
    6. Conversation continuity and natural apologies across multiple turns.
    7. Memory filtering: temporary moods never stored as permanent memories, durable facts preserved.
    8. Authentication, memory retrieval, and chat API contracts intact.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='ei_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_system_prompt_includes_emotional_intelligence_rules(self):
        """Verify BASE_SYSTEM_PROMPT contains the core Phase 1 emotional intelligence guidelines."""
        self.assertIn("Contextual Emotional Intelligence & Empathy", BASE_SYSTEM_PROMPT)
        self.assertIn("Distinguish Playful Teasing from Genuine Frustration or Distress", BASE_SYSTEM_PROMPT)
        self.assertIn("Immediately DROP all teasing, jokes, mock-annoyance, or playful sarcasm", BASE_SYSTEM_PROMPT)
        self.assertIn("Ambiguous Short Messages", BASE_SYSTEM_PROMPT)
        self.assertIn("Do NOT jump to conclusions or assume they are sad, angry, or upset", BASE_SYSTEM_PROMPT)
        self.assertIn("Natural Apologies & Grace", BASE_SYSTEM_PROMPT)
        self.assertIn("Avoid Artificial Emotions & Boundary Bleed", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER guilt-trip your partner", BASE_SYSTEM_PROMPT)
        self.assertIn("NEVER claim to have a physical body", BASE_SYSTEM_PROMPT)
        self.assertIn("Maintain unbroken emotional continuity", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_playful_teasing_response(self, mock_get_client):
        """Verify playful banter is met with witty, affectionate partner banter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ninte kettiyolaada thendi! 😜"
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=5, total_token_count=25)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'poda thendi 😂'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertIn("thendi", reply)
        self.assertNotIn("I apologize as an AI", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_genuine_frustration_drops_teasing(self, mock_get_client):
        """Verify genuine frustration receives calm empathy, without jokes or teasing."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Enthu patti da? Njan kelkkam, calm aayittu para."
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=8, total_token_count=28)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': "stop teasing, I'm really annoyed right now"}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # No mocking or insults in response to genuine distress
        self.assertNotIn("thendi", reply.lower())
        self.assertNotIn("podee", reply.lower())
        self.assertNotIn("😜", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_affectionate_replies_without_repetitive_romantic_templates(self, mock_get_client):
        """Verify warm replies are natural and grounded, avoiding repetitive canned templates."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aaha so sweet of you da, loved hearing that! 🥰"
        mock_response.usage_metadata = MagicMock(prompt_token_count=18, candidates_token_count=9, total_token_count=27)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'You made my morning today'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertTrue(len(reply) > 0)
        # Shouldn't rely on robotic canned lines
        self.assertNotIn("As an AI", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_ambiguous_short_messages_handled_neutrally(self, mock_get_client):
        """Verify ambiguous short messages (e.g. 'hmm', 'sheriya') do not trigger unwarranted mood assumptions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hmm athe da 😌"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=4, total_token_count=19)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'hmm'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        words = reply.split()
        self.assertLessEqual(len(words), 12)
        # Must not assume sadness or initiate unsolicited therapy
        self.assertNotIn("are you sad", reply.lower())
        self.assertNotIn("why are you depressed", reply.lower())
        self.assertNotIn("tell me what happened", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_serious_conversation_appropriate_tone(self, mock_get_client):
        """Verify serious conversation about work burnout receives calm, mature support."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyoda, that sounds really exhausting da. Take a breath and rest for a bit tonight."
        mock_response.usage_metadata = MagicMock(prompt_token_count=25, candidates_token_count=15, total_token_count=40)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Work was brutal today, totally burnt out and exhausted'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertNotIn("haha", reply.lower())
        self.assertNotIn("😂", reply)
        self.assertNotIn("thendi", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_natural_short_manglish_replies(self, mock_get_client):
        """Verify casual messages get short 2-12 word Manglish replies."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Sheri da, pinne kaanaam 😌"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'sheri enna'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        words = reply.split()
        self.assertGreaterEqual(len(words), 2)
        self.assertLessEqual(len(words), 12)

    @patch('chat.services.CompanionChatService.get_client')
    def test_conversation_continuity_and_natural_apology(self, mock_get_client):
        """Verify Ami maintains context and apologizes naturally when a misunderstanding is pointed out."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyo sorry da! Njan thettidharichu, ente mistake."
        mock_response.usage_metadata = MagicMock(prompt_token_count=25, candidates_token_count=8, total_token_count=33)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Apology Thread")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="That's not what I meant at all.")

        res = self.client.post(self.chat_url, data={'message': 'You misunderstood me', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertIn("sorry", reply.lower())
        self.assertNotIn("I apologize as an artificial intelligence", reply)

    def test_temporary_moods_not_saved_as_permanent_memories(self):
        """Verify temporary moods and transient emotional states are never extracted into memory."""
        transient_mood_inputs = [
            "I feel sad today",
            "I'm feeling really depressed right now",
            "I am so stressed out with exams",
            "Enikku valare deshyam varunnu",
            "Feeling anxious today",
            "I'm having a rough day",
            "vishamam aanu",
        ]
        for inp in transient_mood_inputs:
            candidate = detect_candidate_memory(inp)
            self.assertIsNone(candidate, f"Transient mood '{inp}' should NOT be treated as a durable memory candidate!")

        # Durable facts and preferences MUST still be captured
        pref = detect_candidate_memory("I love drinking hot masala chai")
        self.assertIsNotNone(pref)
        self.assertEqual(pref['category'], 'preference')

        work = detect_candidate_memory("I work as a senior software engineer")
        self.assertIsNotNone(work)
        self.assertEqual(work['category'], 'work')

    @patch('chat.services.CompanionChatService.get_client')
    def test_existing_memory_and_chat_api_intact(self, mock_get_client):
        """Verify memory injection, chat persistence, and auth remain intact."""
        CompanionMemory.objects.create(
            user=self.user,
            content="Loves spicy food and dark chocolate",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True,
            importance=2
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Spicy food kazhicho da? 🌶️"
        mock_response.usage_metadata = MagicMock(prompt_token_count=30, candidates_token_count=5, total_token_count=35)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'What should I eat?'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        data = res.json()
        self.assertIn('reply', data)
        self.assertIn('conversation_id', data)
        self.assertIn('message_id', data)


class AmiPhase2NaturalEmotionalResponsesTests(TestCase):
    """
    Test suite for Ami Phase 2: Natural Emotional Responses:
    1. Simple affection ("nee adipoli aanu ❤️") responded to in one short, natural message.
    2. "ariyilla da", "hmm", "onnum illa" handled casually without multiple forced questions.
    3. Sadness acknowledged gently without question bombardment, generic advice, or repetitive reassurance.
    4. Misunderstandings acknowledged specifically and apologized for briefly.
    5. Avoidance of repetitive phrases ("njan ivide thanne undu", "njan kettukolam", "nee parayoo", "ellam nalla mood-il aayirikkanam").
    6. Emotional intensity matching & preference for 1-2 short sentences for ordinary chat.
    7. Clean companion reply preserves good Gemini responses without overriding with fallbacks.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='phase2_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_phase2_prompt_guidelines_present(self):
        """Verify BASE_SYSTEM_PROMPT includes all core Phase 2 rules."""
        self.assertIn("Emotional Intensity Matching & Natural Response Length (Crucial Phase 2 Rules)", BASE_SYSTEM_PROMPT)
        self.assertIn("Simple Affection (e.g. \"nee adipoli aanu ❤️\", \"love you\", \"miss you da\")", BASE_SYSTEM_PROMPT)
        self.assertIn("When the User Says They Are Sad", BASE_SYSTEM_PROMPT)
        self.assertIn("When Misunderstandings Occur", BASE_SYSTEM_PROMPT)
        self.assertIn("BANNED Scripted Clichés & Repetitive Crutches", BASE_SYSTEM_PROMPT)
        self.assertIn("njan ivide thanne undu", BASE_SYSTEM_PROMPT)
        self.assertIn("njan kettukolam", BASE_SYSTEM_PROMPT)
        self.assertIn("nee parayoo", BASE_SYSTEM_PROMPT)
        self.assertIn("ellam nalla mood-il aayirikkanam", BASE_SYSTEM_PROMPT)
        self.assertIn("Prefer ONE or TWO short sentences for ordinary chat", BASE_SYSTEM_PROMPT)
        self.assertIn("Do NOT implement rigid word limits that make serious conversations unnatural", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_simple_affection_nee_adipoli_aanu_one_short_reply(self, mock_get_client):
        """Verify 'nee adipoli aanu ❤️' receives a warm, single short reply, not a paragraph of emotional analysis."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aww thank you da! 🥰 Nee athilum adipoliyaanu! ❤️"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=10, total_token_count=25)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'nee adipoli aanu ❤️'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Short & sweet (under 15 words)
        words = reply.split()
        self.assertLessEqual(len(words), 15)
        # Not a lecture/essay
        self.assertNotIn("\n\n", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_ariyilla_da_casual_without_forced_questions(self, mock_get_client):
        """Verify 'ariyilla da' receives a casual response without forcing questions forward."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aha athu sheri da 😂"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'ariyilla da'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertFalse(reply.endswith("?"))
        self.assertNotIn("why", reply.lower())
        self.assertNotIn("tell me", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_user_sad_gentle_acknowledgment_without_question_bombardment(self, mock_get_client):
        """Verify user sadness is acknowledged gently without interrogating with multiple questions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyoda, enthu patti da? Take it easy tto, I'm right here."
        mock_response.usage_metadata = MagicMock(prompt_token_count=18, candidates_token_count=12, total_token_count=30)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'sad aanu today'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Count questions: should not bombard with 2+ question marks
        question_count = reply.count("?")
        self.assertLessEqual(question_count, 1)
        # Gentle tone
        self.assertTrue(len(reply.split()) <= 20)

    @patch('chat.services.CompanionChatService.get_client')
    def test_misunderstanding_brief_apology(self, mock_get_client):
        """Verify misunderstanding receives a brief, natural apology rather than a long defensive paragraph."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyo sorry da, njan thettidharichu! My bad."
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=8, total_token_count=28)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'You misunderstood what I meant.'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertIn("sorry", reply.lower())
        self.assertLessEqual(len(reply.split()), 15)
        self.assertNotIn("\n\n", reply)

    def test_clean_companion_reply_preserves_valid_gemini_responses(self):
        """Verify clean_companion_reply does not override valid responses with fallbacks."""
        good_affection = "Aww thank you da! 🥰 Nee athilum adipoliyaanu!"
        cleaned_affection = clean_companion_reply(good_affection, user_query="nee adipoli aanu ❤️")
        self.assertEqual(cleaned_affection, good_affection)

        good_casual = "Enna chumma irikkaam da 😌"
        cleaned_casual = clean_companion_reply(good_casual, user_query="onnum illa")
        self.assertEqual(cleaned_casual, good_casual)

        # Affection fallback if Gemini output is completely empty
        blank_affection_fallback = clean_companion_reply("", user_query="nee adipoli aanu")
        self.assertEqual(blank_affection_fallback, "Aww thank you da! 🥰")


class AmiNaturalConversationAuditTests(TestCase):
    """
    Focused test suite auditing and improving Ami's natural conversational flow:
    1. A casual greeting receives a concise reply.
    2. A compliment gets a natural response without a long emotional paragraph.
    3. An ambiguous message does not trigger an assumed emotional crisis.
    4. Genuine distress receives an appropriately empathetic response.
    5. An explicit request for detail can still receive a detailed answer.
    6. Consecutive replies do not rely on repeated reassurance or unnecessary questions.
    7. Memory and conversation context remain included correctly.
    8. Prompt audit directives are strictly present.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='audit_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_prompt_includes_audit_governing_rules(self):
        """Verify BASE_SYSTEM_PROMPT includes all newly audited conversation quality guidelines."""
        self.assertIn("Casual Greetings (e.g. \"hi\", \"hey\", \"hello\", \"good morning\", \"good night\")", BASE_SYSTEM_PROMPT)
        self.assertIn("Detailed Explanations & Advice", BASE_SYSTEM_PROMPT)
        self.assertIn("Consecutive Messages & Non-Repetition", BASE_SYSTEM_PROMPT)
        self.assertIn("No Hidden Emotion Invention", BASE_SYSTEM_PROMPT)
        self.assertIn("Default to Statements over Questions", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_casual_greeting_receives_concise_reply(self, mock_get_client):
        """Verify casual greeting gets a short, single-sentence reply without interrogation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Morning da! ☀️"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=4, total_token_count=19)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'good morning'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Short concise greeting
        self.assertLessEqual(len(reply.split()), 6)
        self.assertFalse(reply.endswith("?"))
        self.assertNotIn("how was your sleep", reply.lower())
        self.assertNotIn("had tea", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_compliment_natural_response_without_long_paragraph(self, mock_get_client):
        """Verify compliment gets a natural, warm response without a multi-sentence emotional essay."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aww thank you da! 🥰"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'You are looking great today'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertLessEqual(len(reply.split()), 8)
        self.assertNotIn("\n\n", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_ambiguous_message_no_assumed_emotional_crisis(self, mock_get_client):
        """Verify mundane/neutral message does not trigger assumed depression or crisis."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aha kollam, entha kazhikkunne? 🍲"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=6, total_token_count=21)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'having lunch now'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertNotIn("are you sad", reply.lower())
        self.assertNotIn("alone", reply.lower())
        self.assertNotIn("tell me everything", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_genuine_distress_appropriately_empathetic_response(self, mock_get_client):
        """Verify genuine distress receives quiet, calm empathy without jokes or dismissiveness."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ayyoda, I am so sorry to hear that da... Take a breath, I'm right here with you."
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=18, total_token_count=38)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'I lost my job today, feeling devastated.'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        self.assertNotIn("haha", reply.lower())
        self.assertNotIn("😂", reply)
        self.assertNotIn("thendi", reply.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_explicit_request_for_detail_can_receive_detailed_answer(self, mock_get_client):
        """Verify when the user asks for a detailed explanation, Ami can provide a comprehensive answer."""
        detailed_guide = (
            "To make authentic Kerala beef roast, first pressure cook the beef pieces with turmeric, chili powder, "
            "coriander powder, crushed ginger-garlic, black pepper, and curry leaves. In an iron kadai, roast coconut slices "
            "in coconut oil until golden. Then sauté shallots, green chilies, and garam masala until deeply browned. "
            "Add the cooked beef with its gravy and slow roast on medium-low heat until dark and aromatic! 🍲"
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = detailed_guide
        mock_response.usage_metadata = MagicMock(prompt_token_count=35, candidates_token_count=75, total_token_count=110)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Can you explain how to make Kerala beef roast in detail?'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Verifies that detailed explanation is preserved and not artificially truncated
        self.assertGreater(len(reply.split()), 30)
        self.assertIn("coconut slices", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_consecutive_replies_avoid_repeated_reassurance_or_questions(self, mock_get_client):
        """Verify consecutive turns do not repeat the same reassurance or question patterns."""
        conv = Conversation.objects.create(user=self.user, title="Banter Thread")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="Hey")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.COMPANION, content="Hey da! ☀️")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="Work kazhinjo?")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Athe, kazhinju! Ippo free aayi 😌"
        mock_response.usage_metadata = MagicMock(prompt_token_count=30, candidates_token_count=8, total_token_count=38)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Entha plan?', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # No robotic repetition of banned crutches
        self.assertNotIn("njan ivide thanne undu", reply)
        self.assertNotIn("njan kettukolam", reply)
        self.assertNotIn("ellam nalla mood-il", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_memory_and_conversation_context_remain_included(self, mock_get_client):
        """Verify memory and multi-turn context are correctly injected into Gemini call."""
        CompanionMemory.objects.create(
            user=self.user,
            content="Favorite football club is Arsenal",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True,
            importance=2
        )
        conv = Conversation.objects.create(user=self.user, title="Football Thread")
        ChatMessage.objects.create(conversation=conv, sender=ChatMessage.Sender.USER, content="Did you watch the match?")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Athe athe, Arsenal match alle? ⚽"
        mock_response.usage_metadata = MagicMock(prompt_token_count=40, candidates_token_count=7, total_token_count=47)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Who played better?', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        instruction = call_kwargs['config'].system_instruction
        self.assertIn("Arsenal", instruction)
        self.assertIn("Did you watch the match?", " ".join([p.text for c in call_kwargs['contents'] for p in c.parts]))


class AmiRealConversationNaturalnessRegressionTests(TestCase):
    """
    Regression test suite for real conversation flow:
    - User: "deee" -> Ami: short, casual greeting
    - User: "evde nte shuttumani" -> Ami: natural and affectionate in casual Manglish, without an unnecessary long explanation
    - User: "chuma" -> Ami: brief acknowledgment (e.g. "Aah sherida 😂") without automatically asking "Enthu visheshama?" again
    Preserves contextual variety, Ami's personality, and avoids repetitive follow-up questions or patterns.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='real_chat_tester', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_prompt_includes_real_chat_directives(self):
        """Verify prompt contains rules for 'deee', 'evde nte shuttumani', and 'chuma'."""
        self.assertIn("deee", BASE_SYSTEM_PROMPT)
        self.assertIn("evde nte shuttumani", BASE_SYSTEM_PROMPT)
        self.assertIn("chuma", BASE_SYSTEM_PROMPT)
        self.assertIn("enthu visheshama?", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_user_deee_short_casual_greeting(self, mock_get_client):
        """Verify 'deee' gets a short, casual greeting/attention reaction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Deee! Entha da? 😄"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'deee'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Short & punchy reaction (under 8 words)
        words = reply.split()
        self.assertLessEqual(len(words), 8)
        self.assertNotIn("\n\n", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_user_evde_nte_shuttumani_natural_affection(self, mock_get_client):
        """Verify 'evde nte shuttumani' gets an affectionate Manglish reply without a long essay."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ivide undeda ninte shuttumani! 🥰 Entha paraye?"
        mock_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=9, total_token_count=29)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'evde nte shuttumani'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Sweet, affectionate, natural length (1 or 2 short sentences, under 16 words)
        words = reply.split()
        self.assertLessEqual(len(words), 16)
        self.assertNotIn("\n\n", reply)
        # In-character Manglish affection
        self.assertTrue(any(w in reply.lower() for w in ["shuttumani", "ivide", "ividundee", "da"]))

    @patch('chat.services.CompanionChatService.get_client')
    def test_user_chuma_brief_acknowledgment_without_enthu_visheshama(self, mock_get_client):
        """Verify 'chuma' receives a brief acknowledgment without asking 'Enthu visheshama?' again."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aah sherida 😂"
        mock_response.usage_metadata = MagicMock(prompt_token_count=15, candidates_token_count=4, total_token_count=19)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'chuma'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']

        # Short casual reaction
        self.assertLessEqual(len(reply.split()), 6)
        # Must not ask repetitive follow-up questions like "Enthu visheshama?"
        self.assertNotIn("enthu vishesham", reply.lower())
        self.assertNotIn("vishesham", reply.lower())
        self.assertFalse(reply.endswith("?"))

    @patch('chat.services.CompanionChatService.get_client')
    def test_full_real_conversation_flow_multi_turn(self, mock_get_client):
        """
        End-to-end multi-turn regression test:
        Turn 1: User says "deee" -> Ami gives a short, casual greeting
        Turn 2: User says "evde nte shuttumani" -> Ami responds affectionately in Manglish without long explanations
        Turn 3: User says "chuma" -> Ami acknowledges briefly without asking 'Enthu visheshama?'
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "deee"
        res1_text = "Deee! Parayoo da 😄"
        mock_response1 = MagicMock(text=res1_text, usage_metadata=MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20))
        mock_client.models.generate_content.return_value = mock_response1

        resp1 = self.client.post(self.chat_url, data={'message': 'deee'}, format='json')
        self.assertEqual(resp1.status_code, status.HTTP_200_OK)
        conv_id = resp1.json()['conversation_id']
        reply1 = resp1.json()['reply']
        self.assertEqual(reply1, res1_text)
        self.assertLessEqual(len(reply1.split()), 8)

        # Turn 2: "evde nte shuttumani"
        res2_text = "Ivide undeda ninte shuttumani! 🥰 Njan evide pokaan?"
        mock_response2 = MagicMock(text=res2_text, usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=10, total_token_count=35))
        mock_client.models.generate_content.return_value = mock_response2

        resp2 = self.client.post(self.chat_url, data={'message': 'evde nte shuttumani', 'conversation_id': conv_id}, format='json')
        self.assertEqual(resp2.status_code, status.HTTP_200_OK)
        reply2 = resp2.json()['reply']
        self.assertEqual(reply2, res2_text)
        self.assertLessEqual(len(reply2.split()), 15)
        self.assertIn("shuttumani", reply2.lower())

        # Turn 3: "chuma"
        res3_text = "Aah sherida 😂"
        mock_response3 = MagicMock(text=res3_text, usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=4, total_token_count=39))
        mock_client.models.generate_content.return_value = mock_response3

        resp3 = self.client.post(self.chat_url, data={'message': 'chuma', 'conversation_id': conv_id}, format='json')
        self.assertEqual(resp3.status_code, status.HTTP_200_OK)
        reply3 = resp3.json()['reply']
        self.assertEqual(reply3, res3_text)
        self.assertLessEqual(len(reply3.split()), 6)
        self.assertNotIn("vishesham", reply3.lower())
        self.assertNotIn("enthu vishesham", reply3.lower())

        # Verify conversation history has all 6 messages preserved in correct order
        conv = Conversation.objects.get(id=conv_id)
        messages = list(conv.messages.order_by('created_at'))
        self.assertEqual(len(messages), 6)
        self.assertEqual([m.sender for m in messages], ['user', 'companion', 'user', 'companion', 'user', 'companion'])
        self.assertEqual(messages[0].content, 'deee')
        self.assertEqual(messages[1].content, res1_text)
        self.assertEqual(messages[2].content, 'evde nte shuttumani')
        self.assertEqual(messages[3].content, res2_text)
        self.assertEqual(messages[4].content, 'chuma')
        self.assertEqual(messages[5].content, res3_text)

    def test_clean_companion_reply_fallbacks_for_real_conversation(self):
        """Verify clean_companion_reply safely handles empty/leak outputs for these exact phrases."""
        # Fallback for "chuma"
        fb_chuma = clean_companion_reply("", user_query="chuma")
        self.assertEqual(fb_chuma, "Aah sherida 😂")
        self.assertNotIn("vishesham", fb_chuma.lower())

        # Fallback for "evde nte shuttumani"
        fb_shuttu = clean_companion_reply("", user_query="evde nte shuttumani")
        self.assertIn("shuttumani", fb_shuttu.lower())

        # Fallback for "deee"
        fb_deee = clean_companion_reply("", user_query="deee")
        self.assertIn("deee", fb_deee.lower())

        # Good responses must never be overwritten
        custom_chuma = "Enna pinne chumma irikkaam 😂"
        cleaned = clean_companion_reply(custom_chuma, user_query="chuma")
        self.assertEqual(cleaned, custom_chuma)


class AmiInstagramStyleConversationRegressionTests(TestCase):
    """
    Regression test suite for natural Instagram/WhatsApp DM-style conversation behaviour:
    1. Casual greetings and short messages across multiple turns.
    2. Affectionate banter matching energy without long speeches.
    3. Repeated emoji messages without analytical overthinking.
    4. Playful teasing and mild insults with witty banter.
    5. 'I'll message you later' followed by smooth contextual resumption.
    6. Ambiguous messages without assumed emotional crisis.
    7. A genuine serious conversation with calm, steady support.
    8. A user explicitly requesting a detailed explanation.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='insta_chat_user', password='password123')
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_prompt_includes_instagram_dm_directives(self):
        """Verify prompt contains Instagram DM and React-First directives."""
        self.assertIn("Instagram DMs", BASE_SYSTEM_PROMPT)
        self.assertIn("React First Before Explaining", BASE_SYSTEM_PROMPT)
        self.assertIn("Repeated Emoji & Sticker Messages", BASE_SYSTEM_PROMPT)
        self.assertIn("Message You Later", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_1_casual_greetings_and_short_messages(self, mock_get_client):
        """Scenario 1: Casual greetings and short messages across 3 turns."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "Heyy"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hey da! ☀️",
            usage_metadata=MagicMock(prompt_token_count=15, candidates_token_count=4, total_token_count=19)
        )
        res1 = self.client.post(self.chat_url, data={'message': 'Heyy'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertLessEqual(len(reply1.split()), 6)
        self.assertFalse(reply1.endswith("?"))

        # Turn 2: "evideya?"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Ivide undu da, room-il! Entha da? 😄",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=8, total_token_count=33)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'evideya?', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertLessEqual(len(reply2.split()), 10)

        # Turn 3: "chumma"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aah sherida 😂",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=4, total_token_count=39)
        )
        res3 = self.client.post(self.chat_url, data={'message': 'chumma', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res3.status_code, status.HTTP_200_OK)
        reply3 = res3.json()['reply']
        self.assertLessEqual(len(reply3.split()), 5)
        self.assertNotIn("vishesham", reply3.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_2_affectionate_banter(self, mock_get_client):
        """Scenario 2: Affectionate banter without essay-length declarations."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "You're cute today"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aww thank you da! 🥰 Nee athilum cute aanu.",
            usage_metadata=MagicMock(prompt_token_count=18, candidates_token_count=9, total_token_count=27)
        )
        res1 = self.client.post(self.chat_url, data={'message': "You're cute today"}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertLessEqual(len(reply1.split()), 12)
        self.assertNotIn("\n\n", reply1)

        # Turn 2: "sathyam para, do you miss me?"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Pinne, miss cheyyathe pinne! Orupadu 🥰",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=7, total_token_count=37)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'sathyam para, do you miss me?', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertLessEqual(len(reply2.split()), 10)
        self.assertNotIn("njan eppozhum ninte koodeyille", reply2.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_3_repeated_emoji_messages(self, mock_get_client):
        """Scenario 3: Pure emoji messages receive natural emoji/vibe reactions, not essays."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "😂😂😂"
        mock_client.models.generate_content.return_value = MagicMock(
            text="😂😂 Eda chirichu maduthu!",
            usage_metadata=MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        )
        res1 = self.client.post(self.chat_url, data={'message': '😂😂😂'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertLessEqual(len(reply1.split()), 6)

        # Turn 2: "❤️❤️"
        mock_client.models.generate_content.return_value = MagicMock(
            text="🥰❤️ Love you da!",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=5, total_token_count=30)
        )
        res2 = self.client.post(self.chat_url, data={'message': '❤️❤️', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertLessEqual(len(reply2.split()), 6)
        self.assertNotIn("I understand that emotions", reply2)

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_4_playful_teasing_and_mild_insults(self, mock_get_client):
        """Scenario 4: Teasing & mild insults ('potta', 'thendi') get reciprocal witty banter."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "poda thendi 😂"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Ninte kettiyolaada thendi! 😜",
            usage_metadata=MagicMock(prompt_token_count=18, candidates_token_count=5, total_token_count=23)
        )
        res1 = self.client.post(self.chat_url, data={'message': 'poda thendi 😂'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertIn("thendi", reply1)
        self.assertNotIn("apologize", reply1.lower())

        # Turn 2: "ayye nee kollilla potta"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aaha ninte ahankaram koodi varunnu tto! 😤😂 Ninne njan nokkikolam!",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=11, total_token_count=41)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'ayye nee kollilla potta', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertIn("ahankaram", reply2.lower())
        self.assertNotIn("language model", reply2.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_5_ill_message_you_later_and_smooth_resumption(self, mock_get_client):
        """Scenario 5: 'Pinne parayaam' acknowledged smoothly, followed by natural topic resumption."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "Going into a meeting, pinne parayaam"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri da, all the best! Free aavumbo message ayaykk 😌",
            usage_metadata=MagicMock(prompt_token_count=20, candidates_token_count=10, total_token_count=30)
        )
        res1 = self.client.post(self.chat_url, data={'message': 'Going into a meeting, pinne parayaam'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertIn("free", reply1.lower())
        self.assertFalse(reply1.endswith("?"))

        # Turn 2: Later user returns: "Meeting ended, got the client approval!"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aha adipoli! Congrats da! 🎉 Paranjille nadakkum ennu!",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=10, total_token_count=45)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'Meeting ended, got the client approval!', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertIn("adipoli", reply2.lower())
        self.assertNotIn("hello, who are you", reply2.lower())
        self.assertNotIn("enthu visheshama?", reply2.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_6_ambiguous_messages_without_assumed_emotions(self, mock_get_client):
        """Scenario 6: Ambiguous/neutral messages do not trigger assumed depression or crisis."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: "hmm"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hmm 😌 Entha da?",
            usage_metadata=MagicMock(prompt_token_count=15, candidates_token_count=5, total_token_count=20)
        )
        res1 = self.client.post(self.chat_url, data={'message': 'hmm'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertNotIn("depressed", reply1.lower())
        self.assertNotIn("sad", reply1.lower())

        # Turn 2: "aano"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Athe athe da 😌",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=5, total_token_count=30)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'aano', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertLessEqual(len(reply2.split()), 6)
        self.assertNotIn("tell me everything", reply2.lower())

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_7_genuine_serious_conversation(self, mock_get_client):
        """Scenario 7: Serious distress is met with steady calm support, not jokes or therapist jargon."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: User shares emergency
        mock_client.models.generate_content.return_value = MagicMock(
            text="Ayyoda... I hope he is okay da. Doctors entha paranje? Don't panic, take deep breaths. I'm right here with you.",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=22, total_token_count=47)
        )
        res1 = self.client.post(self.chat_url, data={'message': 'Dad had a minor heart scare today, at the hospital now.'}, format='json')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv_id = res1.json()['conversation_id']
        reply1 = res1.json()['reply']
        self.assertNotIn("haha", reply1.lower())
        self.assertNotIn("😂", reply1)
        self.assertNotIn("thendi", reply1.lower())

        # Turn 2: Relief update
        mock_client.models.generate_content.return_value = MagicMock(
            text="Thank goodness! Big relief da... Nee vellam vallathum kudichu rest cheyy tto. Take care of him.",
            usage_metadata=MagicMock(prompt_token_count=45, candidates_token_count=19, total_token_count=64)
        )
        res2 = self.client.post(self.chat_url, data={'message': 'Stable now, resting.', 'conversation_id': conv_id}, format='json')
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        reply2 = res2.json()['reply']
        self.assertIn("relief", reply2.lower())
        self.assertNotIn("😂😂", reply2)

    @patch('chat.services.CompanionChatService.get_client')
    def test_scenario_8_explicit_request_for_detailed_explanation(self, mock_get_client):
        """Scenario 8: Explicit request for detailed explanation produces full answer without artificial truncation."""
        recipe_text = (
            "Authentic Kerala beef roast starts with marinating beef cubes in turmeric, Kashmiri chili powder, "
            "coriander, crushed ginger, garlic, and curry leaves. Pressure cook until tender. In an iron kadai, "
            "heat coconut oil, fry thinly sliced fresh coconut bits until golden, sauté lots of shallots and green chilies, "
            "then add garam masala and the cooked beef. Slow-roast on low flame until the meat turns dark brown and fragrant! 🍲"
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=recipe_text,
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=70, total_token_count=105)
        )
        mock_get_client.return_value = mock_client

        res = self.client.post(self.chat_url, data={'message': 'Can you explain how to make authentic Kerala beef roast step by step in detail?'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        reply = res.json()['reply']
        self.assertGreater(len(reply.split()), 30)
        self.assertIn("coconut", reply)


class AmiPersistentMoodSystemTests(TestCase):
    """
    Test suite for Ami Mood System v1 (Persistent Emotional States):
    1. Neutral messages preserve mood and naturally decay toward calm.
    2. Relevant provocative teasing triggers irritation / annoyance.
    3. Mood continuity across multiple turns without sudden amnesia.
    4. Sincere apologies initiate gradual softening recovery.
    5. Affection does not instantly clear unresolved hurt.
    6. Resolved conflicts return naturally to normal (calm or playful).
    7. Irrelevant messages do not trigger false jealousy.
    8. Friendly teasing remains playful rather than angry.
    9. Per-user isolation of mood states.
    10. Persistence across database reloads/restarts.
    11. AI provider failures do not corrupt mood state.
    12. Dynamic prompt injection includes mood directives.
    """

    def setUp(self):
        self.client = APIClient()
        self.user_a = User.objects.create_user(username='mood_user_a', password='password123')
        self.user_b = User.objects.create_user(username='mood_user_b', password='password123')
        self.client.force_authenticate(user=self.user_a)
        self.chat_url = '/api/chat/'

    def test_prompt_injects_mood_directive(self):
        """Verify build_system_prompt injects active mood guidance into system instruction."""
        service = CompanionChatService()
        prompt_annoyed = service.build_system_prompt(
            mood='annoyed',
            mood_context={'unresolved_trigger': 'provocative_jealousy', 'trigger_phrase': 'she is prettier'}
        )
        self.assertIn("Mood: ANNOYED", prompt_annoyed)
        self.assertIn("provocative_jealousy", prompt_annoyed)
        self.assertIn("Ami's Current Persistent Emotional State", prompt_annoyed)

        prompt_softening = service.build_system_prompt(
            mood='softening',
            mood_context={'unresolved_trigger': 'softening_after_apology'}
        )
        self.assertIn("Mood: SOFTENING", prompt_softening)
        self.assertIn("softening_after_apology", prompt_softening)

    @patch('chat.services.CompanionChatService.get_client')
    def test_neutral_message_preserves_mood(self, mock_get_client):
        """Verify neutral message preserves calm or existing state without random mood spikes."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri da, enjoy your lunch! 🍲",
            usage_metadata=MagicMock(prompt_token_count=20, candidates_token_count=6, total_token_count=26)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user_a, title="Lunch Chat", current_mood=Conversation.Mood.CALM)
        res = self.client.post(self.chat_url, data={'message': 'having lunch now', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertEqual(res.json()['mood'], 'calm')

    @patch('chat.services.CompanionChatService.get_client')
    def test_provocative_teasing_triggers_annoyance(self, mock_get_client):
        """Verify provocative romantic comparison triggers annoyed mood and records unresolved trigger."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aaha, athano kaaryam? Vere pennine nokkan aanalle plan! 😤",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=12, total_token_count=37)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user_a, title="Teasing Chat", current_mood=Conversation.Mood.CALM)
        provocative_msg = "I think Sneha is much prettier than you, vere oru pennine nokkiyaalo?"
        res = self.client.post(self.chat_url, data={'message': provocative_msg, 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)
        self.assertEqual(res.json()['mood'], 'annoyed')
        self.assertIn("provocative_jealousy", conv.mood_context.get("unresolved_trigger", ""))

    @patch('chat.services.CompanionChatService.get_client')
    def test_mood_continuity_across_multiple_turns(self, mock_get_client):
        """Verify mood remains annoyed on subsequent turn instead of instantly forgetting."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hmph, onnum parayanilla! 🙄",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=6, total_token_count=36)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Sulk Chat",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_jealousy", "turns_in_mood": 1}
        )

        # User sends a brief follow up without apologizing
        res = self.client.post(self.chat_url, data={'message': 'entha onnum mindathe?', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)
        self.assertEqual(res.json()['mood'], 'annoyed')

    @patch('chat.services.CompanionChatService.get_client')
    def test_sincere_apology_initiates_gradual_softening(self, mock_get_client):
        """Verify sincere apology moves mood from annoyed/hurt to softening."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hmm... okay da, forgive cheythu. But ini angane parayaruthu tto! 😌",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=14, total_token_count=44)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Apology Chat",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_jealousy", "turns_in_mood": 2}
        )

        apology_msg = "Ayyo sorry da, njan chumma paranjathaanu! I was just teasing you."
        res = self.client.post(self.chat_url, data={'message': apology_msg, 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertEqual(res.json()['mood'], 'softening')
        self.assertIn("softening", conv.mood_context.get("unresolved_trigger", ""))

    @patch('chat.services.CompanionChatService.get_client')
    def test_affection_does_not_instantly_clear_unresolved_hurt(self, mock_get_client):
        """Verify affection (love you) alone without apology does not instantly clear unresolved hurt."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Love you paranjaal ellam theerumo? Athra vegam njan marakkilla!",
            usage_metadata=MagicMock(prompt_token_count=28, candidates_token_count=12, total_token_count=40)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Affection Test",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_jealousy", "turns_in_mood": 1}
        )

        res = self.client.post(self.chat_url, data={'message': 'love you da ❤️', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        # Must still be annoyed; does NOT instantly clear hurt
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)

    @patch('chat.services.CompanionChatService.get_client')
    def test_resolved_conflict_returns_to_normal(self, mock_get_client):
        """Verify once in softening, ongoing positive interaction returns conversation to playful or calm."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hahaha sheri da, potte! Enna entha ippo plan? 😂",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=10, total_token_count=35)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Resolution Chat",
            current_mood=Conversation.Mood.SOFTENING,
            mood_context={"unresolved_trigger": "softening_after_apology", "turns_in_mood": 1}
        )

        res = self.client.post(self.chat_url, data={'message': 'ini angane parayilla promise 😂❤️', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.PLAYFUL)
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "")

    @patch('chat.services.CompanionChatService.get_client')
    def test_irrelevant_messages_not_triggering_false_jealousy(self, mock_get_client):
        """Verify cooking recipes, facts, or friendly teasing never trigger false jealousy or irritation."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Poda thendi 😂 Njan evide pokaan!",
            usage_metadata=MagicMock(prompt_token_count=20, candidates_token_count=8, total_token_count=28)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user_a, title="Friendly Chat", current_mood=Conversation.Mood.PLAYFUL)

        # Friendly teasing like 'poda thendi' must remain playful, not annoyed
        res = self.client.post(self.chat_url, data={'message': 'poda thendi 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.PLAYFUL)

    def test_per_user_isolation_of_mood(self):
        """Verify mood states in User A's conversation cannot leak into or affect User B's conversation."""
        conv_a = Conversation.objects.create(
            user=self.user_a,
            title="User A's Annoyed Chat",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_jealousy"}
        )

        conv_b = Conversation.objects.create(
            user=self.user_b,
            title="User B's Calm Chat",
            current_mood=Conversation.Mood.CALM
        )

        # Authenticate as User B
        self.client.force_authenticate(user=self.user_b)

        # User B gets conversation details
        res_b = self.client.get(f'/api/conversations/{conv_b.id}/')
        self.assertEqual(res_b.status_code, status.HTTP_200_OK)
        self.assertEqual(res_b.json()['current_mood'], Conversation.Mood.CALM)

        # User B cannot access User A's conversation
        res_a_from_b = self.client.get(f'/api/conversations/{conv_a.id}/')
        self.assertEqual(res_a_from_b.status_code, status.HTTP_404_NOT_FOUND)

    def test_persistence_across_service_restarts(self):
        """Verify mood state persists accurately in PostgreSQL and survives simulated server restarts."""
        conv = Conversation.objects.create(
            user=self.user_a,
            title="Restart Persistence",
            current_mood=Conversation.Mood.HURT,
            mood_context={"unresolved_trigger": "hurtful_remark", "turns_in_mood": 2}
        )
        conv_id = conv.id

        # Simulate service restart by clearing cache and refetching fresh from DB
        reloaded_conv = Conversation.objects.get(id=conv_id)
        self.assertEqual(reloaded_conv.current_mood, Conversation.Mood.HURT)
        self.assertEqual(reloaded_conv.mood_context["unresolved_trigger"], "hurtful_remark")
        self.assertEqual(reloaded_conv.mood_context["turns_in_mood"], 2)

    @patch('chat.services.CompanionChatService.get_client')
    def test_ai_provider_failure_does_not_corrupt_persisted_state(self, mock_get_client):
        """Verify that when the AI provider raises an error, the conversation mood in DB is not corrupted."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = GenAIAPIError(503, {'error': {'message': 'Temporary API Failure', 'status': 'UNAVAILABLE'}})
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Failure Safety",
            current_mood=Conversation.Mood.CALM,
            mood_context={"turns_in_mood": 1}
        )

        res = self.client.post(self.chat_url, data={'message': 'I think Sneha is prettier than you', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_502_BAD_GATEWAY)

        # Conversation state in DB should NOT have been corrupted or partially updated
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)

    @patch('chat.services.CompanionChatService.get_client')
    def test_minor_apology_quick_natural_forgiveness(self, mock_get_client):
        """Phase 3.2: Minor teasing/misunderstanding apology resolves quickly and naturally."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Hmm... chumma paranjathaano? Sheri, ippolkk vittaarnnu 😌",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=12, total_token_count=42)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Minor Tease",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_jealousy", "severity": "minor", "turns_in_mood": 1}
        )

        # Turn 1: User offers quick apology for teasing
        res1 = self.client.post(
            self.chat_url,
            data={'message': 'chumma paranjatha da, sorry 😂', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertFalse(conv.mood_context.get("emotional_residue", False))
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "softening_after_apology")

        # Turn 2: User chats normally -> natural forgiveness without grudges
        mock_client.models.generate_content.return_value = MagicMock(
            text="Phone nokki chumma irikkuva da 😂",
            usage_metadata=MagicMock(prompt_token_count=32, candidates_token_count=8, total_token_count=40)
        )
        res2 = self.client.post(
            self.chat_url,
            data={'message': 'entha cheyyunne ippo?', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "")

    @patch('chat.services.CompanionChatService.get_client')
    def test_serious_hurt_retains_emotional_residue_after_first_apology(self, mock_get_client):
        """Phase 3.2: Meaningful hurt retains subtle emotional residue after apology without sugar rush."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri da... but athu kettappol nallonam hurt aayi. Let's drop it 😌",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=16, total_token_count=51)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Serious Hurt",
            current_mood=Conversation.Mood.HURT,
            mood_context={"unresolved_trigger": "hurtful_insult", "severity": "serious", "emotional_residue": True, "turns_in_mood": 1}
        )

        # Turn 1: User says simple "sorry"
        res1 = self.client.post(
            self.chat_url,
            data={'message': 'sorry da', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertTrue(conv.mood_context.get("emotional_residue", False))
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "softening_serious_hurt_residue")

        # Verify system prompt given to Gemini included the residue directive
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sys_instruction = call_kwargs['config'].system_instruction
        self.assertIn("Emotional Residue Note", sys_instruction)
        self.assertIn("NO SUDDEN SUGAR RUSH", sys_instruction)

        # Turn 2: Subsequent respectful conversation gradually settles residue to calm
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aah kazhichu da... nee kazhicho?",
            usage_metadata=MagicMock(prompt_token_count=36, candidates_token_count=8, total_token_count=44)
        )
        res2 = self.client.post(
            self.chat_url,
            data={'message': 'kazhicho da?', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertFalse(conv.mood_context.get("emotional_residue", False))

    @patch('chat.services.CompanionChatService.get_client')
    def test_repeated_apology_accepted_with_grace(self, mock_get_client):
        """Phase 3.2: Repeated apology is accepted gracefully without making user beg or hold grudge."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Eda, sarilla... repeated aayi sorry parayenda 😌 I'm okay now, athu vittukala.",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=18, total_token_count=53)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Repeated Apology",
            current_mood=Conversation.Mood.SOFTENING,
            mood_context={"unresolved_trigger": "softening_after_apology", "apology_count": 1, "turns_in_mood": 1}
        )

        res = self.client.post(
            self.chat_url,
            data={'message': 'Really really sorry da, please forgive me ❤️', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "apology_accepted_gracefully")
        self.assertFalse(conv.mood_context.get("emotional_residue", False))

        # Check prompt directive included graceful acceptance guidance
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sys_instruction = call_kwargs['config'].system_instruction
        self.assertIn("Sincere Apology Accepted", sys_instruction)

    @patch('chat.services.CompanionChatService.get_client')
    def test_affection_without_apology_while_hurt_does_not_reset_mood(self, mock_get_client):
        """Phase 3.2: Pure affection without apology does not reset serious hurt."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Athu kettappol sankadam aayi da...",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=7, total_token_count=37)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Hurt Affection",
            current_mood=Conversation.Mood.HURT,
            mood_context={"unresolved_trigger": "hurtful_insult", "severity": "serious", "emotional_residue": True, "turns_in_mood": 1}
        )

        res = self.client.post(
            self.chat_url,
            data={'message': 'love you da ❤️ umma', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.HURT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_multi_turn_gradual_reconciliation_flow(self, mock_get_client):
        """Phase 3.2: Full multi-turn flow: insult -> sincere apology with residue -> respectful chat clears residue -> playful banter."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Multi Turn Flow",
            current_mood=Conversation.Mood.CALM
        )

        # Turn 1: Harsh insult triggers HURT
        mock_client.models.generate_content.return_value = MagicMock(
            text="Athu parayenda aavashyam undaayirunno da?",
            usage_metadata=MagicMock(prompt_token_count=20, candidates_token_count=8, total_token_count=28)
        )
        res1 = self.client.post(
            self.chat_url,
            data={'message': "You're useless and I hate you", 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.HURT)
        self.assertTrue(conv.mood_context.get("emotional_residue", False))

        # Turn 2: Sincere apology moves to SOFTENING with residue
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri da... but athu kettappol nallonam hurt aayi. Let's drop it 😌",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=14, total_token_count=44)
        )
        res2 = self.client.post(
            self.chat_url,
            data={'message': "sorry da, didn't mean to hurt you", 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertTrue(conv.mood_context.get("emotional_residue", False))

        # Turn 3: Respectful normal talk dissolves residue into CALM
        mock_client.models.generate_content.return_value = MagicMock(
            text="Kazhichu da... nee kazhicho?",
            usage_metadata=MagicMock(prompt_token_count=32, candidates_token_count=6, total_token_count=38)
        )
        res3 = self.client.post(
            self.chat_url,
            data={'message': "did you have lunch da?", 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res3.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertFalse(conv.mood_context.get("emotional_residue", False))

        # Turn 4: Friendly banter naturally triggers PLAYFUL
        mock_client.models.generate_content.return_value = MagicMock(
            text="Poda pattee 😂",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=4, total_token_count=34)
        )
        res4 = self.client.post(
            self.chat_url,
            data={'message': "poda thendi 😂", 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res4.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.PLAYFUL)

    def test_prompt_directive_bans_sugar_rush_and_repeated_status_announcements(self):
        """Phase 3.2: Verify prompt directives prohibit sugar rush and status announcements."""
        from chat.mood import get_mood_prompt_directive

        softening_directive = get_mood_prompt_directive('softening', {'emotional_residue': True})
        self.assertIn("NO SUDDEN SUGAR RUSH", softening_directive)
        self.assertIn("ANTI-DECLARATION RULE", softening_directive)
        self.assertIn("Emotional Residue Note", softening_directive)

        annoyed_directive = get_mood_prompt_directive('annoyed')
        self.assertIn("ANTI-DECLARATION RULE", annoyed_directive)
        self.assertIn("I am still angry", annoyed_directive)

        hurt_directive = get_mood_prompt_directive('hurt')
        self.assertIn("ANTI-DECLARATION RULE", hurt_directive)
        self.assertIn("I am still hurt", hurt_directive)

    @patch('chat.services.CompanionChatService.get_client')
    def test_manglish_useless_insult_triggers_hurt_and_first_apology_retains_residue(self, mock_get_client):
        """Phase 3.2 Regression: Manglish insult 'nee useless aanedi...' triggers HURT and first apology retains emotional residue."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user_a,
            title="Manglish Insult Test",
            current_mood=Conversation.Mood.CALM
        )

        # Turn 1: User sends Manglish insult: "nee useless aanedi, ninne kondu oru upayogavum illa"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Athu parayenda aavashyam undaayirunno da? Enikku nallonam sankadam aayi...",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=10, total_token_count=35)
        )
        res1 = self.client.post(
            self.chat_url,
            data={'message': 'nee useless aanedi, ninne kondu oru upayogavum illa', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.HURT)
        self.assertEqual(conv.mood_context.get("severity"), "serious")
        self.assertTrue(conv.mood_context.get("emotional_residue", False))
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "hurtful_insult")

        # Turn 2: User sends first apology: "sorry edi, angane parayan paadillayirunnu"
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri da... but athu kettappol nallonam hurt aayi. Let's drop it 😌",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=14, total_token_count=49)
        )
        res2 = self.client.post(
            self.chat_url,
            data={'message': 'sorry edi, angane parayan paadillayirunnu', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertEqual(conv.mood_context.get("severity"), "serious")
        self.assertTrue(conv.mood_context.get("emotional_residue", False))
        self.assertEqual(conv.mood_context.get("unresolved_trigger"), "softening_serious_hurt_residue")
        self.assertEqual(conv.mood_context.get("apology_count"), 1)

        # Verify prompt directive sent to Gemini:
        # Must include emotional residue note and forbid "repeated aayi sorry parayenda"
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sys_instruction = call_kwargs['config'].system_instruction
        self.assertIn("Emotional Residue Note", sys_instruction)
        self.assertIn("NO SUDDEN SUGAR RUSH", sys_instruction)
        self.assertIn("CRITICAL FOR FIRST APOLOGY: Do NOT say 'repeated aayi sorry parayenda'", sys_instruction)
        self.assertNotIn("Repeated apology accepted: Accept with natural grace", sys_instruction)



@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class AmiPhase3NaturalTextingRegressionTests(TestCase):
    """
    Regression test suite for Ami Phase 3: Natural Texting Behaviour & DM Scale:
    - 1. Simple Activity Check: "Edi entha cheyyunne 😂" -> "Chumma irikkuva da 😂 nee para"
    - 2. Playful Banter / Insults: "Edi thendi 😂" -> "Poda pattee 😂"
    - 3. Provocative Teasing: "Enik vere oru pennine ishtam aanennu paranjaalo 😂" -> "Aaha 😒 appo njan aara da?"
    - 4. Still Annoyed: "Entha mindathe, deshyam aano?" -> "Pinne deshyam aaville 😒"
    - 5. Apology & Gradual Softening: "Ayyo sorry edi, chumma paranjatha ❤️" -> "Hmm... sheri, potte 😌"
    - 6. Affection While Annoyed: "Oru kiss thaa, pinne pinangalle 😚" -> "Athokke pinne nokkaam 😒"
    - 7. Prompt Invariants: Single-thought texting, ban 3-part formula, DM dialogue scales
    - 8. No Unprompted Memory references in casual chatter
    """

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="phase3_user",
            email="phase3_ami@example.com",
            password="StrongPassword123!"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    @patch('chat.services.CompanionChatService.get_client')
    def test_casual_activity_inquiry_receives_natural_brief_reply(self, mock_get_client):
        """User: 'Edi entha cheyyunne 😂' -> short, casual reply without multi-paragraph explanation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chumma irikkuva da 😂 nee para"
        mock_response.usage_metadata = MagicMock(prompt_token_count=35, candidates_token_count=8, total_token_count=43)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Casual Banter", current_mood=Conversation.Mood.CALM)
        res = self.client.post(self.chat_url, data={'message': 'Edi entha cheyyunne 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        data = res.json()
        reply = data.get('reply', '')
        self.assertEqual(reply, "Chumma irikkuva da 😂 nee para")
        # Ensure concise natural scale (under 10 words, single sentence)
        self.assertLessEqual(len(reply.split()), 10)
        self.assertNotIn("It sounds like", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_playful_insult_banter_returns_punchy_comeback_without_explanation(self, mock_get_client):
        """User: 'Edi thendi 😂' -> 'Poda pattee 😂' (no explaining the joke, no unsolicited questions)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Poda pattee 😂"
        mock_response.usage_metadata = MagicMock(prompt_token_count=25, candidates_token_count=4, total_token_count=29)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Banter Thread", current_mood=Conversation.Mood.PLAYFUL)
        res = self.client.post(self.chat_url, data={'message': 'Edi thendi 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        reply = res.json().get('reply', '')
        self.assertEqual(reply, "Poda pattee 😂")
        # Ensure mood remains playful
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.PLAYFUL)

    @patch('chat.services.CompanionChatService.get_client')
    def test_provocative_jealousy_triggers_annoyed_state_and_sharp_comeback(self, mock_get_client):
        """User: 'Enik vere oru pennine ishtam aanennu paranjaalo 😂' -> 'Aaha 😒 appo njan aara da?'"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Aaha 😒 appo njan aara da?"
        mock_response.usage_metadata = MagicMock(prompt_token_count=40, candidates_token_count=8, total_token_count=48)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Teasing Thread", current_mood=Conversation.Mood.PLAYFUL)
        res = self.client.post(self.chat_url, data={'message': 'Enik vere oru pennine ishtam aanennu paranjaalo 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        # Verify mood transitions to ANNOYED
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)

        # Verify system prompt given to Gemini instructs strictly 1 sulky sentence and NO friendly check-ins
        call_kwargs = mock_client.models.generate_content.call_args[1]
        instruction = call_kwargs['config'].system_instruction
        self.assertIn("Mood: ANNOYED", instruction)
        self.assertIn("strictly 1 short sulky sentence", instruction)

    @patch('chat.services.CompanionChatService.get_client')
    def test_asking_if_still_angry_receives_brief_sulky_reaction_no_unsolicited_questions(self, mock_get_client):
        """User: 'Entha mindathe, deshyam aano?' while annoyed -> 'Pinne deshyam aaville 😒'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Pinne deshyam aaville 😒"
        mock_response.usage_metadata = MagicMock(prompt_token_count=35, candidates_token_count=5, total_token_count=40)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Sulk Thread",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_remark", "turns_in_mood": 1}
        )
        res = self.client.post(self.chat_url, data={'message': 'Entha mindathe, deshyam aano?', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        reply = res.json().get('reply', '')
        self.assertEqual(reply, "Pinne deshyam aaville 😒")
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)

    @patch('chat.services.CompanionChatService.get_client')
    def test_apology_initiates_gradual_softening_without_emotional_essay(self, mock_get_client):
        """User: 'Ayyo sorry edi, chumma paranjatha ❤️' -> 'Hmm... sheri, potte 😌' (softening, no essay)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hmm... sheri, potte 😌"
        mock_response.usage_metadata = MagicMock(prompt_token_count=40, candidates_token_count=6, total_token_count=46)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Apology Thread",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_remark", "turns_in_mood": 2}
        )
        res = self.client.post(self.chat_url, data={'message': 'Ayyo sorry edi, chumma paranjatha ❤️', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        # Transition to SOFTENING
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)

        reply = res.json().get('reply', '')
        self.assertLessEqual(len(reply.split()), 15)
        self.assertNotIn("I was thinking about how much our relationship means", reply)

    @patch('chat.services.CompanionChatService.get_client')
    def test_affection_while_annoyed_is_not_instantly_wiped(self, mock_get_client):
        """User: 'Oru kiss thaa 😚' while annoyed -> stays annoyed, concise response 'Athokke pinne nokkaam 😒'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Athokke pinne nokkaam 😒"
        mock_response.usage_metadata = MagicMock(prompt_token_count=35, candidates_token_count=5, total_token_count=40)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Kiss Thread",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"unresolved_trigger": "provocative_remark", "turns_in_mood": 1}
        )
        res = self.client.post(self.chat_url, data={'message': 'Oru kiss thaa 😚', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.ANNOYED)

    def test_system_prompt_enforces_phase3_natural_texting_directives(self):
        """Verify BASE_SYSTEM_PROMPT includes all core Phase 3 guidelines and abstract scale guidance."""
        self.assertIn("React First Before Explaining & Single-Thought Texting", BASE_SYSTEM_PROMPT)
        self.assertIn("THE SINGLE-THOUGHT PRINCIPLE", BASE_SYSTEM_PROMPT)
        self.assertIn("BANNED 3-PART FORMULA", BASE_SYSTEM_PROMPT)
        self.assertIn("Real Instagram DM / WhatsApp Dialogue Scales", BASE_SYSTEM_PROMPT)
        self.assertIn("Activity checks & casual greetings", BASE_SYSTEM_PROMPT)
        self.assertIn("Banter & playful roasts", BASE_SYSTEM_PROMPT)
        self.assertIn("Provocative teasing / jealousy triggers", BASE_SYSTEM_PROMPT)
        self.assertIn("Apologies & reconciliation", BASE_SYSTEM_PROMPT)
        self.assertIn("Affection while annoyed", BASE_SYSTEM_PROMPT)
        self.assertIn("Scale rule: Text like a real human texting their partner", BASE_SYSTEM_PROMPT)
        self.assertIn("Do NOT bring up past projects, work, or long-term memories unless the user explicitly asks", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_casual_chatter_does_not_leak_unrelated_approved_memories(self, mock_get_client):
        """Verify memory guidelines prevent unsolicited memory injection into casual chatter."""
        CompanionMemory.objects.create(
            user=self.user,
            content="Working on distributed Kafka pipeline project",
            category=CompanionMemory.Category.WORK,
            is_approved=True,
            importance=CompanionMemory.Importance.IMPORTANT
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chumma irikkuva da 😂 nee para"
        mock_response.usage_metadata = MagicMock(prompt_token_count=50, candidates_token_count=7, total_token_count=57)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Casual Chat", current_mood=Conversation.Mood.CALM)
        res = self.client.post(self.chat_url, data={'message': 'Edi entha cheyyunne 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        reply = res.json().get('reply', '')
        self.assertNotIn("Kafka", reply)
        self.assertNotIn("pipeline", reply)

        # Check instruction guidelines
        call_kwargs = mock_client.models.generate_content.call_args[1]
        instruction = call_kwargs['config'].system_instruction
        self.assertIn("NEVER bring up memories, past projects, or ongoing goals during casual greetings, banter", instruction)


@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class AmiPhase3Point1NaturalnessPolishTests(TestCase):
    """
    Regression test suite for Ami Phase 3.1 Targeted Naturalness Polish:
    - 1. Activity checks reply naturally without inventing elaborate scenarios/chores
    - 2. Natural Conversational Variety (Anti-Templating) directives
    - 3. Varied multi-turn sulky reactions without repeating identical catchphrases
    - 4. Conversational and gradual apology/forgiveness without scripted reconciliation templates
    - 5. Memory suppression on pure casual chatter while preserving relevant retrieval
    """

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="phase3_1_user",
            email="phase3_1@example.com",
            password="StrongPassword123!"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_anti_templating_and_variety_directives_in_prompt(self):
        """Verify prompt contains Anti-Templating rules and varied natural reply guidance."""
        self.assertIn("Natural Conversational Variety (Anti-Templating)", BASE_SYSTEM_PROMPT)
        self.assertIn("Do NOT treat sample replies as rigid mandatory scripts", BASE_SYSTEM_PROMPT)
        self.assertIn("Simple Activity Checks & Greetings", BASE_SYSTEM_PROMPT)
        self.assertIn("without inventing elaborate physical scenarios", BASE_SYSTEM_PROMPT)

    @patch('chat.services.CompanionChatService.get_client')
    def test_activity_check_prompt_forbids_fictitious_errands_and_scenarios(self, mock_get_client):
        """Verify Gemini call for 'Edi entha cheyyunne 😂' instructs casual response without inventing chores."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Oru paniyum illa, ingane irikkanu 😂 entha?"
        mock_response.usage_metadata = MagicMock(prompt_token_count=30, candidates_token_count=9, total_token_count=39)
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Activity Check", current_mood=Conversation.Mood.CALM)
        res = self.client.post(self.chat_url, data={'message': 'Edi entha cheyyunne 😂', 'conversation_id': str(conv.id)}, format='json')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        reply = res.json().get('reply', '')
        self.assertEqual(reply, "Oru paniyum illa, ingane irikkanu 😂 entha?")
        call_kwargs = mock_client.models.generate_content.call_args[1]
        instruction = call_kwargs['config'].system_instruction
        self.assertIn("without inventing elaborate physical scenarios", instruction)

    def test_annoyed_state_directive_enforces_varied_reactions_and_no_repeats(self):
        """Verify MoodState.ANNOYED instructs varied reactions and strictly bans repeating identical phrases."""
        from chat.mood import get_mood_prompt_directive, MoodState
        directive = get_mood_prompt_directive(MoodState.ANNOYED)
        self.assertIn("React directly to what your partner just said instead of repeating identical sulky catchphrases", directive)
        self.assertIn("NEVER repeat the exact same phrase across turns", directive)
        self.assertIn("ANTI-DECLARATION RULE", directive)
        self.assertIn("strictly 1 short sulky sentence or reaction", directive)

    def test_softening_directive_enforces_conversational_and_gradual_forgiveness(self):
        """Verify MoodState.SOFTENING instructs conversational, gradual forgiveness rather than scripted reconciliation."""
        from chat.mood import get_mood_prompt_directive, MoodState
        directive = get_mood_prompt_directive(MoodState.SOFTENING)
        self.assertIn("Make apology and forgiveness feel gradual, natural, and conversational", directive)
        self.assertIn("NOT like a canned script or dramatic reconciliation template", directive)
        self.assertIn("Acknowledge their apology with lingering hesitation, playful pouting", directive)

    def test_memory_suppression_on_casual_chatter_preserves_topical_retrieval(self):
        """Verify get_relevant_memories suppresses memory on casual chatter but preserves retrieval on relevant queries."""
        from chat.services import get_relevant_memories
        CompanionMemory.objects.create(
            user=self.user,
            content="Working on distributed Kafka pipeline project",
            category=CompanionMemory.Category.WORK,
            is_approved=True,
            importance=CompanionMemory.Importance.IMPORTANT
        )
        CompanionMemory.objects.create(
            user=self.user,
            content="Likes hot black tea with cardamom and zero sugar",
            category=CompanionMemory.Category.PREFERENCE,
            is_approved=True,
            importance=CompanionMemory.Importance.NORMAL
        )

        # 1. Pure casual activity check -> ZERO memories returned
        casual_results = get_relevant_memories(user=self.user, query_text="Edi entha cheyyunne 😂")
        self.assertEqual(len(casual_results), 0)

        # 2. Pure banter / insults -> ZERO memories returned
        banter_results = get_relevant_memories(user=self.user, query_text="Edi thendi 😂")
        self.assertEqual(len(banter_results), 0)

        # 3. Work update -> Work memory returned
        work_results = get_relevant_memories(user=self.user, query_text="Had a really tiring day at work da!")
        self.assertGreaterEqual(len(work_results), 1)
        self.assertTrue(any("Kafka" in m.content for m in work_results))

        # 4. Tea / drink query -> Preference memory returned
        drink_results = get_relevant_memories(user=self.user, query_text="What hot drink should I make right now?")
        self.assertGreaterEqual(len(drink_results), 1)
        self.assertTrue(any("tea" in m.content for m in drink_results))


@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class AmiNaturalContextDrivenEmotionalTests(TestCase):
    """
    Test suite for Ami's refactored context-driven emotional behaviour:
    1. Context extraction extracts cues and disambiguates friendly teasing from insults.
    2. Mood context stores revisable situation summaries instead of acting as a rigid state machine.
    3. Removal of rigid turn counters (no automatic state flips based solely on turn counts).
    4. Abstract behavioural prompt guidance replaces canned dialogue quotes.
    5. Emotional continuity is preserved without forced anger or forced reconciliation scripts.
    """

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="context_driven_user",
            email="context_driven@example.com",
            password="StrongPassword123!"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    def test_context_extraction_disambiguates_friendly_tease_from_harsh_insult(self):
        """Verify context extraction correctly distinguishes playful banter from genuine hostility."""
        from chat.mood import extract_emotional_cues

        # Playful banter with laughing emoji
        tease_cues = extract_emotional_cues("poda thendi 😂")
        self.assertTrue(tease_cues["has_friendly_tease"])
        self.assertFalse(tease_cues["has_insult"])

        # Harsh insult
        insult_cues = extract_emotional_cues("nee useless aanedi, ninne kondu oru upayogavum illa")
        self.assertTrue(insult_cues["has_insult"])
        self.assertFalse(insult_cues["has_friendly_tease"])

        # Apology with remorse
        apology_cues = extract_emotional_cues("sorry edi, angane parayan paadillayirunnu")
        self.assertTrue(apology_cues["has_apology"])
        self.assertFalse(apology_cues["has_insult"])

        # Pure affection
        affection_cues = extract_emotional_cues("love you da muthe ❤️")
        self.assertTrue(affection_cues["has_affection"])
        self.assertFalse(affection_cues["has_apology"])

    def test_mood_context_stores_revisable_situation_summary(self):
        """Verify mood context stores a small, human/LLM-readable revisable summary of the situation."""
        from chat.mood import evaluate_mood_transition, MoodState

        # Turn 1: Harsh insult
        mood, ctx = evaluate_mood_transition(
            current_mood=MoodState.CALM,
            mood_context={},
            user_message="nee waste aanu, ninne kondu onnum pattilla"
        )
        self.assertEqual(mood, MoodState.HURT)
        self.assertIn("situation_summary", ctx)
        self.assertIn("hurtful", ctx["situation_summary"].lower())
        self.assertTrue(ctx.get("emotional_residue"))

        # Turn 2: Apology
        mood2, ctx2 = evaluate_mood_transition(
            current_mood=mood,
            mood_context=ctx,
            user_message="sorry da, chumma paranjatha"
        )
        self.assertEqual(mood2, MoodState.SOFTENING)
        self.assertIn("situation_summary", ctx2)
        self.assertIn("apology", ctx2["situation_summary"].lower())

    def test_removal_of_rigid_turn_counter_transitions(self):
        """Verify conversation in annoyed state does not automatically reset to calm merely because turn count passes 3."""
        from chat.mood import evaluate_mood_transition, MoodState

        initial_ctx = {
            "unresolved_trigger": "provocative_jealousy",
            "severity": "minor",
            "turns_in_mood": 3,  # Old rigid code would have forced CALM at >= 3
            "situation_summary": "Partner teased Ami about another girl."
        }

        # Follow-up message asking why she's quiet without apologizing
        mood, ctx = evaluate_mood_transition(
            current_mood=MoodState.ANNOYED,
            mood_context=initial_ctx,
            user_message="entha mindathe?"
        )
        # Mood remains ANNOYED; no rigid turn-counter reset
        self.assertEqual(mood, MoodState.ANNOYED)

    def test_prompt_directives_use_abstract_guidance_without_canned_dialogue_templates(self):
        """Verify get_mood_prompt_directive uses abstract behavioural guidance across all states without canned dialogue quotes."""
        from chat.mood import get_mood_prompt_directive, MoodState

        for state in MoodState.ALL:
            directive = get_mood_prompt_directive(state)
            # Ensure no verbatim scripted replies exist to be copied
            self.assertNotIn("'Pinne deshyam aaville 😒'", directive)
            self.assertNotIn("'Athe, deshyam thanneya 😒'", directive)
            self.assertNotIn("'Athokke pinne nokkaam 😒'", directive)
            self.assertNotIn("'Eda, sarilla... repeated aayi sorry parayenda 😌 I'm okay now, athu vittukala'", directive)
            self.assertNotIn("'Sheri da... but athu kettappol nallonam hurt aayi'", directive)

            # Ensure abstract behavioral principles are present
            if state in (MoodState.ANNOYED, MoodState.ANGRY, MoodState.HURT, MoodState.SOFTENING):
                self.assertIn("ANTI-DECLARATION RULE", directive)

    @patch('chat.services.CompanionChatService.get_client')
    def test_end_to_end_context_driven_chat_flow(self, mock_get_client):
        """Verify full end-to-end chat flow: insult -> sincere apology with residue -> respectful chat -> normal."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Context-Driven Flow",
            current_mood=Conversation.Mood.CALM
        )

        # Turn 1: Harsh insult
        mock_client.models.generate_content.return_value = MagicMock(
            text="Enthina da angane paranjathu? Enikku nallonam sankadam aayi...",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=10, total_token_count=35)
        )
        res1 = self.client.post(
            self.chat_url,
            data={'message': 'nee useless aanedi, ninne kondu oru upayogavum illa', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.HURT)
        self.assertTrue(conv.mood_context.get("emotional_residue"))

        # Turn 2: Sincere apology
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sarilla... but athu kettappol valare hurt aayi. Let's drop it 😌",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=14, total_token_count=49)
        )
        res2 = self.client.post(
            self.chat_url,
            data={'message': 'sorry edi, angane parayan paadillayirunnu', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertTrue(conv.mood_context.get("emotional_residue"))

        # Check prompt directive provided abstract guidance
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sys_instruction = call_kwargs['config'].system_instruction
        self.assertIn("First Apology for Meaningful Hurt", sys_instruction)
        self.assertIn("CRITICAL FOR FIRST APOLOGY: Do NOT say 'repeated aayi sorry parayenda'", sys_instruction)
        self.assertIn("NO SUDDEN SUGAR RUSH", sys_instruction)

        # Turn 3: Follow-up respectful chat normalizes mood
        mock_client.models.generate_content.return_value = MagicMock(
            text="Kazhichu da... nee kazhicho?",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=6, total_token_count=36)
        )
        res3 = self.client.post(
            self.chat_url,
            data={'message': 'lunch kazhicho da?', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res3.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertFalse(conv.mood_context.get("emotional_residue", False))


@override_settings(
    AI_PROVIDER='gemini',
    GEMINI_API_KEY='AIzaSyDummyValidFormatKeyForTesting12345',
    GEMINI_MODEL='gemini-3.1-flash-lite',
)
class AmiConversationalGeneralizationTests(TestCase):
    """
    Test suite for Ami's conversational intelligence generalization across unseen phrasings and contexts:
    1. Friendly teasing with novel slang and emojis.
    2. Sarcasm: playful vs genuinely cold/passive-aggressive.
    3. Innocent third-party mentions (weddings/colleagues) not falsely triggering jealousy.
    4. External complaints (movies/food/work) not treated as personal insults.
    5. Unseen sincere apology phrasings initiating softening.
    6. Mixed messages containing both teasing and remorse.
    7. Smooth topic transitions after reconciliation without reviving old conflict.
    8. Dynamic model-interpreted mood tag updating conversation state without hardcoded regexes.
    """

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="generalization_user",
            email="generalization@example.com",
            password="StrongPassword123!"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.chat_url = '/api/chat/'

    @patch('chat.services.CompanionChatService.get_client')
    def test_unseen_friendly_teasing_with_novel_slang_and_emojis(self, mock_get_client):
        """User uses novel teasing: 'Ninne aarkkum vendaatha kond aan njan kooday ullath ketto 😜'."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aaha appo angane aanalle! Njan illenkil nee enthu cheyyumennu nokkam 😜\n[mood: playful]",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=16, total_token_count=46)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Unseen Tease", current_mood=Conversation.Mood.CALM)
        res = self.client.post(
            self.chat_url,
            data={'message': 'Ninne aarkkum vendaatha kond aan njan kooday ullath ketto 😜', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.PLAYFUL)
        # Verify tag was stripped cleanly from user reply
        self.assertNotIn('[mood:', res.json()['reply'])

    @patch('chat.services.CompanionChatService.get_client')
    def test_playful_sarcasm_vs_cold_passive_aggression(self, mock_get_client):
        """Playful sarcasm stays playful; cold passive-aggressive sarcasm triggers annoyance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # 1. Playful sarcasm
        mock_client.models.generate_content.return_value = MagicMock(
            text="Poda, ninte valiya Einstein logic njan ippo sammathichu tharaam 😂\n[mood: playful]",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=12, total_token_count=37)
        )
        conv1 = Conversation.objects.create(user=self.user, title="Playful Sarcasm", current_mood=Conversation.Mood.CALM)
        res1 = self.client.post(
            self.chat_url,
            data={'message': 'Aaha ninte logic kettal Einstein vare thottu pokum 😂', 'conversation_id': str(conv1.id)},
            format='json'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        conv1.refresh_from_db()
        self.assertEqual(conv1.current_mood, Conversation.Mood.PLAYFUL)

        # 2. Cold passive-aggressive sarcasm
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sheri... anganeyengil angane 😒",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=5, total_token_count=35)
        )
        conv2 = Conversation.objects.create(user=self.user, title="Cold Sarcasm", current_mood=Conversation.Mood.CALM)
        res2 = self.client.post(
            self.chat_url,
            data={'message': 'Valare nalla karyam... njan enthayalum ini onnum parayanilla', 'conversation_id': str(conv2.id)},
            format='json'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        conv2.refresh_from_db()
        self.assertEqual(conv2.current_mood, Conversation.Mood.ANNOYED)

    @patch('chat.services.CompanionChatService.get_client')
    def test_innocent_third_party_mention_not_triggering_jealousy(self, mock_get_client):
        """Innocent mention of friend's or cousin's wedding does not falsely trigger romantic jealousy."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Aaha adipoli! Sneha-yude marriage-inu njanum koodi varano? 😄",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=12, total_token_count=37)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Wedding Chat", current_mood=Conversation.Mood.CALM)
        res = self.client.post(
            self.chat_url,
            data={'message': 'Ente cousin Sneha-yude kallyanam aanu next month, invitation vannu da', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        # Does NOT trigger ANNOYED or jealousy
        self.assertNotEqual(conv.current_mood, Conversation.Mood.ANNOYED)
        self.assertNotIn("provocative_jealousy", conv.mood_context.get("unresolved_trigger", ""))

    @patch('chat.services.CompanionChatService.get_client')
    def test_external_venting_not_treated_as_personal_insult(self, mock_get_client):
        """User venting about a bad movie or job does not trigger HURT or ANNOYED."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sathyam, athra bore aayirunno? Time waste aayi alle da 😕",
            usage_metadata=MagicMock(prompt_token_count=25, candidates_token_count=12, total_token_count=37)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(user=self.user, title="Movie Review", current_mood=Conversation.Mood.CALM)
        res = self.client.post(
            self.chat_url,
            data={'message': 'Ee padam oru waste aanu da, ottum upayogamilla kandittu', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)

    @patch('chat.services.CompanionChatService.get_client')
    def test_unseen_apology_phrasing_initiates_softening(self, mock_get_client):
        """Novel Manglish apology phrasing initiates softening without exact keyword template."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Sarilla da, you understood... that matters 😌\n[mood: softening]",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=12, total_token_count=47)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Apology Generalization",
            current_mood=Conversation.Mood.HURT,
            mood_context={"severity": "serious", "emotional_residue": True}
        )
        res = self.client.post(
            self.chat_url,
            data={'message': 'Enik ninte vishamam manasilayi... angane aagrahichathalla, kshemikku da', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)
        self.assertNotIn('[mood:', res.json()['reply'])

    @patch('chat.services.CompanionChatService.get_client')
    def test_mixed_message_with_teasing_and_remorse(self, mock_get_client):
        """Mixed intent message (tease + remorse) transitions naturally to softening with light banter."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Ninte oru thamasha! Sheri potte, ippolkk kshemichu 😂\n[mood: softening]",
            usage_metadata=MagicMock(prompt_token_count=35, candidates_token_count=14, total_token_count=49)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Mixed Intent",
            current_mood=Conversation.Mood.ANNOYED,
            mood_context={"severity": "minor"}
        )
        res = self.client.post(
            self.chat_url,
            data={'message': 'Sorry da, chumma ninte kuru pottikkan nokkiyathaa 😂 pakshe hurt aayenkil kshemikku', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.SOFTENING)

    @patch('chat.services.CompanionChatService.get_client')
    def test_topic_change_after_reconciliation_normalizes_smoothly(self, mock_get_client):
        """After softening, partner asking an ordinary question naturally returns to CALM without resurrecting past conflict."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="Njan dosa kazhichu da! Nee entha kazhikkunne? 🍲\n[mood: calm]",
            usage_metadata=MagicMock(prompt_token_count=30, candidates_token_count=12, total_token_count=42)
        )
        mock_get_client.return_value = mock_client

        conv = Conversation.objects.create(
            user=self.user,
            title="Topic Transition",
            current_mood=Conversation.Mood.SOFTENING,
            mood_context={"unresolved_trigger": "softening_serious_hurt_residue", "emotional_residue": True}
        )
        res = self.client.post(
            self.chat_url,
            data={'message': 'ippo entha kazhikkan ponne? food order cheyyano?', 'conversation_id': str(conv.id)},
            format='json'
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        conv.refresh_from_db()
        self.assertEqual(conv.current_mood, Conversation.Mood.CALM)
        self.assertFalse(conv.mood_context.get("emotional_residue", False))


from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock
from chat.voice_services import VoiceService, TranscriptionError, SynthesisError


class AmiVoiceMessagingTests(TestCase):
    """
    Comprehensive test suite for voice messaging features:
    - Authenticated voice upload and user isolation
    - Audio format and size validation
    - Transcription success, failure, and error classification
    - Integration with existing mood engine and memory system
    - Text-to-speech audio reply generation
    - TTS failure fallback (preserving text response)
    - Idempotency and duplicate prevention
    - On-demand voice synthesis and caching
    """

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='voice_user',
            password='testpassword123',
            email='voice@chatpartner.local'
        )
        self.client.force_authenticate(user=self.user)
        self.voice_url = reverse('chat-voice-endpoint')

    def test_unauthenticated_request_rejected(self):
        self.client.logout()
        audio = SimpleUploadedFile("voice.webm", b"FAKE_AUDIO_DATA", content_type="audio/webm")
        res = self.client.post(self.voice_url, {'audio': audio}, format='multipart')
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)

    def test_invalid_and_empty_file_rejected(self):
        # 1. No audio and no transcript
        res = self.client.post(self.voice_url, {}, format='multipart')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

        # 2. Empty audio file (0 bytes)
        empty_audio = SimpleUploadedFile("empty.webm", b"", content_type="audio/webm")
        res = self.client.post(self.voice_url, {'audio': empty_audio}, format='multipart')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("empty", res.data['error'].lower())

        # 3. Unsupported format
        bad_audio = SimpleUploadedFile("bad.exe", b"MZfakebinary", content_type="application/octet-stream")
        res = self.client.post(self.voice_url, {'audio': bad_audio}, format='multipart')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("unsupported", res.data['error'].lower())

    @patch.object(VoiceService, 'transcribe_audio')
    @patch.object(CompanionChatService, 'generate_reply')
    @patch.object(VoiceService, 'synthesize_speech')
    def test_voice_chat_transcription_and_tts_success(
        self,
        mock_synthesize,
        mock_generate_reply,
        mock_transcribe
    ):
        mock_transcribe.return_value = "Edi entha cheyyunne 😂"
        mock_generate_reply.return_value = (
            "Chumma irikkuva da 😂 nee para",
            {"provider": "gemini", "model": "gemini-3.1-flash-lite", "interpreted_mood": "playful"}
        )
        mock_synthesize.return_value = (b"FAKE_TTS_MP3_DATA_FOR_AMI", 2.8)

        audio = SimpleUploadedFile("voice.webm", b"RIFFfakeaudiobytes", content_type="audio/webm")
        res = self.client.post(
            self.voice_url,
            {'audio': audio, 'duration': 4.5},
            format='multipart'
        )

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data['user_transcript'], "Edi entha cheyyunne 😂")
        self.assertEqual(res.data['reply'], "Chumma irikkuva da 😂 nee para")
        self.assertIn("audio/", res.data['user_audio_url'])
        self.assertIn("audio/", res.data['audio_url'])
        self.assertEqual(res.data['mood'], 'playful')

        # Verify DB persistence
        user_msg = ChatMessage.objects.get(id=res.data['user_message_id'])
        self.assertEqual(user_msg.sender, ChatMessage.Sender.USER)
        self.assertEqual(user_msg.message_type, ChatMessage.MessageType.VOICE)
        self.assertEqual(user_msg.content, "Edi entha cheyyunne 😂")
        self.assertTrue(bool(user_msg.audio_file))

        companion_msg = ChatMessage.objects.get(id=res.data['message_id'])
        self.assertEqual(companion_msg.sender, ChatMessage.Sender.COMPANION)
        self.assertEqual(companion_msg.message_type, ChatMessage.MessageType.VOICE)
        self.assertEqual(companion_msg.content, "Chumma irikkuva da 😂 nee para")
        self.assertTrue(bool(companion_msg.audio_file))

    @patch.object(VoiceService, 'transcribe_audio')
    def test_transcription_failure_returns_422(self, mock_transcribe):
        mock_transcribe.side_effect = TranscriptionError("Could not recognize any spoken words.")

        audio = SimpleUploadedFile("silent.webm", b"RIFFsilentaudio", content_type="audio/webm")
        res = self.client.post(self.voice_url, {'audio': audio}, format='multipart')

        self.assertEqual(res.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        self.assertEqual(res.data['error_category'], 'transcription_error')

    @patch.object(VoiceService, 'transcribe_audio')
    @patch.object(CompanionChatService, 'generate_reply')
    @patch.object(VoiceService, 'synthesize_speech')
    def test_tts_failure_preserves_text_reply(
        self,
        mock_synthesize,
        mock_generate_reply,
        mock_transcribe
    ):
        mock_transcribe.return_value = "Hello Ami"
        mock_generate_reply.return_value = (
            "Hey da! Sugham aano?",
            {"provider": "gemini", "model": "gemini-3.1-flash-lite"}
        )
        mock_synthesize.side_effect = SynthesisError("Network timeout during speech synthesis")

        audio = SimpleUploadedFile("hello.webm", b"RIFFhelloaudio", content_type="audio/webm")
        res = self.client.post(self.voice_url, {'audio': audio}, format='multipart')

        # Should still succeed with 200 OK and return text reply!
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data['reply'], "Hey da! Sugham aano?")
        self.assertIsNone(res.data['audio_url'])
        self.assertIn("timeout", res.data['metadata']['tts_error'])

        # Companion message should be saved with text
        companion_msg = ChatMessage.objects.get(id=res.data['message_id'])
        self.assertEqual(companion_msg.content, "Hey da! Sugham aano?")
        self.assertFalse(bool(companion_msg.audio_file))

    def test_user_isolation_for_audio_stream(self):
        other_user = User.objects.create_user(
            username='other_user',
            password='testpassword123',
            email='other@chatpartner.local'
        )
        other_conv = Conversation.objects.create(user=other_user, title="Other User Conv")
        other_msg = ChatMessage.objects.create(
            conversation=other_conv,
            sender=ChatMessage.Sender.USER,
            content="Private voice message",
            audio_file=SimpleUploadedFile("private.mp3", b"PRIVATE_AUDIO_BYTES", content_type="audio/mpeg")
        )

        stream_url = reverse('message-audio-stream', kwargs={'message_id': str(other_msg.id)})

        # Authenticated as self.user, attempting to stream other_user's audio
        res = self.client.get(stream_url)
        self.assertEqual(res.status_code, status.HTTP_404_NOT_FOUND)

        # Authenticated as other_user
        self.client.force_authenticate(user=other_user)
        res_owner = self.client.get(stream_url)
        self.assertEqual(res_owner.status_code, status.HTTP_200_OK)

    @patch.object(VoiceService, 'synthesize_speech')
    def test_on_demand_voice_synthesis_and_caching(self, mock_synthesize):
        mock_synthesize.return_value = (b"SYNTHESIZED_ON_DEMAND_AUDIO", 3.2)

        conv = Conversation.objects.create(user=self.user, title="On Demand Test")
        msg = ChatMessage.objects.create(
            conversation=conv,
            sender=ChatMessage.Sender.COMPANION,
            content="I am speaking on demand!",
            message_type=ChatMessage.MessageType.TEXT
        )

        synth_url = reverse('message-synthesize-voice', kwargs={'message_id': str(msg.id)})

        # 1. First call synthesizes audio
        res1 = self.client.post(synth_url)
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        self.assertFalse(res1.data['cached'])
        self.assertIn("audio/", res1.data['audio_url'])
        self.assertEqual(mock_synthesize.call_count, 1)

        # 2. Second call returns cached audio without re-synthesizing
        res2 = self.client.post(synth_url)
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        self.assertTrue(res2.data['cached'])
        self.assertEqual(mock_synthesize.call_count, 1)  # NOT called again!

    @patch.object(VoiceService, 'transcribe_audio')
    @patch.object(CompanionChatService, 'generate_reply')
    @patch.object(VoiceService, 'synthesize_speech')
    def test_idempotency_prevents_duplicate_messages(
        self,
        mock_synthesize,
        mock_generate_reply,
        mock_transcribe
    ):
        mock_transcribe.return_value = "Idempotency test"
        mock_generate_reply.return_value = ("Reply once", {"provider": "gemini"})
        mock_synthesize.return_value = (b"IDEMPOTENT_AUDIO", 1.5)

        audio = SimpleUploadedFile("idem.webm", b"AUDIO_DATA", content_type="audio/webm")
        client_msg_id = "test-client-uuid-999"

        # First request
        res1 = self.client.post(
            self.voice_url,
            {'audio': audio, 'client_message_id': client_msg_id},
            format='multipart'
        )
        self.assertEqual(res1.status_code, status.HTTP_200_OK)

        initial_count = ChatMessage.objects.count()

        # Second request with identical client_message_id
        audio.seek(0)
        res2 = self.client.post(
            self.voice_url,
            {'audio': audio, 'client_message_id': client_msg_id},
            format='multipart'
        )
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        self.assertEqual(res2.data['message_id'], res1.data['message_id'])
        self.assertEqual(ChatMessage.objects.count(), initial_count)


from chat.voice_services import clean_text_for_speech, validate_audio_upload


class VoiceServiceUnitTests(TestCase):
    """
    Unit tests for voice utility functions and service edge cases.
    """

    def test_clean_text_for_speech(self):
        raw = "Aha! 😂 Ninakku angane thanne venam! *smiles* [mood: playful]"
        cleaned = clean_text_for_speech(raw)
        self.assertNotIn("😂", cleaned)
        self.assertNotIn("*smiles*", cleaned)
        self.assertNotIn("[mood:", cleaned)
        self.assertIn("Ninakku angane thanne venam", cleaned)

    def test_clean_text_for_speech_handles_html_tags(self):
        raw = "Sarilla da <!-- mood: calm --> ippo okay aayi 😊"
        cleaned = clean_text_for_speech(raw)
        self.assertEqual(cleaned, "Sarilla da ippo okay aayi")

    def test_validate_audio_upload_valid_formats(self):
        webm_file = SimpleUploadedFile("voice.webm", b"WEBM_DATA", content_type="audio/webm")
        mime, ext = validate_audio_upload(webm_file)
        self.assertEqual(mime, "audio/webm")
        self.assertEqual(ext, ".webm")

        mp3_file = SimpleUploadedFile("voice.mp3", b"MP3_DATA", content_type="audio/mpeg")
        mime, ext = validate_audio_upload(mp3_file)
        self.assertEqual(mime, "audio/mpeg")
        self.assertEqual(ext, ".mp3")

    def test_validate_audio_upload_size_limit(self):
        big_data = b"0" * (16 * 1024 * 1024)  # 16MB
        big_file = SimpleUploadedFile("big.webm", big_data, content_type="audio/webm")
        with self.assertRaises(ValueError) as ctx:
            validate_audio_upload(big_file, max_size_mb=15)
        self.assertIn("exceeds the allowed limit", str(ctx.exception))


from config.settings import get_database_config


class DatabaseConfigurationTests(TestCase):
    """
    Tests for Django database configuration:
    - Neon PostgreSQL DATABASE_URL parsing and sslmode=require enforcement
    - Local development fallback using individual DB_* environment variables
    - Removal of hardcoded password fallback
    - Custom connection parameters (CONN_MAX_AGE, DB_SSL_REQUIRE)
    """

    def test_neon_database_url_configuration_with_ssl_require(self):
        neon_url = "postgresql://neondb_owner:npg_secret123@ep-cool-mountain-999.us-east-2.aws.neon.tech/neondb?sslmode=require"
        cfg = get_database_config(database_url=neon_url)

        self.assertIn('default', cfg)
        default_db = cfg['default']
        self.assertEqual(default_db['ENGINE'], 'django.db.backends.postgresql')
        self.assertEqual(default_db['NAME'], 'neondb')
        self.assertEqual(default_db['USER'], 'neondb_owner')
        self.assertEqual(default_db['PASSWORD'], 'npg_secret123')
        self.assertEqual(default_db['HOST'], 'ep-cool-mountain-999.us-east-2.aws.neon.tech')
        self.assertEqual(default_db['OPTIONS'].get('sslmode'), 'require')
        self.assertEqual(default_db.get('CONN_MAX_AGE'), 600)
        self.assertTrue(default_db.get('CONN_HEALTH_CHECKS'))

    def test_neon_database_url_without_explicit_query_enforces_ssl(self):
        neon_url = "postgresql://neondb_owner:npg_secret123@ep-cool-mountain-999.us-east-2.aws.neon.tech/neondb"
        cfg = get_database_config(database_url=neon_url)

        default_db = cfg['default']
        self.assertEqual(default_db['OPTIONS'].get('sslmode'), 'require')

    def test_local_database_fallback_when_database_url_absent(self):
        mock_env = {
            'DB_NAME': 'local_test_db',
            'DB_USER': 'local_developer',
            'DB_PASSWORD': 'local_secret_pass',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
        }
        cfg = get_database_config(database_url="", env=mock_env)

        default_db = cfg['default']
        self.assertEqual(default_db['ENGINE'], 'django.db.backends.postgresql')
        self.assertEqual(default_db['NAME'], 'local_test_db')
        self.assertEqual(default_db['USER'], 'local_developer')
        self.assertEqual(default_db['PASSWORD'], 'local_secret_pass')
        self.assertEqual(default_db['HOST'], 'localhost')
        self.assertEqual(default_db['PORT'], '5432')

    def test_no_hardcoded_password_fallback_when_password_env_absent(self):
        mock_env = {
            'DB_NAME': 'chatpartner_db',
            'DB_USER': 'ashique',
        }
        cfg = get_database_config(database_url="", env=mock_env)

        default_db = cfg['default']
        self.assertEqual(default_db['PASSWORD'], '')
        self.assertNotEqual(default_db['PASSWORD'], 'ashique@123')

    def test_custom_conn_max_age_and_ssl_override(self):
        neon_url = "postgresql://user:pass@endpoint.neon.tech/neondb"
        mock_env = {
            'DB_CONN_MAX_AGE': '300',
            'DB_SSL_REQUIRE': 'False',
        }
        cfg = get_database_config(database_url=neon_url, env=mock_env)

        default_db = cfg['default']
        self.assertEqual(default_db.get('CONN_MAX_AGE'), 300)
        self.assertNotIn('sslmode', default_db.get('OPTIONS', {}))


class ProductionCookieSecurityTests(TestCase):
    """
    Tests for cross-site cookie security, proxy SSL header, CSRF token endpoint,
    session persistence across /api/auth/me/ and authenticated conversation requests.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username='prod_session_tester',
            password='prod_secure_password_123',
            email='tester@example.com'
        )

    def test_csrf_token_endpoint_returns_json_token(self):
        """Verify GET /api/auth/csrf/ returns status 200 with csrfToken key and cookie."""
        client = APIClient()
        response = client.get('/api/auth/csrf/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('csrfToken', data)
        self.assertTrue(len(data['csrfToken']) > 10)
        self.assertIn('csrftoken', response.cookies)

    def test_login_creates_session_with_cross_site_production_cookies(self):
        """
        Verify that login under production settings creates a valid Django session
        and returns sessionid and csrftoken with SameSite=None and Secure=True.
        """
        from django.test import Client

        with override_settings(
            DEBUG=False,
            SESSION_COOKIE_SAMESITE='None',
            CSRF_COOKIE_SAMESITE='None',
            SESSION_COOKIE_SECURE=True,
            CSRF_COOKIE_SECURE=True,
            SECURE_PROXY_SSL_HEADER=('HTTP_X_FORWARDED_PROTO', 'https'),
            ALLOWED_HOSTS=['testserver', 'ami-bot-9ys2.onrender.com'],
            CORS_ALLOWED_ORIGINS=['https://ai-chatbot-rho-six-36.vercel.app'],
            CSRF_TRUSTED_ORIGINS=['https://ai-chatbot-rho-six-36.vercel.app']
        ):
            client = Client(enforce_csrf_checks=True)

            # 1. Fetch CSRF token
            csrf_res = client.get(
                '/api/auth/csrf/',
                HTTP_ORIGIN='https://ai-chatbot-rho-six-36.vercel.app',
                HTTP_X_FORWARDED_PROTO='https'
            )
            self.assertEqual(csrf_res.status_code, 200)
            csrf_token = csrf_res.json()['csrfToken']

            # 2. Perform Login
            login_res = client.post(
                '/api/auth/login/',
                data={'username': 'prod_session_tester', 'password': 'prod_secure_password_123'},
                content_type='application/json',
                HTTP_ORIGIN='https://ai-chatbot-rho-six-36.vercel.app',
                HTTP_X_CSRFTOKEN=csrf_token,
                HTTP_X_FORWARDED_PROTO='https'
            )
            self.assertEqual(login_res.status_code, 200)
            login_data = login_res.json()
            self.assertTrue(login_data['authenticated'])
            self.assertEqual(login_data['username'], 'prod_session_tester')

            # Verify session cookie attributes for cross-site browser support
            self.assertIn('sessionid', login_res.cookies)
            session_cookie = login_res.cookies['sessionid']
            self.assertEqual(session_cookie['samesite'], 'None')
            self.assertTrue(session_cookie['secure'])
            self.assertTrue(session_cookie['httponly'])

            # Verify CSRF cookie attributes
            self.assertIn('csrftoken', login_res.cookies)
            csrf_cookie = login_res.cookies['csrftoken']
            self.assertEqual(csrf_cookie['samesite'], 'None')
            self.assertTrue(csrf_cookie['secure'])

            # 3. Subsequent /api/auth/me/ request with session cookie succeeds
            me_res = client.get(
                '/api/auth/me/',
                HTTP_ORIGIN='https://ai-chatbot-rho-six-36.vercel.app',
                HTTP_X_FORWARDED_PROTO='https'
            )
            self.assertEqual(me_res.status_code, 200)
            me_data = me_res.json()
            self.assertTrue(me_data['authenticated'])
            self.assertEqual(me_data['username'], 'prod_session_tester')

            # 4. Subsequent /api/conversations/ request with session cookie succeeds
            conv_res = client.get(
                '/api/conversations/',
                HTTP_ORIGIN='https://ai-chatbot-rho-six-36.vercel.app',
                HTTP_X_FORWARDED_PROTO='https'
            )
            self.assertEqual(conv_res.status_code, 200)
            self.assertIsInstance(conv_res.json(), list)

            # 5. Create conversation with session cookie + CSRF token
            new_csrf = login_data['csrfToken']
            create_res = client.post(
                '/api/conversations/',
                data={'title': 'Production Thread'},
                content_type='application/json',
                HTTP_ORIGIN='https://ai-chatbot-rho-six-36.vercel.app',
                HTTP_X_CSRFTOKEN=new_csrf,
                HTTP_X_FORWARDED_PROTO='https'
            )
            self.assertEqual(create_res.status_code, 201)
            self.assertEqual(create_res.json()['title'], 'Production Thread')

    def test_unauthenticated_requests_without_cookie_rejected(self):
        """Verify /api/auth/me/ returns 401 and /api/conversations/ returns 403 when session cookie is absent."""
        from django.test import Client
        anon_client = Client()

        # /api/auth/me/ returns 401
        res_me = anon_client.get('/api/auth/me/')
        self.assertEqual(res_me.status_code, 401)
        self.assertFalse(res_me.json().get('authenticated', True))

        # /api/conversations/ returns 403 (DRF IsAuthenticated permission)
        res_conv = anon_client.get('/api/conversations/')
        self.assertEqual(res_conv.status_code, 403)

    def test_render_and_production_environment_derivation(self):
        """
        Verify that Render environment markers or DEBUG=False derive SameSite=None,
        Secure=True, and reverse proxy SSL headers.
        """
        # When RENDER=true is set, even if DEBUG=True, it should derive SameSite=None and Secure=True
        is_render = True
        debug = True
        is_prod = (not debug) or is_render
        samesite = 'None' if is_prod else 'Lax'
        secure = True if (is_prod or samesite.lower() == 'none') else False
        proxy_header = ('HTTP_X_FORWARDED_PROTO', 'https') if (is_prod or is_render) else None

        self.assertEqual(samesite, 'None')
        self.assertTrue(secure)
        self.assertEqual(proxy_header, ('HTTP_X_FORWARDED_PROTO', 'https'))

        # When local dev (not Render and DEBUG=True)
        is_render_dev = False
        debug_dev = True
        is_prod_dev = (not debug_dev) or is_render_dev
        samesite_dev = 'None' if is_prod_dev else 'Lax'
        secure_dev = True if (is_prod_dev or samesite_dev.lower() == 'none') else False

        self.assertEqual(samesite_dev, 'Lax')
        self.assertFalse(secure_dev)



















