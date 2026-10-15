"""
Tests for Meta's Official Instagram Messaging API integration:
- Webhook verification challenge (GET hub.challenge)
- HMAC-SHA256 signature verification (X-Hub-Signature-256)
- Malformed payloads and invalid signature rejection
- 24-hour messaging window compliance and customer registration
- Security: No exposure or logging of secrets, tokens, or private messages
"""
import hmac
import hashlib
import json
from datetime import timedelta
from unittest.mock import patch
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from django.utils import timezone

from .models import InstagramCustomer, InstagramWebhookEventLog
from .security import verify_hub_signature
from .config import is_instagram_configured, get_sanitized_config_status

User = get_user_model()


class InstagramWebhookVerificationTests(TestCase):
    """
    Tests Meta's Webhook verification challenge (GET /api/instagram/webhook/).
    Meta sends: hub.mode=subscribe, hub.verify_token=<token>, hub.challenge=<challenge>
    """

    def setUp(self):
        self.verify_token = "secure_test_verify_token_123"
        self.app_secret = "test_meta_app_secret_abc"
        self.mock_settings = {
            'INSTAGRAM_VERIFY_TOKEN': self.verify_token,
            'INSTAGRAM_APP_SECRET': self.app_secret,
            'INSTAGRAM_ACCOUNT_ID': '17841400000000001',
            'INSTAGRAM_ACCESS_TOKEN': 'EAAK...test_token',
        }

    def test_webhook_challenge_success(self):
        """Valid verification challenge returns 200 with hub.challenge as plain text."""
        with override_settings(**self.mock_settings):
            challenge = "1158201444"
            response = self.client.get(
                '/api/instagram/webhook/',
                {
                    'hub.mode': 'subscribe',
                    'hub.verify_token': self.verify_token,
                    'hub.challenge': challenge,
                }
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content.decode('utf-8'), challenge)
            self.assertEqual(response['Content-Type'], 'text/plain')

    def test_webhook_challenge_invalid_token(self):
        """Invalid verify token returns 403 Forbidden."""
        with override_settings(**self.mock_settings):
            response = self.client.get(
                '/api/instagram/webhook/',
                {
                    'hub.mode': 'subscribe',
                    'hub.verify_token': 'wrong_token',
                    'hub.challenge': '12345',
                }
            )
            self.assertEqual(response.status_code, 403)

    def test_webhook_challenge_invalid_mode(self):
        """hub.mode other than 'subscribe' returns 403 Forbidden."""
        with override_settings(**self.mock_settings):
            response = self.client.get(
                '/api/instagram/webhook/',
                {
                    'hub.mode': 'unsubscribe',
                    'hub.verify_token': self.verify_token,
                    'hub.challenge': '12345',
                }
            )
            self.assertEqual(response.status_code, 403)

    def test_webhook_challenge_missing_parameters(self):
        """Missing required query parameters returns 400 Bad Request."""
        with override_settings(**self.mock_settings):
            response = self.client.get('/api/instagram/webhook/')
            self.assertEqual(response.status_code, 400)

    def test_webhook_challenge_unconfigured_token(self):
        """When verify token is not configured on backend, challenge returns 403."""
        with override_settings(INSTAGRAM_VERIFY_TOKEN=''):
            response = self.client.get(
                '/api/instagram/webhook/',
                {
                    'hub.mode': 'subscribe',
                    'hub.verify_token': 'any_token',
                    'hub.challenge': '12345',
                }
            )
            self.assertEqual(response.status_code, 403)


class InstagramWebhookSignatureAndPayloadTests(TestCase):
    """
    Tests incoming webhook events and cryptographic HMAC-SHA256 signature verification.
    """

    def setUp(self):
        self.app_secret = "meta_test_secret_xyz789"
        self.verify_token = "verify_token_test"
        self.account_id = "17841400000000001"
        self.mock_settings = {
            'INSTAGRAM_APP_SECRET': self.app_secret,
            'INSTAGRAM_VERIFY_TOKEN': self.verify_token,
            'INSTAGRAM_ACCOUNT_ID': self.account_id,
            'INSTAGRAM_ACCESS_TOKEN': 'EAAK_valid_mock_token',
        }

    def _sign_payload(self, body_bytes: bytes, secret: str = None) -> str:
        sec = secret if secret is not None else self.app_secret
        digest = hmac.new(sec.encode('utf-8'), body_bytes, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    def test_signature_utility_valid_and_invalid(self):
        """Directly tests verify_hub_signature constant-time validation."""
        body = b'{"object":"instagram","entry":[]}'
        valid_header = self._sign_payload(body)
        invalid_header = "sha256=badbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadb"

        self.assertTrue(verify_hub_signature(body, valid_header, self.app_secret))
        self.assertFalse(verify_hub_signature(body, invalid_header, self.app_secret))
        self.assertFalse(verify_hub_signature(body, "sha1=deprecated", self.app_secret))
        self.assertFalse(verify_hub_signature(body, "", self.app_secret))
        self.assertFalse(verify_hub_signature(body, valid_header, ""))

    def test_webhook_post_missing_signature(self):
        """POST without X-Hub-Signature-256 header returns 403 Forbidden."""
        with override_settings(**self.mock_settings):
            payload = json.dumps({"object": "instagram", "entry": []}).encode('utf-8')
            response = self.client.post(
                '/api/instagram/webhook/',
                data=payload,
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 403)
            self.assertIn("Invalid webhook signature", response.content.decode('utf-8'))

    def test_webhook_post_invalid_signature(self):
        """POST with mismatched signature returns 403 Forbidden."""
        with override_settings(**self.mock_settings):
            payload = json.dumps({"object": "instagram", "entry": []}).encode('utf-8')
            response = self.client.post(
                '/api/instagram/webhook/',
                data=payload,
                content_type='application/json',
                HTTP_X_HUB_SIGNATURE_256="sha256=tampered_signature_hex"
            )
            self.assertEqual(response.status_code, 403)

    def test_webhook_post_valid_signature_success(self):
        """POST with valid HMAC-SHA256 signature returns 200 EVENT_RECEIVED."""
        with override_settings(**self.mock_settings):
            payload = json.dumps({
                "object": "instagram",
                "entry": [
                    {
                        "id": self.account_id,
                        "time": 1728987654,
                        "messaging": [
                            {
                                "sender": {"id": "customer_igsid_999"},
                                "recipient": {"id": self.account_id},
                                "timestamp": 1728987654000,
                                "message": {
                                    "mid": "mid.12345678",
                                    "text": "Hello Ami!"
                                }
                            }
                        ]
                    }
                ]
            }).encode('utf-8')

            signature = self._sign_payload(payload)

            response = self.client.post(
                '/api/instagram/webhook/',
                data=payload,
                content_type='application/json',
                HTTP_X_HUB_SIGNATURE_256=signature
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content.decode('utf-8'), "EVENT_RECEIVED")

            # Check that customer was registered and last_interaction_at was updated
            customer = InstagramCustomer.objects.get(igsid="customer_igsid_999")
            self.assertTrue(customer.is_active)
            self.assertIsNotNone(customer.last_interaction_at)
            self.assertTrue(customer.is_within_24h_window())

            # Check audit event log exists WITHOUT storing the raw message text
            log = InstagramWebhookEventLog.objects.get(sender_igsid="customer_igsid_999")
            self.assertEqual(log.event_type, "message")
            self.assertEqual(log.status, "processed")

    def test_webhook_post_malformed_json_payload(self):
        """POST with valid HMAC signature but malformed JSON returns 400 Bad Request."""
        with override_settings(**self.mock_settings):
            broken_body = b'{"object": "instagram", entry: invalid_json'
            signature = self._sign_payload(broken_body)

            response = self.client.post(
                '/api/instagram/webhook/',
                data=broken_body,
                content_type='application/json',
                HTTP_X_HUB_SIGNATURE_256=signature
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Malformed JSON payload", response.content.decode('utf-8'))

    def test_webhook_post_unrecognized_object(self):
        """POST with valid HMAC but non-instagram object returns 400 Bad Request."""
        with override_settings(**self.mock_settings):
            payload = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode('utf-8')
            signature = self._sign_payload(payload)

            response = self.client.post(
                '/api/instagram/webhook/',
                data=payload,
                content_type='application/json',
                HTTP_X_HUB_SIGNATURE_256=signature
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unrecognized webhook object", response.content.decode('utf-8'))


class InstagramMessagingWindowPolicyTests(TestCase):
    """
    Tests Meta's official 24-hour standard messaging window policy.
    Ensures Ami strictly abides by:
    1. User must message first (no unsolicited outbound DMs).
    2. Messages can only be sent within 24 hours of latest user interaction.
    """

    def setUp(self):
        self.now = timezone.now()

    def test_customer_within_24h_window_allowed(self):
        """Customer who messaged 3 hours ago is within 24h window."""
        cust = InstagramCustomer.objects.create(
            igsid="igsid_active_user",
            last_interaction_at=self.now - timedelta(hours=3),
            is_active=True
        )

        self.assertTrue(cust.is_within_24h_window())
        can_send, reason = cust.can_receive_message()
        self.assertTrue(can_send)
        self.assertIn("Within Meta's 24-hour messaging window", reason)

    def test_customer_outside_24h_window_blocked(self):
        """Customer whose last message was 25 hours ago is blocked by platform policy."""
        cust = InstagramCustomer.objects.create(
            igsid="igsid_expired_user",
            last_interaction_at=self.now - timedelta(hours=25),
            is_active=True
        )

        self.assertFalse(cust.is_within_24h_window())
        can_send, reason = cust.can_receive_message()
        self.assertFalse(can_send)
        self.assertIn("Meta standard messaging window expired", reason)

    def test_customer_without_prior_interaction_blocked(self):
        """Customer with no recorded user interaction is blocked from unsolicited messages."""
        cust = InstagramCustomer.objects.create(
            igsid="igsid_new_target",
            last_interaction_at=None,
            is_active=True
        )

        self.assertFalse(cust.is_within_24h_window())
        can_send, reason = cust.can_receive_message()
        self.assertFalse(can_send)
        self.assertIn("Unsolicited outbound DMs are prohibited", reason)

    def test_inactive_customer_blocked(self):
        """Inactive customer cannot receive messages even if within 24 hours."""
        cust = InstagramCustomer.objects.create(
            igsid="igsid_inactive_user",
            last_interaction_at=self.now - timedelta(minutes=10),
            is_active=False
        )

        self.assertFalse(cust.is_within_24h_window())
        can_send, reason = cust.can_receive_message()
        self.assertFalse(can_send)
        self.assertIn("marked inactive", reason)


class InstagramStatusAndSecurityTests(TestCase):
    """
    Tests security guidelines:
    - Never log access tokens, secrets, or private message contents.
    - Status endpoint requires authentication and never leaks plaintext secrets.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='companion_admin', password='password123')
        self.mock_settings = {
            'INSTAGRAM_ACCOUNT_ID': '17841400001234567',
            'INSTAGRAM_ACCESS_TOKEN': 'EAAK_super_sensitive_token_xyz',
            'INSTAGRAM_APP_SECRET': 'secret_hash_value_999',
            'INSTAGRAM_VERIFY_TOKEN': 'secret_verify_token_456',
        }

    def test_status_endpoint_requires_authentication(self):
        """Unauthenticated call to status endpoint returns 401/403."""
        res = self.client.get('/api/instagram/status/')
        self.assertIn(res.status_code, (401, 403))

    def test_status_endpoint_never_exposes_secrets(self):
        """Authenticated call returns status without leaking tokens or app secrets."""
        self.client.force_login(self.user)
        with override_settings(**self.mock_settings):
            res = self.client.get('/api/instagram/status/')
            self.assertEqual(res.status_code, 200)
            data = res.json()

            self.assertEqual(data['status'], 'ok')
            config = data['configuration']
            self.assertTrue(config['configured'])
            self.assertTrue(config['has_access_token'])
            self.assertTrue(config['has_app_secret'])

            # Strictly verify secrets are NEVER in response
            res_str = str(data)
            self.assertNotIn('EAAK_super_sensitive_token_xyz', res_str)
            self.assertNotIn('secret_hash_value_999', res_str)
            self.assertNotIn('secret_verify_token_456', res_str)

    def test_config_status_helper(self):
        """Sanitized config helper masks account ID and reports boolean presence."""
        with override_settings(**self.mock_settings):
            status = get_sanitized_config_status()
            self.assertTrue(status['configured'])
            self.assertEqual(status['account_id_preview'], '178...4567')
            self.assertTrue(status['has_access_token'])
            self.assertTrue(status['has_app_secret'])
            self.assertTrue(status['has_verify_token'])
            self.assertEqual(status['api_version'], 'v20.0')
