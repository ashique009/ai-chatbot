"""
Views for Meta's Official Instagram Messaging API integration.
Implements:
1. Meta Webhook Verification Challenge (GET)
2. Cryptographically signed Webhook Receiver with HMAC-SHA256 validation (POST)
3. 24-hour interaction recording for Instagram customers
4. Sanitized status diagnostic endpoint

Security Guidelines:
- Never log raw access tokens, app secrets, or private message contents.
- Use constant-time comparison for signature validation.
"""
import json
import logging
import hmac
from django.http import HttpResponse, HttpResponseForbidden, HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from .config import (
    get_instagram_verify_token,
    get_instagram_app_secret,
    get_instagram_account_id,
    get_sanitized_config_status,
)
from .security import verify_hub_signature
from .models import InstagramCustomer, InstagramWebhookEventLog

logger = logging.getLogger(__name__)


@csrf_exempt
def instagram_webhook_view(request):
    """
    Official Meta Webhook endpoint for Instagram Messaging API.
    
    GET: Handles Meta's webhook verification handshake challenge.
         Validates 'hub.mode' and 'hub.verify_token' and echoes back 'hub.challenge'.
         
    POST: Receives incoming events (messages, postbacks, deliveries).
          Validates the HMAC-SHA256 signature in 'X-Hub-Signature-256' using INSTAGRAM_APP_SECRET.
          Updates customer last interaction timestamp for the 24-hour messaging window.
    """
    # -------------------------------------------------------------------------
    # 1. Meta Webhook Verification Handshake (GET)
    # -------------------------------------------------------------------------
    if request.method == 'GET':
        mode = request.GET.get('hub.mode')
        verify_token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')

        expected_token = get_instagram_verify_token()
        if not expected_token:
            logger.error("Instagram webhook verification rejected: INSTAGRAM_VERIFY_TOKEN is not configured.")
            return HttpResponseForbidden("Webhook verification token is not configured on this server.")

        if not mode or not verify_token or not challenge:
            logger.warning("Instagram webhook verification rejected: Missing required challenge query parameters.")
            return HttpResponseBadRequest("Missing required challenge parameters.")

        if mode == 'subscribe' and hmac.compare_digest(verify_token, expected_token):
            logger.info("Instagram webhook challenge verified successfully.")
            # Meta requires the raw challenge string echoed back as text/plain
            return HttpResponse(challenge, content_type="text/plain", status=200)

        logger.warning("Instagram webhook verification rejected: Verification token mismatch.")
        return HttpResponseForbidden("Verification token mismatch.")

    # -------------------------------------------------------------------------
    # 2. Incoming Webhook Event Dispatcher (POST)
    # -------------------------------------------------------------------------
    if request.method == 'POST':
        # Retrieve signature header from request
        signature_header = (
            request.headers.get('X-Hub-Signature-256') or
            request.META.get('HTTP_X_HUB_SIGNATURE_256', '')
        )

        app_secret = get_instagram_app_secret()
        if not app_secret:
            logger.error("Instagram webhook rejected: INSTAGRAM_APP_SECRET is not configured.")
            return HttpResponseForbidden("Instagram webhook signing secret is not configured.")

        # Cryptographically verify the payload signature
        if not verify_hub_signature(request.body, signature_header, app_secret):
            return HttpResponseForbidden("Invalid webhook signature.")

        # Parse JSON payload
        try:
            payload = json.loads(request.body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Instagram webhook rejected: Malformed JSON payload.")
            return HttpResponseBadRequest("Malformed JSON payload.")

        if not isinstance(payload, dict):
            logger.warning("Instagram webhook rejected: Payload is not a JSON object.")
            return HttpResponseBadRequest("Payload must be a JSON object.")

        # Meta Instagram Messaging webhooks identify object as 'instagram' or 'page'
        webhook_object = payload.get('object')
        if webhook_object not in ('instagram', 'page'):
            logger.warning("Instagram webhook: Unrecognized object '%s'.", webhook_object)
            return HttpResponseBadRequest(f"Unrecognized webhook object '{webhook_object}'.")

        # Process entries
        entries = payload.get('entry', [])
        account_id = get_instagram_account_id()

        for entry in entries:
            entry_id = str(entry.get('id', ''))
            messaging_events = entry.get('messaging', [])

            for event in messaging_events:
                sender_id = str(event.get('sender', {}).get('id', '')).strip()
                recipient_id = str(event.get('recipient', {}).get('id', '')).strip()

                # Determine event type
                event_type = 'unknown'
                if 'message' in event:
                    event_type = 'message'
                elif 'delivery' in event:
                    event_type = 'delivery'
                elif 'read' in event:
                    event_type = 'read'
                elif 'postback' in event:
                    event_type = 'postback'

                # If this is an incoming interaction from an external user to our professional account
                if sender_id and sender_id != account_id:
                    # Update or register the Instagram customer and record interaction
                    customer, _ = InstagramCustomer.objects.get_or_create(
                        igsid=sender_id,
                        defaults={'is_active': True}
                    )
                    customer.record_interaction()

                    # Audit log without storing private message contents
                    InstagramWebhookEventLog.objects.create(
                        entry_id=entry_id,
                        sender_igsid=sender_id,
                        event_type=event_type,
                        status='processed'
                    )

        # Meta requires a fast 200 OK response to prevent webhook retries
        return HttpResponse("EVENT_RECEIVED", content_type="text/plain", status=200)

    return HttpResponseBadRequest("Method not supported.")


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def instagram_status_view(request):
    """
    Authenticated diagnostic view returning Instagram integration status.
    Never exposes raw tokens, secrets, or sensitive configuration values.
    """
    status_info = get_sanitized_config_status()
    customer_count = InstagramCustomer.objects.filter(is_active=True).count()
    active_24h_count = sum(
        1 for c in InstagramCustomer.objects.filter(is_active=True)
        if c.is_within_24h_window()
    )

    return JsonResponse({
        "status": "ok",
        "configuration": status_info,
        "metrics": {
            "total_customers": customer_count,
            "active_in_24h_window": active_24h_count,
        }
    })
