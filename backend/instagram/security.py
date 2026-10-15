"""
Cryptographic signature verification utilities for Meta's Instagram Webhooks.
Ensures webhook requests originate strictly from Meta's servers.
"""
import hmac
import hashlib
import logging

logger = logging.getLogger(__name__)


def verify_hub_signature(request_body: bytes, hub_signature_header: str, app_secret: str) -> bool:
    """
    Validates Meta's X-Hub-Signature-256 header against the raw request body.

    Meta signs all webhook events using HMAC-SHA256 with the Meta App Secret as the key.
    Header format: "sha256=<hex_encoded_signature>"

    Uses constant-time comparison (hmac.compare_digest) to prevent timing side-channel attacks.
    Never logs the app secret or raw message payloads.
    """
    if not app_secret:
        logger.warning("Instagram signature verification failed: INSTAGRAM_APP_SECRET is not configured.")
        return False

    if not hub_signature_header or not isinstance(hub_signature_header, str):
        logger.warning("Instagram signature verification failed: Missing or invalid X-Hub-Signature-256 header.")
        return False

    header_clean = hub_signature_header.strip()
    if not header_clean.startswith('sha256='):
        logger.warning("Instagram signature verification failed: Signature does not use sha256 prefix.")
        return False

    received_signature = header_clean[7:].strip()
    if not received_signature:
        logger.warning("Instagram signature verification failed: Empty signature hash in header.")
        return False

    try:
        expected_signature = hmac.new(
            key=app_secret.encode('utf-8'),
            msg=request_body or b'',
            digestmod=hashlib.sha256
        ).hexdigest()

        is_valid = hmac.compare_digest(received_signature.lower(), expected_signature.lower())
        if not is_valid:
            logger.warning("Instagram signature verification failed: HMAC digest mismatch.")
        return is_valid
    except Exception as exc:
        logger.error("Instagram signature verification error: %s", type(exc).__name__)
        return False
