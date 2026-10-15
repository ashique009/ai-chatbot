"""
Configuration settings for Meta's Official Instagram Messaging API.
Reads credentials from Django settings / environment variables without exposing secrets.
"""
from django.conf import settings


def get_instagram_account_id() -> str:
    """Returns the Instagram Professional Account ID."""
    return getattr(settings, 'INSTAGRAM_ACCOUNT_ID', '').strip()


def get_instagram_access_token() -> str:
    """Returns the Meta/Instagram Page Access Token."""
    return getattr(settings, 'INSTAGRAM_ACCESS_TOKEN', '').strip()


def get_instagram_app_secret() -> str:
    """Returns the Meta App Secret used for HMAC-SHA256 signature verification."""
    return getattr(settings, 'INSTAGRAM_APP_SECRET', '').strip()


def get_instagram_verify_token() -> str:
    """Returns the custom webhook verification token configured in Meta App Dashboard."""
    return getattr(settings, 'INSTAGRAM_VERIFY_TOKEN', '').strip()


def get_instagram_api_version() -> str:
    """Returns the target Graph API version."""
    return getattr(settings, 'INSTAGRAM_API_VERSION', 'v20.0').strip()


def is_instagram_configured() -> bool:
    """
    Checks if all required Instagram Messaging credentials are configured.
    Requires account ID, access token, app secret, and verify token.
    """
    return bool(
        get_instagram_account_id() and
        get_instagram_access_token() and
        get_instagram_app_secret() and
        get_instagram_verify_token()
    )


def get_sanitized_config_status() -> dict:
    """
    Returns a safe diagnostic summary of Instagram configuration.
    Never exposes raw tokens, secrets, or sensitive credentials.
    """
    account_id = get_instagram_account_id()
    masked_account_id = (
        f"{account_id[:3]}...{account_id[-4:]}" if len(account_id) >= 8 else ("configured" if account_id else "")
    )

    return {
        "configured": is_instagram_configured(),
        "has_account_id": bool(account_id),
        "account_id_preview": masked_account_id,
        "has_access_token": bool(get_instagram_access_token()),
        "has_app_secret": bool(get_instagram_app_secret()),
        "has_verify_token": bool(get_instagram_verify_token()),
        "api_version": get_instagram_api_version(),
    }
