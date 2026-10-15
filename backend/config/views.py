"""
System health check and diagnostic views.
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.db import connection


from chat.services import get_openai_diagnostic


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """
    Diagnostic endpoint verifying API availability, database connectivity, and OpenAI setup.
    Returns 200 OK if healthy, or 503 Service Unavailable if database is unreachable.
    NEVER logs or exposes secret keys.
    """
    db_status = "connected"
    http_status = status.HTTP_200_OK

    try:
        connection.ensure_connection()
    except Exception as exc:
        db_status = f"unreachable: {exc}"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    from chat.services import get_ai_diagnostic
    ai_diag = get_ai_diagnostic()

    payload = {
        "status": "healthy" if http_status == status.HTTP_200_OK else "degraded",
        "timestamp": timezone.now().isoformat(),
        "services": {
            "api": "operational",
            "database": db_status,
            "ai_connected": ai_diag["configured"],
            "ai_provider": ai_diag["active_provider"],
            "gemini": {
                "key_present": ai_diag["gemini"]["key_present"],
                "is_placeholder": ai_diag["gemini"]["is_placeholder"],
                "configured": ai_diag["gemini"]["configured"],
                "model": ai_diag["gemini"]["model"],
            },
            "openai": {
                "key_present": ai_diag["openai"]["key_present"],
                "is_placeholder": ai_diag["openai"]["is_placeholder"],
                "configured": ai_diag["openai"]["configured"],
                "model": ai_diag["openai"]["model"],
            },
        },
        "version": "1.0.0",
    }
    return Response(payload, status=http_status)
