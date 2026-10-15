"""
URL mappings for authentication, chat, conversations, and long-term memory.
"""
from django.urls import path
from .auth_views import (
    get_csrf_token,
    login_view,
    logout_view,
    register_view,
    current_user_view,
)
from .views import (
    chat_endpoint,
    chat_voice_endpoint,
    message_audio_stream,
    message_synthesize_voice,
    conversation_list_create,
    conversation_detail_delete,
    conversation_clear_messages,
    memory_list_create,
    memory_detail_update_delete,
)

urlpatterns = [
    # Session Authentication & CSRF
    path('auth/csrf/', get_csrf_token, name='auth-csrf'),
    path('auth/login/', login_view, name='auth-login'),
    path('auth/logout/', logout_view, name='auth-logout'),
    path('auth/register/', register_view, name='auth-register'),
    path('auth/me/', current_user_view, name='auth-me'),

    # Interactive AI Chat (Authenticated)
    path('chat/', chat_endpoint, name='chat-endpoint'),
    path('chat/voice/', chat_voice_endpoint, name='chat-voice-endpoint'),

    # Audio Streaming & Speech Synthesis (Authenticated, Scoped)
    path('messages/<uuid:message_id>/audio/', message_audio_stream, name='message-audio-stream'),
    path('messages/<uuid:message_id>/synthesize-voice/', message_synthesize_voice, name='message-synthesize-voice'),

    # Conversation Sessions & History Persistence (Authenticated)
    path('conversations/', conversation_list_create, name='conversation-list-create'),
    path('conversations/<uuid:conversation_id>/', conversation_detail_delete, name='conversation-detail-delete'),
    path('conversations/<uuid:conversation_id>/clear/', conversation_clear_messages, name='conversation-clear-messages'),

    # Long-Term Companion Memories (Authenticated)
    path('memories/', memory_list_create, name='memory-list-create'),
    path('memories/<uuid:memory_id>/', memory_detail_update_delete, name='memory-detail-update-delete'),
]
