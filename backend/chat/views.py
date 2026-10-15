"""
Chat, Conversation, and Memory API views.
Provides secure endpoints for conversational messaging, conversation persistence/resumption,
and long-term memory management. All personal endpoints require authentication and enforce
strict conversation/memory ownership.
"""
import os
import logging
import uuid
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.http import FileResponse, Http404
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

from .models import (
    Conversation,
    ChatMessage,
    CompanionMemory,
)
from .services import (
    CompanionChatService,
    ChatConfigurationError,
    ChatServiceError,
    detect_candidate_memory,
    get_relevant_memories,
    save_or_update_memory,
)
from .voice_services import (
    VoiceService,
    VoiceServiceError,
    TranscriptionError,
    SynthesisError,
    validate_audio_upload,
)


# ============================================================================
# Chat Endpoint (Authenticated, Ownership-Enforced)
# ============================================================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_endpoint(request):
    """
    POST /api/chat/
    Requires authentication.
    Accepts: { "message": "...", "conversation_id": "uuid" (optional) }
    Persists user message in PostgreSQL under the authenticated user,
    retrieves recent history and user-approved memories, generates AI reply,
    and returns response with conversation_id.
    """
    if not isinstance(request.data, dict):
        return Response(
            {"error": "Invalid request format. Expected JSON object."},
            status=status.HTTP_400_BAD_REQUEST
        )

    raw_message = request.data.get('message')
    if raw_message is None or not isinstance(raw_message, str) or not raw_message.strip():
        return Response(
            {"error": "Message cannot be empty."},
            status=status.HTTP_400_BAD_REQUEST
        )

    message_text = raw_message.strip()
    user = request.user

    # 1. Resolve or create Conversation belonging strictly to request.user
    conversation_id = request.data.get('conversation_id')
    conversation = None

    if conversation_id:
        try:
            valid_uuid = uuid.UUID(str(conversation_id))
            conversation = Conversation.objects.filter(id=valid_uuid, user=user).first()
            if not conversation:
                return Response(
                    {"error": f"Conversation with id '{conversation_id}' was not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
        except ValueError:
            return Response(
                {"error": f"Invalid UUID format for conversation_id: '{conversation_id}'."},
                status=status.HTTP_400_BAD_REQUEST
            )
    else:
        # Create a new conversation thread owned by authenticated user
        conversation = Conversation.objects.create(
            user=user,
            title="New Conversation"
        )

    # 2. Persist User Message
    user_msg = ChatMessage.objects.create(
        conversation=conversation,
        sender=ChatMessage.Sender.USER,
        content=message_text
    )

    # 3. Update conversation title if default
    if conversation.title == "New Conversation":
        snippet = message_text[:35].strip()
        conversation.title = snippet + ("..." if len(message_text) > 35 else "")
        conversation.save(update_fields=['title', 'updated_at'])
    else:
        conversation.save(update_fields=['updated_at'])

    # 4. Fetch recent conversation history (excluding the current user_msg)
    history_queryset = (
        conversation.messages
        .exclude(pk=user_msg.pk)
        .order_by('-created_at')[:16]
    )
    history = [
        {
            "role": "user" if m.sender == ChatMessage.Sender.USER else "assistant",
            "content": m.content
        }
        for m in reversed(list(history_queryset))
    ]

    # 5. Fetch approved long-term memories relevant to user query and conversational context
    memories = get_relevant_memories(
        user=user,
        query_text=message_text,
        history=history,
        limit=15
    )

    # 5b. Evaluate context-aware mood transition
    from .mood import evaluate_mood_transition
    current_mood = conversation.current_mood or Conversation.Mood.CALM
    mood_context = conversation.mood_context or {}
    new_mood, new_mood_context = evaluate_mood_transition(
        current_mood=current_mood,
        mood_context=mood_context,
        user_message=message_text,
        history=history
    )

    # 6. Generate AI response
    chat_service = CompanionChatService()

    try:
        reply_text, metadata = chat_service.generate_reply(
            message=message_text,
            history=history,
            memories=memories,
            mood=new_mood,
            mood_context=new_mood_context
        )

        # 7. Persist updated mood and AI Companion Message
        interpreted_mood = metadata.get('interpreted_mood')
        if interpreted_mood and hasattr(Conversation.Mood, interpreted_mood.upper()):
            new_mood = interpreted_mood
            conversation.current_mood = interpreted_mood
            if metadata.get('interpreted_summary'):
                new_mood_context['situation_summary'] = metadata['interpreted_summary']
        else:
            conversation.current_mood = new_mood

        conversation.mood_context = new_mood_context
        conversation.save(update_fields=['current_mood', 'mood_context', 'updated_at'])

        companion_msg = ChatMessage.objects.create(
            conversation=conversation,
            sender=ChatMessage.Sender.COMPANION,
            content=reply_text,
            metadata=metadata
        )

        # 8. Check if user message suggested a candidate memory and persist if durable
        suggested_memory = detect_candidate_memory(message_text)
        if suggested_memory:
            save_or_update_memory(
                user=user,
                content=suggested_memory['content'],
                category=suggested_memory.get('category', 'personal'),
                importance=suggested_memory.get('importance', 1),
                is_approved=True,
                source_message=user_msg
            )

        return Response(
            {
                "reply": reply_text,
                "conversation_id": str(conversation.id),
                "mood": new_mood,
                "message_id": str(companion_msg.id),
                "user_message_id": str(user_msg.id),
                "timestamp": companion_msg.created_at.isoformat(),
                "metadata": {
                    "provider": metadata.get("provider", "openai"),
                    "model": metadata.get("model"),
                    "completion_id": metadata.get("completion_id"),
                    "prompt_tokens": metadata.get("prompt_tokens", 0),
                    "completion_tokens": metadata.get("completion_tokens", 0),
                    "total_tokens": metadata.get("total_tokens", 0),
                    "mood": new_mood,
                },
                "suggested_memory": suggested_memory,
            },
            status=status.HTTP_200_OK
        )

    except ChatConfigurationError as exc:
        return Response(
            {
                "error": str(exc),
                "error_category": getattr(exc, 'error_category', 'missing_or_placeholder_key'),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except ChatServiceError as exc:
        return Response(
            {
                "error": str(exc),
                "error_category": getattr(exc, 'error_category', 'api_error'),
            },
            status=status.HTTP_502_BAD_GATEWAY
        )
    except ValueError as exc:
        return Response(
            {"error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as exc:
        return Response(
            {"error": f"An unexpected error occurred: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# Conversation Management Endpoints (Authenticated, Ownership-Enforced)
# ============================================================================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def conversation_list_create(request):
    """
    GET: List all conversations owned strictly by authenticated request.user.
    POST: Create a new conversation thread for request.user.
    """
    user = request.user

    if request.method == 'GET':
        conversations = Conversation.objects.filter(user=user).order_by('-updated_at')
        data = [
            {
                "id": str(c.id),
                "title": c.title,
                "current_mood": c.current_mood,
                "message_count": c.message_count,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in conversations
        ]
        return Response(data, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        title = "New Conversation"
        if isinstance(request.data, dict) and request.data.get('title'):
            title = str(request.data['title']).strip() or "New Conversation"

        conv = Conversation.objects.create(user=user, title=title)
        return Response(
            {
                "id": str(conv.id),
                "title": conv.title,
                "current_mood": conv.current_mood,
                "message_count": 0,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            status=status.HTTP_201_CREATED
        )


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def conversation_detail_delete(request, conversation_id):
    """
    GET: Retrieve conversation and full message history (scoped to request.user).
    DELETE: Delete conversation and its messages (scoped to request.user).
    Returns 404 if conversation does not exist or belongs to another user.
    """
    user = request.user
    conv = get_object_or_404(Conversation, id=conversation_id, user=user)

    if request.method == 'GET':
        messages = [
            {
                "id": str(m.id),
                "sender": m.sender,
                "content": m.content,
                "message_type": m.message_type,
                "audio_url": m.audio_url,
                "audio_duration": m.audio_duration,
                "metadata": m.metadata,
                "created_at": m.created_at.isoformat(),
            }
            for m in conv.messages.order_by('created_at')
        ]
        return Response(
            {
                "id": str(conv.id),
                "title": conv.title,
                "current_mood": conv.current_mood,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "messages": messages,
            },
            status=status.HTTP_200_OK
        )

    elif request.method == 'DELETE':
        conv.delete()
        return Response(
            {"message": "Conversation deleted successfully."},
            status=status.HTTP_200_OK
        )


@api_view(['POST', 'DELETE'])
@permission_classes([IsAuthenticated])
def conversation_clear_messages(request, conversation_id):
    """
    POST/DELETE /api/conversations/<conversation_id>/clear/
    Permanently clears all ChatMessage records for this conversation from PostgreSQL.
    Preserves the conversation entity, user account, and long-term companion memories.
    """
    user = request.user
    conv = get_object_or_404(Conversation, id=conversation_id, user=user)
    deleted_count, _ = conv.messages.all().delete()
    conv.title = "New Conversation"
    conv.save(update_fields=['title', 'updated_at'])
    return Response(
        {
            "cleared": True,
            "message": "Conversation messages cleared successfully.",
            "deleted_count": deleted_count,
            "conversation_id": str(conv.id),
        },
        status=status.HTTP_200_OK
    )



# ============================================================================
# Long-Term Memory Management Endpoints (Authenticated, Ownership-Enforced)
# ============================================================================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def memory_list_create(request):
    """
    GET: List all companion memories belonging to request.user.
    POST: Create a user-approved long-term memory for request.user.
    """
    user = request.user

    if request.method == 'GET':
        memories = CompanionMemory.objects.filter(user=user).order_by('-importance', '-created_at')
        data = [
            {
                "id": str(m.id),
                "category": m.category,
                "category_display": m.get_category_display(),
                "content": m.content,
                "importance": m.importance,
                "is_approved": m.is_approved,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
            }
            for m in memories
        ]
        return Response(data, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        if not isinstance(request.data, dict):
            return Response(
                {"error": "Expected JSON payload."},
                status=status.HTTP_400_BAD_REQUEST
            )

        content = request.data.get('content')
        if not content or not str(content).strip():
            return Response(
                {"error": "Memory content cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST
            )

        category = request.data.get('category', CompanionMemory.Category.PERSONAL)
        if category not in dict(CompanionMemory.Category.choices):
            category = CompanionMemory.Category.PERSONAL

        importance = request.data.get('importance', CompanionMemory.Importance.NORMAL)
        try:
            importance = int(importance)
            if importance not in dict(CompanionMemory.Importance.choices):
                importance = CompanionMemory.Importance.NORMAL
        except (ValueError, TypeError):
            importance = CompanionMemory.Importance.NORMAL

        is_approved = bool(request.data.get('is_approved', True))

        memory, action = save_or_update_memory(
            user=user,
            content=str(content).strip(),
            category=category,
            importance=importance,
            is_approved=is_approved
        )

        resp_status = status.HTTP_201_CREATED if action == "created" else status.HTTP_200_OK

        return Response(
            {
                "id": str(memory.id),
                "category": memory.category,
                "category_display": memory.get_category_display(),
                "content": memory.content,
                "importance": memory.importance,
                "is_approved": memory.is_approved,
                "action": action,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            status=resp_status
        )


@api_view(['GET', 'PATCH', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def memory_detail_update_delete(request, memory_id):
    """
    GET: Retrieve memory details (scoped to request.user).
    PATCH/PUT: Update memory content, category, importance, or approval (scoped to request.user).
    DELETE: Delete memory (scoped to request.user).
    Returns 404 if memory does not exist or belongs to another user.
    """
    user = request.user
    memory = get_object_or_404(CompanionMemory, id=memory_id, user=user)

    if request.method == 'GET':
        return Response(
            {
                "id": str(memory.id),
                "category": memory.category,
                "category_display": memory.get_category_display(),
                "content": memory.content,
                "importance": memory.importance,
                "is_approved": memory.is_approved,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            status=status.HTTP_200_OK
        )

    elif request.method in ('PATCH', 'PUT'):
        if not isinstance(request.data, dict):
            return Response(
                {"error": "Expected JSON payload."},
                status=status.HTTP_400_BAD_REQUEST
            )

        update_fields = ['updated_at']

        if 'content' in request.data:
            new_content = str(request.data['content']).strip()
            if not new_content:
                return Response(
                    {"error": "Memory content cannot be empty."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            memory.content = new_content
            update_fields.append('content')

        if 'category' in request.data:
            cat = str(request.data['category'])
            if cat in dict(CompanionMemory.Category.choices):
                memory.category = cat
                update_fields.append('category')

        if 'importance' in request.data:
            try:
                imp = int(request.data['importance'])
                if imp in dict(CompanionMemory.Importance.choices):
                    memory.importance = imp
                    update_fields.append('importance')
            except (ValueError, TypeError):
                pass

        if 'is_approved' in request.data:
            memory.is_approved = bool(request.data['is_approved'])
            update_fields.append('is_approved')

        memory.save(update_fields=update_fields)

        return Response(
            {
                "id": str(memory.id),
                "category": memory.category,
                "category_display": memory.get_category_display(),
                "content": memory.content,
                "importance": memory.importance,
                "is_approved": memory.is_approved,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            status=status.HTTP_200_OK
        )

    elif request.method == 'DELETE':
        memory.delete()
        return Response(
            {"message": "Memory deleted successfully."},
            status=status.HTTP_200_OK
        )


# ============================================================================
# Voice Messaging Endpoints (Authenticated, User-Isolated)
# ============================================================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_voice_endpoint(request):
    """
    POST /api/chat/voice/
    Processes an uploaded user voice message:
    1. Validates audio file format and size limits.
    2. Transcribes spoken speech (via Gemini or configured STT provider).
    3. Persists user audio message in PostgreSQL/storage.
    4. Evaluates emotional mood continuity and retrieves relevant memories.
    5. Generates companion AI text response using Gemini.
    6. Synthesizes natural female voice audio for Ami's reply (via edge-tts/TTS provider).
    7. Returns both text reply and streaming audio URL.
    """
    user = request.user
    audio_file = request.FILES.get('audio')
    conversation_id = request.data.get('conversation_id')
    client_message_id = request.data.get('client_message_id')
    duration_raw = request.data.get('duration')
    pre_transcript = (request.data.get('transcript') or '').strip()

    if not audio_file and not pre_transcript:
        return Response(
            {"error": "No audio file or transcript provided."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # 1. Validate audio upload if provided
    clean_mime = 'audio/webm'
    if audio_file:
        try:
            clean_mime, _ = validate_audio_upload(audio_file)
        except ValueError as val_err:
            return Response({"error": str(val_err)}, status=status.HTTP_400_BAD_REQUEST)

    # 2. Idempotency check: if client_message_id is supplied and already processed, return existing
    if client_message_id:
        existing_user_msg = ChatMessage.objects.filter(
            conversation__user=user,
            metadata__client_message_id=client_message_id
        ).select_related('conversation').first()
        if existing_user_msg:
            companion_reply_msg = ChatMessage.objects.filter(
                conversation=existing_user_msg.conversation,
                sender=ChatMessage.Sender.COMPANION,
                created_at__gte=existing_user_msg.created_at
            ).first()
            if companion_reply_msg:
                return Response({
                    "reply": companion_reply_msg.content,
                    "conversation_id": str(existing_user_msg.conversation.id),
                    "mood": existing_user_msg.conversation.current_mood,
                    "message_id": str(companion_reply_msg.id),
                    "user_message_id": str(existing_user_msg.id),
                    "user_transcript": existing_user_msg.content,
                    "user_audio_url": existing_user_msg.audio_url,
                    "user_audio_duration": existing_user_msg.audio_duration,
                    "audio_url": companion_reply_msg.audio_url,
                    "audio_duration": companion_reply_msg.audio_duration,
                    "timestamp": companion_reply_msg.created_at.isoformat(),
                    "metadata": companion_reply_msg.metadata,
                }, status=status.HTTP_200_OK)

    # 3. Speech-to-Text Transcription
    voice_service = VoiceService()
    transcript = pre_transcript
    if not transcript and audio_file:
        try:
            audio_bytes = audio_file.read()
            audio_file.seek(0)
            transcript = voice_service.transcribe_audio(audio_bytes, mime_type=clean_mime)
        except TranscriptionError as t_err:
            return Response(
                {"error": str(t_err), "error_category": "transcription_error"},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        except Exception as exc:
            logger.warning("Transcription failed unexpectedly: %s", exc)
            return Response(
                {"error": f"Transcription error: {exc}", "error_category": "transcription_error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    transcript = (transcript or "").strip()
    if not transcript:
        return Response(
            {"error": "No intelligible speech was recognized in the voice recording.", "error_category": "transcription_empty"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

    # Parse duration
    audio_duration = None
    if duration_raw:
        try:
            audio_duration = max(0.0, float(duration_raw))
        except (ValueError, TypeError):
            pass

    # 4. Resolve or create user's conversation thread
    if conversation_id:
        conversation = get_object_or_404(Conversation, id=conversation_id, user=user)
    else:
        conversation = Conversation.objects.create(user=user, title="Voice Chat")

    # 5. Persist User Voice Message
    user_metadata = {
        "client_message_id": client_message_id,
        "mime_type": clean_mime,
        "is_voice": True,
    }
    if audio_file:
        user_metadata["original_filename"] = getattr(audio_file, 'name', '')

    user_msg = ChatMessage.objects.create(
        conversation=conversation,
        sender=ChatMessage.Sender.USER,
        message_type=ChatMessage.MessageType.VOICE,
        content=transcript,
        audio_file=audio_file,
        audio_duration=audio_duration,
        metadata=user_metadata
    )

    # Update conversation title if default
    if conversation.title in ("New Conversation", "Voice Chat"):
        snippet = transcript[:35].strip()
        conversation.title = snippet + ("..." if len(transcript) > 35 else "")
        conversation.save(update_fields=['title', 'updated_at'])
    else:
        conversation.save(update_fields=['updated_at'])

    # 6. Fetch recent conversation history (excluding current user_msg)
    history_queryset = (
        conversation.messages
        .exclude(pk=user_msg.pk)
        .order_by('-created_at')[:16]
    )
    history = [
        {
            "role": "user" if m.sender == ChatMessage.Sender.USER else "assistant",
            "content": m.content
        }
        for m in reversed(list(history_queryset))
    ]

    # 7. Fetch memories and evaluate mood transition
    memories = get_relevant_memories(
        user=user,
        query_text=transcript,
        history=history,
        limit=15
    )

    from .mood import evaluate_mood_transition
    current_mood = conversation.current_mood or Conversation.Mood.CALM
    mood_context = conversation.mood_context or {}
    new_mood, new_mood_context = evaluate_mood_transition(
        current_mood=current_mood,
        mood_context=mood_context,
        user_message=transcript,
        history=history
    )

    # 8. Generate Companion AI Text Reply
    chat_service = CompanionChatService()
    try:
        reply_text, metadata = chat_service.generate_reply(
            message=transcript,
            history=history,
            memories=memories,
            mood=new_mood,
            mood_context=new_mood_context
        )

        interpreted_mood = metadata.get('interpreted_mood')
        if interpreted_mood and hasattr(Conversation.Mood, interpreted_mood.upper()):
            new_mood = interpreted_mood
            conversation.current_mood = interpreted_mood
            if metadata.get('interpreted_summary'):
                new_mood_context['situation_summary'] = metadata['interpreted_summary']
        else:
            conversation.current_mood = new_mood

        conversation.mood_context = new_mood_context
        conversation.save(update_fields=['current_mood', 'mood_context', 'updated_at'])

        # 9. Synthesize Companion Voice Reply
        companion_audio_file = None
        companion_audio_duration = None
        try:
            tts_bytes, est_dur = voice_service.synthesize_speech(reply_text)
            file_name = f"ami_reply_{uuid.uuid4().hex[:8]}.mp3"
            companion_audio_file = ContentFile(tts_bytes, name=file_name)
            companion_audio_duration = est_dur
            metadata["tts_voice"] = voice_service.default_voice
            metadata["tts_provider"] = voice_service.tts_provider
        except Exception as tts_err:
            logger.warning("TTS generation failed for reply: %s", tts_err)
            metadata["tts_error"] = str(tts_err)

        companion_msg = ChatMessage.objects.create(
            conversation=conversation,
            sender=ChatMessage.Sender.COMPANION,
            message_type=ChatMessage.MessageType.VOICE,
            content=reply_text,
            audio_file=companion_audio_file,
            audio_duration=companion_audio_duration,
            metadata=metadata
        )

        # 10. Check for candidate durable memory
        suggested_memory = detect_candidate_memory(transcript)
        if suggested_memory:
            save_or_update_memory(
                user=user,
                content=suggested_memory['content'],
                category=suggested_memory.get('category', 'personal'),
                importance=suggested_memory.get('importance', 1),
                is_approved=True,
                source_message=user_msg
            )

        return Response(
            {
                "reply": reply_text,
                "conversation_id": str(conversation.id),
                "mood": new_mood,
                "message_id": str(companion_msg.id),
                "user_message_id": str(user_msg.id),
                "user_transcript": transcript,
                "user_audio_url": user_msg.audio_url,
                "user_audio_duration": user_msg.audio_duration,
                "audio_url": companion_msg.audio_url,
                "audio_duration": companion_msg.audio_duration,
                "timestamp": companion_msg.created_at.isoformat(),
                "metadata": {
                    "provider": metadata.get("provider", "gemini"),
                    "model": metadata.get("model"),
                    "prompt_tokens": metadata.get("prompt_tokens", 0),
                    "completion_tokens": metadata.get("completion_tokens", 0),
                    "total_tokens": metadata.get("total_tokens", 0),
                    "mood": new_mood,
                    "tts_error": metadata.get("tts_error"),
                },
                "suggested_memory": suggested_memory,
            },
            status=status.HTTP_200_OK
        )

    except ChatConfigurationError as exc:
        return Response(
            {"error": str(exc), "error_category": getattr(exc, 'error_category', 'missing_or_placeholder_key')},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except ChatServiceError as exc:
        return Response(
            {"error": str(exc), "error_category": getattr(exc, 'error_category', 'api_error')},
            status=status.HTTP_502_BAD_GATEWAY
        )
    except Exception as exc:
        logger.exception("Unexpected error during voice chat: %s", exc)
        return Response(
            {"error": f"An unexpected error occurred: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def message_audio_stream(request, message_id):
    """
    GET /api/messages/<uuid:message_id>/audio/
    Streams audio file for a chat message.
    Strictly enforces user isolation: the message must belong to a conversation owned by request.user.
    """
    user = request.user
    msg = get_object_or_404(ChatMessage, id=message_id, conversation__user=user)

    if not msg.audio_file:
        return Response(
            {"error": "No audio file is associated with this message."},
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        audio_stream = msg.audio_file.open('rb')
    except (FileNotFoundError, OSError):
        return Response(
            {"error": "Audio file not found on server storage."},
            status=status.HTTP_404_NOT_FOUND
        )

    ext = os.path.splitext(msg.audio_file.name)[1].lower()
    content_type = 'audio/mpeg'
    if ext == '.webm':
        content_type = 'audio/webm'
    elif ext == '.ogg':
        content_type = 'audio/ogg'
    elif ext == '.wav':
        content_type = 'audio/wav'
    elif ext in ('.mp4', '.m4a'):
        content_type = 'audio/mp4'

    response = FileResponse(audio_stream, content_type=content_type)
    response['Accept-Ranges'] = 'bytes'
    response['Cache-Control'] = 'private, max-age=3600'
    return response


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def message_synthesize_voice(request, message_id):
    """
    POST /api/messages/<uuid:message_id>/synthesize-voice/
    Generates or retrieves voice audio for a companion message on demand.
    If cached audio already exists, returns it immediately without regenerating.
    """
    user = request.user
    msg = get_object_or_404(
        ChatMessage,
        id=message_id,
        conversation__user=user,
        sender=ChatMessage.Sender.COMPANION
    )

    # 1. If audio is already synthesized and cached, return it directly
    if msg.audio_file:
        return Response({
            "message_id": str(msg.id),
            "audio_url": msg.audio_url,
            "audio_duration": msg.audio_duration,
            "cached": True
        }, status=status.HTTP_200_OK)

    # 2. Synthesize speech on-demand
    voice_service = VoiceService()
    try:
        audio_bytes, est_dur = voice_service.synthesize_speech(msg.content)
        file_name = f"ami_voice_{msg.id}.mp3"
        msg.audio_file.save(file_name, ContentFile(audio_bytes), save=False)
        msg.audio_duration = est_dur
        msg.message_type = ChatMessage.MessageType.VOICE
        msg.save(update_fields=['audio_file', 'audio_duration', 'message_type'])

        return Response({
            "message_id": str(msg.id),
            "audio_url": msg.audio_url,
            "audio_duration": msg.audio_duration,
            "cached": False
        }, status=status.HTTP_200_OK)
    except Exception as exc:
        logger.warning("On-demand voice synthesis failed: %s", exc)
        return Response({
            "error": f"Failed to synthesize voice: {exc}",
            "error_category": "synthesis_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

