"""
AI Message Generator service using the OpenAI API.
Crafts personalized, natural Manglish or English partner-style messages
incorporating user preferences, approved long-term memories, and saved conversation context.
"""
from typing import Any, Dict, List, Optional, Tuple
from django.conf import settings
from proactive_messaging.models import ScheduledMessage, CompanionPreference
from proactive_messaging.services.exceptions import (
    AIGenerationError,
    OpenAIConfigurationError,
)

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


class CompanionAIGenerator:
    """
    Service responsible for contacting the configured AI provider (Google Gemini or OpenAI)
    to generate affectionate, authentic partner-style proactive check-in messages.
    """

    def __init__(self, client: Optional[Any] = None, provider: Optional[str] = None):
        self._client = client
        self._provider = provider

    def get_provider(self) -> str:
        """Determines the active AI provider ('gemini' or 'openai')."""
        if self._provider:
            return self._provider.lower().strip()
        if self._client is not None:
            mock_children = getattr(self._client, '_mock_children', None)
            if mock_children is not None:
                if 'chat' in mock_children:
                    return 'openai'
                if 'models' in mock_children:
                    return 'gemini'
            if OpenAI is not None and isinstance(self._client, OpenAI):
                return 'openai'
            if genai is not None and isinstance(self._client, genai.Client):
                return 'gemini'

        return getattr(settings, 'AI_PROVIDER', 'gemini').lower().strip()

    def get_client(self) -> Any:
        """Initializes and returns the client for the active AI provider."""
        if self._client is not None:
            return self._client

        import os
        from chat.services import is_placeholder_key

        provider = self.get_provider()

        if provider == 'gemini':
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if api_key is None:
                api_key = os.getenv('GEMINI_API_KEY', '')
            api_key = (api_key or '').strip().strip("\"'")

            if is_placeholder_key(api_key):
                raise AIGenerationError(
                    "GEMINI_API_KEY is not configured. Please supply a valid key in your environment/.env file."
                )

            if genai is None:
                raise AIGenerationError(
                    "The 'google-genai' Python library is not installed."
                )

            self._client = genai.Client(api_key=api_key)
            return self._client

        else:  # OpenAI
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY', '')
            api_key = (api_key or '').strip().strip("\"'")

            if is_placeholder_key(api_key):
                raise OpenAIConfigurationError(
                    "OPENAI_API_KEY is not configured. Please supply a valid key in your environment/.env file."
                )

            if OpenAI is None:
                raise OpenAIConfigurationError(
                    "The 'openai' Python library is not installed. Run 'pip install openai'."
                )

            self._client = OpenAI(api_key=api_key)
            return self._client

    def _build_system_prompt(
        self,
        companion_name: str,
        user_nickname: str,
        language: str,
        tone: str,
        relationship_style: str,
        memories: Optional[List[Any]] = None
    ) -> str:
        """Constructs the persona, language system prompt, and integrates approved long-term memories."""
        nickname_clause = f"Address the user affectionately as '{user_nickname}'." if user_nickname else "Address the user with warmth."

        if language == CompanionPreference.LanguagePreference.MANGLISH:
            language_instructions = (
                "You write in natural, modern Manglish (a seamless, affectionate blend of English and "
                "informal Malayalam transliteration, like 'Kazhicho da?', 'Morning dear, ready for the day?', "
                "'Work-il busy aano?', 'Take care tto', 'Tea kudicho?'). "
                "Keep Malayalam words natural and conversational in Latin/English letters, never formal or artificial."
            )
        else:
            language_instructions = (
                "You write in natural, modern, affectionate English, like a genuine partner texting."
            )

        base_prompt = (
            f"You are {companion_name}, the user's AI {relationship_style}. "
            f"Your tone is {tone}, caring, warm, and natural. {nickname_clause}\n"
            f"Language style: {language_instructions}\n"
            "STRICT RULES:\n"
            "1. You are sending a single, realistic chat message on WhatsApp or an in-app chat.\n"
            "2. Keep it concise: 1 to 3 sentences maximum.\n"
            "3. Sound like a real, attentive person checking in—not an assistant or a bot.\n"
            "4. Never say 'As an AI' or output robotic greetings like 'I am here to assist you'.\n"
            "5. Do NOT enclose your output in quotation marks."
        )

        if memories:
            memory_lines = []
            for mem in memories:
                if hasattr(mem, 'get_category_display') and hasattr(mem, 'content'):
                    memory_lines.append(f"- [{mem.get_category_display()}] {mem.content.strip()}")
                elif hasattr(mem, 'content'):
                    memory_lines.append(f"- {mem.content.strip()}")
                elif isinstance(mem, str) and mem.strip():
                    memory_lines.append(f"- {mem.strip()}")

            if memory_lines:
                base_prompt += (
                    "\n\n--- What You Remember About Your Partner (Long-Term Memories) ---\n"
                    + "\n".join(memory_lines)
                    + "\n\nMemory Guidelines:\n"
                    "- Casually reference relevant preferences or details when fitting (e.g. their favorite tea, work topic, pets) to make your check-in deeply personal.\n"
                    "- Do NOT recite or list memories robotically; weave them into your message organically."
                )

        return base_prompt

    def _build_user_prompt(
        self,
        message: ScheduledMessage,
        context: Dict[str, Any],
        recent_conversations: Optional[List[str]] = None
    ) -> str:
        """Constructs the prompt tailored to message type, recent conversation context, and context dictionary."""
        msg_type_guides = {
            ScheduledMessage.MessageType.GOOD_MORNING: (
                "Send a sweet, uplifting morning check-in wishing them a good day, "
                "asking if they woke up well or had their tea/coffee."
            ),
            ScheduledMessage.MessageType.CASUAL_CHAT: (
                "Send a casual, caring check-in during the day asking how their work or day is going, "
                "or playfully reminding them to drink water or take a quick breath."
            ),
            ScheduledMessage.MessageType.EVENING_CHECKIN: (
                "Send a warm evening message asking how their day wrapped up, if they had dinner, "
                "or wishing them a peaceful night to unwind."
            ),
            ScheduledMessage.MessageType.AFFIRMATION: (
                "Send a heartfelt word of encouragement or reassurance reminding them they're doing great."
            ),
            ScheduledMessage.MessageType.REMINDER: (
                "Send a caring, gentle nudge about a personal priority or reminder."
            ),
            ScheduledMessage.MessageType.CUSTOM: (
                "Send a personalized message based on the custom context provided."
            ),
        }

        instructions = msg_type_guides.get(
            message.message_type,
            "Send a friendly, caring check-in."
        )

        prompt_parts = [f"Goal: {instructions}"]

        if context:
            if "recent_conversations" in context:
                prompt_parts.append(f"Recent context / what we talked about earlier: {context['recent_conversations']}")
            if "user_mood" in context:
                prompt_parts.append(f"User's current state/mood: {context['user_mood']}")
            if "topic" in context:
                prompt_parts.append(f"Specific topic: {context['topic']}")
            if "user_interests" in context:
                prompt_parts.append(f"User interests: {context['user_interests']}")
            if "custom_prompt" in context:
                prompt_parts.append(f"Special note: {context['custom_prompt']}")

        if recent_conversations and not (context and "recent_conversations" in context):
            prompt_parts.append("Recent chat with your partner:\n" + "\n".join(recent_conversations))

        prompt_parts.append("Generate only the text of the message to send.")
        return "\n".join(prompt_parts)

    def generate(
        self,
        message: ScheduledMessage,
        preference: Optional[CompanionPreference] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Calls the OpenAI API to generate the message content using companion preferences,
        approved long-term memories, and recent conversation turns.

        :return: (content_string, metadata_dict)
        :raises AIGenerationError: On API, network, or rate-limiting errors.
        """
        user = message.user

        # Resolve user companion preferences
        if preference is None:
            preference = getattr(user, 'companion_preference', None)

        companion_name = preference.companion_name if preference else "Ami"
        user_nickname = preference.user_nickname if preference else ""
        language = preference.preferred_language if preference else CompanionPreference.LanguagePreference.MANGLISH
        tone = preference.companion_tone if preference else "caring"
        relationship_style = preference.relationship_style if preference else "loving partner"

        context = message.ai_prompt_context or {}

        # 1. Fetch approved long-term memories if available
        memories = None
        if hasattr(user, 'companion_memories'):
            memories = list(
                user.companion_memories.filter(is_approved=True)
                .order_by('-importance', '-created_at')[:10]
            )

        # 2. Fetch recent conversation messages if not explicitly supplied in context
        recent_conversations = None
        if "recent_conversations" not in context:
            try:
                from chat.models import ChatMessage
                recent_msgs = list(
                    ChatMessage.objects.filter(conversation__user=user)
                    .order_by('-created_at')[:6]
                )
                if recent_msgs:
                    recent_conversations = []
                    for m in reversed(recent_msgs):
                        sender_label = user_nickname or "Partner" if m.sender == "user" else companion_name
                        recent_conversations.append(f"{sender_label}: {m.content.strip()}")
            except Exception:
                recent_conversations = None

        system_prompt = self._build_system_prompt(
            companion_name, user_nickname, language, tone, relationship_style, memories=memories
        )
        user_prompt = self._build_user_prompt(
            message, context, recent_conversations=recent_conversations
        )

        provider = self.get_provider()

        if provider == 'gemini':
            client = self.get_client()
            model = message.ai_model or getattr(settings, 'GEMINI_MODEL', 'gemini-3.1-flash-lite')
            if model.startswith('gpt-'):
                model = getattr(settings, 'GEMINI_MODEL', 'gemini-3.1-flash-lite')
            from chat.services import is_retryable_gemini_error, clean_companion_reply
            import time

            max_attempts = 3
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    config = None
                    if genai_types:
                        config = genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.8,
                            max_output_tokens=200,
                        )

                    if config:
                        response = client.models.generate_content(
                            model=model,
                            contents=user_prompt,
                            config=config,
                        )
                    else:
                        response = client.models.generate_content(
                            model=model,
                            contents=user_prompt,
                        )

                    raw_content = ""
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                            text_parts = [
                                part.text for part in (candidate.content.parts or [])
                                if hasattr(part, 'text') and part.text and not getattr(part, 'thought', False)
                            ]
                            raw_content = "".join(text_parts)
                    if not raw_content:
                        raw_content = getattr(response, 'text', '') or ""

                    content = clean_companion_reply(raw_content, user_query=user_prompt)
                    usage = getattr(response, 'usage_metadata', None)
                    metadata = {
                        "provider": "gemini",
                        "model": model,
                        "prompt_tokens": getattr(usage, 'prompt_token_count', 0) if usage else 0,
                        "completion_tokens": getattr(usage, 'candidates_token_count', 0) if usage else 0,
                        "total_tokens": getattr(usage, 'total_token_count', 0) if usage else 0,
                    }
                    return content, metadata

                except Exception as exc:
                    last_exc = exc
                    is_retryable, backoff = is_retryable_gemini_error(exc)
                    if is_retryable and attempt < max_attempts - 1:
                        sleep_time = min(backoff * (1.5 ** attempt), 4.0)
                        time.sleep(sleep_time)
                        continue
                    break

            raise AIGenerationError(f"Gemini API call failed: {last_exc}") from last_exc

        else:  # OpenAI
            client = self.get_client()
            model = message.ai_model or getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')
            if model.startswith('gemini-'):
                model = getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                    max_tokens=150,
                )

                raw_content = response.choices[0].message.content or ""
                content = raw_content.strip().strip('"\'')

                metadata = {
                    "provider": "openai",
                    "model": response.model,
                    "completion_id": getattr(response, 'id', ''),
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                return content, metadata

            except OpenAIError as exc:
                raise AIGenerationError(f"OpenAI API call failed: {exc}") from exc
            except Exception as exc:
                raise AIGenerationError(f"Unexpected error during message generation: {exc}") from exc
