"""
Voice messaging services for Ami AI Companion.
Supports:
1. Speech-to-Text (STT): Audio transcription using Google Gemini multimodal capabilities or OpenAI Whisper.
2. Text-to-Speech (TTS): Natural female voice reply generation using edge-tts (neural Indian English/Malayalam female voices) or OpenAI TTS.
3. Audio validation and file handling.
"""
import asyncio
import io
import logging
import os
import re
from typing import Tuple, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class VoiceServiceError(Exception):
    """Base exception for voice messaging errors."""
    def __init__(self, message: str, error_category: str = "voice_error"):
        super().__init__(message)
        self.error_category = error_category


class TranscriptionError(VoiceServiceError):
    """Raised when speech-to-text fails or returns empty."""
    def __init__(self, message: str = "Speech recognition failed."):
        super().__init__(message, error_category="transcription_error")


class SynthesisError(VoiceServiceError):
    """Raised when text-to-speech synthesis fails."""
    def __init__(self, message: str = "Voice synthesis failed."):
        super().__init__(message, error_category="synthesis_error")


def clean_text_for_speech(text: str) -> str:
    """
    Cleans text to produce natural, smooth spoken audio:
    - Removes internal mood tags like [mood: calm]
    - Removes emojis that could be vocalized awkwardly
    - Removes action asterisks like *smiles* or *laughs*
    - Collapses duplicate punctuation and whitespace
    """
    if not text:
        return ""

    # Remove internal mood tags
    cleaned = re.sub(r"\[mood:\s*[^\]]+\]", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<!--\s*mood:\s*[^>]+-->", "", cleaned, flags=re.IGNORECASE)

    # Remove asterisks / action notes
    cleaned = re.sub(r"\*[^*]+\*", "", cleaned)

    # Remove emojis (standard unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess, etc.
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub("", cleaned)

    # Clean redundant punctuation and extra whitespace
    cleaned = re.sub(r"[!?.]{2,}", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def validate_audio_upload(uploaded_file, max_size_mb: Optional[int] = None) -> Tuple[str, str]:
    """
    Validates uploaded audio file:
    - Checks file presence and non-zero size
    - Enforces maximum file size limit
    - Validates allowed audio MIME types and extensions

    Returns (clean_mime_type, normalized_extension).
    """
    if not uploaded_file:
        raise ValueError("No audio file was provided in the upload request.")

    max_mb = max_size_mb or getattr(settings, 'VOICE_MAX_UPLOAD_SIZE_MB', 15)
    max_bytes = max_mb * 1024 * 1024

    if uploaded_file.size <= 0:
        raise ValueError("The uploaded audio file is empty (0 bytes).")

    if uploaded_file.size > max_bytes:
        raise ValueError(f"Audio file size exceeds the allowed limit of {max_mb}MB.")

    content_type = getattr(uploaded_file, 'content_type', '') or ''
    clean_mime = content_type.split(';')[0].strip().lower()

    filename = getattr(uploaded_file, 'name', '') or ''
    ext = os.path.splitext(filename)[1].lower()

    allowed_mimes = {
        'audio/webm',
        'audio/ogg',
        'audio/mp4',
        'audio/wav',
        'audio/x-wav',
        'audio/wave',
        'audio/mpeg',
        'audio/mp3',
        'audio/aac',
        'audio/x-m4a',
        'audio/m4a',
        'video/webm',  # Some browsers label webm audio as video/webm
        'video/mp4',
    }

    allowed_exts = {'.webm', '.ogg', '.mp4', '.wav', '.mp3', '.m4a', '.aac'}

    # If MIME type is generic or missing, check file extension
    is_valid_mime = clean_mime in allowed_mimes
    is_valid_ext = ext in allowed_exts

    if not is_valid_mime and not is_valid_ext:
        raise ValueError(
            f"Unsupported audio format '{clean_mime or ext}'. "
            "Supported formats include WebM, OGG, MP4, WAV, M4A, and MP3."
        )

    # Normalize MIME type and extension
    if not clean_mime or clean_mime not in allowed_mimes:
        ext_to_mime = {
            '.webm': 'audio/webm',
            '.ogg': 'audio/ogg',
            '.mp4': 'audio/mp4',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
        }
        clean_mime = ext_to_mime.get(ext, 'audio/webm')

    if not ext:
        mime_to_ext = {
            'audio/webm': '.webm',
            'video/webm': '.webm',
            'audio/ogg': '.ogg',
            'audio/mp4': '.mp4',
            'video/mp4': '.mp4',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/wave': '.wav',
            'audio/mpeg': '.mp3',
            'audio/mp3': '.mp3',
            'audio/m4a': '.m4a',
            'audio/x-m4a': '.m4a',
            'audio/aac': '.aac',
        }
        ext = mime_to_ext.get(clean_mime, '.webm')

    return clean_mime, ext


class VoiceService:
    """
    Orchestrates speech-to-text transcription and text-to-speech generation.
    """

    def __init__(self):
        self.stt_provider = getattr(settings, 'VOICE_STT_PROVIDER', 'gemini').lower().strip()
        self.tts_provider = getattr(settings, 'VOICE_TTS_PROVIDER', 'edge_tts').lower().strip()
        self.default_voice = getattr(settings, 'VOICE_TTS_VOICE', 'en-IN-NeerjaNeural')
        self.rate = getattr(settings, 'VOICE_TTS_RATE', '+0%')
        self.pitch = getattr(settings, 'VOICE_TTS_PITCH', '+0Hz')

    # =========================================================================
    # Speech-to-Text (Transcription)
    # =========================================================================

    def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribes user voice note bytes into text.
        Supports English, Malayalam, and Manglish.
        """
        clean_mime = mime_type.split(';')[0].strip().lower()
        if not clean_mime:
            clean_mime = "audio/webm"

        if self.stt_provider == 'gemini':
            return self._transcribe_gemini(audio_data, clean_mime)
        elif self.stt_provider == 'openai':
            return self._transcribe_openai(audio_data, clean_mime)
        else:
            # Fallback to gemini if configured, else openai
            gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
            if gemini_key:
                return self._transcribe_gemini(audio_data, clean_mime)
            return self._transcribe_openai(audio_data, clean_mime)

    def _transcribe_gemini(self, audio_data: bytes, mime_type: str) -> str:
        """Transcribe audio using Google Gemini multimodal audio recognition."""
        from .services import CompanionChatService
        from google.genai import types as genai_types

        chat_service = CompanionChatService()
        client = chat_service.get_client()
        model = getattr(settings, 'GEMINI_MODEL', 'gemini-3.1-flash-lite')

        stt_prompt = (
            "You are a professional audio transcriber. Accurately transcribe the spoken voice message. "
            "The speaker may speak in English, Malayalam, or casual Manglish (Malayalam written phonetically in English script). "
            "Preserve Manglish words phonetically as spoken. "
            "Output ONLY the verbatim transcription with appropriate punctuation. "
            "Do NOT output timestamps, formatting tags, explanatory notes, disclaimers, or metadata."
        )

        try:
            # Normalize mime type for Gemini
            normalized_mime = mime_type
            if 'webm' in mime_type:
                normalized_mime = 'audio/webm'
            elif 'ogg' in mime_type:
                normalized_mime = 'audio/ogg'
            elif 'wav' in mime_type:
                normalized_mime = 'audio/wav'
            elif 'mp3' in mime_type or 'mpeg' in mime_type:
                normalized_mime = 'audio/mp3'
            elif 'mp4' in mime_type or 'm4a' in mime_type:
                normalized_mime = 'audio/mp4'

            contents = [
                genai_types.Part.from_bytes(data=audio_data, mime_type=normalized_mime),
                stt_prompt
            ]

            config = genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=300,
            )

            models_to_try = [model]
            for fallback in ['gemini-3.1-flash-lite-preview', 'gemini-3.1-flash-lite', 'gemini-3.5-flash-lite']:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

            last_exc = None
            for current_model in models_to_try:
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=current_model,
                            contents=contents,
                            config=config,
                        )

                        raw_transcript = ""
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                                parts = [
                                    p.text for p in (candidate.content.parts or [])
                                    if hasattr(p, 'text') and p.text and not getattr(p, 'thought', False)
                                ]
                                raw_transcript = "".join(parts)
                        if not raw_transcript:
                            raw_transcript = getattr(response, 'text', '') or ""

                        transcript = raw_transcript.strip().strip("\"'").strip()

                        # Clean any leading 'Transcription:' or 'Transcript:' label
                        transcript = re.sub(r"^(?:transcription|transcript)\s*:\s*", "", transcript, flags=re.IGNORECASE)

                        if not transcript:
                            raise TranscriptionError("Could not recognize any spoken words in the recording.")

                        return transcript

                    except TranscriptionError:
                        raise
                    except Exception as exc:
                        last_exc = exc
                        exc_str = str(exc).lower()
                        if ('503' in exc_str or 'high demand' in exc_str or '429' in exc_str) and attempt < 2:
                            import time
                            time.sleep(1.2 * (attempt + 1))
                            continue
                        break

            if last_exc:
                raise last_exc

        except TranscriptionError:
            raise
        except Exception as exc:
            logger.warning("Gemini audio transcription failed: %s", exc)
            raise TranscriptionError(f"Transcription failed: {exc}")

    def _transcribe_openai(self, audio_data: bytes, mime_type: str) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise TranscriptionError("OpenAI SDK is not installed.")

        api_key = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise TranscriptionError("OpenAI API key is not configured for audio transcription.")

        try:
            client = OpenAI(api_key=api_key)
            ext = '.webm'
            if 'ogg' in mime_type:
                ext = '.ogg'
            elif 'mp3' in mime_type or 'mpeg' in mime_type:
                ext = '.mp3'
            elif 'wav' in mime_type:
                ext = '.wav'
            elif 'mp4' in mime_type:
                ext = '.mp4'

            audio_file = io.BytesIO(audio_data)
            audio_file.name = f"user_voice{ext}"

            transcript_obj = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

            text = getattr(transcript_obj, 'text', '') or str(transcript_obj)
            cleaned = text.strip()
            if not cleaned:
                raise TranscriptionError("Could not recognize any spoken words in the recording.")
            return cleaned

        except TranscriptionError:
            raise
        except Exception as exc:
            logger.warning("OpenAI Whisper transcription failed: %s", exc)
            raise TranscriptionError(f"Transcription failed: {exc}")

    # =========================================================================
    # Text-to-Speech (Synthesis)
    # =========================================================================

    def synthesize_speech(self, text: str, voice: Optional[str] = None) -> Tuple[bytes, float]:
        """
        Synthesizes spoken audio from text using a natural female voice.
        Returns (audio_bytes, estimated_duration_seconds).
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            raise SynthesisError("No text available to synthesize.")

        selected_voice = voice or self.default_voice

        if self.tts_provider == 'edge_tts':
            return self._synthesize_edge_tts(cleaned_text, selected_voice)
        elif self.tts_provider == 'openai':
            return self._synthesize_openai(cleaned_text, selected_voice)
        elif self.tts_provider == 'disabled':
            raise SynthesisError("Voice synthesis is currently disabled.")
        else:
            # Default to edge_tts
            return self._synthesize_edge_tts(cleaned_text, selected_voice)

    def _synthesize_edge_tts(self, text: str, voice: str) -> Tuple[bytes, float]:
        """
        Synthesize speech using edge-tts (Microsoft Neural TTS, 100% free).
        Supports natural Indian English ('en-IN-NeerjaNeural') and Malayalam ('ml-IN-SobhanaNeural').
        """
        try:
            import edge_tts
        except ImportError:
            raise SynthesisError("edge-tts package is not installed.")

        async def _run_communicate():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            audio_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.extend(chunk["data"])
            return bytes(audio_buffer)

        try:
            # Safe event loop handling for Django sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If called inside an existing running event loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        audio_bytes = pool.submit(asyncio.run, _run_communicate()).result(timeout=20)
                else:
                    audio_bytes = loop.run_until_complete(_run_communicate())
            except RuntimeError:
                audio_bytes = asyncio.run(_run_communicate())

            if not audio_bytes:
                raise SynthesisError("Synthesized audio buffer is empty.")

            # Estimate duration based on word count: ~2.4 words/sec (144 words/minute) + 0.5s pause
            words = len(text.split())
            estimated_duration = round(max(1.0, (words / 2.4) + 0.5), 1)

            return audio_bytes, estimated_duration

        except SynthesisError:
            raise
        except Exception as exc:
            logger.warning("edge-tts synthesis failed: %s", exc)
            raise SynthesisError(f"TTS synthesis failed: {exc}")

    def _synthesize_openai(self, text: str, voice: str) -> Tuple[bytes, float]:
        """Synthesize speech using OpenAI TTS API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise SynthesisError("OpenAI SDK is not installed.")

        api_key = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise SynthesisError("OpenAI API key is not configured for voice synthesis.")

        try:
            client = OpenAI(api_key=api_key)
            # Default to 'nova' female voice if edge-tts voice string was passed
            openai_voice = voice if voice in ('nova', 'shimmer', 'alloy', 'echo', 'fable', 'onyx') else 'nova'

            response = client.audio.speech.create(
                model=getattr(settings, 'OPENAI_TTS_MODEL', 'tts-1'),
                voice=openai_voice,
                input=text,
            )

            audio_bytes = response.read()
            if not audio_bytes:
                raise SynthesisError("Synthesized audio buffer is empty.")

            words = len(text.split())
            estimated_duration = round(max(1.0, (words / 2.4) + 0.5), 1)

            return audio_bytes, estimated_duration

        except SynthesisError:
            raise
        except Exception as exc:
            logger.warning("OpenAI TTS synthesis failed: %s", exc)
            raise SynthesisError(f"TTS synthesis failed: {exc}")
