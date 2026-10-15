"""
Django settings for chatpartner project.
"""
from pathlib import Path
import os
import dj_database_url
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file (override=True ensures .env values supersede inherited environment)
load_dotenv(BASE_DIR / '.env', override=True)

# Security settings
SECRET_KEY = os.getenv('SECRET_KEY', 'chatpartner-insecure-dev-secret-key')
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')

allowed_hosts_env = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,testserver')
ALLOWED_HOSTS = [host.strip() for host in allowed_hosts_env.split(',') if host.strip()]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party apps
    'corsheaders',
    'rest_framework',

    # Local apps
    'chat',
    'proactive_messaging',
    'instagram',
]


MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

# Database configuration: Supports Neon PostgreSQL via DATABASE_URL and local fallback
def get_database_config(database_url=None, env=None):
    """
    Builds Django DATABASES configuration dictionary.
    Prioritizes DATABASE_URL (for Neon/production) with secure SSL enforcement.
    Falls back to individual local DB environment variables when DATABASE_URL is absent.
    Credentials are read from environment variables only (no hardcoded password fallback).
    """
    env_lookup = (lambda k, default='': env.get(k, default)) if env is not None else os.getenv
    raw_url = database_url if database_url is not None else env_lookup('DATABASE_URL', '')
    db_url = (raw_url or '').strip()

    if db_url:
        ssl_require = str(env_lookup('DB_SSL_REQUIRE', 'True')).lower() in ('true', '1', 't')
        conn_max_age_val = env_lookup('DB_CONN_MAX_AGE', '600')
        try:
            conn_max_age = int(conn_max_age_val)
        except (ValueError, TypeError):
            conn_max_age = 600

        config = dj_database_url.parse(
            db_url,
            conn_max_age=conn_max_age,
            conn_health_checks=True,
            ssl_require=ssl_require,
        )
        if ssl_require:
            config.setdefault('OPTIONS', {})
            config['OPTIONS'].setdefault('sslmode', 'require')
        return {'default': config}

    db_options = {}
    ssl_mode = env_lookup('DB_SSLMODE', '')
    if ssl_mode:
        db_options['sslmode'] = ssl_mode

    return {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': env_lookup('DB_NAME', 'chatpartner_db'),
            'USER': env_lookup('DB_USER', 'ashique'),
            'PASSWORD': env_lookup('DB_PASSWORD', ''),
            'HOST': env_lookup('DB_HOST', '127.0.0.1'),
            'PORT': env_lookup('DB_PORT', '5433'),
            'OPTIONS': db_options,
        }
    }


DATABASES = get_database_config()

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (Audio attachments, user voice messages, AI voice replies)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Voice Messaging Configuration
VOICE_STT_PROVIDER = os.getenv('VOICE_STT_PROVIDER', 'gemini').lower().strip()
VOICE_TTS_PROVIDER = os.getenv('VOICE_TTS_PROVIDER', 'edge_tts').lower().strip()
VOICE_TTS_VOICE = os.getenv('VOICE_TTS_VOICE', 'en-IN-NeerjaNeural')
VOICE_TTS_RATE = os.getenv('VOICE_TTS_RATE', '+0%')
VOICE_TTS_PITCH = os.getenv('VOICE_TTS_PITCH', '+0Hz')
VOICE_MAX_UPLOAD_SIZE_MB = int(os.getenv('VOICE_MAX_UPLOAD_SIZE_MB', '15'))

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
}

# AI Provider Configuration ('gemini' or 'openai')
AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini').lower().strip()

# Google Gemini API Configuration (stored in environment variables)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')

# OpenAI API Configuration (stored in environment variables)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Delivery Channel Configuration
IN_APP_MESSAGING_ENABLED = os.getenv('IN_APP_MESSAGING_ENABLED', 'True').lower() in ('true', '1', 't')
PUSH_NOTIFICATIONS_ENABLED = os.getenv('PUSH_NOTIFICATIONS_ENABLED', 'False').lower() in ('true', '1', 't')

# WebPush / VAPID Credentials (stored in environment variables)
VAPID_PUBLIC_KEY = os.getenv('VAPID_PUBLIC_KEY', '')
VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY', '')
VAPID_ADMIN_EMAIL = os.getenv('VAPID_ADMIN_EMAIL', '')

# Firebase Cloud Messaging (FCM) Credentials (stored in environment variables)
FCM_SERVER_KEY = os.getenv('FCM_SERVER_KEY', '')

# Meta / Instagram Messaging API Configuration (stored in environment variables)
INSTAGRAM_ACCOUNT_ID = os.getenv('INSTAGRAM_ACCOUNT_ID', '')
INSTAGRAM_ACCESS_TOKEN = os.getenv('INSTAGRAM_ACCESS_TOKEN', '')
INSTAGRAM_APP_SECRET = os.getenv('INSTAGRAM_APP_SECRET', '')
INSTAGRAM_VERIFY_TOKEN = os.getenv('INSTAGRAM_VERIFY_TOKEN', '')
INSTAGRAM_API_VERSION = os.getenv('INSTAGRAM_API_VERSION', 'v20.0')

# CORS Configuration
CORS_ALLOW_ALL_ORIGINS = os.getenv('CORS_ALLOW_ALL_ORIGINS', 'False').lower() in ('true', '1', 't')
cors_origins_env = os.getenv(
    'CORS_ALLOWED_ORIGINS',
    'http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000'
)
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in cors_origins_env.split(',') if origin.strip()]
CORS_ALLOW_CREDENTIALS = True

# CSRF and Session Security Configuration for React Single-Page Application
IS_PROD = not DEBUG

CSRF_COOKIE_HTTPONLY = False  # Allows frontend client to read csrftoken cookie if same-origin
CSRF_COOKIE_NAME = 'csrftoken'
CSRF_HEADER_NAME = 'HTTP_X_CSRFTOKEN'

SESSION_COOKIE_SAMESITE = 'None' if IS_PROD else 'Lax'
CSRF_COOKIE_SAMESITE = 'None' if IS_PROD else 'Lax'
SESSION_COOKIE_SECURE = IS_PROD
CSRF_COOKIE_SECURE = IS_PROD

if IS_PROD:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

CSRF_TRUSTED_ORIGINS = [
    origin for origin in CORS_ALLOWED_ORIGINS
    if origin.startswith('http://') or origin.startswith('https://')
]




