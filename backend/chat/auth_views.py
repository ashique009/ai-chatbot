"""
Session-based authentication views with CSRF protection.
Provides secure login, registration, logout, session verification, and CSRF token provisioning.
"""
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


@api_view(['GET'])
@permission_classes([AllowAny])
@ensure_csrf_cookie
def get_csrf_token(request):
    """
    GET /api/auth/csrf/
    Ensures the CSRF cookie is set on the response and returns the token in JSON.
    """
    token = get_token(request)
    return Response({'csrfToken': token}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """
    POST /api/auth/login/
    Accepts: { "username": "...", "password": "..." }
    Authenticates user against PostgreSQL and begins a secure Django session.
    """
    if not isinstance(request.data, dict):
        return Response({'error': 'Invalid request format. Expected JSON object.'}, status=status.HTTP_400_BAD_REQUEST)

    raw_username = request.data.get('username')
    raw_password = request.data.get('password')

    if raw_username is None or raw_password is None:
        return Response({'error': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    username = str(raw_username).strip()
    password = str(raw_password)

    if not username or not password:
        return Response({'error': 'Username and password cannot be empty.'}, status=status.HTTP_400_BAD_REQUEST)

    # 1. Attempt exact username authentication
    user = authenticate(request, username=username, password=password)

    # 2. If exact match fails, try stripping password in case of trailing spaces
    if user is None and password.strip() != password:
        user = authenticate(request, username=username, password=password.strip())

    # 3. If still None, attempt case-insensitive username lookup
    if user is None:
        candidate = User.objects.filter(username__iexact=username).first()
        if candidate:
            user = authenticate(request, username=candidate.username, password=password)
            if user is None and password.strip() != password:
                user = authenticate(request, username=candidate.username, password=password.strip())

    if user is None:
        return Response({'error': 'Invalid username or password.'}, status=status.HTTP_401_UNAUTHORIZED)

    if not user.is_active:
        return Response({'error': 'User account is inactive. Please contact support.'}, status=status.HTTP_403_FORBIDDEN)

    login(request, user)
    new_csrf = get_token(request)

    user_payload = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'authenticated': True,
    }

    return Response({
        **user_payload,
        'user': user_payload,
        'csrfToken': new_csrf,
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    POST /api/auth/logout/
    Terminates the user session.
    """
    logout(request)
    return Response({'message': 'Logged out successfully.', 'authenticated': False}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    """
    POST /api/auth/register/
    Accepts: { "username": "...", "password": "...", "email": "..." (optional) }
    Creates user account, begins session, and returns user details.
    """
    if not isinstance(request.data, dict):
        return Response({'error': 'Invalid request format. Expected JSON object.'}, status=status.HTTP_400_BAD_REQUEST)

    raw_username = request.data.get('username')
    raw_password = request.data.get('password')
    raw_email = request.data.get('email', '')

    if raw_username is None or raw_password is None:
        return Response({'error': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    username = str(raw_username).strip()
    password = str(raw_password)
    email = str(raw_email).strip() if raw_email else ''

    if not username:
        return Response({'error': 'Username is required.'}, status=status.HTTP_400_BAD_REQUEST)
    if len(username) < 3:
        return Response({'error': 'Username must be at least 3 characters.'}, status=status.HTTP_400_BAD_REQUEST)
    if not password:
        return Response({'error': 'Password is required.'}, status=status.HTTP_400_BAD_REQUEST)
    if len(password) < 6:
        return Response({'error': 'Password must be at least 6 characters.'}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username__iexact=username).exists():
        return Response({'error': 'A user with that username already exists.'}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, password=password, email=email)
    login(request, user)
    new_csrf = get_token(request)

    user_payload = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'authenticated': True,
    }

    return Response({
        **user_payload,
        'user': user_payload,
        'csrfToken': new_csrf,
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([AllowAny])
def current_user_view(request):
    """
    GET /api/auth/me/
    Returns current authenticated user details or 401 if unauthenticated.
    """
    if request.user.is_authenticated:
        user_payload = {
            'id': request.user.id,
            'username': request.user.username,
            'email': request.user.email,
            'first_name': request.user.first_name,
            'authenticated': True,
        }
        return Response({
            **user_payload,
            'user': user_payload,
        }, status=status.HTTP_200_OK)

    return Response({'authenticated': False, 'error': 'Not authenticated.'}, status=status.HTTP_401_UNAUTHORIZED)

