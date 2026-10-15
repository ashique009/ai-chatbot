from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status


class HealthCheckAPITest(TestCase):
    """Tests for system health check endpoint and CORS support."""

    def setUp(self):
        self.client = APIClient()

    def test_health_check_endpoint_returns_healthy(self):
        """Tests that GET /api/health/ returns 200 OK and healthy status."""
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertEqual(data.get('services', {}).get('api'), 'operational')
        self.assertEqual(data.get('services', {}).get('database'), 'connected')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)

    def test_health_check_alias_endpoint(self):
        """Tests that GET /api/health-check/ alias endpoint works."""
        response = self.client.get('/api/health-check/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json().get('status'), 'healthy')

    def test_cors_headers_present_for_allowed_origin(self):
        """Tests that CORS response headers are included for frontend origin."""
        response = self.client.get(
            '/api/health/',
            HTTP_ORIGIN='http://localhost:5173'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.get('Access-Control-Allow-Origin'), 'http://localhost:5173')
