/**
 * API client configuration and companion endpoints.
 * Includes session credentials, in-memory CSRF token handling, and authentication error handling.
 */
const RAW_API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
export const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, '');

let cachedCsrfToken = null;

/**
 * Fetches and caches the CSRF token from Django backend into memory.
 */
export async function fetchCsrfToken() {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/csrf/`, {
      method: 'GET',
      credentials: 'include',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      const data = await response.json();
      if (data?.csrfToken) {
        cachedCsrfToken = data.csrfToken;
      }
      return cachedCsrfToken;
    }
  } catch (err) {
    // Gracefully capture network errors without uncaught promise rejection
    console.warn('Failed to fetch CSRF token:', err);
  }
  return cachedCsrfToken;
}

/**
 * Returns the active CSRF token from in-memory cache.
 */
export function getCsrfToken() {
  return cachedCsrfToken;
}

/**
 * Wraps native fetch with credentials: 'include' and automatic CSRF headers.
 */
export async function secureFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Accept': 'application/json',
    ...(options.headers || {}),
  };

  const method = (options.method || 'GET').toUpperCase();
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    if (!cachedCsrfToken) {
      await fetchCsrfToken();
    }
    const csrfToken = getCsrfToken();
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }
  }

  const response = await fetch(url, {
    ...options,
    headers,
    credentials: 'include',
  });

  return response;
}

// ============================================================================
// Authentication APIs
// ============================================================================

export async function loginUser(username, password) {
  // Ensure we have a fresh CSRF token prior to authentication attempt
  await fetchCsrfToken();
  const response = await secureFetch('/auth/login/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.error || `Login failed (${response.status})`;
    const err = new Error(errorMsg);
    err.status = response.status;
    throw err;
  }
  if (data?.csrfToken) {
    cachedCsrfToken = data.csrfToken;
  }
  return data;
}

export async function registerUser(username, password, email = '') {
  // Ensure we have a fresh CSRF token prior to registration attempt
  await fetchCsrfToken();
  const response = await secureFetch('/auth/register/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, email }),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.error || `Registration failed (${response.status})`;
    const err = new Error(errorMsg);
    err.status = response.status;
    throw err;
  }
  if (data?.csrfToken) {
    cachedCsrfToken = data.csrfToken;
  }
  return data;
}

export async function logoutUser() {
  const response = await secureFetch('/auth/logout/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  cachedCsrfToken = null;
  return response.ok;
}

export async function getCurrentUser() {
  const response = await secureFetch('/auth/me/', {
    method: 'GET',
  });

  if (!response.ok) {
    return null;
  }
  return response.json();
}

// ============================================================================
// Health Check API
// ============================================================================

export async function checkBackendHealth() {
  const response = await secureFetch('/health/', {
    method: 'GET',
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Backend returned status ${response.status}: ${errorBody || response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Interactive Chat API
// ============================================================================

export async function sendChatMessage(message, conversationId = null) {
  const payload = { message };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }

  const response = await secureFetch('/chat/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.error || `Request failed with status ${response.status}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    throw err;
  }

  return data;
}

export async function sendVoiceChatMessage({
  audioBlob,
  conversationId = null,
  duration = null,
  clientMessageId = null,
  transcript = '',
}) {
  const formData = new FormData();
  if (audioBlob) {
    const filename = audioBlob.type.includes('mp4')
      ? 'recording.mp4'
      : audioBlob.type.includes('ogg')
      ? 'recording.ogg'
      : 'recording.webm';
    formData.append('audio', audioBlob, filename);
  }
  if (conversationId) {
    formData.append('conversation_id', conversationId);
  }
  if (duration !== null && duration !== undefined) {
    formData.append('duration', String(duration));
  }
  if (clientMessageId) {
    formData.append('client_message_id', clientMessageId);
  }
  if (transcript) {
    formData.append('transcript', transcript);
  }

  const response = await secureFetch('/chat/voice/', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.error || `Voice request failed with status ${response.status}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.errorCategory = data?.error_category;
    throw err;
  }

  return data;
}

export async function synthesizeMessageVoice(messageId) {
  const response = await secureFetch(`/messages/${messageId}/synthesize-voice/`, {
    method: 'POST',
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.error || `Failed to synthesize voice (${response.status})`;
    const err = new Error(errorMsg);
    err.status = response.status;
    throw err;
  }

  return data;
}

export function getMessageAudioUrl(messageId) {
  return `${API_BASE_URL}/messages/${messageId}/audio/`;
}

// ============================================================================
// Conversation Management APIs
// ============================================================================

export async function listConversations() {
  const response = await secureFetch('/conversations/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load conversations (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function createConversation(title = 'New Conversation') {
  const response = await secureFetch('/conversations/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) {
    const err = new Error(`Failed to create conversation (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function getConversation(conversationId) {
  const response = await secureFetch(`/conversations/${conversationId}/`, {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load conversation history (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function deleteConversation(conversationId) {
  const response = await secureFetch(`/conversations/${conversationId}/`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const err = new Error(`Failed to delete conversation (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function clearConversationMessages(conversationId) {
  const response = await secureFetch(`/conversations/${conversationId}/clear/`, {
    method: 'POST',
  });
  if (!response.ok) {
    const err = new Error(`Failed to clear conversation messages (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}


// ============================================================================
// Long-Term Memory Management APIs
// ============================================================================

export async function listMemories() {
  const response = await secureFetch('/memories/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load memories (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function createMemory({ content, category = 'personal', importance = 1, is_approved = true }) {
  const response = await secureFetch('/memories/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, category, importance, is_approved }),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to create memory (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

export async function updateMemory(memoryId, updates) {
  const response = await secureFetch(`/memories/${memoryId}/`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to update memory (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

export async function deleteMemory(memoryId) {
  const response = await secureFetch(`/memories/${memoryId}/`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const err = new Error(`Failed to delete memory (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

// ============================================================================
// Proactive Companion Messaging APIs (Preferences, Inbox, Scheduling)
// ============================================================================

export async function getCompanionPreferences() {
  const response = await secureFetch('/proactive/preferences/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load preferences (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function updateCompanionPreferences(updates) {
  const response = await secureFetch('/proactive/preferences/', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to update preferences (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

export async function listInboxMessages() {
  const response = await secureFetch('/proactive/inbox/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load in-app inbox (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function markInboxMessageRead(messageId, isRead = true) {
  const response = await secureFetch(`/proactive/inbox/${messageId}/`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ is_read: isRead }),
  });
  if (!response.ok) {
    const err = new Error(`Failed to update message (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function markAllInboxMessagesRead() {
  const response = await secureFetch('/proactive/inbox/read-all/', {
    method: 'POST',
  });
  if (!response.ok) {
    const err = new Error(`Failed to mark all as read (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function deleteInboxMessage(messageId) {
  const response = await secureFetch(`/proactive/inbox/${messageId}/`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const err = new Error(`Failed to delete inbox message (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function listScheduledMessages() {
  const response = await secureFetch('/proactive/scheduled/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to load scheduled messages (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function generateDailySchedule(targetDate = null) {
  const payload = targetDate ? { target_date: targetDate } : {};
  const response = await secureFetch('/proactive/schedule-daily/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to generate daily schedule (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

// ============================================================================
// Web Push Notification APIs
// ============================================================================

export function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

export async function getVapidPublicKey() {
  const response = await secureFetch('/proactive/push/vapid-public-key/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to get VAPID public key (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function subscribePushDevice(subscription) {
  const payload = {
    endpoint: subscription.endpoint,
    keys: subscription.keys || {},
    device_type: 'web',
  };
  const response = await secureFetch('/proactive/push/subscribe/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to register push subscription (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

export async function unsubscribePushDevice(endpoint = null) {
  const payload = endpoint ? { endpoint } : {};
  const response = await secureFetch('/proactive/push/unsubscribe/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = new Error(`Failed to unsubscribe push (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function listPushSubscriptions() {
  const response = await secureFetch('/proactive/push/subscriptions/', {
    method: 'GET',
  });
  if (!response.ok) {
    const err = new Error(`Failed to list push subscriptions (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function deletePushSubscription(subscriptionId) {
  const response = await secureFetch(`/proactive/push/subscriptions/${subscriptionId}/`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const err = new Error(`Failed to delete subscription (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

export async function sendTestPushNotification(message = null, title = null) {
  const payload = {};
  if (message) payload.message = message;
  if (title) payload.title = title;
  const response = await secureFetch('/proactive/push/test/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = new Error(data?.error || `Failed to send test push notification (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return data;
}

export default {
  API_BASE_URL,
  fetchCsrfToken,
  getCsrfToken,
  loginUser,
  registerUser,
  logoutUser,
  getCurrentUser,
  checkBackendHealth,
  sendChatMessage,
  listConversations,
  createConversation,
  getConversation,
  deleteConversation,
  listMemories,
  createMemory,
  updateMemory,
  deleteMemory,
  getCompanionPreferences,
  updateCompanionPreferences,
  listInboxMessages,
  markInboxMessageRead,
  markAllInboxMessagesRead,
  deleteInboxMessage,
  listScheduledMessages,
  generateDailySchedule,
  urlBase64ToUint8Array,
  getVapidPublicKey,
  subscribePushDevice,
  unsubscribePushDevice,
  listPushSubscriptions,
  deletePushSubscription,
  sendTestPushNotification,
};

