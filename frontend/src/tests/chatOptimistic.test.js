import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

/**
 * Pure state reducer helpers modeling App.jsx chat optimistic update flows.
 */
function createOptimisticUserMessage(text) {
  return {
    id: `client-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    sender: 'user',
    text: text.trim(),
    timestamp: '10:00 AM',
    status: 'sending',
  };
}

function reconcileMessageSuccess(messages, tempId, data) {
  const updated = messages.map((m) =>
    m.id === tempId
      ? {
          ...m,
          id: data.user_message_id || m.id,
          status: 'sent',
        }
      : m
  );

  const companionMsg = {
    id: data.message_id || `ami-${Date.now()}`,
    sender: 'companion',
    text: data.reply,
    timestamp: '10:00 AM',
    status: 'sent',
  };

  return [...updated, companionMsg];
}

function reconcileMessageFailure(messages, tempId, errorMessage) {
  return messages.map((m) =>
    m.id === tempId
      ? { ...m, status: 'failed', errorMessage }
      : m
  );
}

function retryFailedMessage(messages, messageId) {
  return messages.map((m) =>
    m.id === messageId
      ? { ...m, status: 'sending', errorMessage: null }
      : m
  );
}

function safeMergeBackgroundSync(prevMessages, serverMessages, isSubmitting, isLoading) {
  // 1. Skip sync if a message is actively in flight
  if (isSubmitting || isLoading || prevMessages.some((m) => m.status === 'sending')) {
    return prevMessages;
  }

  const pendingFailed = prevMessages.filter((m) => m.status === 'failed');

  // 2. If no failed messages and length/last ID match, no update needed
  if (
    pendingFailed.length === 0 &&
    serverMessages.length === prevMessages.length &&
    serverMessages.length > 0 &&
    String(serverMessages[serverMessages.length - 1].id) === String(prevMessages[prevMessages.length - 1].id)
  ) {
    return prevMessages;
  }

  const serverFormatted = serverMessages.map((m) => ({
    ...m,
    status: 'sent',
  }));

  // 3. Preserve failed messages so retry is never lost
  return [...serverFormatted, ...pendingFailed];
}

describe('Ami Chat Optimistic UI State Transitions', () => {
  it('1. Appends optimistic user message immediately with sending status', () => {
    const initialMessages = [];
    const text = 'Hey Ami, how are you?';

    const optimisticMsg = createOptimisticUserMessage(text);
    const messages = [...initialMessages, optimisticMsg];

    assert.equal(messages.length, 1);
    assert.equal(messages[0].sender, 'user');
    assert.equal(messages[0].text, text);
    assert.equal(messages[0].status, 'sending');
    assert.match(messages[0].id, /^client-/);
  });

  it('2. Reconciles optimistic message with server message without creating duplicates', () => {
    const optimisticMsg = createOptimisticUserMessage('Hello!');
    const tempId = optimisticMsg.id;
    let messages = [optimisticMsg];

    // Simulated API response from /api/chat/
    const apiResponse = {
      user_message_id: 'db-user-msg-42',
      message_id: 'db-companion-msg-43',
      reply: 'Njan ivide undu! Sugamaano?',
      conversation_id: 'conv-123',
    };

    messages = reconcileMessageSuccess(messages, tempId, apiResponse);

    // Should contain exactly 2 messages: user message (reconciled) + companion reply
    assert.equal(messages.length, 2);

    // User message should have server user_message_id and sent status
    assert.equal(messages[0].id, 'db-user-msg-42');
    assert.equal(messages[0].sender, 'user');
    assert.equal(messages[0].status, 'sent');
    assert.equal(messages[0].text, 'Hello!');

    // Companion reply appended
    assert.equal(messages[1].id, 'db-companion-msg-43');
    assert.equal(messages[1].sender, 'companion');
    assert.equal(messages[1].text, 'Njan ivide undu! Sugamaano?');
    assert.equal(messages[1].status, 'sent');
  });

  it('3. Handles request failure without losing the message and enables retry', () => {
    const optimisticMsg = createOptimisticUserMessage('Can you hear me?');
    const tempId = optimisticMsg.id;
    let messages = [optimisticMsg];

    // Simulate API network or server error
    messages = reconcileMessageFailure(messages, tempId, 'Network error: 504 Gateway Timeout');

    // Message must NOT be removed from chat
    assert.equal(messages.length, 1);
    assert.equal(messages[0].id, tempId);
    assert.equal(messages[0].status, 'failed');
    assert.equal(messages[0].errorMessage, 'Network error: 504 Gateway Timeout');

    // User clicks Retry
    messages = retryFailedMessage(messages, tempId);
    assert.equal(messages[0].status, 'sending');
    assert.equal(messages[0].errorMessage, null);

    // On retry success
    const retryResponse = {
      user_message_id: 'db-user-msg-99',
      message_id: 'db-companion-msg-100',
      reply: 'Yes, I hear you loud and clear!',
    };
    messages = reconcileMessageSuccess(messages, tempId, retryResponse);

    assert.equal(messages.length, 2);
    assert.equal(messages[0].id, 'db-user-msg-99');
    assert.equal(messages[0].status, 'sent');
    assert.equal(messages[1].id, 'db-companion-msg-100');
  });

  it('4. Prevents duplicate submissions from rapid clicks with synchronous lock', () => {
    let isSubmitting = false;
    let messages = [];

    const submit = (text) => {
      if (isSubmitting || !text.trim()) return false;
      isSubmitting = true; // Synchronous lock
      messages = [...messages, createOptimisticUserMessage(text)];
      return true;
    };

    // First click: accepted
    const firstCall = submit('Quick message');
    assert.equal(firstCall, true);
    assert.equal(messages.length, 1);

    // Second rapid click (before first completes): rejected by synchronous lock
    const secondCall = submit('Quick message');
    assert.equal(secondCall, false);
    assert.equal(messages.length, 1); // No duplicate message added
  });

  it('5. Safe background sync does not overwrite in-flight sending or failed messages', () => {
    const inFlightMsg = createOptimisticUserMessage('Sending right now...');
    const prevMessages = [
      { id: '1', sender: 'user', text: 'Old message', status: 'sent' },
      inFlightMsg,
    ];

    const staleServerMessages = [
      { id: '1', sender: 'user', text: 'Old message' },
    ];

    // Background sync attempts to run while message is sending
    const resultDuringSend = safeMergeBackgroundSync(
      prevMessages,
      staleServerMessages,
      true, // isSubmitting
      true  // isLoading
    );

    // Must return prevMessages untouched
    assert.equal(resultDuringSend.length, 2);
    assert.equal(resultDuringSend[1].status, 'sending');
    assert.equal(resultDuringSend[1].text, 'Sending right now...');

    // Now test with a failed message preserved during sync
    const failedMsg = { id: 'client-fail-1', sender: 'user', text: 'Failed message', status: 'failed' };
    const prevWithFailed = [
      { id: '1', sender: 'user', text: 'Old message', status: 'sent' },
      failedMsg,
    ];

    const resultWithFailed = safeMergeBackgroundSync(
      prevWithFailed,
      staleServerMessages,
      false,
      false
    );

    // Server messages formatted + failed message preserved at the end
    assert.equal(resultWithFailed.length, 2);
    assert.equal(resultWithFailed[0].id, '1');
    assert.equal(resultWithFailed[1].id, 'client-fail-1');
    assert.equal(resultWithFailed[1].status, 'failed');
  });
});
