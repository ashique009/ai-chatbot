import { useState, useEffect, useRef } from 'react';
import {
  sendChatMessage,
  sendVoiceChatMessage,
  synthesizeMessageVoice,
  checkBackendHealth,
  listConversations,
  getConversation,
  createConversation,
  clearConversationMessages,
  loginUser,
  registerUser,
  logoutUser,
  getCurrentUser,
  fetchCsrfToken,
  getCompanionPreferences,
  updateCompanionPreferences,
} from './services/api';
import AudioPlayer from './components/AudioPlayer';
import VoiceRecorder from './components/VoiceRecorder';

export default function App() {
  // Authentication state
  const [currentUser, setCurrentUser] = useState(null);
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [authMode, setAuthMode] = useState('login'); // 'login' | 'register'
  const [authForm, setAuthForm] = useState({ username: '', password: '', email: '' });
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState(null);

  // Chat state
  const [currentConversationId, setCurrentConversationId] = useState(
    () => localStorage.getItem('chatpartner_active_conv_id') || null
  );
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [apiConnected, setApiConnected] = useState(true);

  // Voice player and synthesis state
  const [activePlayerId, setActivePlayerId] = useState(null);
  const [synthesizingMessageId, setSynthesizingMessageId] = useState(null);

  // UI state
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  // Proactive settings modal state
  const [showProactiveSettings, setShowProactiveSettings] = useState(false);
  const [proactivePrefs, setProactivePrefs] = useState({
    frequency: 'daily',
    quiet_hours_enabled: true,
    quiet_hours_start: '22:00',
    quiet_hours_end: '07:00',
  });
  const [isProactiveLoading, setIsProactiveLoading] = useState(false);
  const [isProactiveSaving, setIsProactiveSaving] = useState(false);
  const [proactiveStatus, setProactiveStatus] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const menuRef = useRef(null);

  // Scroll to bottom on messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Close 3-dots menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setIsMenuOpen(false);
      }
    };
    if (isMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isMenuOpen]);

  // Initial check: CSRF token, session authentication check
  useEffect(() => {
    checkBackendHealth()
      .then((health) => {
        const activeProvider = health?.services?.ai_provider || 'gemini';
        const isConfigured = health?.services?.ai_connected ?? health?.services?.[activeProvider]?.configured ?? true;
        setApiConnected(Boolean(isConfigured));
      })
      .catch(() => {
        setApiConnected(false);
      });
    fetchCsrfToken().catch(() => {});

    getCurrentUser()
      .then((user) => {
        if (user && user.authenticated) {
          setCurrentUser(user);
        } else {
          setCurrentUser(null);
        }
      })
      .catch(() => {
        setCurrentUser(null);
      })
      .finally(() => {
        setIsAuthChecking(false);
      });
  }, []);

  // When user is authenticated, ensure single conversation thread is loaded / reused
  useEffect(() => {
    if (!currentUser) return;
    initUserConversation();
  }, [currentUser]);

  const handleAuthError = (err) => {
    if (err && (err.status === 401 || err.status === 403)) {
      setCurrentUser(null);
      setCurrentConversationId(null);
      localStorage.removeItem('chatpartner_active_conv_id');
      setAuthError('Your session has expired. Please log in again.');
      return true;
    }
    return false;
  };

  // Safe conversation initialization & reuse
  const initUserConversation = async () => {
    setIsHistoryLoading(true);
    setErrorMessage(null);

    try {
      const convs = await listConversations();
      let activeId = currentConversationId;

      // Verify if saved conversation ID actually belongs to this user
      const existingConv = activeId && convs.find((c) => c.id === activeId);

      if (!existingConv) {
        if (convs.length > 0) {
          // Reuse user's primary/latest conversation
          activeId = convs[0].id;
        } else {
          // Create the single conversation thread for the user
          const newConv = await createConversation('Chat with Ami');
          activeId = newConv.id;
        }
        setCurrentConversationId(activeId);
        localStorage.setItem('chatpartner_active_conv_id', activeId);
      }

      // Fetch and format messages
      const convData = await getConversation(activeId);
      const formatted = (convData.messages || []).map((m) => ({
        id: m.id,
        sender: m.sender,
        text: m.content,
        messageType: m.message_type || 'text',
        audioUrl: m.audio_url || null,
        audioDuration: m.audio_duration || null,
        timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }));
      setMessages(formatted);
    } catch (err) {
      if (!handleAuthError(err)) {
        setErrorMessage(`Could not load chat: ${err.message}`);
      }
    } finally {
      setIsHistoryLoading(false);
    }
  };

  // Lightweight background sync on window focus and periodic interval
  // Does NOT overwrite unsent inputText or disrupt scrolling while user is typing
  useEffect(() => {
    if (!currentUser || !currentConversationId) return;

    let isMounted = true;

    const refreshSilently = async () => {
      try {
        const convData = await getConversation(currentConversationId);
        if (!isMounted || !convData || !convData.messages) return;

        setMessages((prev) => {
          const newMsgs = convData.messages;
          const hasLengthChanged = newMsgs.length !== prev.length;
          const hasLastIdChanged =
            newMsgs.length > 0 &&
            prev.length > 0 &&
            String(newMsgs[newMsgs.length - 1].id) !== String(prev[prev.length - 1].id);

          if (hasLengthChanged || hasLastIdChanged) {
            return newMsgs.map((m) => ({
              id: m.id,
              sender: m.sender,
              text: m.content,
              messageType: m.message_type || 'text',
              audioUrl: m.audio_url || null,
              audioDuration: m.audio_duration || null,
              timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            }));
          }
          return prev;
        });
      } catch {
        // Silently suppress background refresh errors so typing is never interrupted
      }
    };

    const handleFocus = () => {
      refreshSilently();
    };
    window.addEventListener('focus', handleFocus);
    // Poll every 6 seconds so proactive check-ins appear without manual page refresh
    const interval = setInterval(refreshSilently, 6000);

    return () => {
      isMounted = false;
      window.removeEventListener('focus', handleFocus);
      clearInterval(interval);
    };
  }, [currentUser, currentConversationId]);

  // Proactive settings handlers
  const handleOpenProactiveSettings = async () => {
    setIsMenuOpen(false);
    setShowProactiveSettings(true);
    setIsProactiveLoading(true);
    setProactiveStatus(null);
    try {
      const data = await getCompanionPreferences();
      if (data) {
        setProactivePrefs({
          frequency: data.frequency || (data.proactive_messaging_enabled ? 'daily' : 'disabled'),
          quiet_hours_enabled: data.quiet_hours_enabled ?? true,
          quiet_hours_start: data.quiet_hours_start || '22:00',
          quiet_hours_end: data.quiet_hours_end || '07:00',
        });
      }
    } catch (err) {
      setProactiveStatus({ type: 'error', message: `Failed to load settings: ${err.message}` });
    } finally {
      setIsProactiveLoading(false);
    }
  };

  const handleSaveProactiveSettings = async (e) => {
    if (e) e.preventDefault();
    setIsProactiveSaving(true);
    setProactiveStatus(null);
    try {
      await updateCompanionPreferences({
        frequency: proactivePrefs.frequency,
        proactive_messaging_enabled: proactivePrefs.frequency !== 'disabled',
        quiet_hours_enabled: proactivePrefs.quiet_hours_enabled,
        quiet_hours_start: proactivePrefs.quiet_hours_start,
        quiet_hours_end: proactivePrefs.quiet_hours_end,
      });
      setProactiveStatus({ type: 'success', message: 'Preferences updated successfully!' });
      setTimeout(() => {
        setShowProactiveSettings(false);
        setProactiveStatus(null);
      }, 1200);
    } catch (err) {
      setProactiveStatus({ type: 'error', message: `Could not save settings: ${err.message}` });
    } finally {
      setIsProactiveSaving(false);
    }
  };

  // ============================================================================
  // Authentication Handlers
  // ============================================================================

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError(null);

    const { username, password, email } = authForm;
    if (!username.trim() || !password) {
      setAuthError('Please provide both username and password.');
      return;
    }

    setAuthLoading(true);
    try {
      let data;
      if (authMode === 'login') {
        data = await loginUser(username.trim(), password);
      } else {
        data = await registerUser(username.trim(), password, email.trim());
      }

      const authenticatedUser = data?.user || (data?.authenticated ? data : null);

      if (authenticatedUser && (authenticatedUser.authenticated || authenticatedUser.id)) {
        setCurrentUser(authenticatedUser);
        setAuthForm({ username: '', password: '', email: '' });
        setErrorMessage(null);
        setAuthError(null);
      } else {
        setAuthError('Authentication succeeded but user session was incomplete. Please try again.');
      }
    } catch (err) {
      setAuthError(err.message || 'Authentication failed. Please verify your credentials and try again.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    setIsMenuOpen(false);
    try {
      await logoutUser();
    } catch {
      // ignore
    } finally {
      setCurrentUser(null);
      setCurrentConversationId(null);
      localStorage.removeItem('chatpartner_active_conv_id');
      setMessages([]);
      setErrorMessage(null);
      setAuthError(null);
    }
  };

  // ============================================================================
  // Chat Handlers
  // ============================================================================

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();

    const trimmed = inputText.trim();
    if (!trimmed || isLoading) return;

    // Optimistic user message bubble
    const userMsg = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setErrorMessage(null);
    setIsLoading(true);

    try {
      const data = await sendChatMessage(trimmed, currentConversationId);

      // Ensure active conversation ID is set
      if (!currentConversationId && data.conversation_id) {
        setCurrentConversationId(data.conversation_id);
        localStorage.setItem('chatpartner_active_conv_id', data.conversation_id);
      }

      const companionMsg = {
        id: data.message_id || `ami-${Date.now()}`,
        sender: 'companion',
        text: data.reply,
        timestamp: new Date(data.timestamp || Date.now()).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      };

      setMessages((prev) => [...prev, companionMsg]);
      setApiConnected(true);
    } catch (err) {
      if (err.status === 503 || err.message?.toLowerCase().includes('not configured')) {
        setApiConnected(false);
      }
      if (!handleAuthError(err)) {
        setErrorMessage(err.message || 'Failed to send message. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendVoice = async (audioBlob, duration) => {
    if (isLoading) return;

    const tempId = `voice-${Date.now()}`;
    const userMsg = {
      id: tempId,
      sender: 'user',
      messageType: 'voice',
      audioBlob: audioBlob,
      audioDuration: duration,
      text: '🎤 Voice message (transcribing...)',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setErrorMessage(null);
    setIsLoading(true);

    try {
      const data = await sendVoiceChatMessage({
        audioBlob,
        duration,
        conversationId: currentConversationId,
        clientMessageId: tempId,
      });

      if (!currentConversationId && data.conversation_id) {
        setCurrentConversationId(data.conversation_id);
        localStorage.setItem('chatpartner_active_conv_id', data.conversation_id);
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === tempId
            ? {
                ...m,
                id: data.user_message_id || m.id,
                text: data.user_transcript || m.text,
                audioUrl: data.user_audio_url || m.audioUrl,
                audioDuration: data.user_audio_duration || m.audioDuration,
              }
            : m
        )
      );

      const companionMsg = {
        id: data.message_id || `ami-${Date.now()}`,
        sender: 'companion',
        messageType: 'voice',
        text: data.reply,
        audioUrl: data.audio_url || null,
        audioDuration: data.audio_duration || null,
        timestamp: new Date(data.timestamp || Date.now()).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      };

      setMessages((prev) => [...prev, companionMsg]);
      setApiConnected(true);
    } catch (err) {
      if (err.status === 503 || err.message?.toLowerCase().includes('not configured')) {
        setApiConnected(false);
      }
      if (!handleAuthError(err)) {
        setErrorMessage(err.message || 'Failed to send voice message. Please try again.');
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === tempId
            ? { ...m, text: '⚠️ Failed to send voice message' }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSynthesizeVoice = async (messageId) => {
    setSynthesizingMessageId(messageId);
    try {
      const data = await synthesizeMessageVoice(messageId);
      if (data && data.audio_url) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === messageId
              ? { ...m, audioUrl: data.audio_url, audioDuration: data.audio_duration }
              : m
          )
        );
      }
    } catch (err) {
      setErrorMessage(`Could not generate voice: ${err.message}`);
    } finally {
      setSynthesizingMessageId(null);
    }
  };

  // Clear chat action
  const handleConfirmClearChat = async () => {
    if (!currentConversationId) {
      setMessages([]);
      setShowClearConfirm(false);
      return;
    }

    setIsClearing(true);
    setErrorMessage(null);

    try {
      await clearConversationMessages(currentConversationId);
      // Reset UI to empty chat state
      setMessages([]);
      setShowClearConfirm(false);
    } catch (err) {
      if (!handleAuthError(err)) {
        setErrorMessage(`Failed to clear chat: ${err.message}`);
      }
    } finally {
      setIsClearing(false);
    }
  };

  // Handle Enter to send, Shift+Enter for newline
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Starter chip click helper
  const handleStarterClick = (promptText) => {
    setInputText(promptText);
    inputRef.current?.focus();
  };

  // Loading spinner during initial session check
  if (isAuthChecking) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-slate-100">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-3 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-slate-400 font-medium">Connecting to Ami...</p>
        </div>
      </div>
    );
  }

  // ============================================================================
  // PRE-AUTHENTICATION: Clean Login / Create Account Screen
  // ============================================================================
  if (!currentUser) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
        <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl space-y-6">
          {/* Logo & Header */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-tr from-purple-600 via-pink-500 to-indigo-500 shadow-lg shadow-purple-500/20 mb-1">
              <span className="text-2xl font-bold text-white tracking-wider">A</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Ami</h1>
            <p className="text-sm text-slate-400">
              Your personal AI companion. Supportive, affectionate & thoughtful.
            </p>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex rounded-xl bg-slate-800/80 p-1 border border-slate-700/50">
            <button
              type="button"
              onClick={() => {
                setAuthMode('login');
                setAuthError(null);
              }}
              className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
                authMode === 'login'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setAuthMode('register');
                setAuthError(null);
              }}
              className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
                authMode === 'register'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Create Account
            </button>
          </div>

          {/* Error Banner */}
          {authError && (
            <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs sm:text-sm flex items-start gap-2.5">
              <svg className="w-4 h-4 mt-0.5 shrink-0 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span>{authError}</span>
            </div>
          )}

          {/* Authentication Form */}
          <form onSubmit={handleAuthSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Username
              </label>
              <input
                type="text"
                autoComplete="username"
                required
                value={authForm.username}
                onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-800/90 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                placeholder="e.g. ashique"
              />
            </div>

            {authMode === 'register' && (
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                  Email (Optional)
                </label>
                <input
                  type="email"
                  autoComplete="email"
                  value={authForm.email}
                  onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-800/90 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                  placeholder="e.g. you@example.com"
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Password
              </label>
              <input
                type="password"
                autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                required
                value={authForm.password}
                onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-800/90 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={authLoading}
              className="w-full mt-2 py-3 px-4 rounded-xl font-medium text-sm text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 active:scale-[0.99] transition shadow-md shadow-purple-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {authLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>{authMode === 'login' ? 'Signing in...' : 'Creating account...'}</span>
                </>
              ) : (
                <span>{authMode === 'login' ? 'Sign In to Chat' : 'Create Account & Start Chat'}</span>
              )}
            </button>
          </form>

          {/* Switch Hint */}
          <div className="text-center pt-2">
            <button
              type="button"
              onClick={() => {
                setAuthMode(authMode === 'login' ? 'register' : 'login');
                setAuthError(null);
              }}
              className="text-xs text-purple-400 hover:text-purple-300 font-medium transition"
            >
              {authMode === 'login'
                ? "Don't have an account yet? Create one"
                : 'Already have an account? Sign in'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // POST-AUTHENTICATION: Minimal WhatsApp / Instagram DM Chat View
  // ============================================================================
  return (
    <div className="flex justify-center min-h-screen bg-slate-950 text-slate-100">
      {/* Centered chat container (mobile-first full-width, desktop centered max-w-2xl) */}
      <div className="w-full max-w-2xl h-screen flex flex-col bg-slate-900/40 sm:border-x sm:border-slate-800/80 shadow-2xl relative">
        {/* Compact Header */}
        <header className="h-16 px-4 bg-slate-900/95 backdrop-blur border-b border-slate-800/80 flex items-center justify-between z-20 shrink-0">
          <div className="flex items-center gap-3">
            {/* Avatar with Active Indicator */}
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-purple-500 via-pink-500 to-indigo-500 flex items-center justify-center shadow-sm">
                <span className="text-white font-bold text-sm tracking-wider">A</span>
              </div>
              <span
                className={`absolute bottom-0 right-0 w-3 h-3 border-2 border-slate-900 rounded-full ${
                  apiConnected ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
                title={apiConnected ? 'Active now' : 'API not connected'}
              />
            </div>

            {/* Name & Online Status */}
            <div>
              <h2 className="font-semibold text-slate-100 text-sm sm:text-base leading-tight">Ami</h2>
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    apiConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                  }`}
                />
                <span
                  className={`text-xs font-medium ${
                    apiConnected ? 'text-emerald-400' : 'text-amber-400'
                  }`}
                >
                  {apiConnected ? 'Active now' : 'API not connected'}
                </span>
              </div>
            </div>
          </div>

          {/* 3-Dots Menu */}
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-full transition"
              aria-label="Chat options"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
              </svg>
            </button>

            {/* Dropdown Menu */}
            {isMenuOpen && (
              <div className="absolute right-0 mt-2 w-52 bg-slate-900 border border-slate-800 rounded-xl shadow-xl py-1 z-30 animate-in fade-in duration-150">
                <button
                  type="button"
                  onClick={handleOpenProactiveSettings}
                  className="w-full text-left px-4 py-2.5 text-xs sm:text-sm text-purple-300 hover:bg-purple-500/10 flex items-center gap-2.5 transition"
                >
                  <svg className="w-4 h-4 shrink-0 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Proactive Check-Ins</span>
                </button>

                <div className="h-px bg-slate-800 my-1" />

                <button
                  type="button"
                  onClick={() => {
                    setIsMenuOpen(false);
                    setShowClearConfirm(true);
                  }}
                  className="w-full text-left px-4 py-2.5 text-xs sm:text-sm text-rose-400 hover:bg-rose-500/10 flex items-center gap-2.5 transition"
                >
                  <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                  <span>Clear Chat</span>
                </button>

                <div className="h-px bg-slate-800 my-1" />

                <button
                  type="button"
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2.5 text-xs sm:text-sm text-slate-300 hover:bg-slate-800 hover:text-white flex items-center gap-2.5 transition"
                >
                  <svg className="w-4 h-4 shrink-0 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                    />
                  </svg>
                  <span>Log Out ({currentUser.username})</span>
                </button>
              </div>
            )}
          </div>
        </header>

        {/* Chat Messages Area */}
        <main className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scroll-smooth">
          {/* History Loading Skeleton */}
          {isHistoryLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-2 text-slate-400 text-xs">
                <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                <span>Loading conversation...</span>
              </div>
            </div>
          ) : messages.length === 0 ? (
            /* Empty Chat State */
            <div className="flex flex-col items-center justify-center min-h-[70%] text-center px-4 py-8 space-y-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-purple-600 via-pink-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-purple-500/10">
                <span className="text-2xl font-bold text-white">A</span>
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-semibold text-white">Say hi to Ami! 👋</h3>
                <p className="text-xs sm:text-sm text-slate-400 max-w-xs">
                  Your companion is right here. Chat in English or Manglish — share your day, ask anything, or just say hello!
                </p>
              </div>

              {/* Starter chips */}
              <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => handleStarterClick('Hey Ami! Kazhicho?')}
                  className="px-3 py-1.5 rounded-full bg-slate-800 hover:bg-slate-700/80 text-xs text-purple-300 border border-slate-700/60 transition"
                >
                  Kazhicho? 🍲
                </button>
                <button
                  type="button"
                  onClick={() => handleStarterClick('Hey Ami! How was your day?')}
                  className="px-3 py-1.5 rounded-full bg-slate-800 hover:bg-slate-700/80 text-xs text-purple-300 border border-slate-700/60 transition"
                >
                  How was your day? ✨
                </button>
                <button
                  type="button"
                  onClick={() => handleStarterClick('Tell me something interesting today!')}
                  className="px-3 py-1.5 rounded-full bg-slate-800 hover:bg-slate-700/80 text-xs text-purple-300 border border-slate-700/60 transition"
                >
                  Tell me something nice 💜
                </button>
              </div>
            </div>
          ) : (
            /* Message Bubbles */
            messages.map((msg) => {
              const isUser = msg.sender === 'user';
              const isVoice = msg.messageType === 'voice' || Boolean(msg.audioUrl || msg.audioBlob);

              return (
                <div
                  key={msg.id}
                  className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[85%] sm:max-w-[75%] px-4 py-2.5 shadow-sm text-sm sm:text-base leading-relaxed break-words whitespace-pre-wrap ${
                      isUser
                        ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-2xl rounded-tr-xs'
                        : 'bg-slate-800/90 text-slate-100 border border-slate-700/60 rounded-2xl rounded-tl-xs'
                    }`}
                  >
                    {/* User Voice Message */}
                    {isUser && isVoice ? (
                      <div className="flex flex-col gap-2">
                        <AudioPlayer
                          audioUrl={msg.audioUrl}
                          audioBlob={msg.audioBlob}
                          initialDuration={msg.audioDuration}
                          isUser={true}
                          playerId={msg.id}
                          activePlayerId={activePlayerId}
                          onPlayStarted={setActivePlayerId}
                        />
                        {msg.text && (
                          <div className="text-xs text-purple-100/95 leading-normal border-t border-purple-400/30 pt-1.5 mt-0.5">
                            <span className="font-semibold text-purple-200">Transcript: </span>
                            <span>{msg.text}</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>
                        {msg.text}

                        {/* Ami Voice Reply Controls */}
                        {!isUser && (
                          <div className="mt-2.5 pt-2 border-t border-slate-700/60 flex items-center gap-2">
                            {msg.audioUrl ? (
                              <AudioPlayer
                                audioUrl={msg.audioUrl}
                                initialDuration={msg.audioDuration}
                                isUser={false}
                                playerId={msg.id}
                                activePlayerId={activePlayerId}
                                onPlayStarted={setActivePlayerId}
                              />
                            ) : (
                              <button
                                type="button"
                                onClick={() => handleSynthesizeVoice(msg.id)}
                                disabled={synthesizingMessageId === msg.id}
                                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-500/10 hover:bg-purple-500/20 text-purple-300 text-xs font-medium border border-purple-500/30 transition disabled:opacity-50"
                              >
                                {synthesizingMessageId === msg.id ? (
                                  <>
                                    <div className="w-3 h-3 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                                    <span>Generating voice...</span>
                                  </>
                                ) : (
                                  <>
                                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                                    </svg>
                                    <span>Listen to Ami 🔊</span>
                                  </>
                                )}
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {msg.timestamp && (
                      <span
                        className={`block text-[10px] mt-1 select-none text-right ${
                          isUser ? 'text-purple-200/80' : 'text-slate-400'
                        }`}
                      >
                        {msg.timestamp}
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Typing Indicator */}
          {isLoading && (
            <div className="flex items-start">
              <div className="bg-slate-800/90 border border-slate-700/60 rounded-2xl rounded-tl-xs px-4 py-3 shadow-sm">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          {/* Error Message Toast */}
          {errorMessage && (
            <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs sm:text-sm flex items-center justify-between gap-2">
              <span>{errorMessage}</span>
              <button
                type="button"
                onClick={() => setErrorMessage(null)}
                className="text-rose-400 hover:text-rose-200 font-bold px-1"
              >
                ✕
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>

        {/* Fixed Bottom Message Composer */}
        <footer className="p-3 sm:p-4 bg-slate-900/95 backdrop-blur border-t border-slate-800/80 shrink-0">
          <form
            onSubmit={handleSendMessage}
            className="flex items-end gap-2 w-full"
          >
            <textarea
              ref={inputRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Ami... (Shift+Enter for newline)"
              rows={1}
              className="flex-1 bg-slate-800/90 text-slate-100 placeholder-slate-400 rounded-2xl px-4 py-2.5 text-sm sm:text-base focus:outline-none focus:ring-2 focus:ring-purple-500 border border-slate-700/60 resize-none max-h-32 transition leading-relaxed"
            />

            {/* Voice Recording Microphone Button */}
            <VoiceRecorder
              onSendVoice={handleSendVoice}
              isSending={isLoading}
              onError={(err) => setErrorMessage(err)}
            />

            <button
              type="submit"
              disabled={!inputText.trim() || isLoading}
              className="shrink-0 p-3 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white disabled:opacity-40 disabled:hover:from-purple-600 disabled:hover:to-indigo-600 disabled:cursor-not-allowed shadow-md transition-all active:scale-95 flex items-center justify-center"
              aria-label="Send message"
            >
              <svg className="w-5 h-5 translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            </button>
          </form>
        </footer>

        {/* Proactive Check-Ins Settings Modal */}
        {showProactiveSettings && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-5 sm:p-6 shadow-2xl space-y-5">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2.5">
                  <div className="w-9 h-9 rounded-full bg-purple-500/10 text-purple-400 flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-base font-semibold text-white">Proactive Check-Ins</h4>
                    <p className="text-xs text-slate-400">Manage how and when Ami texts you first</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowProactiveSettings(false)}
                  className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
                >
                  ✕
                </button>
              </div>

              {isProactiveLoading ? (
                <div className="py-8 flex flex-col items-center justify-center gap-2 text-slate-400 text-xs">
                  <div className="w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                  <span>Loading settings...</span>
                </div>
              ) : (
                <form onSubmit={handleSaveProactiveSettings} className="space-y-4">
                  {/* Status Banner */}
                  {proactiveStatus && (
                    <div
                      className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
                        proactiveStatus.type === 'success'
                          ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                          : 'bg-rose-500/10 text-rose-300 border border-rose-500/20'
                      }`}
                    >
                      <span>{proactiveStatus.message}</span>
                    </div>
                  )}

                  {/* Frequency Selection */}
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-slate-300 block">Frequency</label>
                    <div className="space-y-2">
                      {[
                        { id: 'daily', label: 'Once daily', desc: 'Ami sends a friendly check-in once per day.' },
                        { id: 'few_times_week', label: 'Few times a week', desc: 'Ami checks in on Monday, Wednesday, and Friday.' },
                        { id: 'disabled', label: 'Disabled', desc: "Ami won't initiate conversations on her own." },
                      ].map((opt) => (
                        <label
                          key={opt.id}
                          className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition ${
                            proactivePrefs.frequency === opt.id
                              ? 'bg-purple-500/10 border-purple-500/40 text-white'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-300 hover:bg-slate-800'
                          }`}
                        >
                          <input
                            type="radio"
                            name="frequency"
                            value={opt.id}
                            checked={proactivePrefs.frequency === opt.id}
                            onChange={(e) => setProactivePrefs((p) => ({ ...p, frequency: e.target.value }))}
                            className="mt-1 text-purple-600 focus:ring-purple-500"
                          />
                          <div className="text-xs">
                            <span className="font-medium text-slate-100 block">{opt.label}</span>
                            <span className="text-slate-400 block mt-0.5">{opt.desc}</span>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Quiet Hours */}
                  <div className="space-y-2 pt-2 border-t border-slate-800/80">
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-xs font-semibold text-slate-300 block">Quiet Hours</label>
                        <span className="text-[11px] text-slate-400">Pause messages during your sleep hours</span>
                      </div>
                      <input
                        type="checkbox"
                        checked={proactivePrefs.quiet_hours_enabled}
                        onChange={(e) => setProactivePrefs((p) => ({ ...p, quiet_hours_enabled: e.target.checked }))}
                        className="rounded text-purple-600 focus:ring-purple-500 w-4 h-4"
                      />
                    </div>

                    {proactivePrefs.quiet_hours_enabled && (
                      <div className="grid grid-cols-2 gap-3 pt-2">
                        <div>
                          <label className="text-[11px] text-slate-400 block mb-1">Start (Sleep)</label>
                          <input
                            type="time"
                            value={proactivePrefs.quiet_hours_start}
                            onChange={(e) => setProactivePrefs((p) => ({ ...p, quiet_hours_start: e.target.value }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-purple-500"
                          />
                        </div>
                        <div>
                          <label className="text-[11px] text-slate-400 block mb-1">End (Wake up)</label>
                          <input
                            type="time"
                            value={proactivePrefs.quiet_hours_end}
                            onChange={(e) => setProactivePrefs((p) => ({ ...p, quiet_hours_end: e.target.value }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-purple-500"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-slate-800">
                    <button
                      type="button"
                      onClick={() => setShowProactiveSettings(false)}
                      disabled={isProactiveSaving}
                      className="px-4 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 transition"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isProactiveSaving}
                      className="px-4 py-2 rounded-xl text-xs font-medium text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 transition flex items-center gap-2"
                    >
                      {isProactiveSaving ? (
                        <>
                          <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          <span>Saving...</span>
                        </>
                      ) : (
                        <span>Save Settings</span>
                      )}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        )}

        {/* Clear Chat Confirmation Modal */}
        {showClearConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-sm w-full p-5 shadow-2xl space-y-4">
              <div className="w-10 h-10 rounded-full bg-rose-500/10 text-rose-400 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </div>

              <div className="space-y-1">
                <h4 className="text-base font-semibold text-white">Clear chat history?</h4>
                <p className="text-xs sm:text-sm text-slate-400">
                  This will permanently delete all messages in this conversation. Ami's long-term memory and companion settings will remain completely safe.
                </p>
              </div>

              <div className="flex items-center justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={() => setShowClearConfirm(false)}
                  disabled={isClearing}
                  className="px-4 py-2 rounded-xl text-xs sm:text-sm font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 transition"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmClearChat}
                  disabled={isClearing}
                  className="px-4 py-2 rounded-xl text-xs sm:text-sm font-medium text-white bg-rose-600 hover:bg-rose-500 disabled:opacity-50 transition flex items-center gap-2"
                >
                  {isClearing ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Clearing...</span>
                    </>
                  ) : (
                    <span>Clear Chat</span>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
