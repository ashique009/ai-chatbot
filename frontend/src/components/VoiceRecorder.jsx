import { useState, useRef, useEffect } from 'react';

/**
 * Formats seconds into MM:SS format.
 */
function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

/**
 * Dynamically picks the best supported MIME type for MediaRecorder.
 */
function getSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
    'audio/aac',
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return '';
}

export default function VoiceRecorder({
  onSendVoice,
  isSending = false,
  onError,
}) {
  const [recordingState, setRecordingState] = useState('idle'); // 'idle' | 'recording' | 'preview'
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedDuration, setRecordedDuration] = useState(0);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const previewAudioRef = useRef(null);
  const previewUrlRef = useRef(null);
  const startTimeRef = useRef(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current = null;
      }
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  const handleStartRecording = async () => {
    if (isSending) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      onError?.('Voice recording is not supported in this browser. Please use Chrome, Safari, Firefox, or Edge.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = getSupportedMimeType();
      const options = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const actualMime = mediaRecorder.mimeType || mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: actualMime });
        if (blob.size === 0) {
          onError?.('Recording was empty. Please check your microphone.');
          handleDiscard();
          return;
        }

        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        const finalDur = Math.max(1, Math.round(elapsed));

        setRecordedBlob(blob);
        setRecordedDuration(finalDur);
        setRecordingState('preview');

        // Create preview audio element
        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        const url = URL.createObjectURL(blob);
        previewUrlRef.current = url;

        const audio = new Audio(url);
        previewAudioRef.current = audio;
        audio.ontimeupdate = () => setPreviewTime(audio.currentTime);
        audio.onended = () => {
          setPreviewPlaying(false);
          setPreviewTime(0);
        };
      };

      startTimeRef.current = Date.now();
      mediaRecorder.start(250); // collect in 250ms chunks
      setRecordingSeconds(0);
      setRecordingState('recording');

      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        onError?.('Microphone access was denied. Please allow microphone permissions in your browser address bar.');
      } else if (err.name === 'NotFoundError') {
        onError?.('No microphone input device was found on your system.');
      } else {
        onError?.(`Microphone error: ${err.message || 'Could not access audio device.'}`);
      }
      handleDiscard();
    }
  };

  const handleStopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const handleDiscard = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }

    setRecordingState('idle');
    setRecordingSeconds(0);
    setRecordedBlob(null);
    setRecordedDuration(0);
    setPreviewPlaying(false);
    setPreviewTime(0);
  };

  const handleTogglePreviewPlay = () => {
    if (!previewAudioRef.current) return;
    if (previewPlaying) {
      previewAudioRef.current.pause();
      setPreviewPlaying(false);
    } else {
      previewAudioRef.current.play().then(() => {
        setPreviewPlaying(true);
      }).catch(() => {
        setPreviewPlaying(false);
      });
    }
  };

  const handleConfirmSend = () => {
    if (!recordedBlob) return;
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
    onSendVoice(recordedBlob, recordedDuration);
    handleDiscard();
  };

  // 1. IDLE: Just the microphone icon button
  if (recordingState === 'idle') {
    return (
      <button
        type="button"
        onClick={handleStartRecording}
        disabled={isSending}
        className="shrink-0 p-3 rounded-full bg-slate-800 hover:bg-slate-700 text-purple-400 hover:text-purple-300 border border-slate-700/60 shadow-md transition-all active:scale-95 flex items-center justify-center disabled:opacity-40"
        title="Send voice message"
        aria-label="Record voice message"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>
      </button>
    );
  }

  // 2. RECORDING: Pulse dot, timer, audio equalizer, cancel and stop buttons
  if (recordingState === 'recording') {
    return (
      <div className="flex-1 flex items-center justify-between gap-3 bg-slate-800/95 border border-purple-500/50 rounded-2xl px-4 py-2 text-white animate-in fade-in duration-200">
        <div className="flex items-center gap-2.5">
          {/* Pulsing recording light */}
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500" />
          </span>

          <span className="text-sm font-mono text-purple-200 font-semibold tracking-wide">
            {formatDuration(recordingSeconds)}
          </span>

          {/* Equalizer animation */}
          <div className="flex items-center gap-1 ml-2">
            <span className="w-1 bg-purple-400 rounded-full animate-pulse h-4" style={{ animationDelay: '0ms' }} />
            <span className="w-1 bg-purple-400 rounded-full animate-pulse h-6" style={{ animationDelay: '150ms' }} />
            <span className="w-1 bg-purple-400 rounded-full animate-pulse h-3" style={{ animationDelay: '300ms' }} />
            <span className="w-1 bg-purple-400 rounded-full animate-pulse h-5" style={{ animationDelay: '100ms' }} />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Cancel button */}
          <button
            type="button"
            onClick={handleDiscard}
            className="p-2 text-slate-400 hover:text-rose-400 hover:bg-slate-700/60 rounded-full transition"
            title="Cancel recording"
            aria-label="Cancel recording"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>

          {/* Stop and Preview button */}
          <button
            type="button"
            onClick={handleStopRecording}
            className="px-3 py-1.5 rounded-full bg-rose-500 hover:bg-rose-600 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md active:scale-95 transition"
          >
            <div className="w-2.5 h-2.5 bg-white rounded-xs" />
            <span>Done</span>
          </button>
        </div>
      </div>
    );
  }

  // 3. PREVIEW: Listen before sending, discard or send
  return (
    <div className="flex-1 flex items-center justify-between gap-2 sm:gap-3 bg-slate-800/95 border border-purple-500/50 rounded-2xl px-3 sm:px-4 py-2 text-white animate-in fade-in duration-200">
      {/* Play/Pause Preview */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handleTogglePreviewPlay}
          className="w-8 h-8 rounded-full bg-purple-600 hover:bg-purple-500 text-white flex items-center justify-center shrink-0 shadow-sm transition"
          aria-label={previewPlaying ? 'Pause preview' : 'Play preview'}
        >
          {previewPlaying ? (
            <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5 fill-current translate-x-0.5" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <span className="text-xs font-mono text-slate-300">
          {formatDuration(previewTime || recordedDuration)}
        </span>
      </div>

      {/* Discard & Send actions */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handleDiscard}
          className="px-3 py-1.5 rounded-xl bg-slate-700/60 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
        >
          Discard
        </button>

        <button
          type="button"
          onClick={handleConfirmSend}
          disabled={isSending}
          className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md active:scale-95 transition disabled:opacity-40"
        >
          <span>Send</span>
          <svg className="w-3.5 h-3.5 translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}
