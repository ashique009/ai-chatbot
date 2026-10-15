import { useState, useEffect, useRef } from 'react';

/**
 * Formats seconds into MM:SS format.
 */
function formatTime(seconds) {
  if (isNaN(seconds) || seconds === null || seconds === undefined) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

/**
 * Reusable, authenticated audio player for voice notes.
 * Supports both recorded Blobs and authenticated streaming URLs.
 */
export default function AudioPlayer({
  audioUrl = null,
  audioBlob = null,
  initialDuration = null,
  isUser = false,
  activePlayerId = null,
  playerId = null,
  onPlayStarted = null,
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(initialDuration || 0);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const audioRef = useRef(null);
  const objectUrlRef = useRef(null);

  const isCurrentActive = activePlayerId === playerId;

  // If another player became active, pause this one
  useEffect(() => {
    if (!isCurrentActive && isPlaying && audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  }, [isCurrentActive, isPlaying]);

  // Clean up object URLs and audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (objectUrlRef.current && !audioBlob) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [audioBlob]);

  // Resolve audio source (Blob or authenticated URL)
  const getAudioSource = async () => {
    if (audioBlob) {
      if (!objectUrlRef.current) {
        objectUrlRef.current = URL.createObjectURL(audioBlob);
      }
      return objectUrlRef.current;
    }

    if (!audioUrl) return null;

    if (objectUrlRef.current) {
      return objectUrlRef.current;
    }

    setIsLoadingAudio(true);
    setLoadError(null);
    try {
      // Securely fetch with session credentials
      const fullUrl = audioUrl.startsWith('http')
        ? audioUrl
        : `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'}${audioUrl.startsWith('/') ? '' : '/'}${audioUrl}`;

      const res = await fetch(fullUrl, {
        credentials: 'include',
      });

      if (!res.ok) {
        throw new Error(`Could not load audio (${res.status})`);
      }

      const blob = await res.blob();
      const objUrl = URL.createObjectURL(blob);
      objectUrlRef.current = objUrl;
      return objUrl;
    } catch (err) {
      setLoadError(err.message || 'Audio load failed');
      return null;
    } finally {
      setIsLoadingAudio(false);
    }
  };

  const togglePlay = async () => {
    if (isPlaying) {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setIsPlaying(false);
      return;
    }

    try {
      let src = objectUrlRef.current;
      if (!src) {
        src = await getAudioSource();
      }
      if (!src) return;

      if (!audioRef.current) {
        const audio = new Audio(src);
        audioRef.current = audio;

        audio.addEventListener('loadedmetadata', () => {
          if (audio.duration && !isNaN(audio.duration) && isFinite(audio.duration)) {
            setDuration(audio.duration);
          }
        });

        audio.addEventListener('timeupdate', () => {
          setCurrentTime(audio.currentTime);
        });

        audio.addEventListener('ended', () => {
          setIsPlaying(false);
          setCurrentTime(0);
        });

        audio.addEventListener('error', () => {
          setIsPlaying(false);
          setLoadError('Playback error occurred.');
        });
      }

      if (onPlayStarted && playerId) {
        onPlayStarted(playerId);
      }

      await audioRef.current.play();
      setIsPlaying(true);
    } catch (_err) {
      setIsPlaying(false);
      setLoadError('Failed to start playback.');
    }
  };

  const handleSeek = (e) => {
    if (!audioRef.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const pct = Math.max(0, Math.min(1, clickX / width));
    const newTime = pct * duration;
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div
      className={`inline-flex items-center gap-3 py-1.5 px-2 rounded-xl transition-all ${
        isUser ? 'bg-purple-700/40 text-white' : 'bg-slate-700/50 text-slate-200'
      }`}
    >
      {/* Play/Pause Button */}
      <button
        type="button"
        onClick={togglePlay}
        disabled={isLoadingAudio}
        className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition shadow-sm ${
          isUser
            ? 'bg-white text-purple-700 hover:bg-purple-100'
            : 'bg-gradient-to-r from-purple-500 to-indigo-500 text-white hover:from-purple-400 hover:to-indigo-400'
        } disabled:opacity-50`}
        aria-label={isPlaying ? 'Pause' : 'Play voice message'}
      >
        {isLoadingAudio ? (
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : isPlaying ? (
          <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
          </svg>
        ) : (
          <svg className="w-4 h-4 fill-current translate-x-0.5" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      {/* Progress & Time */}
      <div className="flex flex-col gap-1 min-w-[130px] sm:min-w-[160px]">
        {/* Clickable Progress Bar */}
        <div
          onClick={handleSeek}
          className="h-2 w-full bg-slate-900/40 rounded-full cursor-pointer relative overflow-hidden flex items-center"
        >
          <div
            className={`h-full transition-all duration-100 rounded-full ${
              isUser ? 'bg-white' : 'bg-purple-400'
            }`}
            style={{ width: `${progressPct}%` }}
          />
        </div>

        {/* Time Display */}
        <div className="flex justify-between items-center text-[11px] font-mono opacity-80 leading-none">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {loadError && (
        <span className="text-[10px] text-rose-300 font-sans" title={loadError}>
          ⚠️ Error
        </span>
      )}
    </div>
  );
}
