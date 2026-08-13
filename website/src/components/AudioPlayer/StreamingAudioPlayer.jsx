import { useEffect, useRef, useState } from "react";
import { Play, Pause, Volume2 } from "lucide-react";

/**
 * Streaming audio player using Web Audio API.
 * Plays PCM chunks incrementally as they arrive.
 */
export default function StreamingAudioPlayer({ chunks, sampleRate, channels, bitsPerSample }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);
  const audioContextRef = useRef(null);
  const sourceNodesRef = useRef([]); // Track all scheduled sources for proper cleanup
  const scheduledTimeRef = useRef(0);
  const chunksProcessedRef = useRef(0);

  useEffect(() => {
    // Initialize AudioContext on first chunk
    if (!audioContextRef.current && chunks.length > 0) {
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) {
          // Set error via ref to avoid effect setState issue
          setTimeout(() => setError("Web Audio API not supported in this browser"), 0);
          return;
        }
        audioContextRef.current = new AudioContext({ sampleRate });
      } catch (err) {
        setTimeout(() => setError("Failed to initialize audio: " + err.message), 0);
        return;
      }
    }

    // Process new chunks only when playing
    const audioContext = audioContextRef.current;
    if (!audioContext || !isPlaying) return;

    const processChunk = async (chunkData) => {
      try {
        // Decode base64 PCM data
        const binaryString = atob(chunkData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert PCM bytes to AudioBuffer
        const numSamples = bytes.length / (bitsPerSample / 8) / channels;
        const audioBuffer = audioContext.createBuffer(channels, numSamples, sampleRate);

        // Decode 16-bit PCM (currently mono only, matching serve.py output)
        if (bitsPerSample === 16) {
          const view = new DataView(bytes.buffer);
          const channelData = audioBuffer.getChannelData(0);
          for (let i = 0; i < numSamples; i++) {
            const sample = view.getInt16(i * 2, true); // little-endian
            channelData[i] = sample / 32768.0; // normalize to [-1, 1]
          }
        }

        // Schedule playback
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);

        const startTime = Math.max(audioContext.currentTime, scheduledTimeRef.current);
        source.start(startTime);
        scheduledTimeRef.current = startTime + audioBuffer.duration;

        // Track this source for cleanup
        sourceNodesRef.current.push(source);

        // Remove from tracking when it finishes naturally
        source.onended = () => {
          const index = sourceNodesRef.current.indexOf(source);
          if (index > -1) {
            sourceNodesRef.current.splice(index, 1);
          }
        };
      } catch (err) {
        console.error("Failed to process audio chunk:", err);
        setTimeout(() => setError("Failed to play audio chunk"), 0);
      }
    };

    // Process any new chunks that haven't been processed yet
    for (let i = chunksProcessedRef.current; i < chunks.length; i++) {
      processChunk(chunks[i]);
      chunksProcessedRef.current = i + 1;
    }
  }, [chunks, isPlaying, sampleRate, channels, bitsPerSample]);

  useEffect(() => {
    // Cleanup on unmount: stop all sources and close context
    return () => {
      // Stop all scheduled sources
      sourceNodesRef.current.forEach(source => {
        try {
          source.stop();
        } catch {
          // Ignore - may already be stopped
        }
      });
      sourceNodesRef.current = [];

      // Close audio context
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const handlePlayPause = () => {
    if (!audioContextRef.current) return;

    if (isPlaying) {
      // Pause: stop all scheduled sources and reset state
      sourceNodesRef.current.forEach(source => {
        try {
          source.stop();
        } catch {
          // Ignore - source may have already finished or been stopped
        }
      });
      sourceNodesRef.current = [];

      // Reset playback state but NOT chunksProcessedRef
      // This allows resume to skip already-processed chunks
      scheduledTimeRef.current = 0;
      setIsPlaying(false);
    } else {
      // Play/Resume: start from current time
      if (audioContextRef.current.state === "suspended") {
        audioContextRef.current.resume();
      }
      scheduledTimeRef.current = audioContextRef.current.currentTime;
      setIsPlaying(true);
    }
  };

  if (error) {
    return (
      <div className="streaming-audio-error" role="alert">
        <Volume2 size={16} />
        <span>{error}</span>
      </div>
    );
  }

  if (chunks.length === 0) {
    return (
      <div className="streaming-audio-loading">
        <Volume2 size={16} />
        <span>Loading audio...</span>
      </div>
    );
  }

  return (
    <div className="streaming-audio-player">
      <button
        onClick={handlePlayPause}
        aria-label={isPlaying ? "Pause" : "Play"}
        className="audio-play-button"
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <div className="audio-info">
        <Volume2 size={14} />
        <span>Streaming audio ({chunks.length} chunks)</span>
      </div>
    </div>
  );
}
