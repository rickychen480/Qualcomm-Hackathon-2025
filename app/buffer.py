import av
import cv2
import wave
import threading
import collections
import time
import os
from moviepy import VideoFileClip, AudioFileClip


class ShadowReplay:
    """
    A class to continuously record video and audio from a streamlit-webrtc stream
    into a buffer, allowing for the last 'n' seconds to be saved on command.
    """

    def __init__(self, record_seconds=15, fps=30.0):
        """
        Initializes the recorder for a WebRTC stream.

        Args:
            record_seconds (int): The duration of the buffer in seconds.
            fps (float): The target frames per second for video recording.
        """
        # --- Configurations ---
        self.record_seconds = record_seconds
        self.fps = fps
        self.output_dir = "recordings"
        self.temp_dir = ".tmp"

        # --- WebRTC Audio Standard ---
        # streamlit-webrtc typically sends audio frames of 1024 samples at 48000 Hz
        webrtc_audio_rate = 48000
        samples_per_frame = 1024
        audio_frames_per_second = webrtc_audio_rate / samples_per_frame

        # --- Buffers and Thread Control ---
        buffer_video_frames = int(self.record_seconds * self.fps)
        buffer_audio_frames = int(self.record_seconds * audio_frames_per_second) # Corrected buffer size calculation
        self.video_buffer = collections.deque(maxlen=buffer_video_frames)
        self.audio_buffer = collections.deque(maxlen=buffer_audio_frames)

        self.audio_lock = threading.Lock()
        self.is_running = False

    def recv_video(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Callback to receive video frames from the browser."""
        image = frame.to_ndarray(format="bgr24")
        self.video_buffer.append(image)
        return frame

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Callback to receive audio frames from the browser."""
        with self.audio_lock:
            # Each plane in the audio frame is a chunk of audio data
            for p in frame.planes:
                self.audio_buffer.append(p.to_bytes())
        return frame

    def _perform_save(self, video_frames, audio_chunks, output_path):
        """The actual file-saving logic, now more robust."""
        os.makedirs(os.path.join(os.getcwd(), self.temp_dir), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), self.output_dir), exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_video_path = os.path.join(os.getcwd(), self.temp_dir, f"temp_video_{timestamp}.mp4")
        output_path = output_path or os.path.join(os.getcwd(), self.output_dir, f"replay_{timestamp}.mp4")

        # --- Save video frames ---
        if not video_frames:
            print("Error: Video buffer is empty. Cannot save.")
            return
        frame_height, frame_width, _ = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (frame_width, frame_height))
        for frame in video_frames:
            out.write(frame)
        out.release()

        # --- Save audio chunks (ONLY IF AUDIO EXISTS) ---
        temp_audio_path = None
        if audio_chunks:
            temp_audio_path = os.path.join(os.getcwd(), self.temp_dir, f"temp_audio_{timestamp}.wav")
            with wave.open(temp_audio_path, "wb") as wf:
                wf.setnchannels(1)      # WebRTC is typically mono
                wf.setsampwidth(2)      # 16-bit PCM audio
                wf.setframerate(48000)  # Standard WebRTC audio rate
                wf.writeframes(b"".join(audio_chunks))
        else:
            print("Warning: No audio data in buffer. Saving video without audio.")

        # --- Combine files and clean up resources ---
        video_clip = audio_clip = final_clip = None
        try:
            video_clip = VideoFileClip(temp_video_path)
            if temp_audio_path:
                audio_clip = AudioFileClip(temp_audio_path)
                final_clip = video_clip.set_audio(audio_clip)
            else:
                final_clip = video_clip # No audio to combine

            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            print(f"Replay saved successfully to {output_path}")

        except Exception as e:
            print(f"Error combining files: {e}")

        finally:
            # **Crucial:** Close all clips to release file locks
            if final_clip: final_clip.close()
            if video_clip: video_clip.close()
            if audio_clip: audio_clip.close()
            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def start(self):
        self.is_running = True
        print("Shadow Replay is now active and ready to receive frames.")

    def stop(self):
        self.is_running = False
        print("Shadow Replay has been stopped.")

    def get_latest_frame(self):
        return self.video_buffer[-1] if self.video_buffer else None

    def save_replay(self, output_path=None):
        if not self.video_buffer and not self.audio_buffer:
            print("Buffers are empty. Nothing to save.")
            return

        save_thread = threading.Thread(
            target=self._perform_save,
            args=(list(self.video_buffer), list(self.audio_buffer), output_path)
        )
        save_thread.daemon = True
        save_thread.start()