import cv2
import sounddevice  # Unused import necessary to ensure pyaudio import works correctly
import pyaudio
import wave
import threading
import collections
import time
import os
from moviepy import VideoFileClip, AudioFileClip


class ShadowReplay:
    """
    A class to continuously record video and audio into a buffer, allowing for
    the last 'n' seconds to be saved on command.

    Sample usage:
        recorder = ShadowReplay(record_seconds=5)
        recorder.start()
        time.sleep(15)
        recorder.save_replay()
        time.sleep(10)
        recorder.stop()
    """

    def __init__(
        self,
        record_seconds=15,
        camera_index=0,
        fps=30.0,
        frame_width=640,
        frame_height=480,
        channels=1,
        rate=44100,
    ):
        """
        Initializes the recorder with specified configurations.

        Args:
            record_seconds (int): The duration of the buffer in seconds.
            camera_index (int): The index of the camera to use.
            fps (float): The frames per second for video recording.
            frame_width (int): The width of the video frame.
            frame_height (int): The height of the video frame.
            channels (int): The number of audio channels.
            rate (int): The audio sampling rate.
        """
        # --- Configurations ---
        self.record_seconds = record_seconds
        self.camera_index = camera_index
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        # --- Audio Settings ---
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.chunk = 1024  # Audio samples per frame

        # --- Buffers and Thread Control ---
        buffer_video_frames = int(self.record_seconds * self.fps)
        buffer_audio_chunks = int(self.record_seconds * self.rate / self.chunk)
        self.video_buffer = collections.deque(maxlen=buffer_video_frames)
        self.audio_buffer = collections.deque(maxlen=buffer_audio_chunks)

        self.stop_event = threading.Event()
        self.video_thread = None
        self.audio_thread = None
        self.is_running = False

    def _video_recorder(self):
        """Capture video frames and store them in the buffer."""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                self.video_buffer.append(frame)
            else:
                print("Error: Could not read frame from camera.")
                break
        
        cap.release()

    def _audio_recorder(self):
        """Capture audio chunks and store them in the buffer."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        while not self.stop_event.is_set():
            data = stream.read(self.chunk)
            self.audio_buffer.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _perform_save(self, video_frames, audio_chunks, output_path):
        """The actual file-saving logic. This is run in a separate thread."""

        # --- Generate file paths ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_video_path = f"temp_video_{timestamp}.mp4"
        temp_audio_path = f"temp_audio_{timestamp}.wav"
        output_path = output_path or f"replay_{timestamp}.mp4"

        # --- Save video frames to a temporary file ---
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(
            temp_video_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        for frame in video_frames:
            out.write(frame)
        out.release()

        # --- Save audio chunks to a temporary file ---
        wf = wave.open(temp_audio_path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(audio_chunks))
        wf.close()

        # --- Combine video and audio using MoviePy ---
        try:
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = AudioFileClip(temp_audio_path)
            video_clip.audio = audio_clip
            video_clip.write_videofile(
                output_path, codec="libx264", audio_codec="aac", logger=None
            )
            print(f"Replay saved successfully to {output_path}")

        except Exception as e:
            print(f"Error combining files: {e}")

        finally:
            # Clean up temporary files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


    def start(self):
        """Starts the video and audio recording threads."""
        if self.is_running:
            print("Recorder is already running.")
            return

        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._video_recorder)
        self.audio_thread = threading.Thread(target=self._audio_recorder)

        self.video_thread.start()
        self.audio_thread.start()
        self.is_running = True
        print("Shadow recording started.")

    def stop(self):
        """Stops the recording threads and cleans up resources."""
        if not self.is_running:
            print("Recorder is not running.")
            return

        self.stop_event.set()
        self.video_thread.join()
        self.audio_thread.join()
        self.is_running = False
        print("Shadow recording stopped.")

    def get_latest_frame(self):
        """Returns the most recent frame from the video buffer for display."""
        return self.video_buffer[-1] if self.video_buffer else None

    def save_replay(self, output_filename=None):
        """
        Saves the contents of the buffers to a final combined MP4 file.

        Args:
            output_filename (str, optional): The path for the output file.
                If None, a timestamped filename is generated.
        """

        # Copy buffer content to prevent modification during save
        audio_chunks = list(self.audio_buffer)
        video_frames = list(self.video_buffer)

        if not video_frames or not audio_chunks:
            print("Buffers are empty. Nothing to save.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_filename or f"replay_{timestamp}.mp4"

        # Create and start the thread for saving
        save_thread = threading.Thread(
            target=self._perform_save,
            args=(video_frames, audio_chunks, output_path)
        )
        save_thread.daemon = True  # Allows main program to exit even if thread is running
        save_thread.start()
        print(f"Save command received. Replay will be saved in the background.")


# --- Example Usage ---
if __name__ == "__main__":
    # Create an instance of the recorder
    recorder = ShadowReplay(record_seconds=5, fps=30.0)

    # Start the recording threads
    recorder.start()

    print("\n--- Shadow Recording Active ---")
    print("Press 's' to save the last 5 seconds.")
    print("Press 'q' to quit.")

    cv2.namedWindow("Webcam Feed")

    while True:
        # Display the live feed
        frame = recorder.get_latest_frame()
        if frame is not None:
            cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        # Call save_replay() when needed
        if key == ord("s"):
            print("\nSaving replay...")
            recorder.save_replay()
            print("Continuing recording...")

        elif key == ord("q"):
            break

    # Stop the recorder gracefully
    recorder.stop()
    cv2.destroyAllWindows()
