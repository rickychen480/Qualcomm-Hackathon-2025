import cv2
import sounddevice  # Unused import necessary to ensure pyaudio import works correctly
import pyaudio
import wave
import threading
import collections
import time
import os
from moviepy import VideoFileClip, AudioFileClip

# --- Configurations ---
RECORD_SECONDS = 15  # Duration of the buffer in seconds
CAMERA_INDEX = 0  # Change if you have multiple cameras

# Video settings
FPS = 20.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Create circular buffers (deques) with a maximum length
video_buffer = collections.deque(maxlen=int(RECORD_SECONDS * FPS))
audio_buffer = collections.deque(maxlen=int(RECORD_SECONDS * RATE / CHUNK))

# Thread control
stop_event = threading.Event()


def video_recorder():
    """Captures video frames and stores them in the buffer."""

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            video_buffer.append(frame)
        else:
            print("Error: Could not read frame from camera.")
            break
        # Control the frame rate
        time.sleep(1 / FPS)

    cap.release()


def audio_recorder():
    """Captures audio chunks and stores them in the buffer."""

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    while not stop_event.is_set():
        data = stream.read(CHUNK)
        audio_buffer.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


def save_replay():
    """Saves the contents of the buffers to video and audio files, then combines them."""

    # Create a copy to prevent modification while saving
    video_frames = list(video_buffer)
    audio_chunks = list(audio_buffer)

    if not video_frames or not audio_chunks:
        print("Buffers are empty. Nothing to save.")
        return

    # --- Create temporary file paths ---
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    temp_video_path = f"temp_video_{timestamp}.mp4"
    temp_audio_path = f"temp_audio_{timestamp}.wav"
    output_path = f"replay_{timestamp}.mp4"

    # --- Save video frames to a temporary file ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    out = cv2.VideoWriter(temp_video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for frame in video_frames:
        out.write(frame)
    out.release()

    # --- Save audio chunks to a temporary file ---
    wf = wave.open(temp_audio_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(audio_chunks))
    wf.close()

    # --- Combine video and audio using MoviePy ---
    try:
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(temp_audio_path)

        # Set the audio of the video clip to the new audio clip
        video_clip.audio = audio_clip

        # Write the final combined clip to a file
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Replay saved successfully to {output_path}")

    except Exception as e:
        print(f"Error combining files: {e}")

    finally:
        # --- Clean up temporary files ---
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    # Start the recording threads
    video_thread = threading.Thread(target=video_recorder)
    audio_thread = threading.Thread(target=audio_recorder)
    video_thread.start()
    audio_thread.start()

    print("\n--- Shadow Recording Active ---")
    print("Press 's' to save the last 15 seconds.")
    print("Press 'q' to quit.")

    # Use a dummy window to capture key presses
    cv2.namedWindow("Webcam Feed")

    while True:
        # Display the most recent frame
        if video_buffer:
            cv2.imshow("Webcam Feed", video_buffer[-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_replay()
        elif key == ord("q"):
            break

    # Signal threads to stop and wait for them
    stop_event.set()
    video_thread.join()
    audio_thread.join()

    cv2.destroyAllWindows()
