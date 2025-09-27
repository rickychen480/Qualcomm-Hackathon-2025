from __future__ import annotations

import os
import time
import threading as th
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # graceful fallback

from config import Settings


# ----------------------------
# Small ONNX helpers (duplicated locally for simplicity)
# ----------------------------
EP_MAP = {
    "QNN": "QNNExecutionProvider",
    "DML": "DmlExecutionProvider",
    "CPU": "CPUExecutionProvider",
}


def _available_providers() -> List[str]:
    if ort is None:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def _ep_name(token: str) -> Optional[str]:
    return EP_MAP.get(token.upper())


def init_onnx_session(model_path: str, ep_preference: List[str]):
    if ort is None:
        print("[video] onnxruntime not available; skipping:", model_path)
        return None, None
    if not os.path.exists(model_path):
        print(f"[video] model not found: {model_path}")
        return None, None
    avail = _available_providers()
    for token in ep_preference:
        ep = _ep_name(token)
        if ep and ep in avail:
            try:
                so = ort.SessionOptions()
                sess = ort.InferenceSession(model_path, sess_options=so, providers=[ep])
                print(f"[video] loaded {os.path.basename(model_path)} with EP={ep}")
                return sess, ep
            except Exception as e:
                print(f"[video] fail EP={ep} for {model_path}: {e}")
    try:
        sess = ort.InferenceSession(model_path)
        print(f"[video] loaded {os.path.basename(model_path)} with default providers")
        return sess, "default"
    except Exception as e:
        print(f"[video] failed to load {model_path}: {e}")
        return None, None


# ----------------------------
# Buffers & inference utils
# ----------------------------
class VideoRingBuffer:
    def __init__(self, max_seconds: int, fps: int):
        self._lock = th.Lock()
        cap = max_seconds * max(1, fps) * 2
        self.buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=cap)

    def push(self, ts: float, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self.buf.append((ts, frame_bgr.copy()))

    def snapshot(self, last_seconds: int) -> List[Tuple[float, np.ndarray]]:
        cutoff = time.time() - last_seconds
        with self._lock:
            return [(t, fr.copy()) for (t, fr) in self.buf if t >= cutoff]


class MotionEMA:
    def __init__(self, alpha: float = 0.25):
        self.prev_gray: Optional[np.ndarray] = None
        self.ema: float = 0.0
        self.alpha = float(np.clip(alpha, 0.01, 0.99))

    def step(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        score = float(np.mean(diff) / 255.0)
        self.ema = self.alpha * score + (1 - self.alpha) * self.ema
        return self.ema


@dataclass
class FaceAdapter:
    sess: any = None
    conf_thr: float = 0.5
    use_haar: bool = False
    haar: any = None

    def __post_init__(self):
        if self.sess is None:
            try:
                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")  # type: ignore
                self.haar = cv2.CascadeClassifier(cascade_path)
                self.use_haar = True
                print("[video] FaceAdapter using Haar fallback.")
            except Exception:
                print("[video] FaceAdapter: no detector available.")

    def infer(self, frame_bgr: np.ndarray) -> List[Tuple[float, Tuple[int, int, int, int]]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if self.use_haar and self.haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 4)
            return [(1.0, (int(x), int(y), int(x + w), int(y + h))) for (x, y, w, h) in faces]

        if self.sess is None:
            return []

        # NOTE: UltraFace decoding omitted for brevity; returns [] if not implemented.
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (320, 240))
            inp = img.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]
            self.sess.run(None, {self.sess.get_inputs()[0].name: inp})
        except Exception:
            pass
        return []


# ----------------------------
# Video worker
# ----------------------------
class VideoWorker:
    """
    Owns camera, buffering, motion & face inference, and optional live writer.
    Emits:
      - report_cb(dict) for motion/face lines
      - trigger_cb(ts: float, reasons: List[str]) on video-side triggers
    """

    def __init__(
        self,
        cfg: Settings,
        report_cb: Optional[Callable[[dict], None]] = None,
        trigger_cb: Optional[Callable[[float, List[str]], None]] = None,
    ):
        self.cfg = cfg
        self.report_cb = report_cb
        self.trigger_cb = trigger_cb

        self._running = th.Event()
        self._thread: Optional[th.Thread] = None

        self.buf = VideoRingBuffer(cfg.pre_roll_seconds, cfg.fps)
        self.motion = MotionEMA(alpha=0.25)

        # ONNX face (optional)
        face_path = os.path.join(cfg.storage_root, "models", "ultraface.onnx")
        sess, _ = init_onnx_session(face_path, cfg.ep_preference)
        self.face = FaceAdapter(sess=sess, conf_thr=self.cfg.face_conf)

        # State
        self._frame_idx = 0
        self._face_every_n = 4
        self._last_face_ts: float = 0.0
        self._recent_faces: List[Tuple[float, Tuple[int, int, int, int]]] = []

        # Writer sink (set by coordinator)
        self._writer_lock = th.Lock()
        self._writer: Optional[cv2.VideoWriter] = None

    # --- public API for coordinator ---
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = th.Thread(target=self._loop, name="video-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._writer_lock:
            try:
                if self._writer is not None:
                    self._writer.release()
            except Exception:
                pass
            self._writer = None

    def snapshot(self, last_seconds: int) -> List[Tuple[float, np.ndarray]]:
        return self.buf.snapshot(last_seconds)

    def set_video_writer(self, writer: Optional[cv2.VideoWriter]) -> None:
        with self._writer_lock:
            # release previous if any
            try:
                if self._writer is not None and writer is None:
                    self._writer.release()
            except Exception:
                pass
            self._writer = writer

    # --- internal loop ---
    def _loop(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        except Exception:
            pass

        frame_period = 1.0 / float(self.cfg.fps)
        next_tick = time.time()

        while self._running.is_set():
            ok, frame = cap.read()
            ts = time.time()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
                frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))

            # Buffer
            self.buf.push(ts, frame)

            # Motion
            mscore = self.motion.step(frame)
            if self.report_cb:
                self.report_cb({"type": "motion", "ts": ts, "score": mscore})

            # Face (every N frames)
            self._frame_idx += 1
            faces_to_write: List[Tuple[float, Tuple[int, int, int, int]]] = []
            if (self._frame_idx % self._face_every_n) == 0:
                faces = self.face.infer(frame)
                faces = [(c, box) for (c, box) in faces if c >= self.cfg.face_conf]
                if faces:
                    faces_to_write = [(float(c), tuple(map(int, box))) for (c, box) in faces]
                    self._last_face_ts = ts
                    self._recent_faces = faces_to_write
                if self.report_cb:
                    self.report_cb({"type": "face", "ts": ts, "boxes": faces_to_write})

            # Video triggers
            v_trigger = False
            reasons: List[str] = []
            if mscore >= self.cfg.motion_score:
                v_trigger = True
                reasons.append("motion")
            if self._recent_faces and (ts - self._last_face_ts) < 1.0:
                v_trigger = True
                reasons.append("face")

            if v_trigger and self.trigger_cb:
                self.trigger_cb(ts, reasons)

            # Write live video if writer present
            with self._writer_lock:
                if self._writer is not None:
                    try:
                        self._writer.write(frame)
                    except Exception:
                        pass

            # pacing
            next_tick += frame_period
            sleep_d = next_tick - time.time()
            if sleep_d > 0:
                time.sleep(sleep_d)
            else:
                next_tick = time.time()

        cap.release()
