from __future__ import annotations

import os
import time
import wave
import threading as th
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np  # type: ignore

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # graceful fallback

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
        print("[audio] onnxruntime not available; skipping:", model_path)
        return None, None
    if not os.path.exists(model_path):
        print(f"[audio] model not found: {model_path}")
        return None, None
    avail = _available_providers()
    for token in ep_preference:
        ep = _ep_name(token)
        if ep and ep in avail:
            try:
                so = ort.SessionOptions()
                sess = ort.InferenceSession(model_path, sess_options=so, providers=[ep])
                print(f"[audio] loaded {os.path.basename(model_path)} with EP={ep}")
                return sess, ep
            except Exception as e:
                print(f"[audio] fail EP={ep} for {model_path}: {e}")
    try:
        sess = ort.InferenceSession(model_path)
        print(f"[audio] loaded {os.path.basename(model_path)} with default providers")
        return sess, "default"
    except Exception as e:
        print(f"[audio] failed to load {model_path}: {e}")
        return None, None


# ----------------------------
# Buffers & adapters
# ----------------------------
class AudioRingBuffer:
    def __init__(self, rate: int, chunk_ms: int, max_seconds: int):
        self._lock = th.Lock()
        self.rate = rate
        self.chunk_ms = chunk_ms
        self.samples_per_chunk = int(rate * (chunk_ms / 1000.0))
        cap_chunks = int((max_seconds * 1000) / chunk_ms) * 2
        self.buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=cap_chunks)

    def push(self, ts: float, pcm_i16: np.ndarray) -> None:
        with self._lock:
            self.buf.append((ts, pcm_i16.copy()))

    def snapshot(self, last_seconds: int) -> Tuple[np.ndarray, float]:
        cutoff = time.time() - last_seconds
        with self._lock:
            items = [(t, x) for (t, x) in self.buf if t >= cutoff]
        if not items:
            return np.zeros(0, dtype=np.int16), time.time()
        items.sort(key=lambda z: z[0])
        audio = np.concatenate([x for (_, x) in items], axis=0)
        return audio, items[0][0]


@dataclass
class AudioAdapters:
    env_sess: any = None
    vad_sess: any = None
    asr_sess: any = None

    noise_floor: float = 300.0
    nf_alpha: float = 0.01

    def vad(self, pcm_i16: np.ndarray, sensitivity: float, rate: int) -> bool:
        if pcm_i16.size == 0:
            return False
        rms = float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))
        self.noise_floor = (1 - self.nf_alpha) * self.noise_floor + self.nf_alpha * min(self.noise_floor, rms)
        k = 0.5 + (1.5 * (1.0 - float(np.clip(sensitivity, 0.0, 1.0))))
        thr = self.noise_floor * (1.0 + k)
        return rms > thr

    def env_classify(self, pcm_i16: np.ndarray, rate: int) -> List[Tuple[str, float]]:
        if pcm_i16.size == 0:
            return []
        if self.env_sess is None:
            rms = float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))
            conf = float(np.clip(rms / 2000.0, 0.0, 1.0))
            return [("noise", conf)]
        # TODO: implement AST feature pipeline if desired
        return []

    def asr(self, pcm_i16: np.ndarray, rate: int) -> Optional[str]:
        if pcm_i16.size == 0:
            return None
        if self.asr_sess is None:
            return None
        # TODO: implement Parakeet feature pipeline + decoding
        return None


# ----------------------------
# Audio worker
# ----------------------------
class AudioWorker:
    """
    Owns mic capture (sounddevice), buffering, VAD/env/ASR inference, and optional live writer.
    Emits:
      - report_cb(dict) for vad/env/asr lines
      - trigger_cb(ts: float, reasons: List[str]) on audio-side triggers
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
        self._cap_thread: Optional[th.Thread] = None
        self._infer_thread: Optional[th.Thread] = None

        self.buf = AudioRingBuffer(cfg.rate, cfg.chunk_ms, cfg.pre_roll_seconds)
        self.audio = AudioAdapters()

        # sounddevice stream
        self._sd_stream = None

        # Writer sink (wave) controlled by coordinator
        self._writer_lock = th.Lock()
        self._wf: Optional[wave.Wave_write] = None

        # ASR windowing
        self._asr_window_s = 3.0
        self._hold: Deque[np.ndarray] = deque(maxlen=int((self._asr_window_s * 1000) / self.cfg.chunk_ms))

    # --- public API for coordinator ---
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._cap_thread = th.Thread(target=self._capture_loop, name="audio-cap", daemon=True)
        self._cap_thread.start()
        self._infer_thread = th.Thread(target=self._infer_loop, name="audio-infer", daemon=True)
        self._infer_thread.start()

    def stop(self) -> None:
        self._running.clear()
        # stop stream to cease callbacks
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        for t in (self._infer_thread, self._cap_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
        with self._writer_lock:
            try:
                if self._wf is not None:
                    self._wf.close()
            except Exception:
                pass
            self._wf = None

    def snapshot(self, last_seconds: int) -> Tuple[np.ndarray, float]:
        return self.buf.snapshot(last_seconds)

    def set_audio_writer(self, wf: Optional[wave.Wave_write]) -> None:
        with self._writer_lock:
            try:
                if self._wf is not None and wf is None:
                    self._wf.close()
            except Exception:
                pass
            self._wf = wf

    # --- loops ---
    def _capture_loop(self) -> None:
        if sd is None:
            print("[audio] sounddevice unavailable; audio disabled.")
            while self._running.is_set():
                time.sleep(0.25)
            return

        def _cb(indata, frames, time_info, status):
            if status:
                print("[audio] status:", status)
            ts = time.time()
            x = indata[:, 0] if indata.ndim == 2 else indata
            x = np.clip(x, -1.0, 1.0)
            pcm = (x * 32767.0).astype(np.int16)
            self.buf.push(ts, pcm)
            with self._writer_lock:
                if self._wf is not None:
                    try:
                        self._wf.writeframes(pcm.tobytes())
                    except Exception:
                        pass

        block = int(self.cfg.rate * (self.cfg.chunk_ms / 1000.0))
        try:
            self._sd_stream = sd.InputStream(
                channels=1, samplerate=self.cfg.rate, blocksize=block, dtype="float32", callback=_cb
            )
            self._sd_stream.start()
        except Exception as e:
            print("[audio] failed to start stream:", e)
            self._sd_stream = None

        while self._running.is_set():
            time.sleep(0.1)

    def _infer_loop(self) -> None:
        last_env = 0.0
        env_period = 0.5
        vad_active = False
        vad_last_true = 0.0

        while self._running.is_set():
            time.sleep(0.05)
            ts = time.time()

            # VAD on most recent ~20ms
            recent_audio, _ = self.buf.snapshot(max(1, self.cfg.chunk_ms // 10))
            if recent_audio.size > 0:
                tail = recent_audio[-int(self.cfg.rate * (self.cfg.chunk_ms / 1000.0)) :]
            else:
                tail = np.zeros(int(self.cfg.rate * (self.cfg.chunk_ms / 1000.0)), dtype=np.int16)

            vad = self.audio.vad(tail, self.cfg.vad_sensitivity, self.cfg.rate)
            if self.report_cb:
                self.report_cb({"type": "vad", "ts": ts, "flag": vad})

            if vad:
                vad_active = True
                vad_last_true = ts
                self._hold.append(tail)

            # Env classification every ~0.5s
            if (ts - last_env) >= env_period:
                last_env = ts
                audio_1s, _ = self.buf.snapshot(1)
                env = self.audio.env_classify(audio_1s, self.cfg.rate)
                topk = env[:3] if env else []
                if self.report_cb:
                    self.report_cb({"type": "env_audio", "ts": ts, "labels": topk})

                # Triggers
                env_trigger = False
                for lbl, conf in topk:
                    if lbl in self.cfg.env_watchlist or conf >= self.cfg.env_conf:
                        env_trigger = True
                        break

                reasons: List[str] = []
                if vad:
                    reasons.append("vad")
                if env_trigger:
                    reasons.append("env_audio")
                if reasons and self.trigger_cb:
                    self.trigger_cb(ts, reasons)

            # ASR after short VAD window closure
            if vad_active and (ts - vad_last_true) > 0.3:
                if len(self._hold) * int(self.cfg.rate * (self.cfg.chunk_ms / 1000.0)) >= int(
                    self._asr_window_s * self.cfg.rate
                ):
                    wav = np.concatenate(list(self._hold), axis=0)
                    self._hold.clear()
                    text = self.audio.asr(wav, self.cfg.rate)
                    if self.report_cb:
                        self.report_cb({"type": "asr", "ts": ts, "text": text})
                vad_active = False
