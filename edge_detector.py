from __future__ import annotations

import json
import os
import time
import uuid
import wave
import threading as th
from collections import deque
from typing import Deque, List, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from config import Settings
from .interface import IThreatDetector, ThreatEvent
from .video_engine import VideoWorker
from .audio_engine import AudioWorker


def now_s() -> float:
    return time.time()


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]


class EdgeDetector(IThreatDetector):
    """
    Coordinator wiring video+audio workers to implement:
      - threat state machine
      - pre-roll saving
      - live recording
      - report.jsonl and manifest persistence
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self._running = th.Event()
        self._lock = th.RLock()

        # Workers
        self.video = VideoWorker(self.cfg, report_cb=self._report_from_worker, trigger_cb=self._trigger_from_worker)
        self.audio = AudioWorker(self.cfg, report_cb=self._report_from_worker, trigger_cb=self._trigger_from_worker)

        # Event / persistence state
        self._active_event: Optional[ThreatEvent] = None
        self._active_event_dir: Optional[str] = None
        self._report_fp: Optional[open] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._audio_wf: Optional[wave.Wave_write] = None
        self._events: Deque[ThreatEvent] = deque(maxlen=1000)

        # Timers
        self._last_trigger_ts: float = 0.0
        self._had_additional_trigger: bool = False

        # Storage layout
        os.makedirs(self.cfg.storage_root, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.storage_root, "events"), exist_ok=True)

    # ----------------------------
    # IThreatDetector API
    # ----------------------------
    def start(self) -> None:
        with self._lock:
            if self._running.is_set():
                return
            self._running.set()
        self.video.start()
        self.audio.start()
        # small supervisor thread to finalize on timers
        th.Thread(target=self._supervisor, name="edge-supervisor", daemon=True).start()
        print("[edge] detector started.")

    def stop(self) -> None:
        with self._lock:
            if not self._running.is_set():
                return
            self._running.clear()
        self.audio.stop()
        self.video.stop()
        with self._lock:
            if self._active_event:
                self._finalize_event(reason="stop")
        print("[edge] detector stopped.")

    def mark_false_positive(self, event_id: str) -> None:
        with self._lock:
            if self._active_event and self._active_event.id == event_id:
                self._active_event.status = "false_positive"
                self._active_event.last_update = now_s()
                self._write_report({"type": "event_status", "ts": self._active_event.last_update, "status": "false_positive"})
                self._finalize_event(reason="manual_false_positive")
                return
            for ev in list(self._events):
                if ev.id == event_id:
                    ev.status = "false_positive"
                    manifest = os.path.join(self.cfg.storage_root, "events", event_id, "event.json")
                    try:
                        with open(manifest, "w", encoding="utf-8") as fp:
                            json.dump(ev.dict(), fp, indent=2)
                    except Exception:
                        pass
                    break

    def set_thresholds(self, config: dict) -> None:
        with self._lock:
            for k in ["face_conf", "motion_score", "env_conf", "vad_sensitivity"]:
                if k in config:
                    setattr(self.cfg, k, float(config[k]))
            if "env_watchlist" in config and isinstance(config["env_watchlist"], list):
                self.cfg.env_watchlist = [str(x) for x in config["env_watchlist"]]

    def get_active_event(self) -> ThreatEvent | None:
        with self._lock:
            return self._active_event.copy() if self._active_event else None

    def list_events(self, limit: int = 100) -> List[ThreatEvent]:
        with self._lock:
            items = list(self._events)[-int(max(1, limit)) :]
            return [e.copy() for e in items]

    # ----------------------------
    # Worker callbacks
    # ----------------------------
    def _report_from_worker(self, obj: dict) -> None:
        # Forward to report if recording
        with self._lock:
            if self._report_fp is None:
                return
        self._write_report(obj)

    def _trigger_from_worker(self, ts: float, reasons: List[str]) -> None:
        with self._lock:
            if self._active_event is None:
                self._start_event(ts, reasons)
                self._last_trigger_ts = ts
                self._had_additional_trigger = False
            else:
                self._active_event.last_update = ts
                self._active_event.status = "active"
                self._active_event.triggers.extend(reasons)
                self._last_trigger_ts = ts
                self._had_additional_trigger = True
                self._write_report({"type": "event_trigger", "ts": ts, "reasons": reasons})

    # ----------------------------
    # Supervisor: finalize on timers
    # ----------------------------
    def _supervisor(self) -> None:
        while self._running.is_set():
            time.sleep(0.25)
            self._maybe_finalize_from_timers()

    def _maybe_finalize_from_timers(self) -> None:
        with self._lock:
            if self._active_event is None:
                return
            idle = now_s() - self._last_trigger_ts
            if idle >= self.cfg.min_event_seconds:
                self._active_event.status = "closed" if self._had_additional_trigger else "false_positive"
                self._active_event.last_update = now_s()
                self._write_report(
                    {"type": "event_status", "ts": self._active_event.last_update, "status": self._active_event.status}
                )
                self._finalize_event(reason="cooldown")
            elif self._active_event.status == "active" and idle >= 1.0:
                self._active_event.status = "cooldown"
                self._active_event.last_update = now_s()
                self._write_report({"type": "event_status", "ts": self._active_event.last_update, "status": "cooldown"})

    # ----------------------------
    # Event lifecycle
    # ----------------------------
    def _start_event(self, ts: float, reasons: List[str]) -> None:
        eid = new_event_id()
        event_dir = os.path.join(self.cfg.storage_root, "events", eid)
        os.makedirs(event_dir, exist_ok=True)

        v_tmp = os.path.join(event_dir, "video.tmp.mp4")
        a_tmp = os.path.join(event_dir, "audio.tmp.wav")
        rpt = os.path.join(event_dir, "report.jsonl")
        manifest = os.path.join(event_dir, "event.json")

        # Open report
        self._report_fp = open(rpt, "a", encoding="utf-8")

        # Create event record
        self._active_event = ThreatEvent(
            id=eid,
            started_at=ts,
            last_update=ts,
            status="active",
            triggers=list(reasons),
            clip_path_video=os.path.join(event_dir, "video.mp4"),
            clip_path_audio=os.path.join(event_dir, "audio.wav"),
            report_path=rpt,
        )
        self._active_event_dir = event_dir

        # Write manifest early
        try:
            with open(manifest, "w", encoding="utf-8") as fp:
                json.dump(self._active_event.dict(), fp, indent=2)
        except Exception:
            pass

        # Create writers and dump pre-roll
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(v_tmp, fourcc, self.cfg.fps, (self.cfg.width, self.cfg.height))
        self.video.set_video_writer(self._video_writer)

        pre_frames = self.video.snapshot(self.cfg.pre_roll_seconds)
        for _, f in pre_frames:
            try:
                self._video_writer.write(f)
            except Exception:
                pass

        self._audio_wf = wave.open(a_tmp, "wb")
        self._audio_wf.setnchannels(1)
        self._audio_wf.setsampwidth(2)
        self._audio_wf.setframerate(self.cfg.rate)
        self.audio.set_audio_writer(self._audio_wf)

        pre_audio, _ = self.audio.snapshot(self.cfg.pre_roll_seconds)
        if pre_audio.size > 0:
            try:
                self._audio_wf.writeframes(pre_audio.tobytes())
            except Exception:
                pass

        self._write_report({"type": "event_start", "ts": ts, "reasons": reasons})
        print(
            f"[edge] event {eid} started; pre-roll saved "
            f"({len(pre_frames)} frames, {len(pre_audio)} samples)."
        )

    def _finalize_event(self, reason: str) -> None:
        if not self._active_event or not self._active_event_dir:
            return

        event = self._active_event
        event_dir = self._active_event_dir

        v_tmp = os.path.join(event_dir, "video.tmp.mp4")
        a_tmp = os.path.join(event_dir, "audio.tmp.wav")
        v_final = os.path.join(event_dir, "video.mp4")
        a_final = os.path.join(event_dir, "audio.wav")
        manifest = os.path.join(event_dir, "event.json")

        # Detach writers from workers first
        self.video.set_video_writer(None)
        self.audio.set_audio_writer(None)

        # Close here
        try:
            if self._video_writer is not None:
                self._video_writer.release()
        except Exception:
            pass
        finally:
            self._video_writer = None

        try:
            if self._audio_wf is not None:
                self._audio_wf.close()
        except Exception:
            pass
        finally:
            self._audio_wf = None

        # Atomic move
        try:
            if os.path.exists(v_tmp):
                os.replace(v_tmp, v_final)
        except Exception:
            pass
        try:
            if os.path.exists(a_tmp):
                os.replace(a_tmp, a_final)
        except Exception:
            pass

        # Close report
        try:
            if self._report_fp is not None:
                self._report_fp.flush()
                self._report_fp.close()
        except Exception:
            pass
        finally:
            self._report_fp = None

        # Update manifest
        try:
            with open(manifest, "w", encoding="utf-8") as fp:
                json.dump(event.dict(), fp, indent=2)
        except Exception:
            pass

        # History & reset
        self._events.append(event.copy())
        print(f"[edge] event {event.id} finalized ({event.status}); reason={reason}")
        self._active_event = None
        self._active_event_dir = None
        self._last_trigger_ts = 0.0
        self._had_additional_trigger = False

    # ----------------------------
    # Reporting
    # ----------------------------
    def _write_report(self, obj: dict) -> None:
        with self._lock:
            if self._report_fp is None:
                return
            try:
                self._report_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # Keep manifest hot with last_update/triggers
            if self._active_event and self._active_event_dir:
                manifest = os.path.join(self._active_event_dir, "event.json")
                try:
                    with open(manifest, "w", encoding="utf-8") as fp:
                        json.dump(self._active_event.dict(), fp, indent=2)
                except Exception:
                    pass


# Optional singleton if UI wants default
_singleton: Optional[EdgeDetector] = None


def get_default_detector() -> EdgeDetector:
    global _singleton
    if _singleton is None:
        _singleton = EdgeDetector(Settings())
    return _singleton
