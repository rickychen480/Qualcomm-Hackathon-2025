from pydantic import BaseModel, Field
from typing import List


class Settings(BaseModel):
    # Video
    fps: int = Field(20, description="Target capture/encode FPS")
    width: int = Field(1280, description="Capture width")
    height: int = Field(720, description="Capture height")

    # Audio
    rate: int = Field(16000, description="PCM sample rate (Hz)")
    chunk_ms: int = Field(20, description="Audio chunk size in milliseconds")

    # Thresholds (0.0–1.0 where applicable)
    face_conf: float = Field(0.5, description="Face confidence threshold")
    motion_score: float = Field(0.15, description="Normalized motion threshold (0..1)")
    env_conf: float = Field(0.6, description="Env-audio confidence threshold")
    vad_sensitivity: float = Field(
        0.6, description="VAD sensitivity (lower=more sensitive)"
    )

    # Watchlist for environmental sounds (labels depend on the classifier)
    env_watchlist: List[str] = Field(default_factory=list)

    # Timing
    pre_roll_seconds: int = Field(15, description="Pre-roll duration to include (s)")
    min_event_seconds: int = Field(
        300, description="Minimum event duration or cooldown (s)"
    )

    # Storage
    storage_root: str = Field("./data", description="Root directory for saved data")

    # Execution provider preference for onnxruntime
    ep_preference: List[str] = Field(
        default_factory=lambda: ["QNN", "DML", "CPU"],
        description="ONNX EP preference order",
    )
