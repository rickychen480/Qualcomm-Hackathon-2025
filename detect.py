# edge_ai_threat_module.py
# Python 3.11+
# Deps: onnxruntime, numpy, opencv-python, (optional) webrtcvad, (optional) onnx-asr
from __future__ import annotations
from typing import Any, Tuple, List, Dict
import os
import numpy as np
import cv2

# Optional imports (graceful degrade)
try:
    import onnxruntime as ort  # noqa
except Exception:
    ort = None

try:
    import webrtcvad  # noqa
except Exception:
    webrtcvad = None

try:
    import onnx_asr  # noqa
except Exception:
    onnx_asr = None


# ------------------------------ PROVIDERS ------------------------------ #

def _provider_list(ep_preference: Tuple[str, ...]) -> List[str]:
    mapping = {
        "QNN": "QNNExecutionProvider",
        "CUDA": "CUDAExecutionProvider",
        "DML": "DmlExecutionProvider",
        "CoreML": "CoreMLExecutionProvider",
        "OpenVINO": "OpenVINOExecutionProvider",
        "CPU": "CPUExecutionProvider",
    }
    wanted = [mapping.get(x, None) for x in ep_preference]
    wanted = [p for p in wanted if p is not None]
    if "CPUExecutionProvider" not in wanted:
        wanted.append("CPUExecutionProvider")
    return wanted


# ------------------------------ SESSION LOADING ------------------------------ #

def load_sessions(ep_preference=("QNN", "DML", "CPU")) -> dict:
    """
    Load inference "sessions":
      - face: UltraFace ONNX (via onnxruntime)
      - vad: WebRTC VAD instance (Python)
      - env:  AST AudioSet classifier ONNX (via onnxruntime)
      - asr:  Parakeet-TDT (via onnx-asr) if a LOCAL model dir is provided
    Model paths via env (local only; no network):
      EDGEAI_FACE_ONNX  -> file path to UltraFace .onnx
      EDGEAI_ENV_ONNX   -> file path to AST .onnx
      EDGEAI_ASR_LOCAL_DIR -> dir containing Parakeet TDT ONNX export (model.onnx + vocab.txt)
      EDGEAI_ASR_ID     -> onnx-asr model id string (default: nemo-parakeet-tdt-0.6b-v3), used with LOCAL dir
    Returns: {"face": ort_sess|None, "vad": vad_obj|None, "env": ort_sess|None,
              "asr": asr_model_or_None, "env_label_names": list|None, "ep": str}
    """
    providers = _provider_list(ep_preference)

    def _mk_ort(path: str | None):
        if ort is None or not path or not os.path.exists(path):
            return None
        try:
            return ort.InferenceSession(path, providers=providers)
        except Exception:
            return None

    face_path = os.getenv("EDGEAI_FACE_ONNX", r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\version-RFB-640.onnx")
    env_path  = os.getenv("EDGEAI_ENV_ONNX", r"C:C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\ast_audioset.onnx")
    asr_path  = os.getenv("EDGEAI_ASR_ONNX", r"C:\data\models\parakeet_tdt_0.6b_v3.onnx")

    face_sess = _mk_ort(face_path)
    env_sess = _mk_ort(env_path)
    
    asr_sess = _mk_ort(asr_path)
    # Try to fetch label names from env model metadata
    env_label_names = None
    if env_sess is not None:
        try:
            meta = env_sess.get_modelmeta()
            if meta and meta.custom_metadata_map:
                csv = meta.custom_metadata_map.get("labels", "")
                if csv:
                    env_label_names = [s.strip() for s in csv.split(",") if s.strip()]
        except Exception:
            pass

    # VAD
    vad_obj = None
    if webrtcvad is not None:
        try:
            vad_obj = webrtcvad.Vad(2)
        except Exception:
            vad_obj = None

    # ASR via onnx-asr (local only)
    asr_model = None
    if onnx_asr is not None:
        local_dir = os.getenv("EDGEAI_ASR_LOCAL_DIR", "").strip()
        # Back-compat: if EDGEAI_ASR_ONNX points to a file/dir, treat its directory as local_dir
        if not local_dir:
            p = os.getenv("EDGEAI_ASR_ONNX", "").strip()
            if p and os.path.isdir(p):
                local_dir = p
            elif p and os.path.isfile(p):
                local_dir = os.path.dirname(p)

        if local_dir and os.path.isdir(local_dir):
            model_id = os.getenv("EDGEAI_ASR_ID", "nemo-parakeet-tdt-0.6b-v3").strip()
            try:
                # Loads WITHOUT downloading if local_dir provided
                asr_model = onnx_asr.load_model(model_id, local_dir)
            except Exception:
                asr_model = None

    # Pick EP
    ep_used = "CPUExecutionProvider"
    try:
        if face_sess:
            ep_used = face_sess.get_providers()[0]
        elif env_sess:
            ep_used = env_sess.get_providers()[0]
    except Exception:
        pass

    return {
        "face": face_sess,
        "vad": vad_obj,
        "env": env_sess,
        "asr": asr_model,
        "env_label_names": env_label_names,
        "ep": ep_used,
    }


# ------------------------------ VIDEO PIPELINE ------------------------------ #

def preprocess_video(frame: np.ndarray) -> np.ndarray:
    """
    Expect BGR HxWx3. Convert to RGB, resize to (W,H), normalize like the dev code:
      image = (image - [127,127,127]) / 128
    Returns NCHW float32 (1,3,H,W). detect_faces will flip to NHWC if the model needs it.

    Choose size via env:
      EDGEAI_FACE_RES=320x240  (UltraFace RFB-320)
      EDGEAI_FACE_RES=640x480  (UltraFace RFB-640)
    Default: 640x480.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("preprocess_video expects HxWx3 BGR frame.")

    # pick model input size
    res = os.getenv("EDGEAI_FACE_RES", "640x480").lower()
    try:
        w_s, h_s = res.split("x")
        W, H = int(w_s), int(h_s)
    except Exception:
        W, H = 640, 480

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    # (img - 127)/128 as in dev code
    img = (rgb.astype(np.float32) - 127.0) / 128.0

    # CHW → NCHW
    nchw = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)  # (1,3,H,W)
    return nchw.astype(np.float32)

def _iou(a, b) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1, ix2, iy2 = max(xa1, xb1), max(ya1, yb1), min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)

def _ultraface_hard_nms(box_probs: np.ndarray, iou_threshold: float = 0.3, top_k: int = -1) -> np.ndarray:
    """
    Simplified hard-NMS compatible with the dev code.
    box_probs: [N, 5] with columns [x1,y1,x2,y2,score] in the SAME coord space (normalized or pixels).
    Returns the kept rows in original format.
    """
    if box_probs.size == 0:
        return box_probs
    # sort by score ascending, then pop from end
    order = np.argsort(box_probs[:, 4])
    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        order = order[:-1]
        if order.size == 0:
            break
        # IoU with remaining
        xx1 = np.maximum(box_probs[i, 0], box_probs[order, 0])
        yy1 = np.maximum(box_probs[i, 1], box_probs[order, 1])
        xx2 = np.minimum(box_probs[i, 2], box_probs[order, 2])
        yy2 = np.minimum(box_probs[i, 3], box_probs[order, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (box_probs[i, 2] - box_probs[i, 0]) * (box_probs[i, 3] - box_probs[i, 1])
        area_r = (box_probs[order, 2] - box_probs[order, 0]) * (box_probs[order, 3] - box_probs[order, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        # keep only those with IoU <= thr
        order = order[iou <= iou_threshold]
        if top_k > 0 and len(keep) >= top_k:
            break
    return box_probs[keep]

    if boxes.size == 0:
        return boxes, probs
    order = np.argsort(probs)
    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        order = order[:-1]
        if order.size == 0:
            break
        ious = np.array([_iou(boxes[i], boxes[j]) for j in order])
        order = order[ious <= iou_thr]
    return boxes[keep], probs[keep]


def detect_faces(face_sess: Any, frame_input: np.ndarray) -> Tuple[float, List[Tuple[float,float,float,float,float]]]:
    """
    UltraFace runner aligned with the dev sample:
      - Input already preprocessed to (1,3,H,W), (img-127)/128.
      - If the model expects NHWC (1,H,W,3), we auto-transpose.
      - Outputs: confidences [1,N,2], boxes [1,N,4] (normalized corner coords).
      - We run per-class threshold on face class (index 1), then hard-NMS.

    Returns:
      (max_conf, [(x1,y1,x2,y2,conf),...]) with coords in the SAME space as model outputs.
      Since UltraFace emits normalized boxes, we keep them normalized (0..1), which fits your module.
    """
    if face_sess is None or frame_input is None:
        return 0.0, []

    try:
        in0 = face_sess.get_inputs()[0]
        xin = frame_input

        # If model is NHWC and we passed NCHW, transpose
        s = in0.shape
        if len(s) == 4 and s[-1] == 3 and frame_input.shape[1] == 3:
            xin = np.transpose(frame_input, (0, 2, 3, 1))  # NCHW -> NHWC

        outs = face_sess.run(None, {in0.name: xin})

        # Accept (confidences, boxes) in any order; squeeze batch dim if present
        confidences = boxes = None
        for arr in outs:
            a = arr
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 2 and a.shape[-1] == 2:
                confidences = a
            elif a.ndim == 2 and a.shape[-1] == 4:
                boxes = a

        if (confidences is None or boxes is None) and len(outs) == 2:
            a, b = outs
            a = a[0] if a.ndim == 3 and a.shape[0] == 1 else a
            b = b[0] if b.ndim == 3 and b.shape[0] == 1 else b
            if a.ndim == 2 and a.shape[-1] == 4 and b.ndim == 2 and b.shape[-1] == 2:
                boxes, confidences = a, b

        if confidences is None or boxes is None or boxes.size == 0:
            return 0.0, []

        # Dev logic: for class_index in [1..C-1] (faces are index 1)
        face_probs = confidences[:, 1]
        mask = face_probs > 0.7  # dev threshold; your module can still override with thresholds['face_conf']
        if not np.any(mask):
            # even if none pass 0.7, we still want max_conf for scoring
            return float(face_probs.max(initial=0.0)), []

        sel_boxes = boxes[mask]
        sel_probs = face_probs[mask].reshape(-1, 1)
        box_probs = np.concatenate([sel_boxes, sel_probs], axis=1)  # [x1,y1,x2,y2,score] normalized

        kept = _ultraface_hard_nms(box_probs, iou_threshold=0.3, top_k=-1)
        if kept.size == 0:
            return float(face_probs.max(initial=0.0)), []

        max_conf = float(np.max(kept[:, 4]))
        out = [(float(x1), float(y1), float(x2), float(y2), float(c)) for (x1, y1, x2, y2, c) in kept.tolist()]
        return max_conf, out

    except Exception:
        return 0.0, []


def motion_score(prev_frame: np.ndarray | None, frame: np.ndarray) -> float:
    if prev_frame is None:
        return 0.0
    if prev_frame.shape[:2] != frame.shape[:2]:
        prev = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))
    else:
        prev = prev_frame
    g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(g1, g2), (5, 5), 0)
    return float(np.clip(diff.mean() / 255.0, 0.0, 1.0))


# ------------------------------ AUDIO PIPELINE ------------------------------ #

def preprocess_audio(chunk: np.ndarray, target_rate: int = 16000) -> np.ndarray:
    """
    Ensure mono 16 kHz float32; this function assumes the input is already 16 kHz.
    """
    if chunk is None or np.size(chunk) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = np.asarray(chunk, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.reshape(-1)


def vad_score(vad_sess: Any, audio_mono_16k: np.ndarray) -> float:
    if vad_sess is None or audio_mono_16k is None or audio_mono_16k.size == 0:
        return 0.0
    sr = 16000
    frame_len = int(sr * 0.03)  # 30 ms (more robust)
    n = audio_mono_16k.size // frame_len
    if n == 0:
        return 0.0

    energy_gate = float(os.getenv("EDGEAI_VAD_ENERGY", "0.017"))  # ≈ -35 dBFS
    x = audio_mono_16k[: n * frame_len].astype(np.float32)
    frames = x.reshape(n, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    step = frame_len * 2

    voiced = 0
    valid = 0
    for i in range(n):
        if rms[i] < energy_gate:
            continue
        valid += 1
        fb = pcm16[i*step:(i+1)*step]
        try:
            if vad_sess.is_speech(fb, sr):
                voiced += 1
        except Exception:
            pass

    return 0.0 if valid == 0 else float(voiced) / float(valid)



# # ------------------------------ ENV AUDIO (AST) ------------------------------ #

# def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
#     if fmax is None:
#         fmax = sr / 2.0

#     def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
#     def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

#     fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
#     mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num=n_mels + 2)
#     hz = mel_to_hz(mels)
#     bins = np.floor((n_fft + 1) * hz / sr).astype(int)
#     fb = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
#     for m in range(1, n_mels + 1):
#         f1, f2, f3 = bins[m - 1], bins[m], bins[m + 1]
#         if f2 == f1:
#             f2 = min(f2 + 1, 1 + n_fft // 2)
#         if f3 == f2:
#             f3 = min(f3 + 1, 1 + n_fft // 2)
#         for k in range(f1, f2):
#             if 0 <= k < fb.shape[1]:
#                 fb[m - 1, k] = (k - f1) / max(1, (f2 - f1))
#         for k in range(f2, f3):
#             if 0 <= k < fb.shape[1]:
#                 fb[m - 1, k] = (f3 - k) / max(1, (f3 - f2))
#     # optional equal-area normalization omitted for simplicity
#     return fb.astype(np.float32)


# def _logmel_128(audio: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 160, win: int = 400) -> np.ndarray:
#     if audio.size == 0:
#         return np.zeros((128, 0), dtype=np.float32)
#     # pre-emphasis
#     x = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
#     window = np.hanning(win).astype(np.float32)
#     frames = []
#     for start in range(0, max(0, x.size - win + 1), hop):
#         seg = x[start:start + win] * window
#         if win < n_fft:
#             seg = np.pad(seg, (0, n_fft - win))
#         elif win > n_fft:
#             seg = seg[:n_fft]
#         spec = np.fft.rfft(seg, n=n_fft)
#         power = (spec.real**2 + spec.imag**2).astype(np.float32)
#         frames.append(power)
#     if not frames:
#         frames = [np.zeros((1 + n_fft // 2,), dtype=np.float32)]
#     S = np.stack(frames, axis=1)  # (1+n_fft//2, T)
#     fb = _mel_filterbank(sr, n_fft, n_mels=128)
#     M = fb @ S
#     M = np.maximum(M, 1e-10)
#     return np.log(M).astype(np.float32)


# def _ast_normalize(logmel: np.ndarray, mean: float = -4.2677393, std: float = 4.5689974) -> np.ndarray:
#     return (logmel - mean) / (std + 1e-6)


# def env_labels(env_sess: Any, audio_mono_16k: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
#     if env_sess is None or audio_mono_16k.size == 0:
#         return []
#     try:
#         feats = _ast_normalize(_logmel_128(audio_mono_16k))  # (128, T)
#         in0 = env_sess.get_inputs()[0]
#         shape = list(in0.shape)
#         x = feats
#         # Adapt to [1,128,T] or [1,1,128,T] (common AST exports)
#         if len(shape) == 3:
#             x = x[np.newaxis, :, :]
#         elif len(shape) == 4:
#             if shape[1] == 1:
#                 x = x[np.newaxis, np.newaxis, :, :]
#             else:
#                 x = x[np.newaxis, :, :, np.newaxis]
#         else:
#             x = x[np.newaxis, :, :]
#         # Optional center pad/crop if model has fixed T
#         if isinstance(shape[-1], int) and shape[-1] > 0:
#             Texp = shape[-1]
#             curT = x.shape[-1]
#             if curT < Texp:
#                 padL = (Texp - curT) // 2
#                 padR = Texp - curT - padL
#                 x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(padL, padR)], mode="constant")
#             elif curT > Texp:
#                 st = (curT - Texp) // 2
#                 x = x[..., st:st + Texp]
#         logits = env_sess.run(None, {in0.name: x.astype(np.float32)})[0]
#         if logits.ndim == 3:  # [B,T,K]
#             logits = logits.mean(axis=1)[0]
#         elif logits.ndim == 2:
#             logits = logits[0]
#         m = float(np.max(logits))
#         p = np.exp(logits - m)
#         p /= np.sum(p)
#         idx = np.argsort(p)[::-1][:max(1, top_k)]
#         return [(f"class_{int(i)}", float(p[i])) for i in idx]
#     except Exception:
#         return []

# pip install torchaudio transformers onnxruntime soundfile librosa

import os, warnings
import numpy as np
from pathlib import Path
import onnxruntime as ort

from transformers import ASTFeatureExtractor, AutoConfig

# --- silence torchaudio deprecation noise only ---
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
)

# -----------------------
# Paths (ABSOLUTE)
# -----------------------
AUDIO_PATH = Path(r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\cars-honking-sound-311054.mp3")
ONNX_PATH  = Path(r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\ast_audioset.onnx")

# If you want HF label names, set the original HF model id here; else keep None.
MODEL_ID = None  # e.g., "MIT/ast-finetuned-audioset-10-10-0.4593"

# Force layout if auto-guess fails. Options:
#   None  -> try to match ONNX input shape
#   "B1TF" -> (B, 1, T, F)
#   "B1FT" -> (B, 1, F, T)
FORCE_LAYOUT = None

# -----------------------
# Utilities
# -----------------------
def exists(path: Path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return p

def load_audio(filepath, target_sr=16000):
    fp = exists(filepath)
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(fp))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        wav = waveform.squeeze().numpy().astype(np.float32)
        return wav, sr
    except Exception:
        # robust fallback for mp3 on Windows
        import soundfile as sf, librosa
        wav, sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return wav.astype(np.float32), sr

def make_feature_extractor():
    try:
        return ASTFeatureExtractor.from_pretrained(MODEL_ID) if MODEL_ID else ASTFeatureExtractor()
    except Exception:
        return ASTFeatureExtractor()

def extract_inputs(waveform_np, sr, feature_extractor):
    feats = feature_extractor(
        waveform_np,
        sampling_rate=sr,
        padding="max_length",   # AST is trained on ~10s windows
        return_tensors="np",
    )
    x = feats["input_values"].astype(np.float32)   # (B, T, F) or (B, F, T)
    return x

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def is_probabilities(arr):
    # Heuristic: ONNX export might already include sigmoid
    return np.all(arr >= 0.0) and np.all(arr <= 1.0)

from pathlib import Path
import csv

LABEL_CSV = Path(r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\labels.csv")

def load_labels_from_csv(path=LABEL_CSV):
    path = Path(path).expanduser().resolve()
    id2label = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect columns: index, mid, display_name
        rows = sorted(reader, key=lambda r: int(r["index"]))
        for r in rows:
            id2label.append(r["display_name"])
    assert len(id2label) in (527, 521), f"Unexpected labels: {len(id2label)}"
    return id2label


def verify_onnx(onnx_path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print("\n[ONNX I/O]")
    for i, it in enumerate(ins):
        print(f"  input[{i}]: name={it.name} shape={it.shape} dtype={it.type}")
    for i, ot in enumerate(outs):
        print(f"  output[{i}]: name={ot.name} shape={ot.shape} dtype={ot.type}")
    return sess, ins[0].name, ins[0].shape

def arrange_input(x, expected_shape):
    """
    x comes from HF extractor as (B, T, F) or (B, F, T).
    Most AST ONNX exports expect (B, 1, F, T) or (B, 1, T, F).
    """
    # Add channel dim
    if x.ndim == 3:
        pass

    # Force layout if requested
    if FORCE_LAYOUT == "B1TF":
        if x.shape[2] == 128:  # currently (B,1,F,T); swap
            x = np.transpose(x, (0, 1, 3, 2))
    elif FORCE_LAYOUT == "B1FT":
        if x.shape[3] == 128:  # currently (B,1,T,F); swap
            x = np.transpose(x, (0, 1, 3, 2))

    # Try to match expected last two dims if known
    def numeric(v):
        try: return int(v)
        except: return -1
    exp = list(map(numeric, expected_shape))
    if len(exp) == 4 and x.ndim == 4 and exp[-2] != -1 and exp[-1] != -1:
        if (x.shape[-2], x.shape[-1]) != (exp[-2], exp[-1]) and (x.shape[-1], x.shape[-2]) == (exp[-2], exp[-1]):
            x = np.transpose(x, (0, 1, 3, 2))
    return np.ascontiguousarray(x, dtype=np.float32)

def env_labels(audio_path=AUDIO_PATH, onnx_path=ONNX_PATH, k=10):
    audio_path = exists(audio_path)
    onnx_path  = exists(onnx_path)

    print(f"Using audio: {audio_path}")
    print(f"Using model: {onnx_path}")

    # 1) Load audio
    wav, sr = load_audio(audio_path, target_sr=16000)
    print(f"[Audio] sr={sr}, samples={len(wav)}, duration={len(wav)/sr:.2f}s")

    # 2) Features
    fe = make_feature_extractor()
    x = extract_inputs(wav, sr, fe)  # (B, T, F) or (B, F, T)
    print(f"[Extractor] input_values shape: {x.shape}")

    # 3) ONNX session and arrange input
    sess, in_name, exp_shape = verify_onnx(onnx_path)
    x4 = arrange_input(x, exp_shape)
    print(f"[Arrange] feeding to ONNX: {x4.shape}  (expected={exp_shape})")

    # 4) Inference
    outs = sess.run(None, {in_name: x4})
    logits = outs[0]
    print(f"[ONNX] raw output shape: {logits.shape}, min={np.min(logits):.4f}, max={np.max(logits):.4f}")

    # 5) Post-process (probabilities vs logits)
    if logits.ndim == 1:
        logits = logits[None, :]  # ensure batch dimension
    if is_probabilities(logits):
        probs = logits[0]
        print("[Post] Detected probabilities in ONNX output (no activation applied).")
    else:
        probs = sigmoid(logits)[0]
        print("[Post] Applied sigmoid to ONNX logits.")

    # 6) Top-K (no threshold so we always print)
    idx_sorted = np.argsort(-probs)
    id2label = load_labels_from_csv()
    print("\nTop predictions:")
    for i in range(min(k, probs.shape[-1])):
        idx = int(idx_sorted[i])
        name = id2label[idx] if (id2label and idx < len(id2label)) else f"class_{idx}"
        print(f"  {i+1:2d}) {name:<40} prob={float(probs[idx]):.4f}")

    return probs
# ------------------------------ ASR (Parakeet-TDT via onnx-asr) ------------------------------ #

def asr_transcribe(asr_sess: Any, audio_mono_16k: np.ndarray) -> str:
    """
    Use onnx-asr Parakeet-TDT if available. Returns "" if not available.
    - Expects mono float32 @16kHz in [-1,1].
    - No file I/O; sends NumPy array to model.recognize(...).
    """
    if asr_sess is None or audio_mono_16k is None or audio_mono_16k.size == 0:
        return ""
    # onnx-asr accepts NumPy arrays directly: model.recognize(waveform, sample_rate=16000)
    # https://github.com/istupakov/onnx-asr (supports Parakeet TDT v2/v3)
    try:
        if hasattr(asr_sess, "recognize"):
            wav = audio_mono_16k.astype(np.float32).reshape(-1)
            text = asr_sess.recognize(wav, sample_rate=16000)
            if isinstance(text, list):
                text = text[0] if text else ""
            return str(text) if text is not None else ""
    except Exception:
        pass
    return ""


# ------------------------------ DECISION ------------------------------ #

def threat_decision(face_conf: float,
                    motion: float,
                    vad: float,
                    env_top: List[Tuple[str, float]],
                    thresholds: dict) -> Tuple[bool, List[str]]:
    face_thr = float(thresholds.get("face_conf", 0.3))
    mot_thr = float(thresholds.get("motion", 0.25))
    vad_thr = float(thresholds.get("vad", 0.5))
    env_thr = float(thresholds.get("env_conf", 0.5))
    watch = set(str(x).lower() for x in thresholds.get("env_watchlist", []))

    triggers: List[str] = []
    if face_conf >= face_thr:
        triggers.append("face")
    if motion >= mot_thr:
        triggers.append("motion")
    if vad >= vad_thr:
        triggers.append("speech")
    env_hit = False
    for label, conf in env_top or []:
        if conf >= env_thr:
            triggers.append(f"env:{label}")
            env_hit = True
        if label.lower() in watch:
            if f"env:{label}" not in triggers:
                triggers.append(f"env:{label}")
            env_hit = True
    # Dedup keep order
    seen = set()
    triggers = [t for t in triggers if (t not in seen and not seen.add(t))]
    is_threat = bool(triggers)
    return is_threat, triggers


# ------------------------------ SINGLE STEP ------------------------------ #

def step(frame: np.ndarray,
         prev_frame: np.ndarray | None,
         audio_chunk: np.ndarray,
         t: float,
         state: dict,
         sessions: dict,
         thresholds: dict,
         enable_asr: bool = True) -> dict:
    state = dict(state or {})
    # Face
    face_in = preprocess_video(frame)
    face_conf, _boxes = detect_faces(sessions.get("face"), face_in)
    # Motion + light EMA
    mot_raw = motion_score(prev_frame, frame)
    ema_prev = state.get("ema_motion", None)
    ema_mot = mot_raw if ema_prev is None else (0.8 * float(ema_prev) + 0.2 * float(mot_raw))
    motion_val = float(np.clip(ema_mot, 0.0, 1.0))
    # Audio
    audio = preprocess_audio(audio_chunk, 16000)
    vad_val = float(vad_score(sessions.get("vad"), audio)) if audio.size > 0 else 0.0
    env_top = env_labels(sessions.get("env"), audio, top_k=3) if audio.size > 0 else []
    # Optional ASR if speech likely
    transcript = None
    if enable_asr and vad_val >= float(thresholds.get("vad", 0.5)):
        txt = asr_transcribe(sessions.get("asr"), audio)
        transcript = txt if txt else None
    # Decision
    is_threat, triggers = threat_decision(face_conf, motion_val, vad_val, env_top, thresholds)
    # Update minimal state
    state["prev_frame"] = frame.copy()
    state["ema_motion"] = motion_val
    state["last_env_labels"] = env_top if env_top else None
    return {
        "is_threat": bool(is_threat),
        "triggers": triggers,
        "scores": {
            "face_conf": float(face_conf) if face_conf is not None else None,
            "motion": motion_val,
            "vad": vad_val,
            "env_top": env_top if env_top else None,
        },
        "transcript": transcript,
        "t": t,
        "state": state,
    }
