# edge_ai_live_labels.py
# Python 3.11+
# Minimal live video+audio threat detector using ONNX models + labels.csv
# Deps: onnxruntime, opencv-python, numpy, sounddevice
# Opt : webrtcvad (better VAD), transformers (ASTFeatureExtractor)

import os, time, csv
import numpy as np
import cv2

# ---- optional graceful imports
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    from transformers import ASTFeatureExtractor
except Exception:
    ASTFeatureExtractor = None

# ========================================
# CONFIG (edit or set via env vars)
# ========================================
FACE_ONNX = os.getenv("EDGEAI_FACE_ONNX", r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\version-RFB-640.onnx")
ENV_ONNX  = os.getenv("EDGEAI_ENV_ONNX",  r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\ast_audioset.onnx")
ENV_LABELS_FILE = os.getenv("EDGEAI_ENV_LABELS", r"C:\Users\QCWorkshop22\Desktop\Qualcomm-Hackathon-2025\data\models\labels.csv")

CAM_INDEX = int(os.getenv("EDGEAI_CAM_INDEX", "0"))

# thresholds (tune as needed)
THRESHOLDS = {
    "face_conf": 0.30,   # face presence threshold
    "motion":    0.25,   # normalized frame diff
    "vad":       0.50,   # speech fraction (0..1)
    "env_conf":  0.50,   # AST top prob threshold
    "env_watchlist": ["gunshot", "siren", "glass"],  # any hit triggers
}

# audio / loop timing
SR = 16000
CHUNK = 1600            # 0.1s blocks at 16kHz
AST_BUFFER_SECONDS = 10 # rolling window for AST
ENERGY_GATE = 0.017     # fallback VAD (~ -35 dBFS)

# ========================================
# ONNX session loader
# ========================================
def load_onnx_session(path: str):
    if not ort or not path or not os.path.exists(path):
        if not ort:
            print("[warn] onnxruntime not available")
        elif not os.path.exists(path):
            print(f"[warn] model not found: {path}")
        return None
    providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=providers)
        return sess
    except Exception as e:
        print(f"[warn] failed to load ONNX: {path} ({e})")
        return None

# ========================================
# VIDEO: preprocess + face detect (UltraFace)
# ========================================
def preprocess_face_input(frame_bgr: np.ndarray, sess) -> np.ndarray:
    Ht, Wt = 480, 640
    if sess is not None:
        try:
            s = sess.get_inputs()[0].shape  # expect 4D
            if len(s) == 4:
                def num(v):
                    try: return int(v)
                    except: return -1
                a,b,c,d = map(num, s)
                if d == 3 and a == 1:   # NHWC
                    Ht, Wt = (b if b>0 else 480), (c if c>0 else 640)
                elif b == 3 and a == 1: # NCHW
                    Ht, Wt = (c if c>0 else 480), (d if d>0 else 640)
        except Exception:
            pass
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    img = (rgb.astype(np.float32) - 127.0) / 128.0
    x = np.transpose(img, (2,0,1))[None, ...]  # (1,3,H,W)
    return x.astype(np.float32)

def hard_nms(box_probs: np.ndarray, iou_thr: float=0.3) -> np.ndarray:
    # box_probs: [N,5] = [x1,y1,x2,y2,score] (normalized)
    if box_probs.size == 0:
        return box_probs
    order = np.argsort(box_probs[:,4])
    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        order = order[:-1]
        if order.size == 0: break
        xx1 = np.maximum(box_probs[i,0], box_probs[order,0])
        yy1 = np.maximum(box_probs[i,1], box_probs[order,1])
        xx2 = np.minimum(box_probs[i,2], box_probs[order,2])
        yy2 = np.minimum(box_probs[i,3], box_probs[order,3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (box_probs[i,2]-box_probs[i,0])*(box_probs[i,3]-box_probs[i,1])
        area_r = (box_probs[order,2]-box_probs[order,0])*(box_probs[order,3]-box_probs[order,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = order[iou <= iou_thr]
    return box_probs[keep]

def detect_faces(sess, x: np.ndarray, thr: float=0.70):
    if sess is None: return 0.0, []
    try:
        in0 = sess.get_inputs()[0]
        xin = x
        s = in0.shape
        if len(s)==4 and s[-1]==3 and x.shape[1]==3:
            xin = np.transpose(x, (0,2,3,1))  # NCHW->NHWC
        outs = sess.run(None, {in0.name: xin})

        confidences = boxes = None
        for arr in outs:
            a = arr[0] if (arr.ndim==3 and arr.shape[0]==1) else arr
            if a.ndim==2 and a.shape[-1]==2: confidences = a
            if a.ndim==2 and a.shape[-1]==4: boxes = a
        if confidences is None or boxes is None or boxes.size==0:
            return 0.0, []

        face_probs = confidences[:,1]
        if not np.any(face_probs > thr):
            return float(face_probs.max(initial=0.0)), []
        sel = face_probs > thr
        box_probs = np.concatenate([boxes[sel], face_probs[sel,None]], axis=1)
        kept = hard_nms(box_probs, iou_thr=0.3)
        if kept.size==0: return float(face_probs.max(initial=0.0)), []
        return float(np.max(kept[:,4])), kept.tolist()
    except Exception:
        return 0.0, []

def motion_score(prev_bgr, cur_bgr) -> float:
    if prev_bgr is None: return 0.0
    if prev_bgr.shape[:2] != cur_bgr.shape[:2]:
        prev_bgr = cv2.resize(prev_bgr, (cur_bgr.shape[1], cur_bgr.shape[0]))
    g1 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(g1,g2), (5,5), 0)
    return float(np.clip(diff.mean()/255.0, 0.0, 1.0))

# ========================================
# AUDIO: VAD + AST env classification
# ========================================
def vad_fraction(audio_mono_16k: np.ndarray) -> float:
    if audio_mono_16k is None or audio_mono_16k.size==0: return 0.0
    frame = int(SR*0.03)
    n = audio_mono_16k.size // frame
    if n==0: return 0.0
    x = audio_mono_16k[:n*frame].astype(np.float32)
    frames = x.reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    if webrtcvad is None:
        voiced = float((rms >= ENERGY_GATE).sum())
        return voiced / float(n)

    v = webrtcvad.Vad(2)
    pcm16 = (np.clip(x, -1, 1)*32767.0).astype(np.int16).tobytes()
    step = frame*2
    voiced, valid = 0, 0
    for i in range(n):
        if rms[i] < ENERGY_GATE: continue
        valid += 1
        fb = pcm16[i*step:(i+1)*step]
        try:
            if v.is_speech(fb, SR): voiced += 1
        except Exception:
            pass
    return 0.0 if valid==0 else voiced/float(valid)

def load_env_labels(path: str, expected_classes: int | None = None):
    """
    Load label names from labels.csv or plain text.

    Supported formats:
      A) One label per line (optional first line 'label')
      B) Single CSV row with comma-separated labels
      C) CSV with two columns (index,label) with/without header
    """
    if not path or not os.path.exists(path):
        return None

    def is_int(s: str) -> bool:
        try:
            int(s); return True
        except:
            return False

    labels = None
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [[c.strip() for c in row] for row in reader if any(c.strip() for c in row)]

            if not rows:
                return None

            header_like = [c.lower() for c in rows[0]]

            # Case C: (index,label)
            has_header_idx = ("index" in header_like[:1] or "id" in header_like[:1] or "idx" in header_like[:1])
            first_data_row = rows[1] if len(rows) > 1 else []
            looks_indexed = (len(first_data_row) >= 1 and is_int(first_data_row[0]))

            if len(rows[0]) >= 2 and (has_header_idx or looks_indexed):
                start = 1 if has_header_idx else 0
                mapping = {}
                for r in rows[start:]:
                    if len(r) < 2 or not is_int(r[0]): 
                        continue
                    mapping[int(r[0])] = r[1]
                if mapping:
                    size = max(mapping.keys()) + 1
                    if expected_classes is not None:
                        size = max(size, int(expected_classes))
                    labels = [f"class_{i}" for i in range(size)]
                    for i, name in mapping.items():
                        if 0 <= i < size:
                            labels[i] = name

            # Case A: one label per line
            if labels is None and all(len(r) == 1 for r in rows):
                start = 1 if rows and rows[0] and rows[0][0].lower() == "label" else 0
                labels = [r[0] for r in rows[start:] if r and r[0]]

            # Case B: a single CSV row with many labels
            if labels is None and len(rows) == 1 and len(rows[0]) > 1:
                labels = [c for c in rows[0] if c]

        # Fallback: plain text (comma or newline separated)
        if labels is None:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if "," in txt:
                labels = [s.strip() for s in txt.split(",") if s.strip()]
            else:
                labels = [line.strip() for line in txt.splitlines() if line.strip()]

        return labels or None

    except Exception as e:
        print(f"[warn] failed to parse labels from {path}: {e}")
        return None

def env_topk(env_sess, fe, rolling_buf: np.ndarray, k: int=3, label_names=None):
    if env_sess is None or fe is None or rolling_buf.size==0:
        return []
    try:
        iv = fe(rolling_buf, sampling_rate=SR, padding="max_length", return_tensors="np")["input_values"].astype(np.float32)
        in0 = env_sess.get_inputs()[0].name
        exp = env_sess.get_inputs()[0].shape

        # arrange to (B,1,T,F) or (B,T,F) with F=128
        x = iv
        if len(exp)==4:
            if x.ndim==3: x = x[:,None,:,:]
            if exp[-2]>0 and exp[-1]>0:
                if (x.shape[-2],x.shape[-1]) != (exp[-2],exp[-1]) and (x.shape[-1],x.shape[-2])==(exp[-2],exp[-1]):
                    x = np.transpose(x,(0,1,3,2))
        elif len(exp)==3:
            if x.ndim==4: x = x[:,0,:,:]
            if x.shape[-1]!=128 and x.shape[-2]==128:
                x = np.transpose(x,(0,2,1))
        x = np.ascontiguousarray(x, dtype=np.float32)

        outs = env_sess.run(None, {in0: x})
        logits = outs[0]
        if logits.ndim==1: logits = logits[None,:]

        # treat as logits unless already in [0,1]
        if np.all((logits >= 0.0) & (logits <= 1.0)):
            probs = logits[0]
        else:
            probs = 1.0/(1.0+np.exp(-logits[0]))

        order = np.argsort(-probs)
        k = min(k, probs.shape[-1])
        out = []
        for i in range(k):
            idx = int(order[i])
            name = label_names[idx] if (label_names and idx<len(label_names)) else f"class_{idx}"
            out.append((name, float(probs[idx])))
        return out
    except Exception:
        return []

# ========================================
# DECISION
# ========================================
def decide(face_conf, motion_val, vad_val, env_top, thresholds):
    face_thr = float(thresholds.get("face_conf", 0.3))
    mot_thr  = float(thresholds.get("motion",    0.25))
    vad_thr  = float(thresholds.get("vad",       0.5))
    env_thr  = float(thresholds.get("env_conf",  0.5))
    watch    = set(str(x).lower() for x in thresholds.get("env_watchlist", []))

    triggers = []
    if face_conf >= face_thr: triggers.append("face")
    if motion_val >= mot_thr: triggers.append("motion")
    if vad_val >= vad_thr:    triggers.append("speech")
    for label, conf in env_top or []:
        if conf >= env_thr or label.lower() in watch:
            tag = f"env:{label}"
            if tag not in triggers: triggers.append(tag)

    return (len(triggers)>0), triggers

# ========================================
# MAIN
# ========================================
def main():
    # load models
    face_sess = load_onnx_session(FACE_ONNX)
    env_sess  = load_onnx_session(ENV_ONNX)

    # infer expected class count from env model (for index->name CSVs)
    expected_classes = None
    if env_sess is not None:
        try:
            oshape = env_sess.get_outputs()[0].shape
            if isinstance(oshape[-1], int) and oshape[-1] > 0:
                expected_classes = int(oshape[-1])
        except Exception:
            pass

    # load labels.csv first; fallback to metadata
    env_labels = load_env_labels(ENV_LABELS_FILE, expected_classes=expected_classes)
    if env_labels is None and env_sess is not None:
        try:
            meta = env_sess.get_modelmeta()
            if meta and meta.custom_metadata_map:
                csv_str = meta.custom_metadata_map.get("labels","")
                if csv_str:
                    env_labels = [s.strip() for s in csv_str.split(",") if s.strip()]
        except Exception:
            pass
    if env_labels is not None:
        print(f"[info] loaded {len(env_labels)} labels from '{ENV_LABELS_FILE}'")
    elif ENV_LABELS_FILE:
        print(f"[warn] could not load labels from '{ENV_LABELS_FILE}', using class_<idx>")

    # AST feature extractor (only if env model present)
    fe = ASTFeatureExtractor() if (env_sess is not None and ASTFeatureExtractor is not None) else None
    if env_sess is not None and fe is None:
        print("[warn] transformers.ASTFeatureExtractor not available; disabling env audio classifier")

    # audio stream
    if sd is None:
        print("[warn] sounddevice not installed; audio disabled.")
    else:
        print("[info] opening microphone @16kHz mono")
    stream = None
    if sd is not None:
        try:
            stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32")
            stream.start()
        except Exception as e:
            print(f"[warn] could not open microphone: {e}")
            stream = None

    # rolling AST buffer
    ast_cap = SR * AST_BUFFER_SECONDS
    ast_buf = np.zeros(ast_cap, dtype=np.float32)
    ast_len = 0

    # webcam
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[error] cannot open webcam.")
        return

    prev_frame = None
    ema_motion = None
    last_log_t = 0.0

    print("[ready] press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # audio chunk
        if stream is not None:
            try:
                chunk, _ = stream.read(CHUNK)  # (frames,1)
                audio = chunk.reshape(-1).copy()
            except Exception:
                audio = np.zeros((0,), dtype=np.float32)
        else:
            audio = np.zeros((0,), dtype=np.float32)

        # FACE
        face_in = preprocess_face_input(frame, face_sess)
        face_conf, _ = detect_faces(face_sess, face_in, thr=0.70)

        # MOTION (EMA)
        mot_raw = motion_score(prev_frame, frame)
        ema_motion = mot_raw if ema_motion is None else (0.8*ema_motion + 0.2*mot_raw)
        motion_val = float(np.clip(ema_motion, 0.0, 1.0))

        # VAD
        vad_val = vad_fraction(audio) if audio.size>0 else 0.0

        # AST env (rolling)
        env_top = []
        if fe is not None and env_sess is not None and audio.size>0:
            need = min(audio.size, ast_cap)
            if ast_len + need <= ast_cap:
                ast_buf[ast_len:ast_len+need] = audio[-need:]
                ast_len += need
            else:
                shift = (ast_len + need) - ast_cap
                if shift > 0:
                    ast_buf[:-need] = ast_buf[shift:-need]
                ast_buf[-need:] = audio[-need:]
                ast_len = ast_cap
            buf_view = ast_buf[:max(ast_len, 1)]
            env_top = env_topk(env_sess, fe, buf_view, k=3, label_names=env_labels)

        # DECISION
        is_threat, triggers = decide(face_conf, motion_val, vad_val, env_top, THRESHOLDS)

        # HUD
        hud = [
            f"face={face_conf:.2f}  motion={motion_val:.2f}  vad={vad_val:.2f}",
            ("env: " + ", ".join([f"{n}:{p:.2f}" for n,p in (env_top or [])])) if env_top else "env: n/a",
            ("THREAT " + "|".join(triggers)) if is_threat else "OK",
        ]
        y = 22
        for line in hud:
            cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,255,0) if not is_threat else (0,0,255), 2, cv2.LINE_AA)
            y += 26
        cv2.imshow("Edge-AI Threat (labels.csv)", frame)

        # periodic console log
        now = time.time()
        if now - last_log_t > 1.0:
            last_log_t = now
            if is_threat:
                print(f"[THREAT] triggers={triggers} face={face_conf:.2f} motion={motion_val:.2f} vad={vad_val:.2f} env={env_top}")
            else:
                print(f"[ok] face={face_conf:.2f} motion={motion_val:.2f} vad={vad_val:.2f}")

        prev_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    if stream is not None:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
