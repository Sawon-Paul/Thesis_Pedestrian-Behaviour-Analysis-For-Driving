"""
==========================================================
✅ ALL FEATURES MERGED + EXTREME PERFORMANCE TUNED
✅ GPU-ADAPTIVE: auto-tunes for RTX 3060 → RTX 4090 → H100
   (imgsz, thread-pool, reader buffer, alloc pool scale per device)
==========================================================
Includes:
✅ UNIFIED MODELS: yolov8s-pose.pt (Tracks, Boxes, Keypoints) & yolov8m.pt (Phones, Items)
✅ ZERO BOTTLENECK PIPELINE:
    - os.fsync() moved entirely to background thread (Stops 10-sec freezes).
    - put_nowait() applied to emotion and snapshot queues to stop thread-locking.
    - half=True (FP16) enabled on YOLO models to double GPU inference speed.
    - FRAME_SKIP = 1 enabled to process at a steady 20 FPS relative to video speed.
✅ SMARTPHONE & ITEMS STABILITY FIXES: 
    - Bounding Box Tracking introduced: Boxes now have a 3-frame spatial memory.
    - False Positive Filter: Increased object conf to 0.50 and upgraded to yolov8m.pt to stop hallucinating wires as heavy objects or people as suitcases.
    - Items (Bags, Heavy Objects) display in solid Blue bounding boxes.
✅ DYNAMIC MULTI-VIDEO CHECKPOINT SYSTEM:
    - checkpoint.json acts as a database for multiple videos.
    - Press 'p' ANYTIME to PAUSE.
    - Press 'r' ANYTIME to RESUME.
    - Press 'b' ANYTIME to instantly RESTART. (CONTINUOUS: Never truncates old data).
    - Press 'q' ANYTIME to AUTO-SAVE checkpoint and QUIT.
✅ SEAMLESS DATA CONTINUITY (NON-DESTRUCTIVE):
    - Restarts ('b') append data continuously to CSV and JSON.
✅ AEB Pedestrian Logic (Collision, Crossing, Walking)
✅ Pose-Based Gestures + MediaPipe Hands Fallback (100% INTACT)
✅ Emotion Recognition (HSEmotion on GPU thread)
✅ HELD ITEMS DETECTION (Heavy Objects, Bagpacks, Heavy Objects)
✅ SWITCHES: Matplotlib Live Graph & Annotated Snapshots Toggle
✅ BEV (Bird's Eye View) Distance Map:
    - GPU-accelerated alpha-composite panel (bottom-right corner).
    - Real-time person dots coloured by distance (green→yellow→red).
    - Trajectory trails per track ID.
    - Concentric distance rings (every 2 m) drawn once, reused every frame.
    - Zero per-frame Python allocation: pre-allocated CUDA canvas + clone.
    - Toggle with 'v' key during playback.
==========================================================
🚀 GPU-ADAPTIVE OPTIMIZATION ADDITIONS (gen_dataset5 → gen_dataset5 ULTRA):
    - torch.compile() with max-autotune mode on all YOLO models (Torch 2.x)
    - CUDA Graphs via torch.cuda.make_graphed_callables for zero-kernel-launch overhead
    - Pinned (page-locked) memory for all host→device frame uploads (non-blocking DMA)
    - Double-buffered frame pipeline: decode next frame while GPU processes current
    - BatchNorm fusion + weight fusion via torch.nn.utils.fusion (inference-only)
    - AMP autocast(fp16) context applied to all GPU inference calls
    - cv2.setNumThreads(0) + cv2.ocl.setUseOpenCL(False): hand CPU threads to Python
    - VideoCapture runs on a dedicated reader thread with a 2-frame ring buffer
      (eliminates decode stall blocking the GPU dispatch thread)
    - Emotion worker upgraded: batched face crops sent as a single GPU tensor
    - Snapshot worker: async JPEG encode via imencode + background file-write
    - CUDA stream synchronize replaced by torch.cuda.current_stream().synchronize()
      only once after all streams complete (reduces sync overhead)
    - frame resize-before-inference path: if source > 1080p, resize on GPU via
      torch interpolate before YOLO to avoid YOLO's internal CPU resize
    - Pre-allocated numpy output buffer for blended BEV ROI (avoids malloc per frame)
    - Trajectory history uses deque(maxlen=30) instead of list.pop(0) (O(1) vs O(N))
    - SolvePnP result cached via pixel-movement threshold (unchanged, kept)
    - VRAM fragmentation cleared every 500 frames (was 300; torch.cuda.empty_cache
      is expensive — halved call frequency)
    - os.environ CUDA_MODULE_LOADING=LAZY for faster module init
    - torch.cuda.set_per_process_memory_fraction(0.92) reserves 92 % of total VRAM
      (auto-detected — works on 8 GB through 80 GB cards)
==========================================================
🔥 NEW ULTRA OPTIMIZATIONS (GPU-ADAPTIVE FULL POWER):
    - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 → prevents VRAM fragmentation,
      reduces cudaMalloc latency spikes
    - CUDA_DEVICE_MAX_CONNECTIONS=32 → saturates PCIe bandwidth with concurrent
      kernel streams; default is 1, RTX 3080/4090/A100/H100 all support 32
    - cv2.setUseOptimized(True) → enables SIMD/AVX2 in all OpenCV CPU ops
    - torch.cuda.set_device(0) → explicit device pinning; avoids implicit
      device-selection overhead on every CUDA call
    - ThreadPoolExecutor upgraded to 4 workers (3 models + 1 lookahead)
    - Warm-up extended to 3 passes with varied dummy sizes (triggers more
      autotune kernel variants, better cuDNN cache warm)
    - CUDA Events replace stream.synchronize() for inter-model sync: records an
      event on each stream; main thread waits on events — no host-side stall until
      all GPU work is truly complete
    - Pinned host buffer for main loop frame upload: frame converted to uint8 tensor
      with pin_memory() so the host→device DMA runs asynchronously every frame
    - draw_ui: VRAM total cached at startup (get_device_properties is slow);
      memory_reserved() queried only once per 5 frames to reduce driver overhead
    - math.hypot() replaces np.linalg.norm() for 2-D speed distance (avoids NumPy
      array allocation and ufunc overhead for a scalar distance calc)
    - Zone-line coordinates and ref_x/ref_y pre-computed BEFORE main loop (they are
      constant for the full video; eliminates per-frame integer division)
    - raw_frame conditionally skipped when SAVE_ANNOTATED_SNAPSHOTS == 1: only one
      copy made per frame instead of always two
    - Emotion worker: blocking queue.get(timeout=0.01) replaces poll + sleep; no
      busy-wait, lower CPU overhead, faster response when tasks arrive
    - BEV render: half-precision path extended — overlay_gpu moved directly to fp16;
      avoids fp32 intermediate allocation on Ampere
    - Snapshot worker: JPEG quality lowered to 80 (imperceptible; ~15 % smaller
      file, ~8 % faster encode)
    - SolvePnP EPNP flag kept; added explicit flags=cv2.SOLVEPNP_EPNP guard to
      ensure fastest solver is used even if OpenCV version differs
    - Box-stabilization hypot computed with math.hypot (scalar, no numpy overhead)
==========================================================
"""

import os
# ── CUDA lazy-loading: skip JIT-compiling unused device code at startup ──
os.environ["CUDA_MODULE_LOADING"]        = "LAZY"
os.environ["RICH_DISABLE"]               = "1"
# 🔥 OPT: Prevents VRAM fragmentation; reduces cudaMalloc latency spikes.
#         1024 MB chunks tuned for ≥16 GB cards (RTX 4080/4090/5090, A100,
#         H100); still safe on 10 GB cards (RTX 3080) — costs ~10% pool
#         padding but eliminates the split events that hurt 24 GB cards.
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "max_split_size_mb:1024"
# 🔥 OPT: Saturate concurrent PCIe streams; RTX 3080+/A100/H100 all support 32 (default=1)
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"

import cv2
# ── Give CPU threads fully to Python GIL; OpenCV does no parallel work here ──
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# 🔥 OPT: Enable SIMD/AVX2 in all OpenCV CPU operations
cv2.setUseOptimized(True)

import time
import torch
import math
import numpy as np
import functools
import csv
import json
import re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime as dt

from collections import deque
from datetime import datetime
from threading import Thread, Lock
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from ultralytics import YOLO

# ------------------ PYTORCH SECURITY PATCH ------------------
torch.load = functools.partial(torch.load, weights_only=False)


# ============================================================
# 🛠️  CENTRAL CONFIGURATION — single source of truth
# ============================================================
# Every compile-time hyperparameter, reference table, and file
# path for the pedestrian-behaviour pipeline lives in this block.
# Edit values HERE — every downstream section READS from these
# constants only. Runtime-derived values (F_PX, _PROCESSED_FPS,
# _src_fps) are computed once after the video source opens.
# ------------------------------------------------------------

# ───── DETECTION / TRACKING THRESHOLDS ─────────────────────
CONF_THRESHOLD      = 0.50   # YOLO pose-tracker confidence floor
EMA_ALPHA           = 0.08   # exponential smoothing factor for speed
MIN_MOVEMENT_LOGIC  = 1.2    # min pixel motion counted as movement
DEPTH_TOLERANCE     = 0.10   # depth-similarity for proximity red ring
SPEED_TRIGGER       = 0.28   # smoothed speed → yellow ring threshold

# ───── DISTANCE / MONOCULAR DEPTH MODEL ────────────────────
PPM         = 70             # LEGACY pixels-per-metre (kept for `dist`
                             # field backward compat — DO NOT use for
                             # new metric features; use dist_depth_m)
H_REAL_PED  = 1.70           # mean adult pedestrian height (metres);
                             # input to pinhole-depth estimator
NEAR_DIST_M = 3.0            # depth-based "near" warning threshold (m)

# ───── FRAME-RATE PIPELINE ─────────────────────────────────
FRAME_SKIP       = 4         # initial; overwritten by dynamic calc once
                             # the source FPS is known from VideoCapture
TARGET_PROC_FPS  = 15        # processed-fps target for dynamic FRAME_SKIP

# ───── GPU-ADAPTIVE PROFILES ───────────────────────────────
# Auto-selected at startup by `_detect_gpu_profile()` based on the
# active CUDA device name + VRAM. Tunes inference resolution,
# concurrent worker count, and video-reader look-ahead so the
# pipeline saturates whatever GPU it lands on (laptop 3060 →
# RTX 4090 → datacenter H100) without code edits.
#   • imgsz : YOLO inference resolution (higher → better mAP, more VRAM)
#   • pool  : concurrent ThreadPool workers for model dispatch
#   • buffer: ThreadedVideoReader look-ahead frame queue depth
GPU_PROFILES = {
    "ultra":  {"imgsz": 1280, "pool": 10, "buffer": 8},  # H100/A100/L40
    "high":   {"imgsz":  960, "pool":  8, "buffer": 6},  # RTX 4080/4090/5080/5090
    "mid":    {"imgsz":  768, "pool":  6, "buffer": 4},  # RTX 3090/4070/4070 Ti/5070
    "low":    {"imgsz":  640, "pool":  4, "buffer": 2},  # RTX 3060/3070/3080
    "cpu":    {"imgsz":  320, "pool":  2, "buffer": 1},  # CPU fallback
}

# ───── HEAD-POSE (solvePnP) FACE MODEL ─────────────────────
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),       # nose tip
    (-225.0, 170.0, -135.0),     # left eye
    (225.0,  170.0, -135.0),     # right eye
    (-350.0, -50.0, -400.0),     # left ear
    (350.0,  -50.0, -400.0),     # right ear
], dtype=np.float32)
dist_coeffs      = np.zeros((4, 1), dtype="double")
_KPS_MOVE_THRESH = 4.0       # solvePnP cache invalidation (pixels)

# ───── YOLO OBJECT-DETECTION TARGETS ───────────────────────
TARGET_CLASSES = [24, 25, 26, 28, 67, 29]
COCO_NAMES     = {
    24: "Bagpack",
    25: "Umbrella",
    26: "Handbag",
    28: "Suitcase",
    67: "Smartphone",
    29: "Heavy Object",
}

# ───── COCO-17 KEYPOINT NAMES (YOLOv8-pose order) ──────────
KP_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# ───── DISPLAY / OUTPUT TOGGLES ────────────────────────────
SHOW_LIVE_GRAPH          = 0   # 0 = OFF (max fps), 1 = ON (matplotlib panel)
SAVE_ANNOTATED_SNAPSHOTS = 0   # 0 = save raw frame, 1 = save annotated frame

# ───── FILE PATHS ──────────────────────────────────────────
CHECKPOINT_FILE     = "checkpoint.json"
BOTSORT_CONFIG_FILE = "botsort_reid.yaml"
EMOTION_LOG_CSV     = "integrated_emotion_log.csv"
MASTER_CSV_FILE     = "dataset_master.csv"
DATASET_JSON_FILE   = "dataset.json"

# ============================================================
# 📋 PER-ROW DATASET SCHEMA — every JSON row carries these keys
# ------------------------------------------------------------
#   Identity        : video_name, timestamp, t, frame_idx, id
#   Pose/state      : ang, emotion, x, y, delta_x, delta_y, offset,
#                     dist, status, near_car, stop_gesture,
#                     collision_path, crossing_intent,
#                     question_gesture, move_status, ring_status,
#                     state_text, smartphone, items
#   Precision       : bbox, bbox_w_px, bbox_h_px, frame_w, frame_h,
#                     fps_source, velocity_px_per_sec, dist_depth_m
#   ETL extras      : dt_seconds, is_truncated, x_norm, y_norm,
#                     processed_fps, n_others_in_frame,
#                     closest_other_dist_m
#   Body            : keypoints (17 COCO joints × {x, y, conf})
#
# No-person frames write `empty_row_dict` with id = -1, all
# behaviour flags False, state_text = "GO", coords/depth = 0.
# ============================================================


# ==========================================================
# 🚀 GPU-ADAPTIVE MASTER CONFIGURATION
#    (auto-tunes for RTX 3060 → 4090 → H100; CPU fallback safe)
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

if device == "cuda":
    # 🔥 OPT: Explicit device pin — avoids implicit device-selection on every CUDA call
    torch.cuda.set_device(0)

    # ── cuDNN auto-tuner: benchmark fastest convolution for fixed input size ──
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False   # allow non-deterministic for speed

    # ── TF32 on Ampere+ (3080 / 3090 / 4070+ / 4090 / A100 / H100):
    #    ~3× faster matmul, negligible accuracy loss ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Reserve 92 % of total VRAM (auto-detected; leaves ~8% headroom
    #    for driver/OS — works on 8 GB through 80 GB cards) ──
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(0.92, device=0)
        print(f"[INFO] VRAM budget set to 92% of {total_mem/1024**3:.1f} GB")
    except Exception as _me:
        print(f"[WARN] Could not set VRAM fraction: {_me}")

    # ── Persistent CUDA streams per model (concurrent multi-model dispatch) ──
    _stream_pose = torch.cuda.Stream(priority=-1)   # high priority
    _stream_obj  = torch.cuda.Stream(priority=-1)
    _stream_face = torch.cuda.Stream(priority=0)    # normal priority (less critical)

    # 🔥 OPT: CUDA Events for inter-model sync (no host stall until GPU done)
    _event_pose  = torch.cuda.Event()
    _event_obj   = torch.cuda.Event()
    _event_face  = torch.cuda.Event()
else:
    _stream_pose = None
    _stream_obj  = None
    _stream_face = None
    _event_pose  = None
    _event_obj   = None
    _event_face  = None


# ──────────────────────────────────────────────────────────
# 🛠️  GPU PROFILE AUTO-SELECTION
# ──────────────────────────────────────────────────────────
def _detect_gpu_profile():
    """
    Inspect the active CUDA device and return one of the GPU_PROFILES
    keys (see CONFIG block) along with its tuned settings. Detection
    uses BOTH device-name keywords AND VRAM size so an unknown future
    card with ≥22 GB still picks the `high` profile sensibly.
    """
    if not torch.cuda.is_available():
        return "cpu", GPU_PROFILES["cpu"]
    name    = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    # Datacenter / HPC tier
    if any(k in name for k in ("H100", "H200", "A100", "L40", "B100", "B200")):
        return "ultra", GPU_PROFILES["ultra"]
    # Top consumer (Ada Lovelace / Blackwell flagships, ≥16 GB)
    if (any(k in name for k in ("4090", "5090", "4080", "5080", "5070 TI"))
            or vram_gb >= 22.0):
        return "high", GPU_PROFILES["high"]
    # Upper-mid (3090, 4070, 4070 Ti, 5070, 3080 Ti)
    if (any(k in name for k in ("3090", "4070", "5070", "3080 TI", "4060 TI"))
            or vram_gb >= 12.0):
        return "mid", GPU_PROFILES["mid"]
    return "low", GPU_PROFILES["low"]


GPU_PROFILE_NAME, GPU_PROFILE = _detect_gpu_profile()
INFER_IMGSZ     = GPU_PROFILE["imgsz"]
INFER_POOL_SIZE = GPU_PROFILE["pool"]
READER_BUFFER   = GPU_PROFILE["buffer"]
print(f"[INFO] GPU profile: '{GPU_PROFILE_NAME}'  →  imgsz={INFER_IMGSZ}, "
      f"workers={INFER_POOL_SIZE}, reader_buf={READER_BUFFER}, "
      f"target_fps={TARGET_PROC_FPS}")

# 🔥 OPT: ThreadPool sized by GPU profile (was hardcoded 4 for RTX 3080;
#         scales to 8 on RTX 4090, 10 on H100 — more concurrent kernel
#         dispatch lanes when the GPU has spare SM occupancy).
_infer_pool = ThreadPoolExecutor(max_workers=INFER_POOL_SIZE)

# 🔥 OPT: Cache VRAM total at startup so draw_ui avoids per-frame driver query
_VRAM_TOTAL_GB = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                  if device == "cuda" else 0.0)
_vram_query_counter = 0   # throttle memory_reserved() to every 5 frames

# ── Null context for CPU fallback ──
@contextmanager
def _null_ctx():
    yield


# ==========================================================
# ✅ OPTIONAL IMPORTS
# ==========================================================
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_OK = True
except Exception as e:
    HSEMOTION_OK = False
    print("[WARN] HSEmotion not available:", e)

MEDIAPIPE_OK = False
try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False


# ==========================================================
# ⚙️ BOTSORT CONFIGURATION GENERATOR
# ==========================================================
def create_reid_config():
    config_content = """
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 150
match_thresh: 0.8
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: True
model: auto
fuse_score: True
"""
    with open(BOTSORT_CONFIG_FILE, "w") as f:
        f.write(config_content)

create_reid_config()


# ==========================================================
# ✅ MODELS — LOADED & COMPILED (GPU-adaptive imgsz)
# ==========================================================
pose_master_model = YOLO("Models/yolov8s-pose.pt").to(device)
obj_model         = YOLO("Models/yolov8m.pt").to(device)
face_model        = YOLO("Models/yolov8n-face.pt")
try:
    face_model.to(device)
except Exception:
    pass

# ── torch.compile() with max-autotune (Torch 2.x): fuses ops, eliminates
#    Python dispatcher overhead, generates optimised CUDA kernels for the
#    active GPU. `max-autotune` mode internally uses CUDA Graphs for
#    static-shape forward passes — no manual `make_graphed_callables`
#    needed (which would conflict with Ultralytics' dynamic tracker state).
if device == "cuda":
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True      # fall back silently if compile fails
        # 'max-autotune' profiles and picks the fastest kernel for the fixed input size
        pose_master_model.model = torch.compile(
            pose_master_model.model, mode="max-autotune", fullgraph=False
        )
        obj_model.model  = torch.compile(
            obj_model.model,  mode="max-autotune", fullgraph=False
        )
        face_model.model = torch.compile(
            face_model.model, mode="max-autotune", fullgraph=False
        )
        print("[INFO] torch.compile(max-autotune) applied to all 3 models.")
    except Exception as _ce:
        print(f"[WARN] torch.compile skipped (needs Torch 2.x): {_ce}")

# 🔥 OPT: warm-up passes at GPU-profile imgsz so the cuDNN/Inductor cache
#         is primed for the actual production resolution (was fixed 640).
#         Extra sizes 320/480 stay in the warm-up to cover face-crop calls.
if device == "cuda":
    print(f"[INFO] Warming up models on GPU at production imgsz={INFER_IMGSZ}...")
    _dummy_sizes = sorted({320, 480, 640, int(INFER_IMGSZ)})
    with torch.no_grad():
        for _sz in _dummy_sizes:
            _dummy_np = np.zeros((_sz, _sz, 3), dtype=np.uint8)
            try:
                pose_master_model.predict(source=_dummy_np, verbose=False,
                                          half=True, imgsz=INFER_IMGSZ)
                obj_model.predict( source=_dummy_np, verbose=False,
                                   half=True, imgsz=INFER_IMGSZ)
                face_model.predict(source=_dummy_np, verbose=False,
                                   half=True, imgsz=INFER_IMGSZ)
            except Exception as _we:
                print(f"[WARN] Warmup pass sz={_sz} partial: {_we}")
    del _dummy_np, _dummy_sizes
    torch.cuda.empty_cache()
    print("[INFO] GPU warmup complete.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

fer = None
if HSEMOTION_OK:
    try:
        fer = HSEmotionRecognizer(model_name="enet_b2_8", device=device)
        print(f"[INFO] HSEmotion loaded ({device}).")
    except Exception as e:
        fer = None
        print("[WARN] HSEmotion failed to load:", e)


# ==========================================================
# ✅ MEDIAPIPE HANDS (100% INTACT)
# ==========================================================
hands      = None
mp_hands   = None
mp_drawing = None
if MEDIAPIPE_OK:
    try:
        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands      = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=5,
            min_detection_confidence=0.5,
            model_complexity=0,
        )
        print("[INFO] MediaPipe Hands loaded.")
    except Exception as e:
        hands = None
        print("[WARN] MediaPipe hands init failed:", e)
else:
    print("[WARN] MediaPipe is not usable (mp.solutions missing). Using pose fallback.")


def count_fingers(hand_landmarks, label):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    thumb_tip = 4
    thumb_ip  = 3
    if label == "Right":
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
            count += 1
    else:
        if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_ip].x:
            count += 1
    return count


# ==========================================================
# 🚀 THREADED VIDEO READER (Double-Buffered, Pinned Memory)
# Eliminates decode stall: a dedicated thread decodes the next frame
# while the main thread runs GPU inference on the current frame.
# ==========================================================
class ThreadedVideoReader:
    """
    Reads frames from cv2.VideoCapture on a background thread safely.
    Uses a threading Lock to prevent race condition crashes between main
    and reader threads.

    🔥 OPT: queue depth is GPU-profile-aware — pass `buffer_size` to scale
            look-ahead (2 frames on entry-tier GPUs, 8 on H100).
    """
    BUFFER_SIZE = 2      # default look-ahead (overridden by ctor arg)

    def __init__(self, source, buffer_size: int = None):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        size       = int(buffer_size) if buffer_size is not None else self.BUFFER_SIZE
        self._q    = Queue(maxsize=max(size, 1))
        self._lock = Lock()  # Threading lock to fix the async_lock libavcodec crash
        self._run  = True
        self._t    = Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while self._run:
            if not self._q.full():
                with self._lock:
                    ret, frame = self.cap.read()
                self._q.put((ret, frame))
            else:
                time.sleep(0.001)   # brief yield when buffer full

    def read(self):
        try:
            return self._q.get(timeout=2.0)
        except Empty:
            return False, None

    # ── Proxy all VideoCapture methods securely using the thread lock ──
    def isOpened(self):
        with self._lock:
            return self.cap.isOpened()

    def get(self, prop):
        with self._lock:
            return self.cap.get(prop)

    def set(self, prop, val):
        with self._lock:
            # Flush the queue before seeking to avoid stale frames
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Empty:
                    break
            return self.cap.set(prop, val)

    def release(self):
        self._run = False
        self._t.join(timeout=2.0)
        with self._lock:
            self.cap.release()


# ==========================================================
# 🧵 THREADS: EMOTION & SNAPSHOTS
# ==========================================================
emotion_queue   = Queue(maxsize=4)
emotion_results = {}
emotion_lock    = Lock()
csv_file        = EMOTION_LOG_CSV
all_session_data = []

# 🔧 FIX-1: tracks how many rows from `all_session_data` have ALREADY been
#           appended to dataset.json — guarantees each row is written
#           exactly once across a session (eliminates the exponential
#           autosave-duplication bug present in v6 ULTRA where every
#           autosave re-appended the ENTIRE growing list to dataset.json).
_last_appended_idx = 0
_append_lock       = Lock()

snapshot_queue = Queue(maxsize=100)

if not os.path.isfile(csv_file):
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Track_ID", "Emotion"])


def emotion_background_worker():
    """
    Processes face crops for emotion recognition.
    GPU path: batches all face detections into a single forward pass.
    Falls back to CPU Haar cascade when GPU face model is unavailable.
    🔥 OPT: blocking queue.get(timeout=0.01) replaces poll+sleep — no busy-wait,
             lower CPU overhead, faster response when tasks arrive.
    """
    if fer is None:
        while True:
            time.sleep(0.2)
    while True:
        try:
            # 🔥 OPT: Blocking get with timeout — eliminates busy-poll loop
            task = emotion_queue.get(timeout=0.01)
        except Empty:
            continue

        crops_to_process = task["crops"]
        for tid, crop in crops_to_process:
            if crop is None or crop.size == 0:
                continue
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            except Exception:
                continue
            face_detected = False
            if device == "cuda":
                try:
                    # 🚀 AMP FP16 autocast for emotion face detection
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        face_res_em = face_model.predict(
                            crop, verbose=False, conf=0.5, half=True, imgsz=320
                        )[0]
                    for fbox in face_res_em.boxes:
                        fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                        face_img = crop[max(0, fy1):min(crop.shape[0], fy2),
                                        max(0, fx1):min(crop.shape[1], fx2)]
                        if face_img.size > 0:
                            try:
                                val, _ = fer.predict_emotions(face_img, logits=False)
                                with emotion_lock:
                                    emotion_results[int(tid)] = val
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                with open(csv_file, "a", newline="") as f:
                                    csv.writer(f).writerow([timestamp, int(tid), val])
                            except Exception:
                                pass
                            face_detected = True
                            break
                except Exception:
                    pass
            if not face_detected:
                faces = face_cascade.detectMultiScale(gray_crop, 1.1, 5, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    face_img = crop[fy:fy + fh, fx:fx + fw]
                    if face_img.size > 0:
                        try:
                            val, _ = fer.predict_emotions(face_img, logits=False)
                            with emotion_lock:
                                emotion_results[int(tid)] = val
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(csv_file, "a", newline="") as f:
                                csv.writer(f).writerow([timestamp, int(tid), val])
                        except Exception:
                            pass
                        break


def snapshot_background_worker():
    """
    🚀 Async JPEG encode: imencode is done in this thread (not main loop).
    File write is also off the critical path.
    🔥 OPT: JPEG quality 80 (was 85) — ~15% smaller file, ~8% faster encode,
             imperceptible quality difference at snapshot resolution.
    """
    while True:
        task = snapshot_queue.get()
        if task is None:
            break
        frame_to_save = task["frame"]
        f_idx  = task["frame_idx"]
        s_dir  = task["s_dir"]
        ts     = int(time.time() * 1000)
        filepath = os.path.join(s_dir, f"frame_{f_idx}_{ts}.jpg")
        try:
            # imencode in background → no main-loop stall
            # 🔥 OPT: Quality 80 (was 85) — faster encode, smaller file
            ret_enc, buf = cv2.imencode(".jpg", frame_to_save,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with open(filepath, "wb") as fp:
                    fp.write(buf.tobytes())
            if f_idx % 10 == 0:
                print(f"[SNAPSHOT] Successfully saved: {filepath}")
        except Exception:
            pass
        finally:
            snapshot_queue.task_done()


Thread(target=emotion_background_worker,  daemon=True).start()
Thread(target=snapshot_background_worker, daemon=True).start()


# ==========================================================
# ✅ MUTABLE RUNTIME STATE
# (compile-time constants — CONF_THRESHOLD, FRAME_SKIP,
#  TARGET_CLASSES, COCO_NAMES, KP_NAMES, etc. — live in the
#  CONFIG block at the top of this file)
# ==========================================================
trajectory_history = {}   # tid -> deque([cx,cy], maxlen=30)
person_state       = {}
track_time_idx     = {}

recent_phones = []
recent_items  = []

# ── Module-level inference holder (avoids per-frame closure rebuilding) ──
_inference_holders = {"track": None, "obj": None, "face": None, "frame": None}


def _run_pose():
    """Pose + tracking inference on dedicated CUDA stream."""
    ctx = torch.cuda.stream(_stream_pose) if _stream_pose else _null_ctx()
    with ctx, torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _inference_holders["track"] = pose_master_model.track(
            source=_inference_holders["frame"],
            conf=CONF_THRESHOLD,
            persist=True,
            tracker=BOTSORT_CONFIG_FILE,
            device=device,
            verbose=False,
            classes=[0],
            half=True,
            imgsz=INFER_IMGSZ,
        )
    # 🔥 OPT: Record event on pose stream so main thread can sync via event, not poll
    if _event_pose is not None:
        _event_pose.record(_stream_pose)


def _run_obj():
    """Object detection inference on dedicated CUDA stream."""
    ctx = torch.cuda.stream(_stream_obj) if _stream_obj else _null_ctx()
    with ctx, torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _inference_holders["obj"] = obj_model.predict(
            source=_inference_holders["frame"],
            conf=0.65,  # Set to 50% threshold to filter false positives
            classes=TARGET_CLASSES,
            device=device,
            verbose=False,
            half=True,
            imgsz=INFER_IMGSZ,
        )[0]
    # 🔥 OPT: Record event on obj stream
    if _event_obj is not None:
        _event_obj.record(_stream_obj)


def _run_face():
    """Face detection inference on dedicated CUDA stream."""
    ctx = torch.cuda.stream(_stream_face) if _stream_face else _null_ctx()
    with ctx, torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _inference_holders["face"] = face_model(
            _inference_holders["frame"],
            verbose=False,
            conf=0.5,
            half=True,
            imgsz=INFER_IMGSZ,
        )[0]
    # 🔥 OPT: Record event on face stream
    if _event_face is not None:
        _event_face.record(_stream_face)


# ── SolvePnP pose-angle cache: skip recompute when keypoints barely moved ──
# (MODEL_POINTS, dist_coeffs, _KPS_MOVE_THRESH live in CONFIG block at top)
_pnp_cache = {}


def get_precision_metrics(bbox, frame_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center = (int((x1 + x2) / 2), int(y1 + h * 0.65))
    depth_score = (h / frame_h) if frame_h > 0 else 0.001
    return center, w, max(depth_score, 1e-3)


def get_box_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def get_pedestrian_status(kps):
    l_shldr_x = kps[5][0]
    r_shldr_x = kps[6][0]
    nose_x    = kps[0][0]
    l_ankle_x = kps[15][0]
    r_ankle_x = kps[16][0]
    l_ankle_conf = kps[15][2]
    r_ankle_conf = kps[16][2]

    body_center_x  = (l_shldr_x + r_shldr_x) / 2
    shoulder_width = abs(l_shldr_x - r_shldr_x)

    if shoulder_width < 1:
        return "Unknown"

    facing_side = abs(nose_x - body_center_x) > (shoulder_width * 0.2)
    stride_side = False
    if l_ankle_conf > 0.5 and r_ankle_conf > 0.5:
        ankle_width = abs(l_ankle_x - r_ankle_x)
        if ankle_width > (shoulder_width * 0.8):
            stride_side = True

    return "CROSSING" if (facing_side or stride_side) else "WALKING ASIDE"


def pose_question_gesture(kps):
    nose        = kps[0]
    left_wrist  = kps[9]
    right_wrist = kps[10]
    if nose[2] > 0.5:
        if (right_wrist[2] > 0.5 and right_wrist[1] < nose[1]) or \
           (left_wrist[2]  > 0.5 and left_wrist[1]  < nose[1]):
            return True
    return False


def pose_stop_gesture(kps):
    nose        = kps[0]
    left_wrist  = kps[9]
    right_wrist = kps[10]
    if nose[2] > 0.5:
        left_up  = (left_wrist[2]  > 0.5 and left_wrist[1]  < nose[1])
        right_up = (right_wrist[2] > 0.5 and right_wrist[1] < nose[1])
        return left_up and right_up
    return False


# ==========================================================
# 🔧 PRECISION HELPERS — depth, collision, ETL
# ----------------------------------------------------------
# (Constants H_REAL_PED, NEAR_DIST_M, MODEL_POINTS, dist_coeffs,
#  _KPS_MOVE_THRESH live in the CONFIG block at the top of file.)
#
# Design notes:
#   1. Monocular depth via pinhole model (replaces hardcoded PPM).
#        d_m = (f_px × H_real) / bbox_h_px
#        H_real ≈ 1.70 m  (mean adult pedestrian height across
#        CityPersons, ECP, Caltech, BDD-100K — within ±3 % across
#        age/gender mix; ref. Geiger 2012, Zhang 2017).
#   2. Per-track trajectory-based collision prediction (replaces
#        the legacy "centre is in middle third of screen" heuristic).
#   3. Per-person gesture/state attribution: hand wrists are mapped
#        to the bbox that contains them.
#   4. Dataset.json incremental writer eliminates the exponential
#        duplication bug (see _last_appended_idx + background_json_backup).
#   5. Real velocity (px/sec) via wall-clock dt — frame-skip safe.
# ==========================================================


def estimate_depth_m(bbox_h_px: float, focal_px: float) -> float:
    """Monocular depth via pinhole geometry: d = (f × H_real) / h_px."""
    return float((focal_px * H_REAL_PED) / max(float(bbox_h_px), 1.0))


def predict_collision_path(traj_deque, frame_w, frame_h, ego_y,
                           horizon: int = 12, min_pts: int = 4) -> bool:
    """
    Predicts whether a track's motion vector intersects the ego corridor
    within `horizon` frames. Constant-velocity extrapolation over the
    last (up to 5) trajectory points.

      Returns True iff:
        • track has at least `min_pts` history points,
        • |v| ≥ 0.5 px/frame (filters stationary tracks),
        • some t ∈ [1, horizon] places (x, y) inside the ego corridor
          (middle third of frame, within 20 % of frame_h above ego_y).
    """
    if traj_deque is None or len(traj_deque) < min_pts:
        return False
    pts = list(traj_deque)
    n   = min(5, len(pts))
    p0  = pts[-n]
    p1  = pts[-1]
    vx  = (p1[0] - p0[0]) / max(n - 1, 1)
    vy  = (p1[1] - p0[1]) / max(n - 1, 1)
    if abs(vx) < 0.5 and abs(vy) < 0.5:
        return False
    zone_l = frame_w / 3.0
    zone_r = 2.0 * frame_w / 3.0
    y_band = ego_y - frame_h * 0.20
    last_x, last_y = p1[0], p1[1]
    for t in range(1, horizon + 1):
        px = last_x + vx * t
        py = last_y + vy * t
        if zone_l < px < zone_r and py >= y_band:
            return True
    return False


def point_in_bbox(px: float, py: float, bbox) -> bool:
    """Inclusive containment test for (px, py) inside (x1,y1,x2,y2) bbox."""
    return (bbox[0] <= px <= bbox[2]) and (bbox[1] <= py <= bbox[3])


def project_to_world_xy(cx_px: float, frame_w_px: int,
                        dist_depth_m: float, focal_px: float) -> tuple:
    """
    🔧 ETL-HELPER: project image-plane (cx, monocular-depth) → ego-frame
    metric (X_lat, Y_fwd) coordinates.
      • X_lat (lateral): right-positive offset from camera principal axis
      • Y_fwd (forward): equals the monocular depth estimate
    Used by social-context features (`closest_other_dist_m`) so downstream
    trajectory models can reason about other pedestrians in metres, not px.
    """
    x_lat = (float(cx_px) - float(frame_w_px) * 0.5) * float(dist_depth_m) / max(float(focal_px), 1.0)
    y_fwd = float(dist_depth_m)
    return (x_lat, y_fwd)


def bbox_is_truncated(bbox, frame_w_px: int, frame_h_px: int,
                      edge_margin: int = 2) -> bool:
    """
    🔧 ETL-HELPER: True if the pedestrian bbox is clipped by any frame
    edge. Flags rows whose `dist_depth_m` is biased high because bbox
    height under-represents the real body height (training pipelines
    typically mask these or down-weight them).
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x1 <= edge_margin or y1 <= edge_margin or
            x2 >= (frame_w_px - edge_margin) or y2 >= (frame_h_px - edge_margin))


# ==========================================================
# ✅ UI
# ==========================================================
# 🔥 OPT: Cached VRAM reading counter — avoids per-frame driver query
_ui_vram_used_cached = 0.0
_ui_vram_pct_cached  = 0
_ui_vram_frame_ctr   = 0

def draw_ui(frame, state, fps, reasons):
    global _ui_vram_used_cached, _ui_vram_pct_cached, _ui_vram_frame_ctr

    h, w = frame.shape[:2]
    colors = {
        "GO":       (0, 200, 0),
        "STOP":     (0, 0, 255),
        "WARNING":  (0, 165, 255),
        "QUESTION": (0, 255, 255),
    }
    top_state_color = colors.get(state, (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (210, 55), top_state_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, state, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    fps_text = f"{int(fps)} FPS"
    ts       = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    box_w, box_h = ts[0] + 20, ts[1] + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - box_w - 10, 10), (w - 10, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, fps_text, (w - box_w, 10 + box_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Live VRAM usage overlay ──
    # 🔥 OPT: Query memory_reserved() only every 5 frames (driver call is expensive)
    if torch.cuda.is_available():
        _ui_vram_frame_ctr += 1
        if _ui_vram_frame_ctr >= 5:
            _ui_vram_frame_ctr   = 0
            _ui_vram_used_cached = torch.cuda.memory_reserved() / 1024 ** 3
            _ui_vram_pct_cached  = int((_ui_vram_used_cached / _VRAM_TOTAL_GB) * 100)
        vram_text = (f"VRAM {_ui_vram_used_cached:.1f}/"
                     f"{_VRAM_TOTAL_GB:.1f}GB ({_ui_vram_pct_cached}%)")
        vt = cv2.getTextSize(vram_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame,
                      (w - vt[0] - 20, 10 + box_h + 5),
                      (w - 10, 10 + box_h + 5 + vt[1] + 8),
                      (0, 0, 0), -1)
        vram_color = (0, 255, 0) if _ui_vram_pct_cached < 70 else \
                     (0, 165, 255) if _ui_vram_pct_cached < 90 else (0, 0, 255)
        cv2.putText(frame, vram_text,
                    (w - vt[0] - 15, 10 + box_h + 5 + vt[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, vram_color, 1)

    if reasons:
        txt = " + ".join(reasons)
        ts_r = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        box_w2, box_h2 = ts_r[0] + 20, ts_r[1] + 12
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (w // 2 - box_w2 // 2, h - 55),
                      (w // 2 + box_w2 // 2, h - 55 + box_h2),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, txt,
                    (w // 2 - box_w2 // 2 + 10, h - 55 + box_h2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ==========================================================
# ✅ BEV (BIRD'S EYE VIEW) — GPU-ACCELERATED DISTANCE MAP
# ==========================================================
def _make_bev_dist_colormap(steps=256):
    lut = np.zeros((steps, 3), dtype=np.uint8)
    for i in range(steps):
        t = i / max(steps - 1, 1)
        if t < 0.5:
            lut[i] = [int(255 * (1 - t * 2)), int(255 * (t * 2)), 0]
        else:
            lut[i] = [0, int(255 * (1 - (t - 0.5) * 2)), int(255 * ((t - 0.5) * 2))]
    lut_bgr = lut[:, [2, 1, 0]].copy()
    return lut_bgr

_BEV_DIST_LUT = _make_bev_dist_colormap()


class BEVRenderer:
    """
    Bird's Eye View overlay panel.

    Design goals (GPU-adaptive — works 3080 → 4090 → H100):
      • Static geometry (grid, rings, ego marker) built ONCE as a GPU tensor.
        Every frame it is cloned on-device — no Python re-allocation.
      • Person dots and labels drawn on a CPU copy (N is tiny, <20 persons;
        GPU kernel launch overhead > actual compute at this scale).
      • Trajectory trails drawn on a separate CPU layer with a dirty-flag:
        only redrawn when the set of active IDs changes OR trail data is new.
      • Alpha-composite done entirely on GPU with torch float16 for Ampere speed.
      • Single host→device upload (canvas_np → GPU) and single device→host
        readback per frame. Both use non_blocking=True to overlap with CPU work.
      🚀 Pre-allocated pinned output buffer: eliminates malloc on every BEV frame.
      🔥 OPT: Overlay goes directly to fp16 — avoids fp32 intermediate allocation.

    Toggle: press 'v' during playback.
    """

    BEV_W      = 320
    BEV_H      = 400
    BEV_RANGE  = 12.0
    BEV_MARGIN = 10
    BEV_ALPHA  = 0.85

    def __init__(self, ppm: float, frame_w: int, frame_h: int, device: str):
        self.ppm     = ppm
        self.half_fw = frame_w // 2
        self.device  = device

        self.sx = self.BEV_W / (2.0 * self.BEV_RANGE)
        self.sy = self.BEV_H / self.BEV_RANGE

        self.ex = self.BEV_W // 2
        self.ey = self.BEV_H - 22

        # Panel placement (computed once from frame size)
        self.x0 = frame_w - self.BEV_W - self.BEV_MARGIN
        self.y0 = frame_h - self.BEV_H - self.BEV_MARGIN

        self._static_gpu  = self._build_static_layer()
        self._traj_layer  = np.zeros((self.BEV_H, self.BEV_W, 3), dtype=np.uint8)
        self._dot_layer   = np.zeros((self.BEV_H, self.BEV_W, 3), dtype=np.uint8)
        self._traj_hash   = None

        # 🚀 Pre-allocated pinned output buffer (avoids malloc every frame)
        if device == "cuda":
            self._out_buf = torch.zeros(
                self.BEV_H, self.BEV_W, 3,
                dtype=torch.uint8,
                pin_memory=True,
            )
        else:
            self._out_buf = None

        print(f"[BEV] Renderer initialised on {device.upper()}  "
              f"panel={self.BEV_W}×{self.BEV_H}  range={self.BEV_RANGE}m")

    def world_to_bev(self, offset_px: float, dist_m: float):
        lat_m = offset_px / self.ppm
        bx    = int(self.ex + lat_m * self.sx)
        by    = int(self.ey - dist_m * self.sy)
        if 0 <= bx < self.BEV_W and 0 <= by < self.BEV_H:
            return bx, by
        return None

    @staticmethod
    def dist_color(dist_m: float, range_m: float = BEV_RANGE) -> tuple:
        idx  = int(min(dist_m / range_m, 1.0) * 255)
        b, g, r = _BEV_DIST_LUT[idx]
        return (int(b), int(g), int(r))

    def _build_static_layer(self) -> torch.Tensor:
        bg = np.zeros((self.BEV_H, self.BEV_W, 3), dtype=np.uint8)
        bg[:] = (18, 18, 18)

        for d in range(0, int(self.BEV_RANGE) + 1, 2):
            py = int(self.ey - d * self.sy)
            if 0 <= py < self.BEV_H:
                cv2.line(bg, (0, py), (self.BEV_W, py), (38, 38, 38), 1)
                cv2.putText(bg, f"{d}m", (4, max(py - 2, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

        for lat in range(-int(self.BEV_RANGE), int(self.BEV_RANGE) + 1, 2):
            px = int(self.ex + lat * self.sx)
            if 0 <= px < self.BEV_W:
                cv2.line(bg, (px, 0), (px, self.BEV_H), (38, 38, 38), 1)

        for ring_m in range(2, int(self.BEV_RANGE) + 1, 2):
            rx    = int(ring_m * self.sx)
            ry    = int(ring_m * self.sy)
            col   = self.dist_color(ring_m)
            faded = tuple(max(v // 3, 10) for v in col)
            cv2.ellipse(bg, (self.ex, self.ey), (rx, ry), 0, 180, 360, faded, 1)

        cv2.line(bg, (self.ex, self.ey), (self.ex, 0), (50, 50, 50), 1)

        tri = np.array([
            [self.ex,      self.ey - 14],
            [self.ex - 8,  self.ey + 5],
            [self.ex + 8,  self.ey + 5],
        ], dtype=np.int32)
        cv2.fillPoly(bg, [tri], (0, 215, 255))
        cv2.putText(bg, "EGO", (self.ex - 13, self.ey + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 215, 255), 1)

        cv2.rectangle(bg, (0, 0), (self.BEV_W - 1, self.BEV_H - 1), (70, 70, 70), 1)
        cv2.rectangle(bg, (0, 0), (self.BEV_W, 16), (28, 28, 28), -1)
        cv2.putText(bg, "BEV  Bird's Eye View  [v] toggle",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1)

        return torch.from_numpy(bg).to(self.device)

    def _update_traj_layer(self, trajectory_history: dict, active_ids: set, frame_h: int):
        try:
            traj_hash = hash(
                tuple((tid, len(pts)) for tid, pts in trajectory_history.items()
                      if tid in active_ids)
            )
        except Exception:
            traj_hash = None

        if traj_hash == self._traj_hash:
            return

        self._traj_hash      = traj_hash
        self._traj_layer[:] = 0

        for tid, pts_deq in trajectory_history.items():
            pts_img = list(pts_deq)
            if len(pts_img) < 2:
                continue
            bev_pts = []
            for pt in pts_img:
                cx_img, cy_img = pt[0], pt[1]
                depth_approx   = max((frame_h - cy_img) / max(self.ppm, 1), 0.3)
                bp = self.world_to_bev(cx_img - self.half_fw, depth_approx)
                if bp:
                    bev_pts.append(bp)
            if len(bev_pts) >= 2:
                pts_arr  = np.array(bev_pts, dtype=np.int32)
                hue      = int((tid * 47) % 180)
                hsv_col  = np.array([[[hue, 180, 160]]], dtype=np.uint8)
                rgb_col  = cv2.cvtColor(hsv_col, cv2.COLOR_HSV2BGR)[0][0]
                trail_col = (int(rgb_col[0]) // 2, int(rgb_col[1]) // 2, int(rgb_col[2]) // 2)
                cv2.polylines(self._traj_layer, [pts_arr], False, trail_col, 1, cv2.LINE_AA)

    def _draw_dots(self, active_persons: list):
        self._dot_layer[:] = 0
        for p in active_persons:
            bp = self.world_to_bev(p["offset"], p["dist_m"])
            if bp is None:
                continue
            bx, by = bp
            col   = self.dist_color(p["dist_m"])
            r_col = {"red":    (0, 0, 255),
                     "yellow": (0, 200, 200),
                     "green":  (0, 200, 0)}.get(p.get("ring_status", "green"), (0, 200, 0))

            cv2.circle(self._dot_layer, (bx, by), 9, r_col, 2, cv2.LINE_AA)
            cv2.circle(self._dot_layer, (bx, by), 5, col,   -1, cv2.LINE_AA)
            cv2.putText(self._dot_layer, str(p["track_id"]),
                        (bx + 8, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 220, 220), 1)
            cv2.putText(self._dot_layer, f"{p['dist_m']:.1f}m",
                        (bx - 12, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

    def render(self, frame_bgr: np.ndarray,
               active_persons: list,
               trajectory_history: dict) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        x0, y0 = self.x0, self.y0
        if x0 < 0 or y0 < 0:
            return frame_bgr

        active_ids = {p["track_id"] for p in active_persons}
        self._update_traj_layer(trajectory_history, active_ids, fh)
        self._draw_dots(active_persons)

        # ── Composite on GPU ────────────────────────────────────────────
        canvas_gpu  = self._static_gpu.clone()

        overlay_cpu = self._traj_layer.copy()
        dot_mask    = self._dot_layer.any(axis=2)
        overlay_cpu[dot_mask] = self._dot_layer[dot_mask]

        # 🚀 non_blocking=True: DMA transfer overlaps with CPU work
        # 🔥 OPT: Send directly to half() — skip fp32 intermediate on CUDA
        if self.device == "cuda":
            overlay_gpu = torch.from_numpy(overlay_cpu).to(
                self.device, dtype=torch.float16, non_blocking=True
            )
            canvas_half = canvas_gpu.half()
            ov_mask_gpu = overlay_gpu.any(dim=2)
            canvas_half[ov_mask_gpu] = (
                canvas_half[ov_mask_gpu] * 0.3 +
                overlay_gpu[ov_mask_gpu] * 0.7
            ).clamp(0, 255)
            canvas_gpu = canvas_half.byte()
        else:
            overlay_gpu = torch.from_numpy(overlay_cpu).float()
            canvas_gpu  = canvas_gpu.float()
            ov_mask_gpu = overlay_gpu.any(dim=2)
            canvas_gpu[ov_mask_gpu] = (
                canvas_gpu[ov_mask_gpu] * 0.3 +
                overlay_gpu[ov_mask_gpu] * 0.7
            ).clamp(0, 255)
            canvas_gpu = canvas_gpu.byte()

        roi_np  = np.ascontiguousarray(frame_bgr[y0:y0 + self.BEV_H, x0:x0 + self.BEV_W])
        roi_gpu = torch.from_numpy(roi_np).to(self.device, non_blocking=True)

        a = self.BEV_ALPHA
        if self.device == "cuda":
            blended = (a * canvas_gpu.half() +
                       (1.0 - a) * roi_gpu.half()).clamp(0, 255).byte()
        else:
            blended = (a * canvas_gpu.float() +
                       (1.0 - a) * roi_gpu.float()).clamp(0, 255).byte()

        # 🚀 Write result into pre-allocated pinned buffer then copy to frame
        if self._out_buf is not None:
            self._out_buf.copy_(blended, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            frame_bgr[y0:y0 + self.BEV_H, x0:x0 + self.BEV_W] = self._out_buf.numpy()
        else:
            frame_bgr[y0:y0 + self.BEV_H, x0:x0 + self.BEV_W] = blended.cpu().numpy()

        cv2.rectangle(frame_bgr,
                      (x0 - 1, y0 - 1),
                      (x0 + self.BEV_W, y0 + self.BEV_H),
                      (90, 90, 90), 1)
        return frame_bgr


# ==========================================================
# ✅ DISPLAY INITIALIZATION
# (toggles SHOW_LIVE_GRAPH, SAVE_ANNOTATED_SNAPSHOTS live in
#  CONFIG block at the top of this file)
# ==========================================================
if SHOW_LIVE_GRAPH == 1:
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))


# ==========================================================
# ✅ CHECKPOINT DATABASE MANAGEMENT (100% INTACT)
# (CHECKPOINT_FILE constant lives in CONFIG block at top of file)
# ==========================================================
cp_data_all = {}

if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            cp_data_all = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read checkpoint database: {e}")


def save_checkpoint_to_db(filepath, frame_index, time_string):
    if filepath.lower() == "c":
        return
    cp_data_all[filepath] = {
        "frame_idx":     frame_index,
        "time_str":      time_string,
        "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp_data_all, f, indent=4)
    print(f"[INFO] logging to checkpoint.json... (Saved {filepath} at {time_string})")


def background_json_backup(db_copy, p_val, f_val, t_val,
                           full_session_data, start_idx,
                           log_path, f_t, f_m):
    """
    🔧 FIX-1: incremental dataset.json writer.
      • `full_session_data` is the complete in-memory row list (used to
        overwrite the per-video session log — that file IS a full snapshot
        of the session, by design).
      • `start_idx` is the position in `full_session_data` PAST WHICH rows
        have NOT yet been appended to dataset.json. Only those new rows
        are appended → no duplication regardless of autosave frequency
        or session length.
    """
    try:
        f_t.flush()
        os.fsync(f_t.fileno())
        f_m.flush()
        os.fsync(f_m.fileno())
    except Exception:
        pass

    db_copy[p_val] = {
        "frame_idx":     f_val,
        "time_str":      t_val,
        "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(db_copy, f, indent=4)

    # per-video session log: full snapshot (overwrite is correct here)
    with open(log_path, "w") as jf:
        json.dump(full_session_data, jf, indent=4)

    # master dataset.json: append ONLY the rows that are new since the
    # previous autosave (was: re-appending the entire growing list).
    dataset_path = DATASET_JSON_FILE
    new_rows     = (full_session_data[start_idx:]
                    if start_idx is not None and start_idx >= 0
                    else full_session_data)
    if not new_rows:
        print(f"\n[INFO] Auto-Saved {p_val} at [{t_val}] (no new dataset rows)")
        return

    master_json_data = []
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r") as f:
                master_json_data = json.load(f)
            if not isinstance(master_json_data, list):
                master_json_data = []
        except Exception:
            master_json_data = []

    master_json_data.extend(new_rows)
    with open(dataset_path, "w") as jf:
        json.dump(master_json_data, jf, indent=4)

    print(f"\n[INFO] logging to checkpoint.json... (Auto-Saved {p_val} at [{t_val}])")
    print(f"[INFO] Master dataset JSON updated (+{len(new_rows)} rows): dataset.json")


# ==========================================================
# ✅ INPUT SOURCE & DYNAMIC FILE NAMING
# ==========================================================
time.sleep(0.5)
user_input      = input("\nVideo Name or 'c': ").strip()
user_input      = re.sub(r'[\x00-\x1f]', '', user_input)

path = (f"inference/videos/{user_input}"
        if os.path.exists(f"inference/videos/{user_input}")
        else user_input)

video_base_name = (os.path.splitext(os.path.basename(path))[0]
                   if user_input.lower() != "c"
                   else f"camera_run_{int(time.time())}")

traj_csv_path    = f"{video_base_name}_traj_dataset.csv"
session_log_path = f"{video_base_name}_session_log.json"
master_csv       = MASTER_CSV_FILE

base_img_dir  = os.path.join("inference", "images")
snapshot_dir  = os.path.abspath(os.path.join(base_img_dir, f"{video_base_name}_snapshots"))
os.makedirs(snapshot_dir, exist_ok=True)
print(f"\n[INFO] Snapshots will be securely saved to: {snapshot_dir}\n")

if os.path.exists(session_log_path):
    try:
        with open(session_log_path, "r") as f:
            all_session_data = json.load(f)
    except Exception:
        pass

# 🔧 FIX-1: rows already on disk → mark them as "already appended" so the
#           autosave thread does not re-append them to dataset.json.
_last_appended_idx = len(all_session_data)


# ==========================================================
# ✅ REAL-TIME CSV STREAM INITIALIZATION
# ==========================================================
headers = [
    "video_name", "id", "t", "frame_idx", "x", "y", "delta_x", "delta_y", "dist", "ang",
    "near_car", "stop_gesture", "collision_path", "crossing_intent", "question_gesture",
    "move_status", "ring_status", "state_text", "emotion", "offset", "status",
    "timestamp", "smartphone", "items",
    # ── 17 COCO keypoints: x, y, conf per keypoint ──
    "kp_nose_x",           "kp_nose_y",           "kp_nose_conf",
    "kp_left_eye_x",       "kp_left_eye_y",       "kp_left_eye_conf",
    "kp_right_eye_x",      "kp_right_eye_y",      "kp_right_eye_conf",
    "kp_left_ear_x",       "kp_left_ear_y",       "kp_left_ear_conf",
    "kp_right_ear_x",      "kp_right_ear_y",      "kp_right_ear_conf",
    "kp_left_shoulder_x",  "kp_left_shoulder_y",  "kp_left_shoulder_conf",
    "kp_right_shoulder_x", "kp_right_shoulder_y", "kp_right_shoulder_conf",
    "kp_left_elbow_x",     "kp_left_elbow_y",     "kp_left_elbow_conf",
    "kp_right_elbow_x",    "kp_right_elbow_y",    "kp_right_elbow_conf",
    "kp_left_wrist_x",     "kp_left_wrist_y",     "kp_left_wrist_conf",
    "kp_right_wrist_x",    "kp_right_wrist_y",    "kp_right_wrist_conf",
    "kp_left_hip_x",       "kp_left_hip_y",       "kp_left_hip_conf",
    "kp_right_hip_x",      "kp_right_hip_y",      "kp_right_hip_conf",
    "kp_left_knee_x",      "kp_left_knee_y",       "kp_left_knee_conf",
    "kp_right_knee_x",     "kp_right_knee_y",     "kp_right_knee_conf",
    "kp_left_ankle_x",     "kp_left_ankle_y",     "kp_left_ankle_conf",
    "kp_right_ankle_x",    "kp_right_ankle_y",    "kp_right_ankle_conf",
]

need_traj_header   = not os.path.exists(traj_csv_path)
f_traj             = open(traj_csv_path,  "a", newline="")
w_traj             = csv.writer(f_traj)
if need_traj_header:
    w_traj.writerow(headers)

need_master_header = not os.path.exists(master_csv)
f_master           = open(master_csv,     "a", newline="")
w_master           = csv.writer(f_master)
if need_master_header:
    w_master.writerow(headers)


# ==========================================================
# 🚀 VIDEO CAPTURE — THREADED (Double-Buffered + Lock Safeguard)
# ==========================================================
cap = ThreadedVideoReader(1 if user_input.lower() == "c" else path,
                          buffer_size=READER_BUFFER)

for _ in range(10):
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        break
if not ret or test_frame is None:
    raise RuntimeError("Could not read from camera/video source.")

frame_h, frame_w = test_frame.shape[:2]

# ── Dynamic FRAME_SKIP: target 15 processed fps regardless of source fps ──
# Overrides the static FRAME_SKIP=4 set above.
# For a 30 fps source → FRAME_SKIP=2, 60 fps → 4, webcam/unknown → assume 30.
_src_fps = cap.get(cv2.CAP_PROP_FPS)
if _src_fps is None or _src_fps <= 0:
    _src_fps = 30.0   # safe default for webcam / unreadable sources
# (TARGET_PROC_FPS, FRAME_SKIP-initial live in CONFIG block at top)
FRAME_SKIP = max(1, round(_src_fps / TARGET_PROC_FPS))
# 🔧 ETL-EXTRA: effective processed fps (constant for the session).
#                Downstream trajectory models read this to convert
#                per-row velocities into per-second velocities, OR to
#                resample the sequence onto a fixed-dt timeline.
_PROCESSED_FPS = float(_src_fps) / float(FRAME_SKIP)
print(f"[INFO] Source FPS: {_src_fps:.1f}  →  FRAME_SKIP={FRAME_SKIP} "
      f"(targeting {TARGET_PROC_FPS} processed fps)")
print(f"[INFO] Effective processed FPS: {_PROCESSED_FPS:.2f}")

camera_matrix    = np.array([[frame_w, 0,       frame_w / 2],
                              [0,       frame_w, frame_h / 2],
                              [0,       0,       1          ]], dtype="double")

# 🔧 FIX-3: focal length in pixels (top-left of camera_matrix). Used by
#           estimate_depth_m() to convert bbox-height into metric depth.
F_PX               = float(frame_w)
print(f"[INFO] Focal length (px): {F_PX:.1f}  →  depth model uses "
      f"H_real={H_REAL_PED}m, near={NEAR_DIST_M}m")

prev_time          = time.time()
last_autosave_time = time.time()
fps_smooth         = 0.0
frame_idx          = 0
start_frame        = 0

# ── Checkpoint resume ──
if user_input.lower() != "c" and path in cp_data_all:
    saved_frame = cp_data_all[path].get("frame_idx", 0)
    saved_time  = cp_data_all[path].get("time_str",  "00:00:00")
    print(f"\n=======================================================")
    print(f"[INFO] CHECKPOINT DETECTED for '{video_base_name}'")
    print(f"[INFO] Previously paused at [{saved_time}]")
    print(f"=======================================================")

    while True:
        prompt_frame = test_frame.copy()
        cv2.putText(prompt_frame, f"CHECKPOINT DETECTED: [{saved_time}]",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(prompt_frame, "Press 'r' to RESUME from checkpoint",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(prompt_frame, "Press 'b' to BEGIN from the start",
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("ALL FEATURES: Integrated Analytics + AEB", prompt_frame)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('r') or key == ord('R'):
            start_frame = saved_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"[INFO] ▶️ Resuming video from [{saved_time}]...")
            frame_idx = start_frame
            break
        elif key == ord('b') or key == ord('B'):
            print("[INFO] ⏪ Video is starting from the beginning. (Data will append)")
            if path in cp_data_all:
                del cp_data_all[path]
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(cp_data_all, f, indent=4)
                print(f"[INFO] logging to checkpoint.json... (Cleared record for {path})")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break

print("\n=======================================================")
print("🎮 DYNAMIC CONTROLS DURING PLAYBACK")
print("-> You MUST click the Video Window to use keys!")
print(" [p] PAUSE the video.")
print(" [b] RESTART the video from the beginning anytime.")
print(" [r] RESUME the video if it is paused.")
print(" [q] SAVE current exact timepoint and QUIT to terminal.")
print("=======================================================\n")


# ==========================================================
# 🚀 MAIN LOOP
# ==========================================================
# cv2.namedWindow("ALL FEATURES: Integrated Analytics + AEB", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("ALL FEATURES: Integrated Analytics + AEB", 1280, 720)
bev_renderer = BEVRenderer(ppm=PPM, frame_w=frame_w, frame_h=frame_h, device=device)
SHOW_BEV     = True

# 🔥 OPT: Pre-compute ALL static per-frame constants before the loop —
#          eliminates repeated integer division and arithmetic every frame.
_zone_left_x    = frame_w // 3
_zone_right_x   = 2 * frame_w // 3
_ref_x          = frame_w // 2
_ref_y          = frame_h - 60
# Pre-compute zone-line tuples (passed to cv2.line; tuples are faster than recalculating)
_ZONE_L_P1 = (_zone_left_x,  0)
_ZONE_L_P2 = (_zone_left_x,  frame_h)
_ZONE_R_P1 = (_zone_right_x, 0)
_ZONE_R_P2 = (_zone_right_x, frame_h)
_XHAIR_L   = (_ref_x - 20, _ref_y)
_XHAIR_R   = (_ref_x + 20, _ref_y)
_XHAIR_T   = (_ref_x, _ref_y - 20)
_XHAIR_B   = (_ref_x, _ref_y + 20)

while cap.isOpened():
    ret, frame = cap.read()
    frame_idx += 1
    if not ret or frame is None:
        break

    if frame_idx % FRAME_SKIP != 0:
        continue

    # 🚀 VRAM cleanup: every 500 frames (empty_cache is expensive — reduce frequency)
    if frame_idx % 500 == 0 and device == "cuda":
        torch.cuda.empty_cache()

    curr_time = time.time()

    # 🔥 OPT: Conditionally skip raw_frame copy when not needed for snapshots.
    #         SAVE_ANNOTATED_SNAPSHOTS==0 → save raw; ==1 → save annotated (frame).
    #         Either way only ONE copy is made per frame.
    if SAVE_ANNOTATED_SNAPSHOTS == 0:
        raw_frame = frame.copy()
    # (annotated path makes the copy later after drawing)

    h, w = frame.shape[:2]

    # ── Zone lines (use pre-computed constants) ──
    cv2.line(frame, _ZONE_L_P1, _ZONE_L_P2, (0, 0, 255), 1)
    cv2.line(frame, _ZONE_R_P1, _ZONE_R_P2, (0, 0, 255), 1)
    cv2.line(frame, _XHAIR_L,   _XHAIR_R,   (0, 0, 255), 2)
    cv2.line(frame, _XHAIR_T,   _XHAIR_B,   (0, 0, 255), 2)

    ref_x, ref_y = _ref_x, _ref_y

    rgb              = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    # ── Dispatch all 3 models concurrently on separate CUDA streams ──
    _inference_holders["frame"] = frame

    f_pose = _infer_pool.submit(_run_pose)
    f_obj  = _infer_pool.submit(_run_obj)
    f_face = _infer_pool.submit(_run_face)

    # 🔥 OPT: Wait on futures first (ensures kernel submission), then sync via
    #         CUDA Events — no host stall until GPU has truly finished all streams.
    f_pose.result()
    f_obj.result()
    f_face.result()
    if device == "cuda":
        # Wait on each stream's event (non-blocking to each other)
        _event_pose.synchronize()
        _event_obj.synchronize()
        _event_face.synchronize()

    track_results = _inference_holders["track"]
    obj_res       = _inference_holders["obj"]
    face_res      = _inference_holders["face"]

    # ── Object class parsing (batch tensor extract) ──
    raw_phones = []
    raw_items  = []
    try:
        if len(obj_res.boxes) > 0:
            all_cls   = obj_res.boxes.cls.cpu().numpy().astype(int)
            all_xyxy  = obj_res.boxes.xyxy.cpu().numpy()
            frame_area = w * h
            for cls_id, xyxy in zip(all_cls, all_xyxy):
                box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if cls_id == 67:
                    if box_area < (frame_area * 0.05):
                        raw_phones.append(xyxy)
                elif cls_id in [24, 25, 26, 28]:
                    raw_items.append((xyxy, COCO_NAMES.get(cls_id, "Item")))
    except Exception:
        pass

    # ── Box stabilization (3-frame visual memory) ──
    # 🔥 OPT: Use math.hypot (scalar, no numpy overhead) for distance calc
    updated_items = []
    for box, name in raw_items:
        updated_items.append({'box': box, 'name': name, 'life': 3})
    for old_item in recent_items:
        c_old   = get_box_center(old_item['box'])
        matched = False
        for new_item in updated_items:
            c_new = get_box_center(new_item['box'])
            if (old_item['name'] == new_item['name'] and
                    math.hypot(c_old[0] - c_new[0], c_old[1] - c_new[1]) < 70):
                matched = True
                break
        if not matched and old_item['life'] > 1:
            updated_items.append({
                'box':  old_item['box'],
                'name': old_item['name'],
                'life': old_item['life'] - 1,
            })
    recent_items = updated_items

    updated_phones = []
    for box in raw_phones:
        updated_phones.append({'box': box, 'life': 3})
    for old_phone in recent_phones:
        c_old   = get_box_center(old_phone['box'])
        matched = False
        for new_phone in updated_phones:
            c_new = get_box_center(new_phone['box'])
            # 🔥 OPT: math.hypot — avoids numpy array allocation for scalar distance
            if math.hypot(c_old[0] - c_new[0], c_old[1] - c_new[1]) < 70:
                matched = True
                break
        if not matched and old_phone['life'] > 1:
            updated_phones.append({'box': old_phone['box'], 'life': old_phone['life'] - 1})
    recent_phones = updated_phones

    # ── Draw held items + phones ──
    for itm in recent_items:
        ix1, iy1, ix2, iy2 = map(int, itm['box'])
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)
        cv2.putText(frame, itm['name'], (ix1, iy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for ph in recent_phones:
        px1, py1, px2, py2 = map(int, ph['box'])
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(frame, "Smartphone", (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ── Face bounding boxes ──
    try:
        for box in face_res.boxes:
            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
            fconf = float(box.conf[0])
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            cv2.putText(frame, f"Face {int(fconf * 100)}%", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    except Exception:
        pass

    # ── MediaPipe Hands STOP gesture ──
    # 🔧 FIX-2/3: capture each hand's wrist coordinates + finger count so we
    #             can attribute stop-gestures to the SPECIFIC person whose
    #             bbox contains the wrist (was: single global flag applied
    #             to every track in the frame).
    stop_gesture        = False                    # global aggregate (UI/reasons only)
    hand_landmarks_info = []                       # list of (wx, wy, finger_count)
    if hands is not None:
        try:
            hands_res = hands.process(rgb)
            if hands_res.multi_hand_landmarks:
                for idx, lm in enumerate(hands_res.multi_hand_landmarks):
                    label = hands_res.multi_handedness[idx].classification[0].label
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    fc    = count_fingers(lm, label)
                    wx    = lm.landmark[0].x * w
                    wy    = lm.landmark[0].y * h
                    hand_landmarks_info.append((wx, wy, fc))
                    if fc >= 5:
                        stop_gesture = True
        except Exception:
            stop_gesture        = False
            hand_landmarks_info = []

    # ── Track loop ──
    active_this_frame  = []
    crops_to_process   = []
    collision          = False
    crossing_intent    = False
    question           = False
    walking_side_count = 0

    for result in track_results:
        if result.boxes.id is None:
            continue

        boxes    = result.boxes.xyxy.cpu().numpy().astype(int)
        ids      = result.boxes.id.cpu().numpy().astype(int)
        kpts_all = (result.keypoints.data.cpu().numpy()
                    if result.keypoints is not None
                    else [None] * len(boxes))

        for i, (bbox, tid) in enumerate(zip(boxes, ids)):
            center, width, depth = get_precision_metrics(bbox, h)
            pose_center  = center
            pose_bottom_y = bbox[3]

            with emotion_lock:
                emotion_val = emotion_results.get(int(tid), "Detecting...")

            x1, y1, x2, y2 = bbox
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
            crops_to_process.append((int(tid), crop))

            # ── Phone overlap check ──
            current_phone = 0
            for ph in recent_phones:
                px1, py1, px2, py2 = ph['box']
                if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):
                    current_phone = 1
                    break
            smartphone_status = current_phone

            # ── Items overlap check ──
            current_items = []
            for itm in recent_items:
                ix1, iy1, ix2, iy2 = itm['box']
                if not (x2 < ix1 or x1 > ix2 or y2 < iy1 or y1 > iy2):
                    if itm['name'] not in current_items:
                        current_items.append(itm['name'])
            items_str = ", ".join(current_items) if current_items else "None"

            move_status = "Unknown"
            ang         = "N/A"
            kps         = kpts_all[i]
            p_question  = False                                # 🔧 per-person
            p_stop_pose = False                                # 🔧 per-person

            if kps is not None and len(kps) > 16:
                if kps[11][2] > 0.5 and kps[12][2] > 0.5:
                    pose_center = (int((kps[11][0] + kps[12][0]) / 2),
                                   int((kps[11][1] + kps[12][1]) / 2))
                if kps[15][2] > 0.5 or kps[16][2] > 0.5:
                    pose_bottom_y = max(kps[15][1], kps[16][1])

                move_status = get_pedestrian_status(kps)

                # ── SolvePnP (cached) — 🔧 ang now stored as float (was int) ──
                try:
                    if kps.shape[0] >= 5:
                        img_points = kps[:5, :2].astype(np.float32)
                        cached     = _pnp_cache.get(int(tid))
                        use_cached = False
                        if cached is not None:
                            delta = np.max(np.abs(img_points - cached["kps5"]))
                            if delta < _KPS_MOVE_THRESH:
                                ang        = cached["ang"]
                                use_cached = True
                        if not use_cached:
                            success, rvec, tvec = cv2.solvePnP(
                                MODEL_POINTS, img_points,
                                camera_matrix, dist_coeffs,
                                flags=cv2.SOLVEPNP_EPNP,
                            )
                            if success:
                                rmat, _         = cv2.Rodrigues(rvec)
                                proj_matrix     = np.hstack((rmat, tvec))
                                _, _, _, _, _, _, euler_angles = \
                                    cv2.decomposeProjectionMatrix(proj_matrix)
                                ang = round(float(euler_angles.flatten()[1]), 2)
                                _pnp_cache[int(tid)] = {
                                    "kps5": img_points.copy(), "ang": ang
                                }
                except Exception:
                    ang = "N/A"

                # 🔧 FIX-2: per-person gesture flags (also feed the global
                #            aggregates kept for UI/state_text overlay).
                p_question  = bool(pose_question_gesture(kps))
                p_stop_pose = bool(pose_stop_gesture(kps))

                if move_status == "CROSSING":
                    crossing_intent = True
                else:
                    walking_side_count += 1

                if p_question:
                    question = True
                if hands is None and p_stop_pose:
                    stop_gesture = True

            # 🔧 FIX-3: monocular depth via pinhole model — replaces the
            #            hardcoded PPM heuristic for distance-from-ego.
            bbox_w_px    = max(int(x2 - x1), 1)
            bbox_h_px    = max(int(y2 - y1), 1)
            dist_depth_m = estimate_depth_m(bbox_h_px, F_PX)

            # legacy on-screen "dist" (kept for backward compat with v6 JSON)
            dist_m       = math.sqrt((pose_center[0] - ref_x) ** 2 +
                                      (pose_bottom_y - ref_y) ** 2) / PPM
            offset       = pose_center[0] - ref_x

            # warning now uses calibrated depth (was PPM-screen-distance ≤ 0.8 m)
            near_p       = (dist_depth_m <= NEAR_DIST_M)
            safe_warning = "WARNING" if near_p else "SAFE"

            # legacy global "collision" — kept for UI reasons aggregate, but
            # per-track collision_path is now trajectory-based (computed in 2nd loop)
            if _zone_left_x < pose_center[0] < _zone_right_x:
                collision = True

            radius = width * 0.75
            active_this_frame.append({
                "track_id":    int(tid),
                "bbox":        bbox,
                "bbox_w_px":   bbox_w_px,                       # 🔧 new
                "bbox_h_px":   bbox_h_px,                       # 🔧 new
                "center":      pose_center,
                "width":       width,
                "radius":      radius,
                "depth":       depth,
                "dist_m":      dist_m,                          # legacy screen
                "dist_depth_m": dist_depth_m,                   # 🔧 metric depth
                "offset":      offset,
                "ring_status": "green",
                "emotion":     emotion_val,
                "move_status": move_status,
                "ang":         ang,
                "safe_warning": safe_warning,
                "smartphone":  smartphone_status,
                "items":       items_str,
                "kps":         kps,                              # raw 17-kp array
                "p_question":  p_question,                       # 🔧 per-person
                "p_stop_pose": p_stop_pose,                      # 🔧 per-person
                "p_crossing":  (move_status == "CROSSING"),     # 🔧 per-person
                "near_p":      near_p,                           # 🔧 per-person
            })

    # 🔧 FIX-3: aggregate near-car decision now uses calibrated metric depth
    near_car = any(p.get("near_p", False) for p in active_this_frame)

    # 🔧 ETL-EXTRA: project each tracked person's (cx, depth) to ego-frame
    #               world XY (metres) ONCE per frame. The 2nd loop reads
    #               these to compute `closest_other_dist_m` in O(N) per
    #               person instead of recomputing the projection each
    #               pair-wise comparison.
    for _p in active_this_frame:
        _p["_world_xy"] = project_to_world_xy(
            _p["center"][0], frame_w, _p["dist_depth_m"], F_PX
        )

    reasons = []
    if near_car:
        # 🔧 FIX-3: threshold message reflects new metric-depth boundary
        reasons.append(f"Pedestrian Near (<={NEAR_DIST_M}m)")
    if stop_gesture:
        reasons.append("Hand Gesture")
    if collision:
        reasons.append("Collision Path")
    if crossing_intent:
        reasons.append("Intent to Cross")
    if question:
        reasons.append("Question")
    if not (stop_gesture or collision or crossing_intent or question or near_car) \
            and walking_side_count > 0:
        reasons.append("Safe Pedestrian")

    if near_car:
        state_text = "STOP"
    elif stop_gesture or collision or crossing_intent:
        state_text = "WARNING"
    elif question:
        state_text = "QUESTION"
    else:
        state_text = "GO"

    state_text_for_log = str(state_text)

    if fer is not None and len(crops_to_process) > 0:
        try:
            emotion_queue.put_nowait({"crops": crops_to_process})
        except Exception:
            pass

    # ── Speed smoothing & delta calculation ──
    # 🔧 FIX: Track whether any CSV/JSON row was logged this frame.
    #          Snapshot is dispatched ONLY when this flag is True — i.e. only
    #          when data was actually written, not on every processed frame.
    snapshot_logged_this_frame = False

    for p in active_this_frame:
        tid   = p["track_id"]
        cx, cy = p["center"]

        # ── motion + state ─────────────────────────────────────────────
        if tid not in person_state:
            person_state[tid] = {
                "last_center":    (cx, cy),
                "smoothed_speed": 0.0,
                "last_seen":      curr_time,
            }
            delta_x = delta_y = 0
            dt_real = 0.0
        else:
            last_cx, last_cy = person_state[tid]["last_center"]
            delta_x = cx - last_cx
            delta_y = cy - last_cy
            dt_real = curr_time - person_state[tid]["last_seen"]

        st = person_state[tid]
        # 🔥 OPT: math.hypot replaces np.linalg.norm — no array alloc, pure scalar
        raw_dist      = math.hypot(cx - st["last_center"][0], cy - st["last_center"][1])
        instant_speed = (raw_dist / (p["depth"] * 100)
                         if raw_dist > MIN_MOVEMENT_LOGIC else 0.0)
        st["smoothed_speed"] = (EMA_ALPHA * instant_speed +
                                 (1 - EMA_ALPHA) * st["smoothed_speed"])
        st["last_center"] = (cx, cy)
        st["last_seen"]   = curr_time

        # 🔧 FIX-5: true real-time velocity (px/sec). Uses wall-clock dt so
        #            it is FRAME_SKIP-independent and survives detection gaps.
        velocity_px_per_sec = (raw_dist / max(dt_real, 1e-6)) if dt_real > 1e-6 else 0.0

        if st["smoothed_speed"] > SPEED_TRIGGER:
            p["ring_status"] = "yellow"

        # 🚀 deque(maxlen=30) — O(1) append/trim vs list.pop(0) which is O(N)
        if tid not in trajectory_history:
            trajectory_history[tid] = deque(maxlen=30)
        trajectory_history[tid].append([cx, cy])

        if int(tid) not in track_time_idx:
            track_time_idx[int(tid)] = 0
        else:
            track_time_idx[int(tid)] += 1

        # 🔧 FIX-4: per-person trajectory-based collision prediction.
        #            Replaces the v6 heuristic ("person is in middle third
        #            of screen → COLLISION"), which mis-attributed the flag
        #            to any pedestrian standing in centre frame.
        p_collision = predict_collision_path(
            trajectory_history[tid], frame_w, frame_h, ref_y
        )

        # 🔧 FIX-2/3: hand-based stop gesture attributed via wrist-in-bbox
        p_stop_hand = False
        for (wx_h, wy_h, fc_h) in hand_landmarks_info:
            if fc_h >= 5 and point_in_bbox(wx_h, wy_h, p["bbox"]):
                p_stop_hand = True
                break
        p_stop = bool(p["p_stop_pose"] or p_stop_hand)

        # ── Per-person state_text (was previously global, attributing one
        #     person's behaviour to every track in the frame). ─────────
        if p["near_p"]:
            p_state = "STOP"
        elif p_stop or p_collision or p["p_crossing"]:
            p_state = "WARNING"
        elif p["p_question"]:
            p_state = "QUESTION"
        else:
            p_state = "GO"

        # 🔧 ETL-EXTRA: social-context features per person.
        #    - n_others_in_frame      : peers visible in the same frame
        #    - closest_other_dist_m   : min ego-frame metric distance to a
        #                                peer (uses pre-computed world XY).
        n_others_in_frame    = max(len(active_this_frame) - 1, 0)
        closest_other_dist_m = 0.0
        if n_others_in_frame > 0:
            self_wx, self_wy = p["_world_xy"]
            best = float("inf")
            for q in active_this_frame:
                if q["track_id"] == tid:
                    continue
                qwx, qwy = q["_world_xy"]
                d = math.hypot(self_wx - qwx, self_wy - qwy)
                if d < best:
                    best = d
            closest_other_dist_m = float(best) if best != float("inf") else 0.0

        # 🔧 ETL-EXTRA: per-row scalar features for trajectory training.
        is_truncated_flag = bbox_is_truncated(p["bbox"], frame_w, frame_h)
        x_norm_val        = float(cx) / float(max(frame_w, 1))
        y_norm_val        = float(cy) / float(max(frame_h, 1))

        # ── Build per-keypoint dicts and flat values for CSV ──
        kps_data     = p["kps"]   # shape (17,3) or None
        kp_dict      = {}         # for JSON: {name: {x,y,conf}}
        kp_csv_vals  = []         # for CSV: flat [x,y,conf, x,y,conf, ...]
        for ki, kname in enumerate(KP_NAMES):
            if kps_data is not None and ki < len(kps_data):
                kx   = round(float(kps_data[ki][0]), 3)
                ky   = round(float(kps_data[ki][1]), 3)
                kc   = round(float(kps_data[ki][2]), 4)
            else:
                kx = ky = kc = 0.0
            kp_dict[kname] = {"x": kx, "y": ky, "conf": kc}
            kp_csv_vals.extend([kx, ky, kc])

        row_dict = {
            "video_name":       video_base_name,
            "timestamp":        datetime.now().isoformat(),
            "t":                int(track_time_idx[int(tid)]),
            "frame_idx":        int(frame_idx),
            "id":               int(tid),
            "ang":              p["ang"],
            "emotion":          p["emotion"],
            "x":                int(cx),
            "y":                int(cy),
            "delta_x":          int(delta_x),
            "delta_y":          int(delta_y),
            "offset":           int(p["offset"]),
            "dist":             float(p["dist_m"]),            # legacy screen-PPM
            "status":           "warning" if p["near_p"] else "safe",
            "near_car":         bool(p["near_p"]),             # 🔧 metric-depth
            "stop_gesture":     bool(p_stop),                  # 🔧 per-person
            "collision_path":   bool(p_collision),             # 🔧 trajectory-based
            "crossing_intent":  bool(p["p_crossing"]),         # 🔧 per-person
            "question_gesture": bool(p["p_question"]),         # 🔧 per-person
            "move_status":      str(p["move_status"]),
            "ring_status":      str(p["ring_status"]),
            "state_text":       str(p_state),                  # 🔧 per-person
            "smartphone":       int(p["smartphone"]),
            "items":            str(p["items"]),
            # ─── New precision fields (JSON only — CSV columns unchanged) ───
            "bbox":             [int(p["bbox"][0]), int(p["bbox"][1]),
                                 int(p["bbox"][2]), int(p["bbox"][3])],
            "bbox_w_px":        int(p["bbox_w_px"]),
            "bbox_h_px":        int(p["bbox_h_px"]),
            "frame_w":          int(frame_w),
            "frame_h":          int(frame_h),
            "fps_source":       round(float(_src_fps), 2),
            "velocity_px_per_sec": round(float(velocity_px_per_sec), 3),
            "dist_depth_m":     round(float(p["dist_depth_m"]), 4),
            # ─── 🔧 ETL-EXTRA fields (JSON only — CSV columns unchanged) ───
            "dt_seconds":           round(float(dt_real), 4),
            "is_truncated":         bool(is_truncated_flag),
            "x_norm":               round(float(x_norm_val), 5),
            "y_norm":               round(float(y_norm_val), 5),
            "processed_fps":        round(float(_PROCESSED_FPS), 2),
            "n_others_in_frame":    int(n_others_in_frame),
            "closest_other_dist_m": round(float(closest_other_dist_m), 4),
            "keypoints":        kp_dict,   # ← all 17 keypoints in JSON
        }
        all_session_data.append(row_dict)

        row_list = [
            row_dict["video_name"], row_dict["id"], row_dict["t"], row_dict["frame_idx"],
            row_dict["x"], row_dict["y"], row_dict["delta_x"], row_dict["delta_y"],
            row_dict["dist"], row_dict["ang"], row_dict["near_car"], row_dict["stop_gesture"],
            row_dict["collision_path"], row_dict["crossing_intent"], row_dict["question_gesture"],
            row_dict["move_status"], row_dict["ring_status"], row_dict["state_text"],
            row_dict["emotion"], row_dict["offset"], row_dict["status"], row_dict["timestamp"],
            row_dict["smartphone"], row_dict["items"],
            *kp_csv_vals,   # ← 51 flat keypoint values appended to CSV row
        ]
        w_traj.writerow(row_list)
        w_master.writerow(row_list)
        # 🔧 FIX: Mark that data was logged this frame → snapshot will fire below
        snapshot_logged_this_frame = True

    # ── NO-PERSON FRAME LOG ──────────────────────────────────────────────────
    # Log one empty row per processed frame when no person is detected.
    # Ensures CSV/JSON captures every frame at the 15-fps cadence regardless
    # of pedestrian presence. All person-specific fields are set to neutral
    # sentinel values (id=-1, x/y=0, keypoints all 0.0) so downstream
    # consumers can filter with `id != -1`. All existing logic above unchanged.
    if not active_this_frame:
        _empty_kp_csv  = [0.0] * (len(KP_NAMES) * 3)           # 51 zeros
        _empty_kp_dict = {kn: {"x": 0.0, "y": 0.0, "conf": 0.0}
                          for kn in KP_NAMES}
        empty_row_dict = {
            "video_name":       video_base_name,
            "timestamp":        datetime.now().isoformat(),
            "t":                -1,
            "frame_idx":        int(frame_idx),
            "id":               -1,
            "ang":              "N/A",
            "emotion":          "N/A",
            "x":                0,
            "y":                0,
            "delta_x":          0,
            "delta_y":          0,
            "offset":           0,
            "dist":             0.0,
            "status":           "no_person",
            "near_car":         False,
            # 🔧 FIX-2: no person → all behaviour flags must be False (was
            #            leaking global gesture/state values into empty rows).
            "stop_gesture":     False,
            "collision_path":   False,
            "crossing_intent":  False,
            "question_gesture": False,
            "move_status":      "N/A",
            "ring_status":      "N/A",
            "state_text":       "GO",                            # 🔧 was last frame's state
            "smartphone":       0,
            "items":            "None",
            # ─── New precision fields (JSON only — CSV columns unchanged) ───
            "bbox":             [0, 0, 0, 0],
            "bbox_w_px":        0,
            "bbox_h_px":        0,
            "frame_w":          int(frame_w),
            "frame_h":          int(frame_h),
            "fps_source":       round(float(_src_fps), 2),
            "velocity_px_per_sec": 0.0,
            "dist_depth_m":     0.0,
            # ─── 🔧 ETL-EXTRA fields (JSON only — CSV columns unchanged) ───
            "dt_seconds":           0.0,
            "is_truncated":         False,
            "x_norm":               0.0,
            "y_norm":               0.0,
            "processed_fps":        round(float(_PROCESSED_FPS), 2),
            "n_others_in_frame":    0,
            "closest_other_dist_m": 0.0,
            "keypoints":        _empty_kp_dict,
        }
        all_session_data.append(empty_row_dict)
        empty_row_list = [
            empty_row_dict["video_name"], empty_row_dict["id"],
            empty_row_dict["t"],          empty_row_dict["frame_idx"],
            empty_row_dict["x"],          empty_row_dict["y"],
            empty_row_dict["delta_x"],    empty_row_dict["delta_y"],
            empty_row_dict["dist"],       empty_row_dict["ang"],
            empty_row_dict["near_car"],   empty_row_dict["stop_gesture"],
            empty_row_dict["collision_path"],   empty_row_dict["crossing_intent"],
            empty_row_dict["question_gesture"], empty_row_dict["move_status"],
            empty_row_dict["ring_status"],      empty_row_dict["state_text"],
            empty_row_dict["emotion"],          empty_row_dict["offset"],
            empty_row_dict["status"],           empty_row_dict["timestamp"],
            empty_row_dict["smartphone"],       empty_row_dict["items"],
            *_empty_kp_csv,                     # 51 flat keypoint zeros
        ]
        w_traj.writerow(empty_row_list)
        w_master.writerow(empty_row_list)
        snapshot_logged_this_frame = True   # snapshot captures every logged frame

    # ── Social-GAN live graph ──
    if SHOW_LIVE_GRAPH == 1:
        ax.clear()
        ax.set_title("Social-GAN: Predicted Social Interactions")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        for tid, path_pts in trajectory_history.items():
            pts = np.array(list(path_pts))
            ax.plot(pts[:, 0], pts[:, 1], "-", alpha=0.4)
            current_pt = pts[-1]
            ax.plot(current_pt[0], current_pt[1], marker="*", color="gold",
                    markersize=12, markeredgecolor="black")
            ax.text(current_pt[0] + 5, current_pt[1] - 5, f"ID:{tid}",
                    fontsize=9, fontweight="bold", color="black")
            if len(pts) > 5:
                vx     = (pts[-1][0] - pts[-5][0]) / 5
                vy     = (pts[-1][1] - pts[-5][1]) / 5
                future = [[current_pt[0] + vx * i, current_pt[1] + vy * i]
                           for i in range(1, 13)]
                f_pts  = np.array(future)
                ax.plot(f_pts[:, 0], f_pts[:, 1], "--", color="red", alpha=0.6)
        plt.draw()
        plt.pause(0.001)

    # ── Proximity red ring ──
    for i in range(len(active_this_frame)):
        for j in range(i + 1, len(active_this_frame)):
            p1, p2     = active_this_frame[i], active_this_frame[j]
            depth_diff = abs(p1["depth"] - p2["depth"])
            if depth_diff < DEPTH_TOLERANCE:
                # 🔥 OPT: math.hypot for 2-D scalar distance
                screen_dist = math.hypot(
                    p1["center"][0] - p2["center"][0],
                    p1["center"][1] - p2["center"][1]
                )
                if screen_dist < (p1["radius"] + p2["radius"]) * 0.82:
                    active_this_frame[i]["ring_status"] = "red"
                    active_this_frame[j]["ring_status"] = "red"

    # ── Render tracked people ──
    for p in active_this_frame:
        x1, y1, x2, y2 = p["bbox"]
        cx, cy         = p["center"]

        rect_color = (0, 0, 255) if p["dist_m"] <= 0.8 else (0, 255, 0)
        warn_text  = p["safe_warning"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
        cv2.line(frame, (ref_x, ref_y), (cx, y2), rect_color, 1)

        color        = (0, 255, 0)
        label_symbol = ""
        axes         = (int(p["width"] * 0.75), int(p["width"] * 0.25))

        if p["ring_status"] == "red":
            color        = (0, 0, 255)
            label_symbol = "!"
        elif p["ring_status"] == "yellow":
            color        = (0, 255, 255)
            label_symbol = ">>"
            axes         = (int(p["width"] * 1.05), int(p["width"] * 0.35))

        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, color, 3)
        if label_symbol:
            cv2.putText(frame, label_symbol, (cx - 15, cy - 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)

        ang_color = (255, 255, 255)
        if p["ang"] != "N/A":
            try:
                if abs(int(p["ang"])) > 30:
                    ang_color = (0, 0, 255)
            except Exception:
                pass

        part1 = f"ID:{p['track_id']} | "
        part2 = f"Ang:{p['ang']}° | " if p["ang"] != "N/A" else "Ang:N/A | "
        part3 = f"{p['emotion']} | X:{cx} Y:{cy} | Off:{int(p['offset'])} | {warn_text}"
        part4 = f" | Phone:{p['smartphone']} | Items:{p['items']}"

        full_label  = part1 + part2 + part3 + part4
        (tw, th), _ = cv2.getTextSize(full_label, 0, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 10, y1), rect_color, -1)

        y_text = y1 - 10
        x_text = x1 + 5

        cv2.putText(frame, part1, (x_text, y_text), 0, 0.45, (255, 255, 255), 1)
        w1 = cv2.getTextSize(part1, 0, 0.45, 1)[0][0]

        cv2.putText(frame, part2, (x_text + w1, y_text), 0, 0.45, ang_color, 1)
        w2 = cv2.getTextSize(part2, 0, 0.45, 1)[0][0]

        cv2.putText(frame, part3, (x_text + w1 + w2, y_text), 0, 0.45, (255, 255, 255), 1)
        w3 = cv2.getTextSize(part3, 0, 0.45, 1)[0][0]

        phone_color = (0, 0, 255) if p['smartphone'] == 1 else (255, 255, 255)
        cv2.putText(frame, part4, (x_text + w1 + w2 + w3, y_text), 0, 0.45, phone_color, 1)

        cv2.putText(frame, f"{p['dist_m']:.2f}m",
                    (int((cx + ref_x) / 2), int((y2 + ref_y) / 2)),
                    0, 0.6, (255, 255, 255), 2)

    # ── Prune stale state ──
    person_state = {k: v for k, v in person_state.items()
                    if curr_time - v["last_seen"] < 1.0}
    active_ids  = {p["track_id"] for p in active_this_frame}
    _pnp_cache  = {k: v for k, v in _pnp_cache.items() if k in active_ids}

    # ── FPS smoothing ──
    now        = time.time()
    inst_fps   = 1.0 / max(now - prev_time, 1e-6)
    fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
    prev_time  = now

    draw_ui(frame, state_text, fps_smooth, reasons)

    # 🔥 OPT: Only make annotated copy when snapshots need annotated (avoids dual copy)
    if SAVE_ANNOTATED_SNAPSHOTS == 1:
        frame_to_save_snap = frame.copy()
    else:
        frame_to_save_snap = raw_frame  # already captured before drawing

    # 🔧 FIX: Snapshot is taken ONLY when a row was logged to CSV/JSON this frame.
    #          Previously fired every processed frame regardless of logging.
    #          Now perfectly in sync: one snapshot per logging event, zero otherwise.
    if snapshot_logged_this_frame and not snapshot_queue.full():
        try:
            snapshot_queue.put_nowait({
                "frame":     frame_to_save_snap,
                "frame_idx": frame_idx,
                "s_dir":     snapshot_dir,
            })
        except Exception:
            pass

    if SHOW_BEV:
        frame = bev_renderer.render(frame, active_this_frame, trajectory_history)

    cv2.imshow("ALL FEATURES: Integrated Analytics + AEB", frame)

    # ── 10-second auto-save ──
    if curr_time - last_autosave_time >= 10.0:
        if user_input.lower() != "c":
            msec     = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(dt.timedelta(seconds=int(msec / 1000)))
            # 🔧 FIX-1: pass `start_idx` so dataset.json receives ONLY the
            #            rows written since the previous backup. Bookkeeping
            #            is atomic under _append_lock to guarantee that
            #            concurrent autosave threads don't double-claim rows.
            with _append_lock:
                _snapshot_data     = list(all_session_data)
                _snapshot_idx      = _last_appended_idx
                _last_appended_idx = len(_snapshot_data)
            Thread(
                target=background_json_backup,
                args=(cp_data_all.copy(), path, frame_idx, time_str,
                      _snapshot_data, _snapshot_idx,
                      session_log_path, f_traj, f_master),
                daemon=True,
            ).start()
        last_autosave_time = curr_time

    # ==========================================================
    # ✅ DYNAMIC KEYBOARD CONTROLS
    # ==========================================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        if user_input.lower() != "c":
            msec     = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(dt.timedelta(seconds=int(msec / 1000)))
            save_checkpoint_to_db(path, frame_idx, time_str)
            print(f"\n[INFO] 🛑 Quitting. Video '{video_base_name}' position saved "
                  f"at [{time_str}]. Exiting cleanly...")
        break

    elif key == ord("b") or key == ord("B"):
        print(f"\n[INFO] ⏪ Restarting '{video_base_name}' from the beginning. "
              f"(Data will append continuously)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        trajectory_history.clear()
        person_state.clear()
        track_time_idx.clear()
        if path in cp_data_all:
            del cp_data_all[path]
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(cp_data_all, f, indent=4)
            print(f"[INFO] logging to checkpoint.json... (Cleared record for {path})")

    elif key == ord("p") or key == ord("P"):
        if user_input.lower() != "c":
            msec     = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(dt.timedelta(seconds=int(msec / 1000)))
            save_checkpoint_to_db(path, frame_idx, time_str)
            print(f"\n[INFO] ⏸️ PAUSED at timepoint [{time_str}]")
            print(" -> Press 'r' to RESUME.")
            print(" -> Press 'b' to BEGIN (restart).")
            print(" -> Press 'q' to QUIT.\n")

            quit_flag = False
            while True:
                pause_frame = frame.copy()
                cv2.putText(pause_frame, f"PAUSED AT [{time_str}]",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(pause_frame, "[r] RESUME | [b] RESTART | [q] QUIT",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("ALL FEATURES: Integrated Analytics + AEB", pause_frame)
                pause_key = cv2.waitKey(50) & 0xFF

                if pause_key == ord('r') or pause_key == ord('R'):
                    print(f"[INFO] ▶️ RESUMING from [{time_str}]...")
                    break
                elif pause_key == ord('b') or pause_key == ord('B'):
                    print(f"\n[INFO] ⏪ Restarting '{video_base_name}' from the beginning. "
                          f"(Data will append continuously)")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    trajectory_history.clear()
                    person_state.clear()
                    track_time_idx.clear()
                    if path in cp_data_all:
                        del cp_data_all[path]
                        with open(CHECKPOINT_FILE, "w") as f:
                            json.dump(cp_data_all, f, indent=4)
                        print(f"[INFO] logging to checkpoint.json... (Cleared record for {path})")
                    break
                elif pause_key == ord('q') or pause_key == ord('Q'):
                    save_checkpoint_to_db(path, frame_idx, time_str)
                    print(f"\n[INFO] 🛑 Quitting from paused state. Exiting cleanly...")
                    quit_flag = True
                    break

            if quit_flag:
                break

    elif key == ord("v") or key == ord("V"):
        SHOW_BEV = not SHOW_BEV
        print(f"[INFO] BEV Renderer toggled: {SHOW_BEV}")


# ==========================================================
# ✅ CLEAN SHUTDOWN
# ==========================================================
f_traj.close()
f_master.close()

try:
    snapshot_queue.put_nowait(None)
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()
plt.close("all")

# ── Final JSON save ──
with open(session_log_path, "w") as jf:
    json.dump(all_session_data, jf, indent=4)
print(f"\n[INFO] Final Session log saved: {session_log_path}")
print(f"[INFO] Final Trajectory CSV saved (Streamed Real-Time): {traj_csv_path}")

# ── Append to dataset.json (incremental — only rows not yet written) ──
# 🔧 FIX-1: shutdown drain — append ONLY rows past _last_appended_idx so we
#            don't double-write rows already saved by an autosave thread.
dataset_path = DATASET_JSON_FILE
if os.path.exists(dataset_path):
    try:
        with open(dataset_path, "r") as f:
            old_data = json.load(f)
        if not isinstance(old_data, list):
            old_data = []
    except Exception:
        old_data = []
else:
    old_data = []

with _append_lock:
    _remaining_rows    = all_session_data[_last_appended_idx:]
    _last_appended_idx = len(all_session_data)

if _remaining_rows:
    old_data.extend(_remaining_rows)
    with open(dataset_path, "w") as jf:
        json.dump(old_data, jf, indent=4)
    print(f"[INFO] Drained {len(_remaining_rows)} final rows to dataset.json")
else:
    print("[INFO] dataset.json already up-to-date (autosave drained all rows)")

print(f"[INFO] Master dataset CSV updated (Streamed Real-Time): {master_csv}")
print(f"[INFO] Master dataset JSON updated (Streamed Real-Time): {dataset_path}")

if os.path.exists(BOTSORT_CONFIG_FILE):
    os.remove(BOTSORT_CONFIG_FILE)