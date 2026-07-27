"""
Stage 1: Dual-Stream Visual & Audio Encoding (ingestion.py)

This module processes an input .mp4 video through two concurrent streams:
  - Stream A (Visuals): OpenCV frame extraction at 3-5 FPS + YOLO/YOLO-World object detection
    to catch split-second transient visual events.
  - Stream B (Audio): OpenAI Whisper audio transcription with segment-level timestamps.
  - Synchronization: Aligns visual detections and spoken transcript into a single timeline JSON.
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image as PILImage
except ImportError:
    CLIPProcessor = None
    CLIPModel = None
    torch = None
    PILImage = None



def format_timestamp(seconds: float) -> str:
    """Format seconds float into MM:SS string representation."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def generate_clip_image_embedding(image_path: str) -> List[float]:
    """Generates a 512-dimensional normalized visual vector embedding for a local image path using CLIP."""
    if CLIPModel is None or CLIPProcessor is None or torch is None or PILImage is None:
        raise ImportError("transformers and torch packages are required for CLIP image embedding extraction.")
    
    global _clip_model, _clip_processor, _clip_device
    if "_clip_model" not in globals():
        print("[Embedder] Loading CLIP model (openai/clip-vit-base-patch32)...")
        globals()["_clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        globals()["_clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        globals()["_clip_model"] = globals()["_clip_model"].to(device)
        globals()["_clip_device"] = device
    
    device = globals()["_clip_device"]
    processor = globals()["_clip_processor"]
    model = globals()["_clip_model"]

    image = PILImage.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # L2 normalize
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()


def process_stream_a_visuals(
    video_path: str,
    target_fps: float = 4.0,
    yolo_model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.35,
    custom_classes: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Stream A: Extract frames at target_fps using OpenCV, run object detection using YOLO/YOLO-World,
    and return timestamped visual detection logs.
    """
    if cv2 is None or YOLO is None:
        raise ImportError("opencv-python and ultralytics packages are required for Stream A visual processing. Run `pip install opencv-python ultralytics`.")

    print(f"[Stream A] Loading YOLO model: {yolo_model_name}...")
    model = YOLO(yolo_model_name)

    # Enable open-vocabulary mode if custom classes are provided and supported by model
    if custom_classes and hasattr(model, "set_classes"):
        print(f"[Stream A] Setting open-vocabulary target classes: {custom_classes}")
        model.set_classes(custom_classes)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    print(f"[Stream A] Video metadata: {video_fps:.2f} FPS, {total_frames} frames, {duration_sec:.2f}s duration.")

    # Calculate frame step interval to sample at target_fps
    frame_step = max(1, int(round(video_fps / target_fps)))
    
    visual_logs: List[Dict[str, Any]] = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp_sec = round(frame_idx / video_fps, 2)
            timestamp_fmt = format_timestamp(timestamp_sec)

            # Run inference
            results = model.predict(frame, conf=conf_threshold, verbose=False, device="cpu")
            detected_objects = set()

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    detected_objects.add(class_name)

            # Save representative frame to ./saved_frames/
            os.makedirs("./saved_frames", exist_ok=True)
            frame_filename = f"frame_{int(timestamp_sec)}.jpg"
            frame_path = os.path.join("./saved_frames", frame_filename)
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)

            # Generate CLIP vector embedding from the saved image
            try:
                clip_emb = generate_clip_image_embedding(frame_path)
            except Exception as emb_err:
                print(f"[Embedder Warning] Failed to compute CLIP embedding for '{frame_path}': {emb_err}")
                clip_emb = None

            visual_logs.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "timestamp_formatted": timestamp_fmt,
                "objects": sorted(list(detected_objects)),
                "image_path": f"./saved_frames/{frame_filename}",
                "clip_embedding": clip_emb
            })

        frame_idx += 1

    cap.release()
    print(f"[Stream A] Extracted {len(visual_logs)} visual detection frames across video.")
    return visual_logs


def process_stream_b_audio(
    video_path: str,
    whisper_model_name: str = "base"
) -> List[Dict[str, Any]]:
    """
    Stream B: Transcribe audio track from video file using OpenAI Whisper, returning timestamped transcript segments.
    """
    import shutil
    if whisper is None:
        print("[Stream B Warning] 'openai-whisper' package is not installed. Skipping audio stream transcription.")
        return []

    if not shutil.which("ffmpeg"):
        print("[Stream B Warning] 'ffmpeg' executable was not found on your system PATH.")
        print("[Stream B Warning] Audio transcription via Whisper requires 'ffmpeg'. Skipping audio transcription.")
        return []

    try:
        print(f"[Stream B] Loading Whisper audio model: {whisper_model_name}...")
        audio_model = whisper.load_model(whisper_model_name)
        
        print(f"[Stream B] Transcribing audio from {video_path}...")
        result = audio_model.transcribe(video_path)
    except Exception as e:
        print(f"[Stream B Warning] Failed to transcribe audio via Whisper: {e}")
        print("[Stream B Warning] Proceeding with visual stream data only.")
        return []

    transcript_segments: List[Dict[str, Any]] = []
    for segment in result.get("segments", []):
        start_sec = round(segment["start"], 2)
        end_sec = round(segment["end"], 2)
        text = segment["text"].strip()
        
        if text:
            transcript_segments.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "timestamp_formatted": f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}",
                "text": text
            })

    print(f"[Stream B] Extracted {len(transcript_segments)} spoken audio segments.")
    return transcript_segments


def synchronize_streams(
    visual_logs: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    granularity_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Merge Visual Stream A and Audio Stream B into a unified chronological timeline list of time windows.
    """
    print(f"[Merge] Synchronizing visual logs and audio transcript at {granularity_sec}s granularity...")
    
    max_time_sec = 0.0
    if visual_logs:
        max_time_sec = max(max_time_sec, visual_logs[-1]["timestamp_sec"])
    if transcript_segments:
        max_time_sec = max(max_time_sec, transcript_segments[-1]["end_sec"])

    timeline: List[Dict[str, Any]] = []
    curr_time = 0.0

    while curr_time <= max_time_sec:
        window_end = round(curr_time + granularity_sec, 2)
        
        # Collect objects detected within [curr_time, window_end]
        window_objects = set()
        window_image_path = None
        best_log = None
        for log in visual_logs:
            if curr_time <= log["timestamp_sec"] < window_end:
                window_objects.update(log["objects"])
                if log.get("image_path"):
                    if best_log is None or len(log["objects"]) > len(best_log["objects"]):
                        best_log = log

        window_clip_embedding = None
        if best_log:
            window_image_path = best_log["image_path"]
            window_clip_embedding = best_log.get("clip_embedding")

        # Collect spoken text overlapping with [curr_time, window_end]
        window_transcripts = []
        for seg in transcript_segments:
            if not (seg["end_sec"] <= curr_time or seg["start_sec"] >= window_end):
                window_transcripts.append(seg["text"])

        timeline.append({
            "window_start": curr_time,
            "window_end": window_end,
            "timestamp_formatted": f"{format_timestamp(curr_time)} - {format_timestamp(window_end)}",
            "visual_objects": sorted(list(window_objects)),
            "transcript_text": " ".join(window_transcripts).strip(),
            "image_path": window_image_path,
            "clip_embedding": window_clip_embedding
        })

        curr_time = window_end

    print(f"[Merge] Created synchronized timeline with {len(timeline)} time windows.")
    return timeline


def run_ingestion(
    video_path: str,
    output_json: str = "timeline.json",
    target_fps: float = 4.0,
    yolo_model: str = "yolo11n.pt",
    whisper_model: str = "base",
    conf_threshold: float = 0.35,
    custom_classes: Optional[List[str]] = None,
    granularity_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """End-to-end execution of Stage 1 Dual-Stream Ingestion."""
    print("=" * 60)
    print("STAGE 1: DUAL-STREAM VISUAL & AUDIO INGESTION")
    print("=" * 60)

    visual_logs = process_stream_a_visuals(
        video_path=video_path,
        target_fps=target_fps,
        yolo_model_name=yolo_model,
        conf_threshold=conf_threshold,
        custom_classes=custom_classes
    )

    transcript_segments = process_stream_b_audio(
        video_path=video_path,
        whisper_model_name=whisper_model
    )

    timeline = synchronize_streams(
        visual_logs=visual_logs,
        transcript_segments=transcript_segments,
        granularity_sec=granularity_sec
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    print(f"[Stage 1 Complete] Timeline saved to {output_json}")
    return timeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Dual-Stream Video Ingestion (Visual + Audio)")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 video file")
    parser.add_argument("--output_json", type=str, default="timeline.json", help="Path to save output timeline JSON")
    parser.add_argument("--fps", type=float, default=4.0, help="Visual frame sampling rate (FPS)")
    parser.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model (yolo11n.pt, yolov8n.pt, yolov8s-world.pt)")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper audio model size (tiny, base, small, medium)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence threshold")
    parser.add_argument("--custom_classes", nargs="*", help="Optional custom target classes for open-vocabulary detection")
    parser.add_argument("--granularity", type=float, default=5.0, help="Time window granularity in seconds for merged timeline")

    args = parser.parse_args()

    run_ingestion(
        video_path=args.video_path,
        output_json=args.output_json,
        target_fps=args.fps,
        yolo_model=args.yolo_model,
        whisper_model=args.whisper_model,
        conf_threshold=args.conf,
        custom_classes=args.custom_classes,
        granularity_sec=args.granularity
    )
