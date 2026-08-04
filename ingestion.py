"""
Stage 1: Dual-Stream Visual & Audio Encoding (ingestion.py)

This module processes an input .mp4 video through two concurrent streams:
  - Stream A (Visuals): OpenCV frame extraction at 3-5 FPS + YOLO/YOLO-World object detection
    to catch split-second transient visual events.
  - Stream B (Audio): OpenAI Whisper audio transcription with segment-level timestamps.
  - Synchronization: Aligns visual detections and spoken transcript into a single timeline JSON.
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def log_to_terminal(msg: str):
    """Prints log messages directly to standard stdout and forces physical terminal output via sys.__stdout__."""
    print(msg, flush=True)
    if hasattr(sys, "__stdout__") and sys.__stdout__ is not None:
        try:
            sys.__stdout__.write(str(msg) + "\n")
            sys.__stdout__.flush()
        except Exception:
            pass

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
    from shazamio import Shazam
except ImportError:
    Shazam = None

try:
    from scenedetect import detect, AdaptiveDetector
except ImportError:
    detect, AdaptiveDetector = None, None

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig
    import torch
    from PIL import Image as PILImage
    setattr(PretrainedConfig, "forced_bos_token_id", None)
    setattr(PretrainedConfig, "forced_eos_token_id", None)
    if hasattr(AutoModelForCausalLM, "_supports_sdpa"):
        pass
    else:
        setattr(AutoModelForCausalLM, "_supports_sdpa", True)
except ImportError:
    AutoProcessor = None
    AutoModelForCausalLM = None
    PretrainedConfig = None
    torch = None
    PILImage = None


def recognize_background_song(video_path: str) -> Optional[Dict[str, str]]:
    """Uses Shazam acoustic fingerprinting to identify background music in the video."""
    if Shazam is None:
        return None
    
    import asyncio
    async def _shazam_task():
        try:
            print(f"[Stream B] Running Shazam acoustic song recognition on {video_path}...")
            shazam = Shazam()
            out = await shazam.recognize(video_path)
            track = out.get("track", {})
            if track:
                title = track.get("title", "")
                artist = track.get("subtitle", "")
                genre = track.get("genres", {}).get("primary", "Music")
                if title and artist:
                    print(f"[Shazam Recognized] Song: '{title}' by {artist} ({genre})")
                    return {"title": title, "artist": artist, "genre": genre}
        except Exception as e:
            print(f"[Shazam Warning] Song recognition failed: {e}")
        return None

    try:
        return asyncio.run(_shazam_task())
    except Exception:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_shazam_task())
        except Exception:
            return None


def format_timestamp(seconds: float) -> str:
    """Format seconds float into MM:SS string representation."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def generate_dense_caption(image_path: str) -> str:
    """Generates a rich, highly-detailed dense text caption for a local image using Microsoft Florence-2 (or BLIP fallback)."""
    if PILImage is None or torch is None:
        log_to_terminal("[VLM Warning] torch or PIL packages are required for VLM dense captioning.")
        return ""

    if not os.path.exists(image_path):
        return ""

    global _florence_model, _florence_processor, _florence_device
    if "_florence_model" not in globals():
        try:
            import transformers.models.roberta.tokenization_roberta as roberta_tok
            if not hasattr(roberta_tok.RobertaTokenizer, "additional_special_tokens"):
                setattr(roberta_tok.RobertaTokenizer, "additional_special_tokens", property(lambda self: []))
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        florence_loaded = False
        for fl_model_name in ["multimodalart/Florence-2-large-no-flash-attn", "microsoft/Florence-2-large", "microsoft/Florence-2-base"]:
            log_to_terminal(f"[VLM Ingestion] Attempting to load Florence-2 model ({fl_model_name}) on {device}...")
            try:
                processor = AutoProcessor.from_pretrained(fl_model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(fl_model_name, trust_remote_code=True).to(device)
                globals()["_florence_processor"] = processor
                globals()["_florence_model"] = model
                globals()["_florence_device"] = device
                florence_loaded = True
                log_to_terminal(f"[VLM Ingestion] Successfully loaded model '{fl_model_name}' on {device}!")
                break
            except Exception as err:
                log_to_terminal(f"[VLM Warning] {fl_model_name} loading failed: {err}")

        if not florence_loaded:
            log_to_terminal("[VLM Warning] Florence-2 initialization failed. Trying BLIP fallback...")
            globals()["_florence_model"] = None

    device = globals().get("_florence_device", "cuda" if torch.cuda.is_available() else "cpu")
    processor = globals().get("_florence_processor")
    model = globals().get("_florence_model")

    image = PILImage.open(image_path).convert("RGB")

    if model is not None and processor is not None:
        try:
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
            if isinstance(parsed, dict) and "<MORE_DETAILED_CAPTION>" in parsed:
                caption = parsed["<MORE_DETAILED_CAPTION>"].strip()
            else:
                caption = str(generated_text).replace("<MORE_DETAILED_CAPTION>", "").strip()
            return caption
        except Exception as err:
            log_to_terminal(f"[VLM Warning] Florence-2 inference failed for '{image_path}': {err}")

    # Rock-solid BLIP VLM fallback
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        global _blip_model, _blip_processor
        if "_blip_model" not in globals():
            log_to_terminal("[VLM Ingestion] Loading BLIP model (Salesforce/blip-image-captioning-base)...")
            globals()["_blip_processor"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            globals()["_blip_model"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        b_proc = globals()["_blip_processor"]
        b_mod = globals()["_blip_model"]
        inputs = b_proc(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = b_mod.generate(**inputs, max_new_tokens=100)
        caption = b_proc.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as b_err:
        log_to_terminal(f"[VLM Warning] Fallback BLIP captioning failed for '{image_path}': {b_err}")
        return ""





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

    # Clean up old saved frames from previous ingestions to guarantee fresh frame extraction for new uploads
    frames_dir = "./saved_frames"
    if os.path.exists(frames_dir):
        for file in os.listdir(frames_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                    os.remove(os.path.join(frames_dir, file))
                except Exception:
                    pass
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    print(f"[Stream A] Video metadata: {video_fps:.2f} FPS, {total_frames} frames, {duration_sec:.2f}s duration.")

    # Pillar 1: PySceneDetect Dynamic Scene Boundary Detection
    scenes_list = []
    if detect is not None and AdaptiveDetector is not None:
        log_to_terminal("[Stream A] Running PySceneDetect (AdaptiveDetector) for semantic scene boundaries...")
        try:
            detected_scenes = detect(video_path, AdaptiveDetector())
            for sc in detected_scenes:
                s_start = round(sc[0].seconds, 2)
                s_end = round(sc[1].seconds, 2)
                # Safeguard: if scene > 10.0s, split into 10-second sub-chunks
                if s_end - s_start > 10.0:
                    sub_start = s_start
                    while sub_start < s_end:
                        sub_end = min(round(sub_start + 10.0, 2), s_end)
                        if sub_end - sub_start >= 0.5:
                            scenes_list.append((sub_start, sub_end))
                        sub_start = sub_end
                elif s_end - s_start >= 0.5:
                    scenes_list.append((s_start, s_end))
            log_to_terminal(f"[Stream A] PySceneDetect found {len(scenes_list)} semantic scene chunks.")
        except Exception as e:
            log_to_terminal(f"[Stream A Warning] PySceneDetect failed ({e}). Falling back to visual state tracking.")

    visual_chunks: List[Dict[str, Any]] = []

    if scenes_list:
        # Process PySceneDetect scenes directly via Midpoint Keyframe Extraction
        for sc_idx, (sc_start, sc_end) in enumerate(scenes_list):
            midpoint_sec = round((sc_start + sc_end) / 2.0, 2)
            frame_num = int(round(midpoint_sec * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            frame_filename = f"frame_{int(midpoint_sec)}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            detected_objects = set()
            detected_person_ids = set()

            if ret and frame is not None:
                cv2.imwrite(frame_path, frame)
                try:
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, verbose=False, device="cpu")
                except Exception:
                    results = model.predict(frame, conf=conf_threshold, verbose=False, device="cpu")

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        detected_objects.add(class_name)
                        if class_name.lower() == "person" or cls_id == 0:
                            if hasattr(box, "id") and box.id is not None:
                                track_id = int(box.id[0])
                                detected_person_ids.add(f"Person_{track_id}")
                            else:
                                detected_person_ids.add("Person_1")
            
            # Run Dense VLM Captioning (Florence-2 / BLIP) on preserved keyframe
            dense_caption = generate_dense_caption(frame_path)

            visual_chunks.append({
                "start_sec": sc_start,
                "end_sec": sc_end,
                "timestamp_formatted": f"{format_timestamp(sc_start)} - {format_timestamp(sc_end)}",
                "objects": sorted(list(detected_objects)),
                "detected_person_ids": sorted(list(detected_person_ids)),
                "image_path": f"./saved_frames/{frame_filename}",
                "dense_caption": dense_caption
            })
    else:
        # Fallback to YOLO visual state tracking
        frame_step = max(1, int(round(video_fps / target_fps)))
        max_chunk_sec = 10.0 # Safeguard max chunk duration
        current_chunk = {"start": 0.0, "objects": set(), "frames": []}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                timestamp_sec = round(frame_idx / video_fps, 2)
                timestamp_fmt = format_timestamp(timestamp_sec)

                results = model.predict(frame, conf=conf_threshold, verbose=False, device="cpu")
                detected_objects = set()
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        detected_objects.add(model.names[cls_id])

                frame_filename = f"frame_{int(timestamp_sec)}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "timestamp_formatted": timestamp_fmt,
                    "objects": detected_objects,
                    "image_path": f"./saved_frames/{frame_filename}",
                    "num_objects": len(detected_objects)
                }

                objects_changed = (detected_objects != current_chunk["objects"]) and (len(current_chunk["objects"]) > 0)
                duration_exceeded = (timestamp_sec - current_chunk["start"]) >= max_chunk_sec

                if objects_changed or duration_exceeded:
                    chunk_start = current_chunk["start"]
                    chunk_end = timestamp_sec
                    chunk_frames = current_chunk["frames"]
                    midpoint_sec = (chunk_start + chunk_end) / 2.0
                    best_frame = min(
                        chunk_frames,
                        key=lambda f: (-f["num_objects"], abs(f["timestamp_sec"] - midpoint_sec))
                    ) if chunk_frames else frame_data

                    dense_caption = generate_dense_caption(best_frame["image_path"])

                    visual_chunks.append({
                        "start_sec": chunk_start,
                        "end_sec": chunk_end,
                        "timestamp_formatted": f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}",
                        "objects": sorted(list(current_chunk["objects"])),
                        "image_path": best_frame["image_path"],
                        "dense_caption": dense_caption
                    })

                    current_chunk = {"start": timestamp_sec, "objects": set(detected_objects), "frames": [frame_data]}
                else:
                    current_chunk["objects"].update(detected_objects)
                    current_chunk["frames"].append(frame_data)

            frame_idx += 1

        if current_chunk["frames"]:
            chunk_start = current_chunk["start"]
            chunk_end = duration_sec if duration_sec > chunk_start else chunk_start + 1.0
            chunk_frames = current_chunk["frames"]
            midpoint_sec = (chunk_start + chunk_end) / 2.0
            best_frame = min(chunk_frames, key=lambda f: (-f["num_objects"], abs(f["timestamp_sec"] - midpoint_sec)))
            dense_caption = generate_dense_caption(best_frame["image_path"])
            visual_chunks.append({
                "start_sec": chunk_start,
                "end_sec": chunk_end,
                "timestamp_formatted": f"{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}",
                "objects": sorted(list(current_chunk["objects"])),
                "image_path": best_frame["image_path"],
                "dense_caption": dense_caption
            })

    cap.release()
    log_to_terminal(f"[Stream A] Created {len(visual_chunks)} dynamic semantic chunks.")
    return visual_chunks


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

    # Run Shazam acoustic song recognition
    song_info = recognize_background_song(video_path)
    if song_info:
        transcript_segments.append({
            "start_sec": 0.0,
            "end_sec": 5.0,
            "timestamp_formatted": "00:00 - 00:05",
            "text": f"[Background Music Track]: '{song_info['title']}' by {song_info['artist']} ({song_info['genre']})"
        })

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
    visual_chunks: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge Visual Stream A dynamic event chunks and Audio Stream B transcript segments into a unified timeline.
    """
    log_to_terminal(f"[Merge] Synchronizing {len(visual_chunks)} dynamic visual event chunks with audio transcripts...")

    timeline: List[Dict[str, Any]] = []

    for v_chunk in visual_chunks:
        c_start = v_chunk["start_sec"]
        c_end = v_chunk["end_sec"]

        # Collect spoken text overlapping with dynamic window [c_start, c_end]
        window_transcripts = []
        for seg in transcript_segments:
            if not (seg["end_sec"] <= c_start or seg["start_sec"] >= c_end):
                window_transcripts.append(seg["text"])

        timeline.append({
            "window_start": c_start,
            "window_end": c_end,
            "timestamp_formatted": v_chunk["timestamp_formatted"],
            "visual_objects": v_chunk["objects"],
            "detected_person_ids": v_chunk.get("detected_person_ids", []),
            "transcript_text": " ".join(window_transcripts).strip(),
            "image_path": v_chunk["image_path"],
            "dense_caption": v_chunk.get("dense_caption", "")
        })

    log_to_terminal(f"[Merge] Created synchronized timeline with {len(timeline)} dynamic semantic time windows.")
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
        visual_chunks=visual_logs,
        transcript_segments=transcript_segments
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
