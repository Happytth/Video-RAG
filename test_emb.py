from ingestion import generate_clip_image_embedding
import os

path = "saved_frames/frame_7.jpg"
print("Exists:", os.path.exists(path))
if os.path.exists(path):
    try:
        emb = generate_clip_image_embedding(path)
        print("Embedding length:", len(emb) if emb else "None")
    except Exception as e:
        print("Error:", e)
