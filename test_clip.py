import sys
import os
import ingestion

img_path = "./saved_frames/frame_7.jpg"
if not os.path.exists(img_path):
    print(f"File {img_path} does not exist!")
else:
    emb = ingestion.generate_clip_image_embedding(img_path)
    print("CLIP Embedding Length:", len(emb))
    print("First 5 values:", emb[:5])
