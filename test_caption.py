import sys
import ingestion

caption = ingestion.generate_dense_caption("./saved_frames/frame_7.jpg")
print("CAPTION RESULT:")
print(caption)
