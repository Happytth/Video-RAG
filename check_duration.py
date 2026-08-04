import cv2

video_path = "/mnt/c/Users/SOUBHAGYA NAYAK/Downloads/He saw it Clark believe me -Superman Edit -Kendrick Lamar, SZA - All The Stars  #superman #dc - Keshav AE (1080p, h264).mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = count / fps if fps > 0 else 0
print(f"Video metadata -> FPS: {fps:.2f}, Total Frames: {count}, Duration: {duration:.2f} seconds")
cap.release()
