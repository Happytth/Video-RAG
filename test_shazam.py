import asyncio
import json
from shazamio import Shazam

async def main():
    shazam = Shazam()
    video_path = "/mnt/c/Users/SOUBHAGYA NAYAK/Downloads/He saw it Clark believe me -Superman Edit -Kendrick Lamar, SZA - All The Stars  #superman #dc - Keshav AE (1080p, h264).mp4"
    print("Recognizing audio track via Shazam...")
    out = await shazam.recognize(video_path)
    track = out.get("track", {})
    if track:
        print("Title:", track.get("title"))
        print("Subtitle (Artist):", track.get("subtitle"))
        print("Genres:", track.get("genres", {}).get("primary"))
    else:
        print("No Shazam match found.")

if __name__ == "__main__":
    asyncio.run(main())
