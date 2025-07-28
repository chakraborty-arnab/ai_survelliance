import cv2
from PIL import Image
import numpy as np
import llm
import blip
import pandas as pd

# ==== CONFIGURATION ====
VIDEO_PATH = "assets/handshake.mp4"
FRAME_INTERVAL = 1

def extract_key_frames(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    key_frames = []
    for t in np.arange(0, duration, interval_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        key_frames.append((int(t), img))

    cap.release()
    return key_frames

def summarize_video(video_path):
    print("[INFO] Extracting key frames...")
    frames = extract_key_frames(video_path, interval_sec=FRAME_INTERVAL)

    summary = []
    last_description = None

    for timestamp, img in frames:
        print(f"[INFO] Analyzing frame at {timestamp}s...")
        description = blip.caption_frame(img)
        if description != last_description:
            print("[INFO] Description changed — running LLM...")
            llm_description = llm.describe_frame_with_gpt(img, timestamp)
            summary.append(f"[{timestamp}s] {llm_description}")
            last_description = description
        else:
            print("[INFO] Skipping LLM — description unchanged.")

    return "\n".join(summary)

if __name__ == "__main__":
    df = pd.read_csv("recognized_faces.csv")
    df = df[df['Face Name'] != 'Unknown']
    identified_people = df['Face Name'].unique()
    identified_people_str = "\n".join(identified_people)
    scene_descriptions = summarize_video(VIDEO_PATH)
    final_summary = llm.generate_overall_summary(scene_descriptions, identified_people_str)

    full_text = f"=== Final Summary ===\n{final_summary}\n\n=== Scene Descriptions ===\n{scene_descriptions}"
    with open("scene_descriptions.txt", "w") as f:
        f.write(full_text)

    # Optionally print
    print("\n=== Final Summary ===\n")
    print(final_summary)