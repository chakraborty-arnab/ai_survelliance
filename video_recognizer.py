import cv2
import os
import csv
from recognize import init, recognize, label_image

# Parameters
VIDEO_PATH = "assets/handshake.mp4"  
FRAME_INTERVAL = 1               
OUTPUT_DIR = "processed_frames"
CSV_PATH = "recognized_faces.csv"

def process_video(video_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    known_face_encodings, known_face_names = init()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Timestamp (s)", "Face Name", "Top", "Right", "Bottom", "Left"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                print(f"[INFO] Processing frame {frame_count}")
                try:
                    face_names, face_locations = recognize(frame, known_face_encodings, known_face_names)
                    if face_names:
                        labeled_frame = label_image(frame.copy(), face_locations, face_names)
                        # output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.jpg")
                        # cv2.imwrite(output_path, labeled_frame)

                        timestamp = frame_count / fps
                        for name, (top, right, bottom, left) in zip(face_names, face_locations):
                            writer.writerow([frame_count, round(timestamp, 2), name, top, right, bottom, left])
                except Exception as e:
                    print(f"[ERROR] Failed to process frame {frame_count}: {e}")

            frame_count += 1

    cap.release()
    print(f"[INFO] Processing complete. Results saved to {CSV_PATH} and {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_video(VIDEO_PATH)


