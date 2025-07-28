from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io
import base64

# ==== CONFIGURATION ====
MODEL = "gpt-4o-mini"

# ==== INIT ====
load_dotenv()
client = OpenAI()

def pil_image_to_base64(img: Image.Image):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def describe_frame_with_gpt(img, timestamp):
    try:
        image_b64 = pil_image_to_base64(img)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes events and interactions between people in video frames."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"What is happening at {timestamp} seconds in this video frame?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing frame at {timestamp}s: {e}"
def generate_overall_summary(scene_descriptions, df_string):
    print("[INFO] Generating overall summary...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video interaction analyst. "
                    "Your task is to generate a brief and focused summary from scene-level outputs of a video analysis system. "
                    "The system includes face recognition and frame captioning. "
                    "Prioritize recognized individuals, key actions, and any interactions or notable events. "
                )
            },
            {
                "role": "user",
                "content": (
                    "Below are frame-wise scene descriptions. Some frames include enhanced descriptions where changes were detected:\n\n"
                    f"{scene_descriptions}\n\n"
                    "List of people recognized in the video:\n"
                    f"{df_string}\n\n"
                    "Please write a concise and coherent summary of the events, individuals involved, and their interactions."
                )
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

