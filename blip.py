import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_size="large"
model_name = f"Salesforce/blip-image-captioning-{model_size}"
processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(model_name)

def caption_frame(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP image captioning")
    parser.add_argument("--model_size", choices=["base", "large"], default="base", help="Choose BLIP model size")
    parser.add_argument("--image_path", type=str, default="landslide.webp", help="Path to input image")

    args = parser.parse_args()

    raw_image = Image.open(args.image_path).convert("RGB")
    scene_descriptions = caption_frame(raw_image)

    print("\n=== Scene Breakdown ===\n")
    print(scene_descriptions)
