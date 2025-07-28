# AI Surveillance

## 🚀 Features

✅ Face Detection & Recognition
Detect and recognize faces in video frames using face_recognition, and generate structured logs.

✅ Real-time Image Captioning
Generate frame-level captions using BLIP (Salesforce/blip-image-captioning-large).

✅ Visual LLM Enhancement
Automatically invoke a vision-language model (e.g., GPT-4o) to describe a frame only when the caption differs from the previous one.

✅ Smart Scene Summarization
Summarize key frames and face recognition logs using GPT-4o or any compatible vision-enabled LLM for a concise overview of the video.

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/chakraborty-arnab/ai-survelliance.git
cd ai_survelliance
```

### 2. Create and Activate Conda Environment
```bash
conda create -n video_analytics python=3.13 -y
conda activate video_analytics
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add OpenAI API
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the following scripts (It uses handshake.mp4 from assets by default)
```bash
python video_recognizer.py
python interaction_summarizer.py
```

This process will generate the following output files:
* recognized_faces.csv – Contains a list of identified individuals along with relevant metadata.
* scene_descriptions.txt – Includes detailed scene descriptions and the final summary of video content.

