import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")

def speech_to_text(audio_path):

    model = genai.GenerativeModel("gemini-1.5-pro")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = model.generate_content([
        {"mime_type": "audio/wav", "data": audio_bytes},
        "Convert this speech to text"
    ])

    return response.text.strip()