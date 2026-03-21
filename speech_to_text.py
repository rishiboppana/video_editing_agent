# import google.generativeai as genai

# genai.configure(api_key="AIzaSyACoa3A8hr2SpBiQOfp6YtdAf-7gUCFasE")

# def speech_to_text(audio_path):

#     model = genai.GenerativeModel("gemini-2.5-flash")

#     with open(audio_path, "rb") as f:
#         audio_bytes = f.read()

#     response = model.generate_content([
#         {"mime_type": "audio/wav", "data": audio_bytes},
#         "Convert this speech to text"
#     ])

#     return response.text.strip()


import whisper

model = whisper.load_model("base")

def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]