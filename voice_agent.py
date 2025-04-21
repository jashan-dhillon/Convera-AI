# import whisper
# import pyttsx3
# import json
# import sqlite3
# import argparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import speech_recognition as sr  # Ensure this import is present
# import numpy as np
# from scipy.signal import resample
# import time
# import os
# import csv
# import pandas as pd
# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet

# continue_conversation = False
# CONTINUE_FLAG_FILE = "continue_flag.txt"

# # Check for continue flag
# def check_continue_flag():
#     global continue_conversation
#     if os.path.exists(CONTINUE_FLAG_FILE):
#         continue_conversation = True
#         os.remove(CONTINUE_FLAG_FILE)

# # === Load Models ===
# print("🧠 Loading models...")
# whisper_model = whisper.load_model("large")  # Changed to large model for better transcription
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
# print("🧠 Models loaded successfully!")

# # Initialize TTS
# print("🔊 Initializing TTS (offline)...")
# engine = pyttsx3.init()

# # Initialize speech recognition
# recognizer = sr.Recognizer()
# mic = sr.Microphone()

# # Argument parser for OpenAI fallback
# parser = argparse.ArgumentParser(description="Voice-based Conversational AI")
# parser.add_argument("--openai", action="store_true", default=False, help="Use OpenAI as fallback LLM")
# parser.add_argument("--export", action="store_true", help="Export knowledge base")
# parser.add_argument("--format", default="json", help="Export format (json, csv, pdf)")
# args = parser.parse_args()

# # Optional OpenAI
# openai_client = None
# if args.openai:
#     print("🧠 Initializing OpenAI (fallback)...")
#     from openai import OpenAI
#     openai_client = OpenAI(api_key="your_openai_api_key_here")  # Replace with your key

# # SQLite database
# print("🧠 Initializing memory (JSON or local DB)...")
# conn = sqlite3.connect("knowledge_base.db")
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, input TEXT, response TEXT)''')
# conn.commit()

# # Initialize conversation log
# conversation_log_file = "conversation_log.json"
# if os.path.exists(conversation_log_file):
#     try:
#         with open(conversation_log_file, 'r') as f:
#             conversation_log = json.load(f)
#         if not isinstance(conversation_log, list):
#             print("🧠 Warning: conversation_log.json contains invalid data. Initializing with empty list.")
#             conversation_log = []
#     except json.JSONDecodeError:
#         print("🧠 Warning: conversation_log.json is invalid or empty. Initializing with empty list.")
#         conversation_log = []
# else:
#     print("🧠 Warning: conversation_log.json not found. Creating new empty log.")
#     conversation_log = []
# with open(conversation_log_file, 'w') as f:
#     json.dump(conversation_log, f, indent=4)

# # Knowledge base structure
# knowledge_base = {
#     "profession": None,
#     "agent_type": None,
#     "details": {}
# }
# conversation_step = 0
# selected_flow = None
# conversation_complete = False

# # Function to save conversation
# def save_conversation(user_input, agent_response):
#     conversation_log.append({"user": user_input, "agent": agent_response})
#     with open(conversation_log_file, 'w', buffering=1) as f:
#         json.dump(conversation_log, f, indent=4)

# def transcribe_audio():
#     print("🎤 Press Enter to start listening (timeout in 7 seconds)...")
#     input()
#     with mic as source:
#         print("🎤 Listening... Please speak now (timeout in 7 seconds)")
#         recognizer.adjust_for_ambient_noise(source, duration=2.0)
#         try:
#             audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)
#             audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
#             audio_data = audio_data.astype(np.float32) / 32768.0
#             sample_rate = 16000
#             if audio.sample_rate != sample_rate:
#                 num_samples = int(len(audio_data) * sample_rate / audio.sample_rate)
#                 audio_data = resample(audio_data, num_samples)
#             result = whisper_model.transcribe(audio_data, fp16=False, language="en")
#             text = result["text"].strip()
#             if not text or len(text.split()) < 2:
#                 print("🧠 Transcribed: No significant audio detected (ignored)")
#                 return ""
#             print(f"🧠 Transcribed: {text}")
#             return text
#         except sr.WaitTimeoutError:
#             print("🧠 Warning: Listening timed out. Please speak clearly and try again.")
#             return ""
#         except Exception as e:
#             print(f"🧠 Transcription error: {e}. Please check your audio setup or try again.")
#             return ""
# # ... (rest of the file, including generate_response, remains the same as the last version)

# # Function to generate response using phi-2
# def generate_response(text):
#     global conversation_step, selected_flow, knowledge_base, conversation_complete
#     response = None
#     if args.openai and openai_client:
#         try:
#             response = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are a dynamic conversational AI designed to assist users in creating a knowledge base for an AI agent. Adapt the conversation flow based on user input, inferring intent (e.g., personal use vs. professional use) from context. Ask relevant follow-up questions and adjust based on clarifications without relying on fixed phrases."},
#                     {"role": "user", "content": text}
#                 ]
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"🧠 OpenAI error: {e}")

#     text_lower = text.lower()
#     # Dynamic intent detection and context reset
#     if any(prof in text_lower for prof in ["project manager", "doctor", "tutor", "lawyer", "therapist"]) or "agent" in text_lower:
#         # Infer the most likely profession or agent type
#         selected_flow = next((prof for prof in ["project_manager", "doctor", "tutor", "lawyer", "therapist"] if prof.replace("_", " ") in text_lower), None)
#         if not selected_flow and "agent" in text_lower:
#             selected_flow = "therapist" if "therapist" in text_lower else "project_manager"  # Default inference
#         if selected_flow:
#             knowledge_base["profession"] = selected_flow.replace("_", " ")
#             knowledge_base.clear()  # Reset knowledge_base to start fresh
#             knowledge_base["profession"] = selected_flow.replace("_", " ")
#             if "agent" in text_lower and any(task in text_lower for task in ["to", "for", "which will"]):
#                 knowledge_base["agent_type"] = text.split("agent")[1].strip().replace("to", "").replace("for", "").replace("which will", "").strip() or "general assistance"
#                 if selected_flow == "therapist":
#                     response = f"Understood! You want an agent to act as a therapist. Is this for personal use or professional use with clients? Please clarify."
#                     conversation_step = 2
#                 else:  # project_manager
#                     response = f"Got it! You want an agent for {knowledge_base['agent_type']} as a project manager. How many people do you typically manage?"
#                     conversation_step = 2
#             else:
#                 response = f"Great! What type of AI agent would you like to create to assist you as a {knowledge_base['profession']}?"
#                 conversation_step = 1
#     elif selected_flow:
#         if selected_flow == "project_manager":
#             if conversation_step == 2:
#                 if "i manage" in text_lower and "people" in text_lower:
#                     knowledge_base["details"] = knowledge_base.get("details", {})
#                     knowledge_base["details"]["team_size"] = text_lower.split("i manage")[1].split("people")[0].strip()
#                     response = "Thanks! How often do you have meetings, and when do they usually happen?"
#                 else:
#                     response = "Please tell me how many people you typically manage (e.g., 'I manage 5 people')."
#                 conversation_step = 3 if "i manage" in text_lower and "people" in text_lower else conversation_step
#             # ... (rest of project_manager flow remains the same up to step 5)
#         elif selected_flow == "therapist":
#             if conversation_step == 2:
#                 knowledge_base["details"] = knowledge_base.get("details", {})
#                 if "personal" in text_lower or "for me" in text_lower or "not for clients" in text_lower:
#                     knowledge_base["details"]["personal_use"] = True
#                     response = "Got it! This is for your personal use. What tasks would you like the AI therapist to handle (e.g., mood tracking, daily reminders)?"
#                     conversation_step = 3
#                 elif "client" in text_lower and ("see" in text_lower or "work with" in text_lower):
#                     knowledge_base["details"]["client_load"] = text_lower.split("client")[0].split("see")[-1].strip() if "see" in text_lower else text_lower.split("work with")[-1].strip()
#                     response = "Thanks! What tasks would you like the AI to handle for your therapy sessions (e.g., scheduling, note-taking)?"
#                 else:
#                     response = "Please clarify if this is for personal use or professional use with clients (e.g., 'This is for me' or 'I see 10 clients a week')."
#                 conversation_step = 3 if ("personal" in text_lower or "client" in text_lower) else conversation_step
#             elif conversation_step == 3:
#                 if "i’d like it to" in text_lower or "tasks" in text_lower:
#                     knowledge_base["details"]["tasks"] = text.split("i’d like it to ")[1].replace(".", "").strip() if "i’d like it to" in text_lower else text_lower.split("tasks")[-1].strip()
#                     use_type = "personal use" if knowledge_base["details"].get("personal_use") else f"{knowledge_base['details'].get('client_load', 'unknown')} clients"
#                     response = f"Great! Here’s what I’ve gathered: You’re a therapist who wants an AI to help with {knowledge_base['agent_type']} for {use_type}, and it should {knowledge_base['details'].get('tasks', 'unknown')}. Sound good?"
#                 else:
#                     response = "Please specify what tasks you’d like the AI to handle (e.g., 'I’d like it to track my mood')."
#                 conversation_step = 4 if "i’d like it to" in text_lower or "tasks" in text_lower else conversation_step
#             # ... (rest of therapist flow remains the same up to step 5)
#         elif conversation_step >= 5:
#             response = "Got it! Anything else to add or refine?"

#     if not response:
#         prompt = f"### Instruction: You are a dynamic conversational AI assistant. Infer the user’s intent from their input and respond with a relevant question or acknowledgment to build a knowledge base for an AI agent. Adapt to the context without relying on fixed phrases.\n### Input: {text}\n### Response:"
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#             pad_token_id=tokenizer.eos_token_id,
#             no_repeat_ngram_size=3
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

#     print(f"🧠 LLM interpreted: {response}")
#     return response


# # Function to speak response
# def speak_response(text):
#     print("🔊 Agent speaking...")
#     engine.say(text)
#     engine.runAndWait()

# # Function to store in memory
# def store_in_memory(input_text, response_text):
#     print("🧠 Storing in memory...")
#     cursor.execute("INSERT INTO knowledge (input, response) VALUES (?, ?)", (input_text, response_text))
#     conn.commit()

# # Function to export knowledge base
# def export_knowledge_base(format_type):
#     print(f"📦 Exporting knowledge base as {format_type}...")
#     cursor.execute("SELECT input, response FROM knowledge")
#     data = [{"input": row[0], "response": row[1]} for row in cursor.fetchall()]
#     kb_data = {
#         "profession": knowledge_base["profession"],
#         "agent_type": knowledge_base["agent_type"],
#         "details": knowledge_base["details"]
#     }
#     if format_type == "json":
#         with open("../static/knowledge_base.json", "w") as f:
#             json.dump(kb_data, f, indent=4)
#         return {"message": "Knowledge base exported as JSON.", "url": "http://localhost:5000/static/knowledge_base.json"}
#     elif format_type == "csv":
#         with open("../static/knowledge_base.csv", "w", newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=["profession", "agent_type", "details"])
#             writer.writeheader()
#             writer.writerow(kb_data)
#         return {"message": "Knowledge base exported as CSV.", "url": "http://localhost:5000/static/knowledge_base.csv"}
#     elif format_type == "pdf":
#         pdf_file = "../static/knowledge_base.pdf"
#         doc = SimpleDocTemplate(pdf_file, pagesize=letter)
#         styles = getSampleStyleSheet()
#         story = []
#         story.append(Paragraph(f"Profession: {kb_data['profession']}", styles['Heading2']))
#         story.append(Paragraph(f"Agent Type: {kb_data['agent_type']}", styles['Normal']))
#         story.append(Paragraph(f"Details: {json.dumps(kb_data['details'])}", styles['Normal']))
#         story.append(Spacer(1, 12))
#         doc.build(story)
#         return {"message": "Knowledge base exported as PDF.", "url": "http://localhost:5000/static/knowledge_base.pdf"}
#     return {"message": "Unsupported format."}

# # Main loop
# def main():
#     if args.export:
#         result = export_knowledge_base(args.format)
#         print(result["message"])
#         return

#     print("Voice Agent started. Press Enter to speak, say 'that’s all for now' to stop, or 'export' to save knowledge base.")
#     global continue_conversation
#     while True:
#         check_continue_flag()
#         if continue_conversation:
#             continue_conversation = False
#             print("Continuing conversation...")
#         user_input = transcribe_audio()
#         print(f"Received input: {user_input}")
#         if not user_input:
#             print("No valid input detected. Press Enter to try again.")
#             continue

#         if user_input.lower() == "export":
#             result = export_knowledge_base("json")  # Default to JSON for now
#             print(result["message"])
#             continue
#         elif user_input.lower() in ["that’s all for now", "done", "stop"]:
#             response = "Thanks for the info! Your knowledge base is ready. Say 'export' to save or 'continue' to add more."
#             speak_response(response)
#             save_conversation(user_input, response)
#             store_in_memory(user_input, response)
#             break

#         response = generate_response(user_input)
#         store_in_memory(user_input, response)
#         save_conversation(user_input, response)
#         speak_response(response)

#     conn.close()

# if __name__ == "__main__":
#     main()







import whisper
import pyttsx3
import json
import sqlite3
import argparse
import requests  # For Ollama API
import speech_recognition as sr  # Added to fix NameError
import numpy as np
from scipy.signal import resample
import time
import os
import csv
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

continue_conversation = False
CONTINUE_FLAG_FILE = "continue_flag.txt"

# Check for continue flag
def check_continue_flag():
    global continue_conversation
    if os.path.exists(CONTINUE_FLAG_FILE):
        continue_conversation = True
        os.remove(CONTINUE_FLAG_FILE)

# === Load Models ===
print("🧠 Loading models...")
whisper_model = whisper.load_model("large")  # Keep large for better transcription
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"  # Use "mistral:7b-instruct" for instruction-tuned version

# Initialize TTS
print("🔊 Initializing TTS (offline)...")
engine = pyttsx3.init()

# Initialize speech recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Argument parser
parser = argparse.ArgumentParser(description="Voice-based Conversational AI")
parser.add_argument("--export", action="store_true", help="Export knowledge base")
parser.add_argument("--format", default="json", help="Export format (json, csv, pdf)")
args = parser.parse_args()

# SQLite database
print("🧠 Initializing memory (JSON or local DB)...")
conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, input TEXT, response TEXT)''')
conn.commit()

# Initialize conversation log
conversation_log_file = "conversation_log.json"
if os.path.exists(conversation_log_file):
    try:
        with open(conversation_log_file, 'r') as f:
            conversation_log = json.load(f)
        if not isinstance(conversation_log, list):
            print("🧠 Warning: conversation_log.json contains invalid data. Initializing with empty list.")
            conversation_log = []
    except json.JSONDecodeError:
        print("🧠 Warning: conversation_log.json is invalid or empty. Initializing with empty list.")
        conversation_log = []
else:
    print("🧠 Warning: conversation_log.json not found. Creating new empty log.")
    conversation_log = []
with open(conversation_log_file, 'w') as f:
    json.dump(conversation_log, f, indent=4)

# Knowledge base structure
knowledge_base = {
    "profession": None,
    "agent_type": None,
    "details": {},
    "history": []  # Store conversation history for context
}
conversation_step = 0
selected_flow = None
conversation_complete = False

# Function to save conversation
def save_conversation(user_input, agent_response):
    conversation_log.append({"user": user_input, "agent": agent_response})
    with open(conversation_log_file, 'w', buffering=1) as f:
        json.dump(conversation_log, f, indent=4)

def transcribe_audio():
    print("🎤 Press Enter to start listening (timeout in 7 seconds)...")
    input()
    with mic as source:
        print("🎤 Listening... Please speak now (timeout in 7 seconds)")
        recognizer.adjust_for_ambient_noise(source, duration=2.0)
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            sample_rate = 16000
            if audio.sample_rate != sample_rate:
                num_samples = int(len(audio_data) * sample_rate / audio.sample_rate)
                audio_data = resample(audio_data, num_samples)
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            text = result["text"].strip()
            if not text or len(text.split()) < 2:
                print("🧠 Transcribed: No significant audio detected (ignored)")
                return ""
            print(f"🧠 Transcribed: {text}")
            return text
        except sr.WaitTimeoutError:
            print("🧠 Warning: Listening timed out. Please speak clearly and try again.")
            return ""
        except Exception as e:
            print(f"🧠 Transcription error: {e}. Please check your audio setup or try again.")
            return ""

# Function to speak response
def speak_response(text):
    print("🔊 Agent speaking...")
    engine.say(text)
    engine.runAndWait()

# Function to store in memory
def store_in_memory(input_text, response_text):
    print("🧠 Storing in memory...")
    cursor.execute("INSERT INTO knowledge (input, response) VALUES (?, ?)", (input_text, response_text))
    conn.commit()

# Function to export knowledge base
def export_knowledge_base(format_type):
    print(f"📦 Exporting knowledge base as {format_type}...")
    cursor.execute("SELECT input, response FROM knowledge")
    data = [{"input": row[0], "response": row[1]} for row in cursor.fetchall()]
    kb_data = {
        "profession": knowledge_base["profession"],
        "agent_type": knowledge_base["agent_type"],
        "details": knowledge_base["details"]
    }
    if format_type == "json":
        with open("../static/knowledge_base.json", "w") as f:
            json.dump(kb_data, f, indent=4)
        return {"message": "Knowledge base exported as JSON.", "url": "http://localhost:5000/static/knowledge_base.json"}
    elif format_type == "csv":
        with open("../static/knowledge_base.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["profession", "agent_type", "details"])
            writer.writeheader()
            writer.writerow(kb_data)
        return {"message": "Knowledge base exported as CSV.", "url": "http://localhost:5000/static/knowledge_base.csv"}
    elif format_type == "pdf":
        pdf_file = "../static/knowledge_base.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(f"Profession: {kb_data['profession']}", styles['Heading2']))
        story.append(Paragraph(f"Agent Type: {kb_data['agent_type']}", styles['Normal']))
        story.append(Paragraph(f"Details: {json.dumps(kb_data['details'])}", styles['Normal']))
        story.append(Spacer(1, 12))
        doc.build(story)
        return {"message": "Knowledge base exported as PDF.", "url": "http://localhost:5000/static/knowledge_base.pdf"}
    return {"message": "Unsupported format."}

# Function to generate response using Ollama's Mistral 7B
def generate_response(text):
    global conversation_step, selected_flow, knowledge_base, conversation_complete
    response = None
    text_lower = text.lower()
    knowledge_base["history"] = knowledge_base.get("history", []) + [text]  # Store conversation history

    # Dynamic contextual prompting
    prompt = f"### Instruction: You are a conversational AI assisting in creating a knowledge base for any AI agent. Infer the user's intent dynamically from the conversation context (e.g., interest in a therapist agent, personal or professional use) without relying on specific phrases. Ask relevant, iterative questions to gather details (e.g., tasks, preferences) for up to 30 minutes. If the user seems unsure about data, prompt them to retrieve relevant information. Maintain context using the following history: {knowledge_base.get('history', [])}. Current input: {text}\n### Response:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload).json()["response"]
    except Exception as e:
        print(f"🧠 Ollama API error: {e}")
        response = "Sorry, I couldn’t process that. Please try again."

    # Minimal flow to guide context (optional fallback)
    if not selected_flow and any(word in text_lower for word in ["help", "talk", "agent", "ai"]):
        selected_flow = "therapist"
        knowledge_base["profession"] = "therapist"
        knowledge_base["agent_type"] = "therapist assistance"
        if not response or "discuss" in response.lower():
            response = "It seems you might be interested in a therapist AI. Could you share how you’d like to use it—perhaps for yourself or others?"
        conversation_step = 1
    elif selected_flow == "therapist":
        if conversation_step == 1:
            if "myself" in text_lower or "personal" in text_lower:
                knowledge_base["details"] = knowledge_base.get("details", {})
                knowledge_base["details"]["personal_use"] = True
                response = "I gather this is for personal use. What tasks would you like the therapist AI to assist with?"
                conversation_step = 2
            elif "others" in text_lower or "client" in text_lower:
                knowledge_base["details"] = knowledge_base.get("details", {})
                response = "It sounds like this is for others. Could you tell me more about your typical workload or number of people it would serve?"
                conversation_step = 2
        elif conversation_step >= 2:
            if "task" in text_lower or "like" in text_lower:
                knowledge_base["details"]["tasks"] = text.split("like")[-1].strip() if "like" in text_lower else text
                response = f"Nice! You want a therapist AI to {knowledge_base['details'].get('tasks', 'unknown')}. Any other details or preferences to add? We can keep going!"
                conversation_step += 1
            elif "yes" in text_lower or "continue" in text_lower:
                response = "Great! What else would you like to include—maybe timing, features, or interaction style?"
                conversation_step += 1
            elif "no" in text_lower or "done" in text_lower:
                use_type = "personal use" if knowledge_base["details"].get("personal_use") else "professional use"
                response = f"Done! Your therapist AI is for {use_type} to {knowledge_base['details'].get('tasks', 'unknown')}. Say 'export' to save or 'continue'."
                conversation_complete = True

    print(f"🧠 LLM interpreted: {response}")
    return response

# Main loop
def main():
    if args.export:
        result = export_knowledge_base(args.format)
        print(result["message"])
        return

    print("Voice Agent started. Press Enter to speak, say 'done' to stop, or 'export' to save knowledge base.")
    global continue_conversation
    while True:
        check_continue_flag()
        if continue_conversation:
            continue_conversation = False
            print("Continuing conversation...")
        user_input = transcribe_audio()
        print(f"Received input: {user_input}")
        if not user_input:
            print("No valid input detected. Press Enter to try again.")
            continue

        if user_input.lower() == "export":
            result = export_knowledge_base(args.format)
            print(result["message"])
            continue
        elif user_input.lower() in ["done", "stop"]:
            response = "Thanks for the info! Your knowledge base is ready. Say 'export' to save or 'continue' to add more."
            speak_response(response)
            save_conversation(user_input, response)
            store_in_memory(user_input, response)
            break

        response = generate_response(user_input)
        store_in_memory(user_input, response)
        save_conversation(user_input, response)
        speak_response(response)

    conn.close()

if __name__ == "__main__":
    main()
    