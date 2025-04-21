import speech_recognition as sr
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize FastAPI
app = FastAPI()

# Initialize SQLite database
conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        raw_text TEXT,
        processed_text TEXT,
        metadata JSON
    )
""")
conn.commit()

# Step 1: Voice Input Capture
def capture_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice input...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Transcribed text: {text}")
            return text
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand audio")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Speech recognition service unavailable")

# Step 2: Text Preprocessing
def preprocess_text(text):
    # Clean text (remove noise, URLs, special characters)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()

    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove duplicates
    tokens = list(dict.fromkeys(tokens))

    # Join tokens back to text
    processed_text = " ".join(tokens)

    return processed_text

# Step 3: Entity Extraction
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "persons": [],
        "organizations": [],
        "dates": [],
        "locations": []
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["locations"].append(ent.text)
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    return entities

# Step 4: Summarization
def summarize_text(text):
    if len(text) < 50:
        return text
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Step 5: Topic Modeling
def extract_topics(texts, n_topics=3):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

# Step 6: Structure Metadata
def create_metadata(raw_text, processed_text):
    entities = extract_entities(raw_text)
    summary = summarize_text(raw_text)
    topics = extract_topics([processed_text])
    metadata = {
        "entities": entities,
        "summary": summary,
        "topics": topics,
        "timestamp": datetime.now().isoformat(),
        "source": "voice_input"
    }
    return metadata

# Step 7: Store in Knowledge Base
def store_in_knowledge_base(raw_text, processed_text, metadata):
    cursor.execute(
        """
        INSERT INTO knowledge_base (timestamp, raw_text, processed_text, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (metadata["timestamp"], raw_text, processed_text, json.dumps(metadata))
    )
    conn.commit()

# FastAPI Endpoint for Voice Input
class VoiceInput(BaseModel):
    trigger: str  # Placeholder to trigger voice input (e.g., "start")

@app.post("/process_voice")
async def process_voice(input: VoiceInput):
    try:
        # Capture voice input
        raw_text = capture_voice_input()

        # Preprocess text
        processed_text = preprocess_text(raw_text)

        # Create metadata
        metadata = create_metadata(raw_text, processed_text)

        # Store in knowledge base
        store_in_knowledge_base(raw_text, processed_text, metadata)

        return {
            "raw_text": raw_text,
            "processed_text": processed_text,
            "metadata": metadata
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI Endpoint to Retrieve Knowledge Base
@app.get("/knowledge_base")
async def get_knowledge_base():
    cursor.execute("SELECT id, timestamp, raw_text, processed_text, metadata FROM knowledge_base")
    rows = cursor.fetchall()
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "raw_text": row[2],
            "processed_text": row[3],
            "metadata": json.loads(row[4])
        }
        for row in rows
    ]

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    conn.close()