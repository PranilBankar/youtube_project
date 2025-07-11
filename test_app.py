import os
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API - IMPORTANT: Replace with your actual API key
genai.configure(api_key="AIzaSyCHBEkq4m_c7a2gPGTwuOZntZBL2sUqNE8")

@app.route('/')
def home():
    return render_template('ind.html')

def get_transcript(video_url):
    """
    Efficiently extract transcript from YouTube video
    """
    try:
        # More robust video ID extraction
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript with intelligent spacing
        return " ".join([entry['text'].strip() for entry in transcript if entry['text'].strip()])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def generate_summary(transcript):
    """
    Generate summary using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Trim and clean text
        transcript = transcript.strip()
        
        # Very short text handling
        if not transcript or len(transcript.split()) < 30:
            return "The transcript is too short to generate a meaningful summary."
        
        # Prompt for summary generation
        prompt = f"""Please provide a concise and comprehensive summary of the following transcript. 
        Focus on the key points, main ideas, and most important information. 
        The summary should be clear, coherent, and capture the essence of the content:

        {transcript}"""
        
        # Generate summary
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def translate_to_hindi(text):
    """
    Translate text to Hindi using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare prompt for translation
        prompt = f"""Translate the following English text to Hindi. 
        Ensure the translation is natural, fluent, and maintains the original meaning:

        {text}"""
        
        # Generate translation
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error translating text: {str(e)}"

def answer_question(transcript, question):
    """
    Answer questions using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare prompt for question answering
        prompt = f"""Based on the following transcript, please answer the question as accurately and concisely as possible:

Transcript:
{transcript}

Question: {question}

Answer:"""
        
        # Generate answer
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error answering question: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Summarization route using Gemini
    """
    data = request.json
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    # Fetch transcript
    transcript = get_transcript(video_url)
    
    if "Error" in transcript:
        return jsonify({"error": transcript}), 400
    
    # Generate summary
    summary = generate_summary(transcript)
    
    if "Error" in summary:
        return jsonify({"error": summary}), 400
    
    return jsonify({"summary": summary})

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translation route for converting English to Hindi
    """
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided for translation"}), 400
    
    # Translate text to Hindi
    hindi_text = translate_to_hindi(text)
    
    if "Error" in hindi_text:
        return jsonify({"error": hindi_text}), 400
    
    return jsonify({"translated_text": hindi_text})

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Question answering route using Gemini
    """
    data = request.json
    video_url = data.get('video_url')
    question = data.get('question')
    
    if not video_url or not question:
        return jsonify({"error": "Video URL or question not provided"}), 400
    
    # Fetch transcript
    transcript = get_transcript(video_url)
    
    if "Error" in transcript:
        return jsonify({"error": transcript}), 400
    
    # Answer question
    answer = answer_question(transcript, question)
    
    if "Error" in answer:
        return jsonify({"error": answer}), 400
    
    return jsonify({"answer": answer})

# Performance optimization for Flask
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

if __name__ == "__main__":
    # Use a production WSGI server in production
    app.run(threaded=True)