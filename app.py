from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route('/')
def home():
    return render_template('index.html')

def get_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def summarize_text(text, max_length=500, min_length=30):
    try:
        if len(text.split()) < min_length:
            return "Transcript is too short to summarize."
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        tokens = tokenizer.encode(text, truncation=False, return_tensors="pt")[0]
        chunk_size = 500
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        
        summaries = []
        for chunk in text_chunks:
            if len(chunk.split()) >= min_length:
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    print("Fetching transcript...")
    transcript = get_transcript(video_url)
    if "Error" in transcript:
        return jsonify({"error": transcript}), 400
    
    print("Generating summary...")
    summary = summarize_text(transcript)
    if "Error" in summary:
        return jsonify({"error": summary}), 400
    
    return jsonify({"summary": summary})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    video_url = data.get('video_url')
    question = data.get('question')
    
    if not video_url or not question:
        return jsonify({"error": "Video URL or question not provided"}), 400
    
    print("Fetching transcript...")
    transcript = get_transcript(video_url)
    if "Error" in transcript:
        return jsonify({"error": transcript}), 400
    
    print("Answering question...")
    try:
        # Use the QA pipeline to answer the question
        answer = qa_pipeline(question=question, context=transcript)
        return jsonify({"answer": answer['answer']})
    except Exception as e:
        return jsonify({"error": f"Error answering question: {str(e)}"}), 400

if __name__ == "_main_":
    app.run(debug=True)