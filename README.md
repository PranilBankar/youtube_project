# YouTube Insights Pro 🎥✨

A powerful Flask-based web application that leverages Google's Gemini AI to analyze YouTube videos. Extract transcripts, generate intelligent summaries, translate content to Hindi, and ask questions about any YouTube video.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **📝 Smart Summarization**: Generate concise, comprehensive summaries of YouTube videos using Gemini 2.0 Flash
- **🌐 Hindi Translation**: Translate summaries to Hindi with natural, fluent translations using Gemini 1.5 Pro
- **❓ Q&A System**: Ask specific questions about video content and get accurate AI-powered answers
- **🎨 Modern UI**: Beautiful glassmorphism design with smooth animations and responsive layout
- **⚡ Fast Processing**: Efficient transcript extraction and AI-powered analysis

## 🚀 Demo

### Main Interface
The application provides three core functionalities:
1. Generate video summaries
2. Translate summaries to Hindi with toggle switch
3. Ask questions about video content

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Internet connection for API calls

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube_project.git
   cd youtube_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Gemini API key**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   
   Or set it as an environment variable:
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export GEMINI_API_KEY=your_api_key_here
   ```

## 🎯 Usage

1. **Start the Flask application**
   ```bash
   python test_app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Analyze a YouTube video**
   - Paste a YouTube video URL
   - Click "Generate Summary" to get an AI-powered summary
   - Click "Translate" to convert the summary to Hindi
   - Use the toggle switch to switch between English and Hindi
   - Ask questions about the video content in the Q&A section

## 📁 Project Structure

```
youtube_project/
│
├── test_app.py              # Main Flask application
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
└── templates/
    └── ind.html            # Frontend HTML template
```

## 🛠️ Technologies Used

### Backend
- **Flask**: Lightweight web framework
- **YouTube Transcript API**: Extract video transcripts
- **Google Gemini AI**: Advanced language model for summarization, translation, and Q&A

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Tailwind CSS**: Utility-first CSS framework
- **AOS**: Animate On Scroll library
- **Font Awesome**: Icon library
- **Axios**: HTTP client for API requests

## 🔑 API Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/` | GET | Render home page | - |
| `/summarize` | POST | Generate video summary | `{"video_url": "string"}` |
| `/translate` | POST | Translate text to Hindi | `{"text": "string"}` |
| `/ask` | POST | Answer questions about video | `{"video_url": "string", "question": "string"}` |

### Example API Usage

**Summarize a video:**
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Translate text:**
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Your summary text here"}'
```

**Ask a question:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "question": "What is the main topic?"}'
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes |
| `FLASK_ENV` | Flask environment (development/production) | No |
| `FLASK_DEBUG` | Enable debug mode (True/False) | No |

### Flask Configuration

The application includes performance optimizations:
- `JSON_SORT_KEYS = False`: Faster JSON serialization
- `JSONIFY_PRETTYPRINT_REGULAR = False`: Reduced response size
- `threaded=True`: Handle multiple requests concurrently

## 🚨 Important Security Notes

> **⚠️ WARNING**: Never commit your API key to version control!

1. Always use environment variables for sensitive data
2. Add `.env` to your `.gitignore` file
3. Use a production WSGI server (like Gunicorn) for deployment
4. Enable HTTPS in production

## 🐛 Troubleshooting

### Common Issues

**"No transcript available"**
- The video may not have captions/subtitles
- Try a different video with available transcripts

**"API key error"**
- Verify your Gemini API key is correct
- Check if the API key has proper permissions
- Ensure the key is properly set in environment variables

**"Module not found"**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python environment

## 🚀 Deployment

### Using Gunicorn (Production)

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Run the application:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 test_app:app
   ```

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV GEMINI_API_KEY=your_key_here
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "test_app:app"]
```

Build and run:
```bash
docker build -t youtube-insights .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key youtube-insights
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction
- [Google Gemini AI](https://deepmind.google/technologies/gemini/) for powerful language processing
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the UI components

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Flask and Google Gemini AI**
