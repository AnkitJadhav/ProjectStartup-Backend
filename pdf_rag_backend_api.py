from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib
from werkzeug.utils import secure_filename

# Core libraries
import PyPDF2
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DeepSeekClient:
    """DeepSeek API client for free R1 model"""

    def __init__(self):
        # API key for OpenRouter
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP_REFERER": "http://localhost:3000",  # Required by OpenRouter
            "X-Title": "PDF RAG System",  # Required by OpenRouter
            "User-Agent": "PDF RAG System/1.0.0",  # Required by OpenRouter
            "Accept": "application/json"  # Required by OpenRouter
        }

    def chat_completion(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Create chat completion using free DeepSeek R1 model"""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",  # Free model
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "headers": {  # Include headers in the request body as well
                "HTTP_REFERER": "http://localhost:3000",
                "X-Title": "PDF RAG System"
            }
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            print(f"OpenRouter API Response Status: {response.status_code}")
            print(f"OpenRouter API Response: {response.text}")
            
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "Error: Invalid response from API"

        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"

class PDFRAGBackend:
    """
    PDF RAG System Backend API
    - Upload PDF and ask questions via REST API
    - Smart chunking and TF-IDF based search
    - Student-friendly explanations
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.deepseek_client = None
        self.chunks = []
        self.chunk_vectors = None
        self.pdf_loaded = False
        self.pdf_name = ""
        self.pdf_path = ""
        self.session_id = str(uuid.uuid4())

        self._initialize()

    def _initialize(self):
        """Initialize components"""
        print("🚀 Initializing PDF RAG System...")

        try:
            # Initialize DeepSeek client
            self.deepseek_client = DeepSeekClient()
            print("✅ DeepSeek API ready!")

        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF file and return status"""
        try:
            if not os.path.exists(pdf_path):
                return {"success": False, "error": "PDF file not found"}

            if not pdf_path.lower().endswith('.pdf'):
                return {"success": False, "error": "Please provide a PDF file"}

            # Extract text from PDF
            text = self._extract_pdf_text(pdf_path)
            if not text.strip():
                return {"success": False, "error": "No text found in PDF"}

            # Create chunks
            self.chunks = self._create_chunks(text)

            # Generate TF-IDF vectors
            chunk_texts = [chunk['text'] for chunk in self.chunks]
            self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)

            self.pdf_loaded = True
            self.pdf_name = os.path.basename(pdf_path)
            self.pdf_path = pdf_path

            return {
                "success": True,
                "message": f"PDF '{self.pdf_name}' processed successfully",
                "chunks_created": len(self.chunks),
                "pages_processed": max(chunk['page'] for chunk in self.chunks) if self.chunks else 0,
                "session_id": self.session_id
            }

        except Exception as e:
            return {"success": False, "error": f"Error processing PDF: {str(e)}"}

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n=== PAGE {page_num + 1} ===\n\n{page_text}"
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
        return text

    def _create_chunks(self, text: str) -> List[Dict]:
        """Create smart text chunks using sentence tokenization"""
        # Clean text
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)

        chunks = []
        chunk_size = 800
        overlap = 150

        # Split by pages first
        pages = re.split(r'=== PAGE \d+ ===', text)

        for page_num, page_content in enumerate(pages[1:], 1):
            if not page_content.strip():
                continue

            # Split into sentences
            sentences = sent_tokenize(page_content)
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding sentence exceeds chunk size, save current chunk
                if len(current_chunk + sentence) > chunk_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page': page_num,
                        'chunk_id': len(chunks)
                    })

                    # Start new chunk with overlap
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add the last chunk if it exists
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num,
                    'chunk_id': len(chunks)
                })

        return chunks

    def find_relevant_chunks(self, question: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find most relevant chunks using TF-IDF similarity"""
        if not self.chunks or self.chunk_vectors is None:
            return []

        # Get question vector
        question_vector = self.vectorizer.transform([question])

        # Calculate similarities
        similarities = np.dot(self.chunk_vectors, question_vector.T).toarray().flatten()

        # Sort by similarity and return top results
        chunk_scores = list(zip(self.chunks, similarities))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores[:top_k]

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question about the PDF and return structured response"""
        try:
            if not self.pdf_loaded:
                return {
                    "success": False,
                    "error": "No PDF loaded. Please upload a PDF first.",
                    "session_id": self.session_id
                }

            if not question.strip():
                return {
                    "success": False,
                    "error": "Please provide a valid question.",
                    "session_id": self.session_id
                }

            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(question)

            if not relevant_chunks or relevant_chunks[0][1] < 0.2:
                return {
                    "success": False,
                    "error": "No relevant information found for your question.",
                    "suggestions": [
                        "Try using different keywords",
                        "Ask about topics covered in the document",
                        "Be more specific about what you want to know"
                    ],
                    "session_id": self.session_id
                }

            # Prepare context for API call
            context_parts = []
            for i, (chunk, score) in enumerate(relevant_chunks):
                context_parts.append(f"[Source {i+1} - Page {chunk['page']} - Score: {score:.2f}]\n{chunk['text']}")

            context = "\n\n--- SECTION ---\n\n".join(context_parts)

            # Create prompt for DeepSeek
            prompt = f"""You are a helpful AI tutor. Based on the PDF content below, answer the student's question clearly and simply.

**PDF CONTENT:**
{context}

**STUDENT'S QUESTION:** {question}

**Instructions:**
1. Answer based ONLY on the PDF content provided
2. Give a clear, direct answer first
3. Then explain it in simple terms
4. If the answer isn't in the content, say so clearly
5. Be helpful and educational

**Answer:**"""

            # Get response from DeepSeek
            messages = [
                {"role": "system", "content": "You are a helpful AI tutor that explains things clearly to students."},
                {"role": "user", "content": prompt}
            ]

            ai_response = self.deepseek_client.chat_completion(messages, temperature=0.3, max_tokens=1000)

            # Prepare source information
            pages_used = list(set([chunk['page'] for chunk, _ in relevant_chunks]))
            relevance_scores = [round(score, 3) for _, score in relevant_chunks]

            return {
                "success": True,
                "answer": ai_response,
                "sources": sorted(pages_used),
                "relevance_scores": relevance_scores,
                "chunks_used": len(relevant_chunks),
                "session_id": self.session_id,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing question: {str(e)}",
                "session_id": self.session_id
            }

    def get_pdf_info(self) -> Dict[str, Any]:
        """Get information about loaded PDF"""
        if not self.pdf_loaded:
            return {
                "success": False,
                "error": "No PDF loaded",
                "session_id": self.session_id
            }

        return {
            "success": True,
            "pdf_name": self.pdf_name,
            "chunks_count": len(self.chunks),
            "pages_count": max(chunk['page'] for chunk in self.chunks) if self.chunks else 0,
            "status": "ready",
            "session_id": self.session_id,
            "loaded_at": datetime.now().isoformat()
        }

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure upload settings
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create Data directory if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize RAG system
rag_system = PDFRAGBackend()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API ENDPOINTS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "PDF RAG Backend is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "cors_enabled": True
    })

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_pdf():
    """Upload and process PDF file"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Only PDF files are allowed"
            }), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"📁 File saved to: {file_path}")

        # Process the PDF
        result = rag_system.process_pdf(file_path)

        if result["success"]:
            print(f"✅ PDF processed successfully: {result}")
            return jsonify(result), 200
        else:
            print(f"❌ PDF processing failed: {result}")
            return jsonify(result), 400

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ Upload error: {error_msg}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    """Ask a question about the uploaded PDF"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "error": "Question is required"
            }), 400

        question = data['question'].strip()

        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400

        print(f"❓ Question received: {question}")

        result = rag_system.ask_question(question)

        if result["success"]:
            print(f"✅ Answer generated successfully")
            return jsonify(result), 200
        else:
            print(f"❌ Question processing failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 400

    except Exception as e:
        error_msg = f"Failed to process question: {str(e)}"
        print(f"❌ Question error: {error_msg}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route('/info', methods=['GET'])
def get_pdf_info():
    """Get information about the currently loaded PDF"""
    try:
        result = rag_system.get_pdf_info()

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({
            "success": True,
            "error": f"Failed to get PDF info: {str(e)}"
        }), 500

@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset_session():
    """Reset the current session and clear loaded PDF"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        global rag_system
        rag_system = PDFRAGBackend()  # Create new instance

        print("🔄 Session reset successfully")

        return jsonify({
            "success": True,
            "message": "Session reset successfully",
            "session_id": rag_system.session_id,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to reset session: {str(e)}"
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB."
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    print("🚀 Starting PDF RAG Backend API...")
    print("📚 Available endpoints:")
    print("   GET  /health     - Health check")
    print("   POST /upload     - Upload PDF file")
    print("   POST /ask        - Ask question")
    print("   GET  /info       - Get PDF info")
    print("   POST /reset      - Reset session")
    
    # Get the local IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🌐 Server starting on:")
    print(f"   Local:   http://localhost:5000")
    print(f"   Network: http://{local_ip}:5000")
    print("\n🔧 CORS enabled for all origins")

    # Check if required packages are installed
    try:
        import flask
        import flask_cors
        import sentence_transformers
        import PyPDF2
        import numpy
        import requests
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install: pip install flask flask-cors sentence-transformers PyPDF2 numpy requests")

    app.run(debug=True, host='0.0.0.0', port=5000)
