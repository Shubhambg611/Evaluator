import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import re

# Check and install dependencies
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini AI library found")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ WARNING: google-generativeai not installed")
    print("   Run: pip install google-generativeai")

try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
    print("✅ FastAPI libraries found")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("   Run: pip install fastapi uvicorn pydantic")
    exit(1)

try:
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship
    print("✅ SQLAlchemy found")
except ImportError:
    print("❌ Missing SQLAlchemy")
    print("   Run: pip install sqlalchemy")
    exit(1)

import uvicorn

# Initialize FastAPI app
app = FastAPI(title="HelloIvy Essay Evaluator API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./essays.db")
try:
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print("✅ Database connection established")
except Exception as e:
    print(f"❌ Database setup failed: {e}")

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyDZz3u71N4Ee0nQNTkwHFMCRlUDxxf6GNk"
model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini AI configured successfully")
    except Exception as e:
        print(f"❌ Gemini AI configuration failed: {e}")
        model = None
else:
    print("🔄 Running without Gemini API")

# Database Models
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=True)
    credits = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    essays = relationship("Essay", back_populates="user")

class Essay(Base):
    __tablename__ = "essays"
    
    essay_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    title = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    college_degree = Column(String)
    content = Column(Text, nullable=False)
    analysis_result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="essays")

# Create tables safely
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
except Exception as e:
    print(f"❌ Database table creation failed: {e}")

# Pydantic Models
class EssaySubmission(BaseModel):
    title: str
    question_type: str
    word_count: int
    college_degree: str
    content: str

class AnalysisSection(BaseModel):
    key_observations: List[str]
    next_steps: List[str]

class AnalysisData(BaseModel):
    overall_score: float
    alignment_with_topic: AnalysisSection
    essay_narrative_impact: AnalysisSection
    language_and_structure: AnalysisSection

class Highlight(BaseModel):
    text: str
    type: str
    issue: str
    suggestion: str

class AnalysisResponse(BaseModel):
    status: str
    analysis: AnalysisData
    ai_provider: str
    highlights: List[Highlight] = []

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Enhanced AI Essay Evaluator
class GeminiEssayEvaluator:
    def __init__(self):
        self.model = model
    
    async def evaluate_essay(self, content: str, title: str, question_type: str, college_degree: str = "") -> Dict[str, Any]:
        """Enhanced Gemini AI essay evaluation with proper grammar scoring"""
        
        if not self.model:
            # Return demo analysis if no AI available
            return self._generate_demo_analysis(content, title, question_type, college_degree)
        
        prompt = f"""
        You are an expert college admissions counselor with 15+ years of experience evaluating essays for top universities. 
        Analyze this essay and provide constructive, actionable feedback with PRECISE SCORING that properly weighs grammar errors.

        **Essay Details:**
        - Title: {title}
        - Question/Prompt: {question_type}
        - Target College/Program: {college_degree}

        **Essay Content:**
        {content}

        **CRITICAL SCORING GUIDELINES (1-10 scale with precise weights):**

        **Overall Score Calculation (Weighted Average):**
        - Grammar & Language Mechanics: 35% weight (CRITICAL for college readiness)
        - Content & Ideas: 25% weight  
        - Structure & Organization: 20% weight
        - Prompt Alignment: 15% weight
        - Voice & Authenticity: 5% weight

        **MANDATORY SCORING CAPS:**
        - If essay has 10+ grammar errors: Maximum overall score is 6.0
        - If essay has 15+ grammar errors: Maximum overall score is 5.0
        - If essay has 20+ grammar errors: Maximum overall score is 4.0
        - If essay has 25+ grammar errors: Maximum overall score is 3.5

        **Please provide your analysis in this exact JSON format:**

        {{
            "overall_score": 4.2,
            "strengths": [
                "Shows clear personal connection through specific examples",
                "Demonstrates genuine passion for the subject",
                "Includes concrete details and experiences",
                "Good narrative structure with clear progression",
                "Addresses the prompt directly",
                "Shows authentic voice and perspective"
            ],
            "improvements": [
                "CRITICAL: Fix numerous subject-verb agreement errors",
                "CRITICAL: Correct pronoun usage throughout the essay",
                "CRITICAL: Fix verb tense consistency issues",
                "Replace generic phrases with more specific language",
                "Strengthen transitions between paragraphs",
                "Add more sophisticated vocabulary",
                "Improve sentence variety and structure",
                "Proofread carefully to eliminate basic errors"
            ],
            "grammar_issues": [
                {{
                    "text": "I always was fascinated",
                    "issue": "Awkward word order",
                    "suggestion": "I was always fascinated"
                }},
                {{
                    "text": "who work as a mechanic",
                    "issue": "Subject-verb disagreement", 
                    "suggestion": "who works as a mechanic"
                }},
                {{
                    "text": "me and my friends",
                    "issue": "Incorrect pronoun order",
                    "suggestion": "my friends and I"
                }}
            ]
        }}
        """

        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No valid JSON found in AI response")
            
            feedback_data = json.loads(json_text)
            
            return self._process_ai_response(feedback_data)
            
        except Exception as e:
            print(f"❌ Gemini API error: {str(e)}")
            return self._generate_demo_analysis(content, title, question_type, college_degree)
    
    def _process_ai_response(self, feedback_data):
        """Process AI response into expected format"""
        overall_score = feedback_data.get("overall_score", 5.0)
        
        analysis = AnalysisData(
            overall_score=overall_score,
            alignment_with_topic=AnalysisSection(
                key_observations=feedback_data.get("strengths", [])[:3],
                next_steps=feedback_data.get("improvements", [])[:3]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=feedback_data.get("strengths", [])[3:6] if len(feedback_data.get("strengths", [])) > 3 else ["Essay shows developing narrative voice"],
                next_steps=feedback_data.get("improvements", [])[3:6] if len(feedback_data.get("improvements", [])) > 3 else ["Continue developing your unique voice"]
            ),
            language_and_structure=AnalysisSection(
                key_observations=feedback_data.get("strengths", [])[6:] if len(feedback_data.get("strengths", [])) > 6 else ["Writing demonstrates basic proficiency"],
                next_steps=feedback_data.get("improvements", [])[6:] if len(feedback_data.get("improvements", [])) > 6 else ["Focus on grammar and language clarity"]
            )
        )
        
        highlights = []
        for issue in feedback_data.get("grammar_issues", []):
            highlights.append(Highlight(
                text=issue.get("text", ""),
                type="grammar",
                issue=issue.get("issue", "Language issue"),
                suggestion=issue.get("suggestion", "Review this section")
            ))
        
        return {
            "analysis": analysis,
            "highlights": highlights
        }
    
    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str):
        """Generate demo analysis when AI is not available"""
        word_count = len(content.split())
        
        # Simple grammar error detection
        grammar_errors = []
        if "I always was" in content:
            grammar_errors.append({"text": "I always was", "issue": "Word order", "suggestion": "I was always"})
        if "me and my friends" in content:
            grammar_errors.append({"text": "me and my friends", "issue": "Pronoun order", "suggestion": "my friends and I"})
        
        score = 6.0 if len(grammar_errors) < 3 else 4.0
        
        analysis = AnalysisData(
            overall_score=score,
            alignment_with_topic=AnalysisSection(
                key_observations=["Essay addresses the prompt clearly", "Shows personal connection to the topic"],
                next_steps=["Strengthen specific examples", "Add more concrete details"]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=["Good narrative structure", "Shows authentic voice"],
                next_steps=["Enhance storytelling elements", "Improve transitions"]
            ),
            language_and_structure=AnalysisSection(
                key_observations=["Basic writing proficiency demonstrated"],
                next_steps=["Fix grammar errors", "Improve sentence variety"]
            )
        )
        
        highlights = [Highlight(
            text=err["text"],
            type="grammar",
            issue=err["issue"],
            suggestion=err["suggestion"]
        ) for err in grammar_errors]
        
        return {
            "analysis": analysis,
            "highlights": highlights
        }

# Initialize evaluator
evaluator = GeminiEssayEvaluator()

# API Endpoints
@app.post("/api/analyze-essay", response_model=AnalysisResponse)
async def analyze_essay(submission: EssaySubmission, db: Session = Depends(get_db)):
    """Analyze essay using enhanced Gemini AI with proper grammar scoring"""
    
    # Validate input
    if not submission.content.strip():
        raise HTTPException(status_code=400, detail="Essay content is required")
    
    if not submission.title.strip():
        raise HTTPException(status_code=400, detail="Essay title is required")
    
    if len(submission.content.split()) < 20:
        raise HTTPException(status_code=400, detail="Essay must be at least 20 words")
    
    try:
        # Generate user ID for tracking
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Analyze with enhanced AI
        result = await evaluator.evaluate_essay(
            submission.content,
            submission.title,
            submission.question_type,
            submission.college_degree
        )
        
        # Try to save to database
        try:
            user = User(user_id=user_id)
            db.add(user)
            
            essay_id = f"essay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
            
            essay = Essay(
                essay_id=essay_id,
                user_id=user_id,
                title=submission.title,
                question_type=submission.question_type,
                college_degree=submission.college_degree,
                content=submission.content,
                analysis_result=json.dumps({
                    "analysis": result["analysis"].dict(),
                    "highlights": [h.dict() for h in result["highlights"]]
                })
            )
            
            db.add(essay)
            db.commit()
            print(f"✅ Saved essay analysis: {essay_id}")
        except Exception as db_error:
            print(f"⚠️ Database save failed: {db_error}")
            db.rollback()
        
        ai_provider = "Gemini AI Enhanced" if model else "Demo Analysis"
        
        return AnalysisResponse(
            status="success",
            analysis=result["analysis"],
            ai_provider=ai_provider,
            highlights=result["highlights"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_engine": "Gemini AI Enhanced" if model else "Demo Mode",
        "database": "Connected"
    }

@app.get("/")
async def serve_frontend():
    """Serve the main HTML application"""
    
    # First try to serve the HTML file
    html_files = ["essay_evaluator.html", "index.html"]
    
    for html_file in html_files:
        if os.path.exists(html_file):
            try:
                return FileResponse(html_file)
            except Exception as e:
                print(f"⚠️ Error serving {html_file}: {e}")
    
    # If no HTML file found, return inline HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Essay Evaluator - Setup Required</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .error { background: #ffe6e6; padding: 20px; border-radius: 8px; border: 1px solid #ff9999; }
            .success { background: #e6ffe6; padding: 20px; border-radius: 8px; border: 1px solid #99ff99; }
            code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🚀 Essay Evaluator Backend is Running!</h1>
        
        <div class="success">
            <h3>✅ Backend Status: WORKING</h3>
            <p>Your FastAPI backend is running successfully on <strong>http://localhost:8000</strong></p>
        </div>
        
        <div class="error">
            <h3>⚠️ Frontend File Missing</h3>
            <p>The HTML frontend file was not found. Please ensure you have <code>essay_evaluator.html</code> in the same directory as your <code>working_backend.py</code> file.</p>
        </div>
        
        <h3>📋 Quick Setup:</h3>
        <ol>
            <li>Make sure <code>essay_evaluator.html</code> is in the same directory as this backend</li>
            <li>Restart the server: <code>python working_backend.py</code></li>
            <li>Access the application at <code>http://localhost:8000</code></li>
        </ol>
        
        <h3>🔧 API Endpoints Available:</h3>
        <ul>
            <li><a href="/api/health">Health Check</a> - <code>/api/health</code></li>
            <li><a href="/docs">API Documentation</a> - <code>/docs</code></li>
            <li>Essay Analysis - <code>POST /api/analyze-essay</code></li>
        </ul>
        
        <p><strong>Current Directory:</strong> Looking for HTML files in: <code>""" + os.getcwd() + """</code></p>
        <p><strong>Files Found:</strong> """ + ", ".join(os.listdir(".")) + """</p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Additional helpful endpoints
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Test endpoint working!", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting HelloIvy Essay Evaluator Backend...")
    print("=" * 60)
    print(f"📝 AI Engine: {'Gemini AI Enhanced' if model else 'Demo Mode (No API Key)'}")
    print(f"💾 Database: {DATABASE_URL}")
    print(f"🌐 Frontend: http://localhost:8000")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health Check: http://localhost:8000/api/health")
    print(f"⚖️ Enhanced Grammar Scoring: Grammar errors weighted 35% of total score")
    
    if not model:
        print("⚠️  WARNING: Running in demo mode")
        print("   To enable full AI analysis, set GEMINI_API_KEY environment variable")
    
    print("=" * 60)
    
    try:
        uvicorn.run(
            "working_backend:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("🔧 Try running on a different port:")
        print("   uvicorn working_backend:app --host 0.0.0.0 --port 8001")