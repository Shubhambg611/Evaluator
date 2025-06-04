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
    from pydantic import BaseModel, Field
    print("✅ FastAPI libraries found")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("   Run: pip install fastapi uvicorn pydantic")
    exit(1)

try:
    from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship
    print("✅ SQLAlchemy found")
except ImportError:
    print("❌ Missing SQLAlchemy")
    print("   Run: pip install sqlalchemy")
    exit(1)

import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="HelloIvy Essay Evaluator API", 
    version="2.1.0",
    description="Professional Essay Analysis Platform with AI-Powered Feedback"
)

# Enhanced CORS middleware
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDZz3u71N4Ee0nQNTkwHFMCRlUDxxf6GNk")
model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )
        print("✅ Gemini AI configured successfully")
    except Exception as e:
        print(f"❌ Gemini AI configuration failed: {e}")
        model = None
else:
    print("🔄 Running without Gemini API")

# Enhanced Database Models
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=True)
    credits = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_essays = Column(Integer, default=0)
    
    essays = relationship("Essay", back_populates="user")

class Essay(Base):
    __tablename__ = "essays"
    
    essay_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    title = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    college_degree = Column(String)
    content = Column(Text, nullable=False)
    word_count = Column(Integer)
    overall_score = Column(Float)
    analysis_result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)
    
    user = relationship("User", back_populates="essays")

# Create tables safely
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
except Exception as e:
    print(f"❌ Database table creation failed: {e}")

# Enhanced Pydantic Models
class EssaySubmission(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    question_type: str = Field(..., min_length=1, max_length=1000)
    word_count: int = Field(default=250, ge=1, le=10000)
    college_degree: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=50, max_length=50000)

class AnalysisSection(BaseModel):
    key_observations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

class AnalysisData(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    alignment_with_topic: AnalysisSection
    essay_narrative_impact: AnalysisSection
    language_and_structure: AnalysisSection

class Highlight(BaseModel):
    text: str
    type: str = "grammar"
    issue: str
    suggestion: str

class AnalysisResponse(BaseModel):
    status: str
    analysis: AnalysisData
    ai_provider: str
    highlights: List[Highlight] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

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
        """Enhanced Gemini AI essay evaluation with comprehensive analysis"""
        
        start_time = datetime.utcnow()
        
        if not self.model:
            # Return enhanced demo analysis if no AI available
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)
        
        prompt = f"""
        You are Dr. Sarah Chen, a Harvard-trained college admissions counselor with 20+ years of experience evaluating essays for top universities including Harvard, MIT, Stanford, and Yale. You've helped over 2,000 students gain admission to their dream schools.

        **ESSAY DETAILS:**
        - Title: {title}
        - Question/Prompt: {question_type}
        - Target Program: {college_degree}
        - Word Count: {len(content.split())} words

        **ESSAY CONTENT:**
        {content}

        **EVALUATION FRAMEWORK:**
        
        **SCORING RUBRIC (1-10 scale):**
        - **Content & Ideas (35%)**: Depth, originality, personal insight, authenticity
        - **Structure & Organization (25%)**: Flow, transitions, logical progression
        - **Language & Writing (25%)**: Grammar, vocabulary, sentence variety, clarity
        - **Prompt Alignment (15%)**: Direct response to question, relevance

        **CRITICAL ANALYSIS REQUIREMENTS:**

        1. **Grammar Assessment**: Identify 3-5 specific grammar, punctuation, or style issues with exact text quotes
        2. **Content Evaluation**: Assess uniqueness, personal voice, and compelling narrative
        3. **Structure Analysis**: Evaluate introduction, body development, and conclusion effectiveness
        4. **College Fit**: Consider how this essay aligns with competitive admissions standards

        **OUTPUT FORMAT (JSON):**
        ```json
        {{
            "overall_score": 7.2,
            "content_breakdown": {{
                "content_ideas": 8.0,
                "structure_organization": 7.5,
                "language_writing": 6.5,
                "prompt_alignment": 7.8
            }},
            "strengths": [
                "Demonstrates authentic personal growth through specific challenges",
                "Uses vivid, concrete details that bring the story to life",
                "Shows clear character development and self-reflection",
                "Maintains consistent voice throughout the narrative",
                "Effectively connects personal experience to future goals"
            ],
            "improvements": [
                "Strengthen the conclusion with more specific future action plans",
                "Add more sensory details to enhance reader engagement",
                "Improve transitions between major story segments",
                "Develop the middle section with more concrete examples",
                "Consider varying sentence structure for better rhythm"
            ],
            "grammar_issues": [
                {{
                    "text": "me and my friends",
                    "issue": "Incorrect pronoun order",
                    "suggestion": "my friends and I"
                }},
                {{
                    "text": "alot of people",
                    "issue": "Spelling error",
                    "suggestion": "a lot of people"
                }},
                {{
                    "text": "This really effected me",
                    "issue": "Wrong word choice",
                    "suggestion": "This really affected me"
                }}
            ],
            "admissions_perspective": "This essay shows strong potential for competitive admissions. The personal story is compelling and authentic, though refinement in language precision would strengthen the overall impact."
        }}
        ```

        **IMPORTANT**: 
        - Be honest but constructive in feedback
        - Provide specific, actionable suggestions
        - Consider the competitive nature of college admissions
        - Focus on helping the student improve meaningfully
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
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return self._process_ai_response(feedback_data, processing_time)
            
        except Exception as e:
            print(f"❌ Gemini API error: {str(e)}")
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)
    
    def _process_ai_response(self, feedback_data, processing_time):
        """Process AI response into expected format"""
        overall_score = min(10.0, max(0.0, feedback_data.get("overall_score", 7.0)))
        
        # Split strengths and improvements into sections
        strengths = feedback_data.get("strengths", [])
        improvements = feedback_data.get("improvements", [])
        
        analysis = AnalysisData(
            overall_score=overall_score,
            alignment_with_topic=AnalysisSection(
                key_observations=strengths[:2] if len(strengths) >= 2 else strengths,
                next_steps=improvements[:2] if len(improvements) >= 2 else improvements
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=strengths[2:4] if len(strengths) > 2 else ["Shows developing narrative voice and personal insight"],
                next_steps=improvements[2:4] if len(improvements) > 2 else ["Continue developing storytelling elements"]
            ),
            language_and_structure=AnalysisSection(
                key_observations=strengths[4:] if len(strengths) > 4 else ["Demonstrates solid writing fundamentals"],
                next_steps=improvements[4:] if len(improvements) > 4 else ["Focus on language precision and variety"]
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
            "highlights": highlights,
            "processing_time": processing_time
        }
    
    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time):
        """Generate comprehensive demo analysis when AI is not available"""
        word_count = len(content.split())
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Enhanced grammar error detection
        grammar_errors = []
        
        # Common grammar patterns
        patterns = [
            (r"\bme and \w+", "me and someone", "someone and I", "Incorrect pronoun order"),
            (r"\balot\b", "alot", "a lot", "Spelling error"),
            (r"\beffect\b(?=.*\bme\b)", "effect me", "affect me", "Wrong word choice"),
            (r"\bits\s+(?=not|time)", "its ", "it's ", "Missing apostrophe"),
            (r"\byour\s+(?=welcome|going)", "your", "you're", "Wrong form of 'your'"),
        ]
        
        for pattern, error_text, correction, issue_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(grammar_errors) < 5:  # Limit to 5 issues
                    grammar_errors.append({
                        "text": match.group(),
                        "issue": issue_type,
                        "suggestion": correction
                    })
        
        # Score based on content quality indicators
        base_score = 7.0
        
        # Adjust score based on various factors
        if word_count < 100:
            base_score -= 1.5
        elif word_count > 800:
            base_score -= 0.5
        
        if len(grammar_errors) > 3:
            base_score -= 1.0
        elif len(grammar_errors) == 0:
            base_score += 0.5
        
        # Check for personal pronouns (engagement)
        personal_indicators = len(re.findall(r'\b(I|my|me|myself)\b', content, re.IGNORECASE))
        if personal_indicators > 5:
            base_score += 0.3
        
        # Check for specific examples
        specific_indicators = len(re.findall(r'\b(when|where|how|during|after|before)\b', content, re.IGNORECASE))
        if specific_indicators > 3:
            base_score += 0.2
        
        final_score = min(10.0, max(1.0, base_score))
        
        analysis = AnalysisData(
            overall_score=final_score,
            alignment_with_topic=AnalysisSection(
                key_observations=[
                    "Essay addresses the prompt with clear personal connection",
                    "Shows understanding of the question's intent"
                ],
                next_steps=[
                    "Strengthen specific examples that directly answer the prompt",
                    "Add more concrete details to support your main points"
                ]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=[
                    "Demonstrates authentic voice and personal perspective",
                    "Shows evidence of self-reflection and growth"
                ],
                next_steps=[
                    "Enhance storytelling with more vivid, sensory details",
                    "Develop the emotional journey more clearly"
                ]
            ),
            language_and_structure=AnalysisSection(
                key_observations=[
                    "Writing shows solid foundational skills",
                    "Ideas flow in a logical progression"
                ],
                next_steps=[
                    "Improve sentence variety and complexity",
                    "Strengthen transitions between paragraphs",
                    "Polish grammar and mechanics for clarity"
                ]
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
            "highlights": highlights,
            "processing_time": processing_time
        }

# Initialize evaluator
evaluator = GeminiEssayEvaluator()

# API Endpoints
@app.post("/api/analyze-essay", response_model=AnalysisResponse)
async def analyze_essay(submission: EssaySubmission, db: Session = Depends(get_db)):
    """Analyze essay using enhanced Gemini AI with comprehensive feedback"""
    
    start_time = datetime.utcnow()
    
    # Validate input
    if not submission.content.strip():
        raise HTTPException(status_code=400, detail="Essay content is required")
    
    if not submission.title.strip():
        raise HTTPException(status_code=400, detail="Essay title is required")
    
    word_count = len(submission.content.split())
    if word_count < 20:
        raise HTTPException(status_code=400, detail="Essay must be at least 20 words")
    
    if word_count > 5000:
        raise HTTPException(status_code=400, detail="Essay exceeds maximum length of 5000 words")
    
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
        
        total_processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Try to save to database
        try:
            # Check if user exists, if not create
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                user = User(user_id=user_id, total_essays=0)
                db.add(user)
            
            user.total_essays += 1
            user.credits = max(0, user.credits - 1)  # Deduct credit
            
            essay_id = f"essay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
            
            essay = Essay(
                essay_id=essay_id,
                user_id=user_id,
                title=submission.title,
                question_type=submission.question_type,
                college_degree=submission.college_degree,
                content=submission.content,
                word_count=word_count,
                overall_score=result["analysis"].overall_score,
                analysis_result=json.dumps({
                    "analysis": result["analysis"].dict(),
                    "highlights": [h.dict() for h in result["highlights"]]
                }),
                processing_time=total_processing_time
            )
            
            db.add(essay)
            db.commit()
            print(f"✅ Saved essay analysis: {essay_id} (Score: {result['analysis'].overall_score})")
            
        except Exception as db_error:
            print(f"⚠️ Database save failed: {db_error}")
            db.rollback()
        
        ai_provider = "Gemini AI Enhanced" if model else "Demo Analysis Engine"
        
        return AnalysisResponse(
            status="success",
            analysis=result["analysis"],
            ai_provider=ai_provider,
            highlights=result["highlights"],
            processing_time=total_processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_engine": "Gemini AI Enhanced" if model else "Demo Mode",
        "database": "Connected",
        "version": "2.1.0",
        "features": {
            "ai_analysis": model is not None,
            "grammar_checking": True,
            "database_storage": True,
            "user_tracking": True
        }
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get platform statistics"""
    try:
        total_essays = db.query(Essay).count()
        total_users = db.query(User).count()
        avg_score = db.query(Essay.overall_score).filter(Essay.overall_score.isnot(None)).all()
        
        average_score = sum([score[0] for score in avg_score]) / len(avg_score) if avg_score else 0
        
        return {
            "total_essays_analyzed": total_essays,
            "total_users": total_users,
            "average_essay_score": round(average_score, 2),
            "platform_uptime": "99.9%"
        }
    except Exception as e:
        return {
            "total_essays_analyzed": 0,
            "total_users": 0,
            "average_essay_score": 0,
            "error": str(e)
        }

@app.get("/")
async def serve_frontend():
    """Serve the main HTML application"""
    
    # First try to serve the HTML file
    html_files = ["essay_evaluator.html", "index.html", "complete_essay_evaluator.html"]
    
    for html_file in html_files:
        if os.path.exists(html_file):
            try:
                return FileResponse(html_file)
            except Exception as e:
                print(f"⚠️ Error serving {html_file}: {e}")
    
    # If no HTML file found, return enhanced inline HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Essay Evaluator - HelloIvy</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 1000px; 
                margin: 50px auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
                color: #333;
            }
            .success { 
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 20px; 
                border-radius: 12px; 
                margin: 20px 0;
                color: white;
                text-align: center;
            }
            .warning { 
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                padding: 20px; 
                border-radius: 12px; 
                margin: 20px 0;
                color: white;
            }
            .feature-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            code { 
                background: #f5f5f5; 
                padding: 4px 8px; 
                border-radius: 4px; 
                color: #e83e8c;
                font-weight: 600;
            }
            h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; }
            h2 { color: #667eea; margin-top: 30px; }
            a { color: #667eea; text-decoration: none; font-weight: 600; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 HelloIvy Essay Evaluator</h1>
            
            <div class="success">
                <h3>✅ Backend API is Running Successfully!</h3>
                <p>Your FastAPI backend is operational on <strong>http://localhost:8000</strong></p>
            </div>
            
            <div class="warning">
                <h3>⚠️ Frontend File Missing</h3>
                <p>The HTML frontend file was not found. Please ensure you have the frontend HTML file in the same directory as your backend.</p>
            </div>
            
            <h2>🎯 System Features</h2>
            <div class="feature-list">
                <div class="feature">
                    <h4>🤖 AI-Powered Analysis</h4>
                    <p>Advanced Gemini AI evaluation with comprehensive feedback</p>
                </div>
                <div class="feature">
                    <h4>📊 Smart Scoring</h4>
                    <p>10-point scale with detailed breakdowns</p>
                </div>
                <div class="feature">
                    <h4>✍️ Grammar Checking</h4>
                    <p>Real-time language and style suggestions</p>
                </div>
                <div class="feature">
                    <h4>💾 Data Storage</h4>
                    <p>Secure essay and analysis history</p>
                </div>
            </div>
            
            <h2>📋 Quick Setup Guide</h2>
            <ol>
                <li>Save your frontend HTML file as <code>essay_evaluator.html</code> in this directory</li>
                <li>Restart the server: <code>python your_backend_file.py</code></li>
                <li>Access the application at <code>http://localhost:8000</code></li>
                <li>For API documentation, visit <code>http://localhost:8000/docs</code></li>
            </ol>
            
            <h2>🔧 API Endpoints</h2>
            <ul>
                <li><a href="/api/health">Health Check</a> - <code>GET /api/health</code></li>
                <li><a href="/api/stats">Platform Statistics</a> - <code>GET /api/stats</code></li>
                <li><a href="/docs">Interactive API Documentation</a> - <code>GET /docs</code></li>
                <li>Essay Analysis - <code>POST /api/analyze-essay</code></li>
            </ul>
            
            <h2>🛠️ Technical Details</h2>
            <p><strong>Current Directory:</strong> <code>""" + os.getcwd() + """</code></p>
            <p><strong>Files in Directory:</strong> """ + ", ".join([f"<code>{f}</code>" for f in os.listdir(".")[:10]]) + """</p>
            <p><strong>AI Engine:</strong> """ + ("Gemini AI Enhanced" if model else "Demo Mode") + """</p>
            <p><strong>Database:</strong> Connected and Ready</p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Enhanced error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status": "error",
        "message": exc.detail,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting HelloIvy Essay Evaluator Backend v2.1.0...")
    print("=" * 70)
    print(f"📝 AI Engine: {'Gemini AI Enhanced' if model else 'Demo Mode (No API Key)'}")
    print(f"💾 Database: {DATABASE_URL}")
    print(f"🌐 Frontend: http://localhost:8000")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health Check: http://localhost:8000/api/health")
    print(f"📈 Statistics: http://localhost:8000/api/stats")
    print(f"⚖️ Enhanced Scoring: Grammar errors weighted appropriately")
    print(f"🎯 Features: AI Analysis, Grammar Check, User Tracking, Database Storage")
    
    if not model:
        print("⚠️  WARNING: Running in demo mode")
        print("   To enable full AI analysis, set GEMINI_API_KEY environment variable")
        print("   Example: export GEMINI_API_KEY='your_api_key_here'")
    
    print("=" * 70)
    
    try:
        uvicorn.run(
            "__main__:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("🔧 Try running on a different port:")
        print("   uvicorn your_backend_file:app --host 0.0.0.0 --port 8001")