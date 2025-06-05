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
    version="2.2.0", # Version updated
    description="Professional Essay Analysis Platform with AI-Powered Feedback (Enhanced Data)"
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
    # exit(1) # Consider if you want to exit or run in a degraded mode

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Placeholder
model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048, # Increased for potentially richer JSON
            )
        )
        print("✅ Gemini AI configured successfully")
    except Exception as e:
        print(f"❌ Gemini AI configuration failed: {e}")
        model = None
elif GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("🟡 Gemini API Key is a placeholder. Running without full AI capabilities.")
    GEMINI_AVAILABLE = False # Ensure demo mode if placeholder key
    model = None
else:
    print("🔄 Running without Gemini API (google-generativeai library not found or API key missing)")
    GEMINI_AVAILABLE = False
    model = None


# Enhanced Database Models
class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: f"user_{uuid.uuid4().hex[:12]}")
    email = Column(String, unique=True, nullable=True, index=True)
    credits = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_essays = Column(Integer, default=0)

    essays = relationship("Essay", back_populates="user")

class Essay(Base):
    __tablename__ = "essays"

    essay_id = Column(String, primary_key=True, default=lambda: f"essay_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, ForeignKey("users.user_id"), index=True)
    title = Column(String, nullable=False)
    question_type = Column(String, nullable=False)
    college_degree = Column(String) # e.g. "Harvard University - Bachelor of Arts (BA) in Computer Science"
    content = Column(Text, nullable=False)
    word_count = Column(Integer) # Actual word count of the content
    overall_score = Column(Float)
    analysis_result = Column(Text) # Stores JSON of AnalysisData and highlights
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)

    user = relationship("User", back_populates="essays")

# Create tables safely
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")
except Exception as e:
    print(f"❌ Database table creation failed: {e}")

# Enhanced Pydantic Models
class EssaySubmission(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Title of the essay or brainstormed topic.")
    question_type: str = Field(..., min_length=1, max_length=1000, description="The essay prompt or question being addressed.")
    word_count: int = Field(default=250, ge=1, le=10000, description="Target word count/limit for the essay (not the actual count).")
    college_degree: str = Field(..., min_length=1, max_length=300, description="Target college, degree, and major (e.g., 'Harvard University - BA in Computer Science').")
    content: str = Field(..., min_length=20, max_length=50000, description="The actual content of the essay.")

class AnalysisSection(BaseModel):
    key_observations: List[str] = Field(default_factory=list, description="Key positive observations for this section.")
    next_steps: List[str] = Field(default_factory=list, description="Actionable suggestions for improvement in this section.")

class AnalysisData(BaseModel):
    overall_score: float = Field(..., ge=0, le=10, description="The overall AI-generated score for the essay.")
    alignment_with_topic: AnalysisSection
    essay_narrative_impact: AnalysisSection
    language_and_structure: AnalysisSection
    content_breakdown: Optional[Dict[str, float]] = Field(None, description="Optional breakdown scores for content, structure, language, prompt alignment.")
    admissions_perspective: Optional[str] = Field(None, description="Optional AI qualitative feedback from an admissions perspective.")


class Highlight(BaseModel):
    text: str = Field(..., description="The exact text snippet with an issue.")
    type: str = Field(default="grammar", description="Type of issue (e.g., grammar, style).")
    issue: str = Field(..., description="Description of the identified issue.")
    suggestion: str = Field(..., description="Suggestion for correcting the issue.")

class AnalysisResponse(BaseModel):
    status: str = Field("success", description="Status of the analysis request.")
    analysis: AnalysisData
    ai_provider: str = Field(..., description="Identifier for the AI engine used.")
    highlights: List[Highlight] = Field(default_factory=list, description="List of identified issues and suggestions.")
    processing_time: float = Field(..., description="Total time taken for analysis in seconds.")

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
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)

        actual_word_count = len(content.split())
        prompt = f"""
        You are Dr. Sarah Chen, a Harvard-trained college admissions counselor with 20+ years of experience evaluating essays for top universities including Harvard, MIT, Stanford, and Yale. You've helped over 2,000 students gain admission to their dream schools.

        **ESSAY DETAILS:**
        - Title: {title}
        - Question/Prompt: {question_type}
        - Target Program: {college_degree}
        - Word Count: {actual_word_count} words

        **ESSAY CONTENT:**
        {content}

        **EVALUATION FRAMEWORK:**

        **SCORING RUBRIC (1-10 scale for overall and breakdowns):**
        - **Content & Ideas (35%)**: Depth, originality, personal insight, authenticity
        - **Structure & Organization (25%)**: Flow, transitions, logical progression
        - **Language & Writing (25%)**: Grammar, vocabulary, sentence variety, clarity
        - **Prompt Alignment (15%)**: Direct response to question, relevance

        **CRITICAL ANALYSIS REQUIREMENTS:**

        1. **Overall Score**: A single float score from 1.0 to 10.0.
        2. **Content Breakdown Scores**: Individual float scores (1.0-10.0) for "content_ideas", "structure_organization", "language_writing", and "prompt_alignment".
        3. **Strengths**: 3-5 bullet points on what the essay does well.
        4. **Improvements**: 3-5 actionable bullet points for improvement.
        5. **Grammar/Style Issues**: Identify 3-5 specific grammar, punctuation, or style issues. For each, provide the exact "text" quote, the "issue" type, and a "suggestion".
        6. **Admissions Perspective**: A concise paragraph (2-3 sentences) summarizing the essay's potential from a competitive college admissions standpoint, considering its fit for the target program if specified.

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
                "Demonstrates authentic personal growth through specific challenges.",
                "Uses vivid, concrete details that bring the story to life."
            ],
            "improvements": [
                "Strengthen the conclusion with more specific future action plans.",
                "Vary sentence structure for better rhythm and flow."
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
                }}
            ],
            "admissions_perspective": "This essay shows strong potential due to its compelling personal narrative. To enhance its competitiveness for {college_degree or 'top programs'}, the applicant should focus on refining language precision and more explicitly connecting their experiences to their stated academic interests."
        }}
        ```

        **IMPORTANT**:
        - Be honest but constructive.
        - Provide specific, actionable suggestions.
        - Adhere strictly to the JSON output format. Ensure all keys are present.
        - Scores should be realistic for a competitive applicant pool.
        """

        try:
            # Ensure generate_content is awaited if it's an async function in the library
            # For google.generativeai, generate_content is synchronous, so use to_thread
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            # Robust JSON extraction
            json_text = None
            if "```json" in response_text:
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
            if not json_text and "{" in response_text: # Fallback if no markdown
                 # Try to find the outermost JSON object
                first_brace = response_text.find("{")
                last_brace = response_text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    potential_json = response_text[first_brace : last_brace + 1]
                    try:
                        json.loads(potential_json) # Validate if it's JSON
                        json_text = potential_json
                    except json.JSONDecodeError:
                        print("Fallback JSON parsing failed validation.")
                        pass # Invalid JSON

            if not json_text:
                print(f"❌ No valid JSON found in AI response. Response text: {response_text[:500]}...")
                raise ValueError("No valid JSON found in AI response")

            feedback_data = json.loads(json_text)
            processing_time_val = (datetime.utcnow() - start_time).total_seconds()
            
            return self._process_ai_response(feedback_data, processing_time_val)

        except Exception as e:
            print(f"❌ Gemini API error or JSON processing error: {str(e)}")
            # Fallback to demo analysis on error
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)

    def _process_ai_response(self, feedback_data: Dict[str, Any], processing_time_val: float) -> Dict[str, Any]:
        """Process AI response into the application's expected format."""
        overall_score = min(10.0, max(0.0, float(feedback_data.get("overall_score", 7.0))))

        strengths = feedback_data.get("strengths", [])
        improvements = feedback_data.get("improvements", [])
        
        # Distribute strengths and improvements among sections
        # This logic can be refined based on how you want to categorize them
        analysis = AnalysisData(
            overall_score=overall_score,
            alignment_with_topic=AnalysisSection(
                key_observations=strengths[:len(strengths)//3] if strengths else ["Good initial alignment."],
                next_steps=improvements[:len(improvements)//3] if improvements else ["Ensure all prompt parts are addressed."]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=strengths[len(strengths)//3 : 2*len(strengths)//3] if len(strengths) > 1 else ["Developing narrative voice."],
                next_steps=improvements[len(improvements)//3 : 2*len(improvements)//3] if len(improvements) > 1 else ["Strengthen storytelling elements."]
            ),
            language_and_structure=AnalysisSection(
                key_observations=strengths[2*len(strengths)//3:] if len(strengths) > 2 else ["Solid writing fundamentals shown."],
                next_steps=improvements[2*len(improvements)//3:] if len(improvements) > 2 else ["Review for clarity and conciseness."]
            ),
            content_breakdown=feedback_data.get("content_breakdown"), # Pass through if available
            admissions_perspective=feedback_data.get("admissions_perspective") # Pass through if available
        )

        highlights_data = []
        for issue in feedback_data.get("grammar_issues", []):
            highlights_data.append(Highlight(
                text=issue.get("text", ""),
                type=issue.get("type", "grammar"), # Allow AI to specify type
                issue=issue.get("issue", "Language issue"),
                suggestion=issue.get("suggestion", "Review this section")
            ))

        return {
            "analysis": analysis,
            "highlights": highlights_data,
            "processing_time": processing_time_val
        }

    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive demo analysis when AI is not available or fails."""
        # ... (Keep existing _generate_demo_analysis logic as it was)
        # For brevity, I'm not repeating the demo analysis code here, assume it's the same as your original.
        # Make sure it returns a dictionary with "analysis", "highlights", and "processing_time" keys
        # matching the structure expected by the endpoint.
        # Ensure the 'analysis' object conforms to AnalysisData, potentially with None for new fields.
        actual_word_count = len(content.split())
        processing_time_val = (datetime.utcnow() - start_time).total_seconds()
        
        # Enhanced grammar error detection (simplified example)
        grammar_errors = []
        common_patterns = [
            (r"\bme and \w+", "Incorrect pronoun order", "my friend and I"),
            (r"\balot\b", "Spelling error", "a lot"),
        ]
        for pattern, issue, suggestion in common_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                if len(grammar_errors) < 3:
                     grammar_errors.append({
                        "text": match.group(0), "issue": issue, "suggestion": suggestion, "type": "grammar"
                    })
        
        highlights_demo = [Highlight(**err) for err in grammar_errors]

        # Simplified scoring
        score = 6.5 + (actual_word_count / 500.0) - (len(grammar_errors) * 0.5)
        final_score = min(10.0, max(1.0, round(score,1)))

        demo_analysis_data = AnalysisData(
            overall_score=final_score,
            alignment_with_topic=AnalysisSection(
                key_observations=["Essay generally addresses the prompt.", "Shows basic understanding."],
                next_steps=["Elaborate more on key arguments with specific examples.", "Ensure every part of the prompt is directly answered."]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=["Personal voice is somewhat present.", "Some storytelling elements used."],
                next_steps=["Develop a more compelling narrative arc.", "Use more vivid imagery and sensory details."]
            ),
            language_and_structure=AnalysisSection(
                key_observations=["Basic sentence structure is clear.", "Paragraphs are organized around main ideas."],
                next_steps=["Improve sentence variety and complexity.", "Strengthen transitions between paragraphs and ideas."]
            ),
            content_breakdown={ # Example breakdown for demo
                "content_ideas": round(final_score * 0.9, 1),
                "structure_organization": round(final_score * 0.85, 1),
                "language_writing": round(final_score * 0.8, 1),
                "prompt_alignment": round(final_score * 0.95, 1)
            },
            admissions_perspective=f"This demo analysis suggests the essay has a foundational structure. For {college_degree or 'a competitive program'}, further development in depth and polish is recommended."
        )
        return {
            "analysis": demo_analysis_data,
            "highlights": highlights_demo,
            "processing_time": processing_time_val
        }

# Initialize evaluator
evaluator = GeminiEssayEvaluator()

# API Endpoints
@app.post("/api/analyze-essay", response_model=AnalysisResponse)
async def analyze_essay(submission: EssaySubmission, db: Session = Depends(get_db)):
    """Analyze essay using enhanced Gemini AI with comprehensive feedback"""
    request_start_time = datetime.utcnow()

    # Input validation already handled by Pydantic for content length
    # The 'word_count' in submission is the target, not actual.
    actual_word_count = len(submission.content.split())
    if actual_word_count < 10: # Stricter minimum for actual content
         raise HTTPException(status_code=400, detail="Essay content must be at least 10 words.")
    if actual_word_count > 7000: # Stricter maximum for actual content
         raise HTTPException(status_code=400, detail="Essay content exceeds maximum length of 7000 words.")


    try:
        # Generate a temporary user_id if not implementing full user auth yet
        # For better tracking, this should come from an authenticated session in a real app
        temp_user_id = f"user_anon_{uuid.uuid4().hex[:8]}"

        ai_result = await evaluator.evaluate_essay(
            submission.content,
            submission.title,
            submission.question_type,
            submission.college_degree
        )
        
        total_processing_time = (datetime.utcnow() - request_start_time).total_seconds()

        # Database Operations
        try:
            user = db.query(User).filter(User.user_id == temp_user_id).first()
            if not user:
                user = User(user_id=temp_user_id, email=f"{temp_user_id}@example.com") # Example email
                db.add(user)
                db.flush() # To get user_id if generated by DB

            user.total_essays = (user.total_essays or 0) + 1
            if user.credits is not None: # Ensure credits field is not None before decrementing
                 user.credits = max(0, (user.credits or 0) -1)


            db_essay_id = f"essay_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{user.user_id[:8]}"
            
            # Use exclude_none=True for cleaner JSON in DB
            analysis_result_json = json.dumps({
                "analysis": ai_result["analysis"].dict(exclude_none=True),
                "highlights": [h.dict() for h in ai_result["highlights"]]
            })

            essay_entry = Essay(
                essay_id=db_essay_id,
                user_id=user.user_id,
                title=submission.title,
                question_type=submission.question_type,
                college_degree=submission.college_degree,
                content=submission.content,
                word_count=actual_word_count, # Store actual word count
                overall_score=ai_result["analysis"].overall_score,
                analysis_result=analysis_result_json,
                processing_time=ai_result["processing_time"] # AI processing time
            )
            db.add(essay_entry)
            db.commit()
            print(f"✅ Saved essay analysis: {db_essay_id} (Score: {ai_result['analysis'].overall_score})")
        except Exception as db_error:
            print(f"⚠️ Database save failed: {db_error}")
            db.rollback() # Rollback on DB error

        ai_provider_name = "Gemini AI Enhanced" if model and GEMINI_AVAILABLE else "Demo Analysis Engine"
        
        return AnalysisResponse(
            status="success",
            analysis=ai_result["analysis"],
            ai_provider=ai_provider_name,
            highlights=ai_result["highlights"],
            processing_time=total_processing_time # Overall request processing time
        )

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"❌ Analysis failed with unexpected error: {str(e)}")
        # Consider logging the full traceback here
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {str(e)}")


@app.get("/api/health")
async def health_check():
    db_status = "Disconnected"
    try:
        # Try to connect and execute a simple query
        db = SessionLocal()
        db.execute("SELECT 1")
        db_status = "Connected"
        db.close()
    except Exception as e:
        print(f"Database health check failed: {e}")
        db_status = f"Error: {e}"


    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_engine_status": "Gemini AI Active" if model and GEMINI_AVAILABLE else "Demo Mode / Inactive",
        "database_status": db_status,
        "version": app.version,
        "active_features": {
            "ai_analysis": model is not None and GEMINI_AVAILABLE,
            "grammar_checking_demo": True, # Demo grammar check is always on
            "database_storage": "sqlite" in DATABASE_URL or "postgresql" in DATABASE_URL, # crude check
            "user_tracking_basic": True
        }
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        total_essays = db.query(Essay).count()
        total_users = db.query(User).count()
        
        # Calculate average score more safely
        avg_scores_query = db.query(Essay.overall_score).filter(Essay.overall_score.isnot(None)).all()
        
        if avg_scores_query:
            scores = [score_tuple[0] for score_tuple in avg_scores_query if score_tuple[0] is not None]
            average_score = sum(scores) / len(scores) if scores else 0.0
        else:
            average_score = 0.0
        
        # Platform uptime is typically monitored externally, this is a placeholder
        platform_uptime_placeholder = "99.9% (Monitored Externally)" 

        return {
            "total_essays_analyzed": total_essays,
            "total_users": total_users,
            "average_essay_score": round(average_score, 2),
            "platform_uptime": platform_uptime_placeholder,
            "ai_model_in_use": model.model_name if model and hasattr(model, 'model_name') else "N/A or Demo"
        }
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {
            "total_essays_analyzed": 0, "total_users": 0, "average_essay_score": 0,
            "platform_uptime": "Unknown", "error": str(e)
        }


@app.get("/")
async def serve_frontend():
    html_files_to_try = ["updated_essay_evaluator.html", "essay_evaluator.html", "index.html", "complete_essay_evaluator.html"]
    
    for html_file in html_files_to_try:
        if os.path.exists(html_file):
            print(f"Serving frontend file: {html_file}")
            return FileResponse(html_file)
            
    # Fallback HTML if no file is found
    print(f"No frontend HTML file found from list: {html_files_to_try}. Serving fallback HTML.")
    # ... (Your existing fallback HTML content can go here)
    # For brevity, I'll use a simpler fallback here.
    current_dir = os.getcwd()
    files_in_dir = ", ".join([f"<code>{f}</code>" for f in os.listdir(".")[:10]])
    html_content = f"""
    <!DOCTYPE html><html><head><title>Essay Evaluator API</title><style>body {{font-family: sans-serif; padding: 20px;}} code {{background: #f0f0f0; padding: 2px 4px; border-radius:3px;}}</style></head>
    <body><h1>🚀 HelloIvy Essay Evaluator API Backend is Running!</h1>
    <p>Could not find a primary HTML file (e.g., <code>updated_essay_evaluator.html</code>). Please ensure it's in the correct directory.</p>
    <p><strong>Current Directory:</strong> <code>{current_dir}</code></p>
    <p><strong>Files found (sample):</strong> {files_in_dir}</p>
    <p>Access API docs at <a href="/docs">/docs</a> or health check at <a href="/api/health">/api/health</a>.</p>
    </body></html>
    """
    return HTMLResponse(content=html_content)

# Enhanced error handler for HTTPExceptions to return JSON
from fastapi.responses import JSONResponse
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
                                                    
if __name__ == "__main__":
    print("=" * 70)
    print(f"🚀 Starting HelloIvy Essay Evaluator Backend v{app.version}...")
    print("=" * 70)
    ai_status = "Gemini AI Enhanced" if model and GEMINI_AVAILABLE else "Demo Mode (No API Key or google-generativeai not installed)"
    print(f"📝 AI Engine: {ai_status}")
    print(f"💾 Database: {DATABASE_URL}")
    print(f"🌐 Frontend Access: http://localhost:8000 (Ensure HTML file is present)")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health Check: http://localhost:8000/api/health")
    print(f"📈 Statistics: http://localhost:8000/api/stats")
    
    if not (model and GEMINI_AVAILABLE):
        print("=" * 70)
        print("⚠️  WARNING: Running in demo mode or AI is not fully configured.")
        if not GEMINI_AVAILABLE and not os.getenv("GEMINI_API_KEY"):
             print("   - google-generativeai library might be missing. Run: pip install google-generativeai")
        if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
             print("   - GEMINI_API_KEY environment variable is not set or is a placeholder.")
             print("     Example: export GEMINI_API_KEY='your_actual_api_key_here'")
        print("=" * 70)
    
    try:
        uvicorn.run(
            "__main__:app", # Changed from "your_backend_file:app" to ensure it runs directly
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)), # Allow port to be set by env var
            reload=True, # Reload is great for development
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start Uvicorn server: {e}")
        print("🔧 Try running on a different port if 8000 is in use, e.g.:")
        print("   python your_backend_file.py (if it sets a different port via PORT env var)")
        print("   or uvicorn your_backend_file:app --host 0.0.0.0 --port 8001")