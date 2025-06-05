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
    version="2.3.0", # Version updated for new suggestion feature
    description="Professional Essay Analysis Platform with AI-Powered Feedback (Enhanced Data & Suggestions)"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend domain
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
    # Consider if you want to exit or run in a degraded mode

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # No default placeholder here, rely on env
model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash', # Or your preferred Gemini model
            generation_config=genai.types.GenerationConfig(
                temperature=0.6, # Slightly lower for more factual editorial suggestions
                top_p=0.85,
                top_k=40,
                max_output_tokens=3072, # Increased for potentially richer JSON and more suggestions
            ),
            safety_settings={ # Adjust safety settings as needed
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )
        print("✅ Gemini AI configured successfully")
    except Exception as e:
        print(f"❌ Gemini AI configuration failed: {e}")
        model = None
elif not GEMINI_API_KEY:
    print("🟡 Gemini API Key (GEMINI_API_KEY env var) not found. Running without full AI capabilities.")
    GEMINI_AVAILABLE = False 
    model = None
else: # google-generativeai library not found
    print("🔄 Running without Gemini API (google-generativeai library not found)")
    GEMINI_AVAILABLE = False
    model = None


# Enhanced Database Models
class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: f"user_{uuid.uuid4().hex[:12]}")
    email = Column(String, unique=True, nullable=True, index=True) # Optional for now
    credits = Column(Integer, default=10) # Example credit system
    created_at = Column(DateTime, default=datetime.utcnow)
    total_essays = Column(Integer, default=0)

    essays = relationship("Essay", back_populates="user")

class Essay(Base):
    __tablename__ = "essays"

    essay_id = Column(String, primary_key=True, default=lambda: f"essay_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, ForeignKey("users.user_id"), index=True)
    title = Column(String, nullable=False)
    question_type = Column(String, nullable=False) # The prompt
    college_degree = Column(String) # e.g. "Harvard University - Bachelor of Arts (BA) in Computer Science"
    content = Column(Text, nullable=False)
    word_count = Column(Integer) # Actual word count of the content
    overall_score = Column(Float) # AI overall score
    analysis_result = Column(Text) # Stores JSON of AnalysisData and highlights
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float) # AI processing time

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
    text: str = Field(..., description="The exact text snippet with an issue. For additions, text immediately PRECEDING the suggested addition.")
    type: str = Field(default="grammar", description="Type of issue (e.g., spelling, replace_candidate, remove_candidate, add_candidate, grammar, style).")
    issue: str = Field(..., description="Description of the identified issue.")
    suggestion: str = Field(..., description="Suggestion for correcting the issue. For 'remove_candidate', can be empty if 'issue' is clear.")

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
        """Enhanced Gemini AI essay evaluation with comprehensive analysis and specific editorial suggestions."""
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

        1.  **Overall Score**: A single float score from 1.0 to 10.0.
        2.  **Content Breakdown Scores**: Individual float scores (1.0-10.0) for "content_ideas", "structure_organization", "language_writing", and "prompt_alignment".
        3.  **Alignment with Topic Feedback**:
            - "alignment_topic_observations": [List of 2-3 key positive observations related to prompt alignment, relevance, and addressing the question directly.]
            - "alignment_topic_next_steps": [List of 2-3 actionable suggestions to improve topic alignment.]
        4.  **Essay Narrative Impact Feedback**:
            - "narrative_impact_observations": [List of 2-3 key positive observations on storytelling, personal voice, depth of reflection, and impact.]
            - "narrative_impact_next_steps": [List of 2-3 actionable suggestions to enhance narrative impact and engagement.]
        5.  **Language and Structure Feedback**:
            - "language_structure_observations": [List of 2-3 key positive observations regarding clarity, flow, grammar, vocabulary, and organization.]
            - "language_structure_next_steps": [List of 2-3 actionable suggestions to improve language precision, sentence structure, and overall coherence.]
        
        6.  **Specific Editorial Suggestions (Populate these into the "grammar_issues" list)**:
            Identify 4-6 distinct, highly specific editorial suggestions from the essay. For each:
            -   "text": The exact, verbatim text snippet from the essay that needs attention. For additions (like a comma), this 'text' should be the word or short phrase immediately PRECEDING where the addition is needed.
            -   "type": Categorize the suggestion using one of these exact strings:
                *   "spelling": For clear misspellings.
                *   "replace_candidate": For phrases that could be improved by replacing them.
                *   "remove_candidate": For redundant, awkward, or unnecessary phrases/sentences that should be removed.
                *   "add_candidate": For suggesting additions, typically punctuation (like a comma) or very short connecting words.
                *   "grammar": For other specific grammatical errors (e.g., subject-verb agreement, pronoun errors).
                *   "style": For minor stylistic improvements (e.g., word choice, conciseness not covered by replace/remove).
            -   "issue": A concise description of why the text needs attention (e.g., "Misspelled word.", "Awkward phrasing.", "Redundant.", "Missing comma after introductory phrase.", "Incorrect verb tense.").
            -   "suggestion": 
                *   For "spelling": The correctly spelled word.
                *   For "replace_candidate": The improved phrase.
                *   For "remove_candidate": An empty string ("") or a brief note like "Consider removing this phrase for conciseness." if the `issue` field isn't sufficient. An empty string is preferred if the `issue` is clear.
                *   For "add_candidate": The exact punctuation or short word(s) to add (e.g., ",", "and").
                *   For "grammar" / "style": The corrected phrase or a descriptive suggestion.
            Ensure these suggestions are genuinely helpful for improving clarity, conciseness, correctness, and impact. Prioritize common student errors.

        7.  **Admissions Perspective**: A concise paragraph (2-3 sentences) summarizing the essay's potential from a competitive college admissions standpoint, considering its fit for the target program if specified.

        **OUTPUT FORMAT (JSON EXAMPLE - focus on `grammar_issues` for new types):**
        ```json
        {{
            "overall_score": 7.2,
            "content_breakdown": {{
                "content_ideas": 8.0,
                "structure_organization": 7.5,
                "language_writing": 6.5,
                "prompt_alignment": 7.8
            }},
            "alignment_topic_observations": ["Effectively addresses the core question."],
            "alignment_topic_next_steps": ["Consider explicitly stating the connection to your long-term goals mentioned in the prompt."],
            "narrative_impact_observations": ["The personal anecdote is compelling and emotionally resonant."],
            "narrative_impact_next_steps": ["Expand slightly on the 'aha!' moment to deepen its impact."],
            "language_structure_observations": ["The essay is generally well-organized with clear paragraphs."],
            "language_structure_next_steps": ["Proofread carefully for minor agreement errors and vary sentence beginnings."],
            "grammar_issues": [
                {{
                    "text": "pasion",
                    "type": "spelling",
                    "issue": "Misspelled word.",
                    "suggestion": "passion"
                }},
                {{
                    "text": "However this", 
                    "type": "add_candidate",
                    "issue": "Missing comma after introductory element.",
                    "suggestion": "," 
                }},
                {{
                    "text": "my voice myself", 
                    "type": "replace_candidate",
                    "issue": "A bit redundant; 'myself' implies finding your voice in this context.",
                    "suggestion": "myself" 
                }},
                {{
                    "text": "The support and encouragement I received from the audience bolstered my self-belief, pushing me to further explore my creative potential.",
                    "type": "remove_candidate",
                    "issue": "This sentence is lengthy and its core idea might be shown more effectively or concisely.",
                    "suggestion": "" 
                }},
                {{
                    "text": "me and my friends",
                    "type": "grammar",
                    "issue": "Incorrect pronoun order.",
                    "suggestion": "my friends and I"
                }}
            ],
            "admissions_perspective": "This essay shows strong potential due to its compelling personal narrative. To enhance its competitiveness for {college_degree or 'top programs'}, the applicant should focus on refining language precision and more explicitly connecting their experiences to their stated academic interests."
        }}
        ```

        **IMPORTANT**:
        - Be honest but constructive.
        - Provide specific, actionable suggestions.
        - Adhere strictly to the JSON output format. Ensure all keys, including all specified "type" values for `grammar_issues`, are used correctly.
        - Scores should be realistic for a competitive applicant pool.
        """

        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            json_text = None
            if "```json" in response_text:
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
            if not json_text and "{" in response_text:
                first_brace = response_text.find("{")
                last_brace = response_text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    potential_json = response_text[first_brace : last_brace + 1]
                    try:
                        json.loads(potential_json)
                        json_text = potential_json
                    except json.JSONDecodeError:
                        print("Fallback JSON parsing failed validation.")
                        pass

            if not json_text:
                print(f"❌ No valid JSON found in AI response. Response text: {response_text[:500]}...")
                raise ValueError("No valid JSON found in AI response")

            feedback_data = json.loads(json_text)
            processing_time_val = (datetime.utcnow() - start_time).total_seconds()
            
            return self._process_ai_response(feedback_data, processing_time_val)

        except Exception as e:
            print(f"❌ Gemini API error or JSON processing error: {str(e)}")
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)

    def _process_ai_response(self, feedback_data: Dict[str, Any], processing_time_val: float) -> Dict[str, Any]:
        """Process AI response into the application's expected format using direct categorization."""
        overall_score = min(10.0, max(0.0, float(feedback_data.get("overall_score", 7.0))))

        analysis = AnalysisData(
            overall_score=overall_score,
            alignment_with_topic=AnalysisSection(
                key_observations=feedback_data.get("alignment_topic_observations", ["Review for prompt alignment."]),
                next_steps=feedback_data.get("alignment_topic_next_steps", ["Ensure all parts of the prompt are addressed."])
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=feedback_data.get("narrative_impact_observations", ["Assess narrative strength."]),
                next_steps=feedback_data.get("narrative_impact_next_steps", ["Strengthen storytelling elements."])
            ),
            language_and_structure=AnalysisSection(
                key_observations=feedback_data.get("language_structure_observations", ["Check clarity and organization."]),
                next_steps=feedback_data.get("language_structure_next_steps", ["Review for grammar and flow."])
            ),
            content_breakdown=feedback_data.get("content_breakdown"),
            admissions_perspective=feedback_data.get("admissions_perspective")
        )

        highlights_data = []
        for issue in feedback_data.get("grammar_issues", []): # This now includes new types
            highlights_data.append(Highlight(
                text=issue.get("text", ""),
                type=issue.get("type", "grammar"), 
                issue=issue.get("issue", "Editorial suggestion"),
                suggestion=issue.get("suggestion", "") # Default to empty for remove_candidate if not provided
            ))

        return {
            "analysis": analysis,
            "highlights": highlights_data,
            "processing_time": processing_time_val
        }

    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive demo analysis when AI is not available or fails, now with new suggestion types."""
        actual_word_count = len(content.split())
        processing_time_val = (datetime.utcnow() - start_time).total_seconds()
        
        demo_highlights = []
        # Example spelling
        if "teh" in content.lower():
            demo_highlights.append(Highlight(text="teh", type="spelling", issue="Misspelled word.", suggestion="the"))
        # Example remove
        if "in actual fact" in content.lower():
             demo_highlights.append(Highlight(text="in actual fact", type="remove_candidate", issue="Redundant phrase.", suggestion="")) # Empty suggestion is fine
        # Example replace
        if "utilize" in content.lower():
             demo_highlights.append(Highlight(text="utilize", type="replace_candidate", issue="Consider simpler word.", suggestion="use"))
        # Example add (punctuation) - harder to demo robustly with regex
        if "However " in content and not "However," in content : # Simplified check
            demo_highlights.append(Highlight(text="However", type="add_candidate", issue="Missing comma after introductory element.", suggestion=","))
        # Generic grammar if few specifics found
        if len(demo_highlights) < 2 and len(content.split()) > 30:
             demo_highlights.append(Highlight(text=content.split()[5] if len(content.split()) > 5 else "essay", type="grammar", issue="General grammar review needed.", suggestion="Check this section for clarity."))

        final_score = min(10.0, max(1.0, round(6.5 + (actual_word_count / 500.0) - (len(demo_highlights) * 0.3),1)))

        demo_analysis_data = AnalysisData(
            overall_score=final_score,
            alignment_with_topic=AnalysisSection(
                key_observations=["Essay generally addresses the prompt (Demo).", "Shows basic understanding (Demo)."],
                next_steps=["Elaborate more on key arguments with specific examples (Demo).", "Ensure every part of the prompt is directly answered (Demo)."]
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=["Personal voice is somewhat present (Demo).", "Some storytelling elements used (Demo)."],
                next_steps=["Develop a more compelling narrative arc (Demo).", "Use more vivid imagery and sensory details (Demo)."]
            ),
            language_and_structure=AnalysisSection(
                key_observations=["Basic sentence structure is clear (Demo).", "Paragraphs are organized around main ideas (Demo)."],
                next_steps=["Improve sentence variety and complexity (Demo).", "Strengthen transitions between paragraphs and ideas (Demo)."]
            ),
            content_breakdown={ 
                "content_ideas": round(final_score * 0.9, 1), "structure_organization": round(final_score * 0.85, 1),
                "language_writing": round(final_score * 0.8, 1), "prompt_alignment": round(final_score * 0.95, 1)
            },
            admissions_perspective=f"This demo analysis suggests the essay has a foundational structure. For {college_degree or 'a competitive program'}, further development in depth and polish is recommended."
        )
        return {
            "analysis": demo_analysis_data,
            "highlights": demo_highlights[:4], # Limit demo highlights
            "processing_time": processing_time_val
        }

# Initialize evaluator
evaluator = GeminiEssayEvaluator()

# API Endpoints
@app.post("/api/analyze-essay", response_model=AnalysisResponse)
async def analyze_essay(submission: EssaySubmission, db: Session = Depends(get_db)):
    """Analyze essay using enhanced Gemini AI with comprehensive feedback and specific editorial suggestions."""
    request_start_time = datetime.utcnow()

    actual_word_count = len(submission.content.split())
    if actual_word_count < 10:
         raise HTTPException(status_code=400, detail="Essay content must be at least 10 words.")
    if actual_word_count > 7000:
         raise HTTPException(status_code=400, detail="Essay content exceeds maximum length of 7000 words.")

    try:
        # TODO: Replace with actual authenticated user ID in production
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
            if not user: # Create a demo user if not found
                user = User(user_id=temp_user_id, email=f"{temp_user_id}@example.com") 
                db.add(user)
                db.flush() 

            user.total_essays = (user.total_essays or 0) + 1
            if user.credits is not None:
                 user.credits = max(0, (user.credits or 0) -1)

            db_essay_id = f"essay_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{user.user_id[:6]}"
            
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
                word_count=actual_word_count,
                overall_score=ai_result["analysis"].overall_score,
                analysis_result=analysis_result_json,
                processing_time=ai_result["processing_time"] 
            )
            db.add(essay_entry)
            db.commit()
            print(f"✅ Saved essay analysis: {db_essay_id} (Score: {ai_result['analysis'].overall_score})")
        except Exception as db_error:
            print(f"⚠️ Database save failed: {db_error}")
            db.rollback() 

        ai_provider_name = "Gemini AI Enhanced" if model and GEMINI_AVAILABLE else "Demo Analysis Engine"
        
        return AnalysisResponse(
            status="success",
            analysis=ai_result["analysis"],
            ai_provider=ai_provider_name,
            highlights=ai_result["highlights"],
            processing_time=total_processing_time
        )

    except HTTPException: 
        raise
    except Exception as e:
        print(f"❌ Analysis failed with unexpected error: {str(e)}")
        # In production, log the full traceback: import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {str(e)}")


@app.get("/api/health")
async def health_check():
    db_status = "Disconnected"
    try:
        db = SessionLocal()
        db.execute("SELECT 1") # Use text("SELECT 1") with SQLAlchemy 2.0+
        db_status = "Connected"
        db.close()
    except Exception as e:
        print(f"Database health check failed: {e}")
        db_status = f"Error: {str(e)[:100]}" # Truncate long error messages

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_engine_status": "Gemini AI Active" if model and GEMINI_AVAILABLE else "Demo Mode / Inactive",
        "database_status": db_status,
        "version": app.version,
        "active_features": {
            "ai_analysis": model is not None and GEMINI_AVAILABLE,
            "editorial_suggestions": model is not None and GEMINI_AVAILABLE, # New feature flag
            "database_storage": "sqlite" in DATABASE_URL or "postgresql" in DATABASE_URL,
            "user_tracking_basic": True
        }
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        total_essays = db.query(Essay).count()
        total_users = db.query(User).count()
        
        avg_scores_query = db.query(Essay.overall_score).filter(Essay.overall_score.isnot(None)).all()
        
        if avg_scores_query:
            scores = [score_tuple[0] for score_tuple in avg_scores_query if score_tuple[0] is not None]
            average_score = sum(scores) / len(scores) if scores else 0.0
        else:
            average_score = 0.0
        
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
        # Consider logging the full traceback here
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/")
async def serve_frontend():
    # Prioritize the most up-to-date HTML file if multiple exist
    html_files_to_try = ["essay_evaluator.html", "updated_essay_evaluator.html", "index.html", "complete_essay_evaluator.html"]
    
    for html_file in html_files_to_try:
        if os.path.exists(html_file):
            print(f"Serving frontend file: {html_file}")
            return FileResponse(html_file)
            
    print(f"No primary HTML file found from list: {html_files_to_try}. Serving fallback HTML.")
    current_dir = os.getcwd()
    files_in_dir = os.listdir(".")
    sample_files = ", ".join([f"<code>{f}</code>" for f in files_in_dir[:10]]) + ("..." if len(files_in_dir) > 10 else "")
    
    html_content = f"""
    <!DOCTYPE html><html><head><title>Essay Evaluator API</title>
    <style>body {{font-family: sans-serif; padding: 20px; line-height: 1.6;}} code {{background: #f0f0f0; padding: 2px 4px; border-radius:3px;}} h1 {{ color: #333; }} p {{ margin-bottom: 10px; }}</style></head>
    <body><h1>🚀 HelloIvy Essay Evaluator API Backend is Running!</h1>
    <p>Could not find a primary HTML file (e.g., <code>essay_evaluator.html</code>). Please ensure it's in the correct directory.</p>
    <p><strong>Current Directory:</strong> <code>{current_dir}</code></p>
    <p><strong>Files found (sample):</strong> {sample_files}</p>
    <hr><p>To run the application:</p>
    <ol>
        <li>Ensure you have an HTML file (like <code>essay_evaluator.html</code> from previous steps) in the same directory as this Python script.</li>
        <li>Set your <code>GEMINI_API_KEY</code> environment variable: <code>export GEMINI_API_KEY='your_actual_api_key_here'</code> (or equivalent for your OS).</li>
        <li>Run the backend: <code>python {os.path.basename(__file__)}</code></li>
        <li>Open your browser to <a href="http://localhost:{os.getenv("PORT", 8000)}">http://localhost:{os.getenv("PORT", 8000)}</a></li>
    </ol>
    <p>API docs at <a href="/docs">/docs</a> | Health check at <a href="/api/health">/api/health</a>.</p>
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
    port = int(os.getenv("PORT", 8000))
    print("=" * 70)
    print(f"🚀 Starting HelloIvy Essay Evaluator Backend v{app.version}...")
    print("=" * 70)
    
    ai_status_message = "Gemini AI Enhanced (Full Features)" if model and GEMINI_AVAILABLE else \
                       ("Demo Mode (GEMINI_API_KEY missing or invalid)" if GEMINI_AVAILABLE else 
                        "Demo Mode (google-generativeai library not installed AND/OR API Key missing)")
    print(f"📝 AI Engine: {ai_status_message}")
    print(f"💾 Database: {DATABASE_URL}")
    print(f"🌐 Frontend Access: http://localhost:{port} (Ensure HTML file is present)")
    print(f"📊 API Docs: http://localhost:{port}/docs")
    print(f"🔧 Health Check: http://localhost:{port}/api/health")
    print(f"📈 Statistics: http://localhost:{port}/api/stats")
    
    if not (model and GEMINI_AVAILABLE):
        print("=" * 70)
        print("⚠️  WARNING: Running in demo mode. AI features will be limited.")
        if not GEMINI_AVAILABLE:
             print("   - `google-generativeai` library might be missing. Install it: `pip install google-generativeai`")
        if not os.getenv("GEMINI_API_KEY"):
             print("   - `GEMINI_API_KEY` environment variable is not set.")
             print("     To enable full AI features, set this variable with your Google AI Studio API key.")
             print("     Example (Linux/macOS): export GEMINI_API_KEY='your_api_key_here'")
             print("     Example (Windows CMD): set GEMINI_API_KEY=your_api_key_here")
             print("     Example (Windows PowerShell): $env:GEMINI_API_KEY='your_api_key_here'")
        print("=" * 70)
    
    try:
        # Assuming this script is named working_backend.py or similar
        # For Uvicorn, the string should be "filename_without_py:app_variable_name"
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        uvicorn.run(
            f"{script_name}:app", 
            host="0.0.0.0",
            port=port,
            reload=True, 
            log_level="info",
            access_log=True
        )
    except RuntimeError as e: # Catch common Uvicorn error if run directly with `python file.py` in some setups
        if "Cannot run ASGI application" in str(e) and "uvicorn.run" in str(e):
             print(f"❌ Error: {e}")
             print("💡 Try running with Uvicorn directly: ")
             print(f"   uvicorn {script_name}:app --host 0.0.0.0 --port {port} --reload")
        else:
             print(f"❌ Failed to start Uvicorn server: {e}")
    except Exception as e:
        print(f"❌ Failed to start Uvicorn server: {e}")