# ==============================================================================
# HelloIvy Essay Evaluator - All-in-One Backend with Authentication
# Version: 3.0.3 (Single-File Edition - Verified Feature Complete)
# ==============================================================================
import os
import json
import asyncio
import uuid
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# --- Dependency Imports ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ WARNING: google-generativeai not installed. AI features will be disabled.")

# --- Initial Setup ---
load_dotenv()

# ==============================================================================
# 1. DATABASE SETUP
# ==============================================================================
DATABASE_URL = "mysql+pymysql://helloivy_user:YourStrongPassword_123!@localhost:3306/helloivy_db"

print("="*60)
print(f"!!! USING HARDCODED DATABASE URL FOR TESTING !!!")
print(f"URL: {DATABASE_URL}")
print("="*60)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================================================================
# 2. AUTHENTICATION HELPERS (Passwords & JWT Tokens)
# ==============================================================================
# IMPORTANT: Generate a new secret key for production: `openssl rand -hex 32`
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ==============================================================================
# 3. Pydantic Schemas (Data Validation Models)
# ==============================================================================
class EssayBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Title of the essay or brainstormed topic.")
    question_type: str = Field(..., min_length=1, max_length=1000, description="The essay prompt or question being addressed.")
    college_degree: str = Field(..., min_length=1, max_length=300, description="Target college, degree, and major.")
    content: str = Field(..., min_length=20, max_length=50000, description="The actual content of the essay.")

class EssaySubmission(EssayBase):
    pass

class EssayResponseSchema(EssayBase):
    id: str
    user_id: str
    created_at: datetime
    overall_score: Optional[float] = None
    class Config: from_attributes = True

class AnalysisSection(BaseModel):
    key_observations: List[str]
    next_steps: List[str]

class AnalysisData(BaseModel):
    overall_score: float
    alignment_with_topic: AnalysisSection
    essay_narrative_impact: AnalysisSection
    language_and_structure: AnalysisSection
    content_breakdown: Optional[Dict[str, float]] = None
    admissions_perspective: Optional[str] = None

class Highlight(BaseModel):
    text: str; type: str; issue: str; suggestion: str

class AnalysisResponse(BaseModel):
    status: str; analysis: AnalysisData; ai_provider: str
    highlights: List[Highlight]; processing_time: float

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserSchema(UserBase):
    id: str
    is_active: bool
    credits: int
    essays: List[EssayResponseSchema] = []
    class Config: from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# ==============================================================================
# 4. SQLAlchemy MODELS (Database Tables)
# ==============================================================================
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    credits = Column(Integer, default=10, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1, nullable=False)
    essays = relationship("Essay", back_populates="user", cascade="all, delete-orphan")

class Essay(Base):
    __tablename__ = "essays"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    question_type = Column(Text, nullable=False)
    college_degree = Column(String(300))
    content = Column(Text, nullable=False)
    word_count = Column(Integer)
    overall_score = Column(Float)
    analysis_result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)
    user = relationship("User", back_populates="essays")

# Create database tables on startup
Base.metadata.create_all(bind=engine)

# ==============================================================================
# 5. CRUD (Create, Read, Update, Delete) Database Operations
# ==============================================================================
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_essays_by_user(db: Session, user_id: str, skip: int = 0, limit: int = 20):
    return db.query(Essay).filter(Essay.user_id == user_id).order_by(Essay.created_at.desc()).offset(skip).limit(limit).all()

def create_user_essay(db: Session, essay: EssaySubmission, user_id: str, analysis_details: dict):
    analysis_result_json = json.dumps({
        "analysis": analysis_details["analysis"].model_dump(exclude_none=True),
        "highlights": [h.model_dump() for h in analysis_details["highlights"]]
    })
    db_essay = Essay(
        **essay.model_dump(),
        user_id=user_id,
        word_count=len(essay.content.split()),
        overall_score=analysis_details["analysis"].overall_score,
        analysis_result=analysis_result_json,
        processing_time=analysis_details["processing_time"]
    )
    db.add(db_essay)
    db.commit()
    db.refresh(db_essay)
    return db_essay

# ==============================================================================
# 6. AUTHENTICATION DEPENDENCY
# ==============================================================================
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==============================================================================
# 7. AI EVALUATOR CLASS
# ==============================================================================
gemini_model = None
if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(temperature=0.6, top_p=0.85, top_k=40, max_output_tokens=3072),
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )
        print("✅ Gemini AI configured successfully")
    except Exception as e:
        print(f"❌ Gemini AI configuration failed: {e}")
elif not os.getenv("GEMINI_API_KEY"):
    print("🟡 Gemini API Key not found. Running in demo mode.")
else:
    print("🔄 Running in demo mode (google-generativeai library not found)")

class GeminiEssayEvaluator:
    def __init__(self):
        self.model = gemini_model

    def is_active(self) -> bool:
        return self.model is not None

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
            -   "type": Categorize the suggestion using one of these exact strings: "spelling", "replace_candidate", "remove_candidate", "add_candidate", "grammar", "style".
            -   "issue": A concise description of why the text needs attention (e.g., "Misspelled word.", "Awkward phrasing.", "Redundant.", "Missing comma after introductory phrase.", "Incorrect verb tense.").
            -   "suggestion": The corrected phrase or a descriptive suggestion. For "remove_candidate", this can be empty. For "add_candidate", it should be the text to add.
        7.  **Admissions Perspective**: A concise paragraph (2-3 sentences) summarizing the essay's potential from a competitive college admissions standpoint.

        **OUTPUT FORMAT (JSON EXAMPLE):**
        ```json
        {{
            "overall_score": 7.2,
            "content_breakdown": {{"content_ideas": 8.0, "structure_organization": 7.5, "language_writing": 6.5, "prompt_alignment": 7.8}},
            "alignment_topic_observations": ["Effectively addresses the core question."],
            "alignment_topic_next_steps": ["Consider explicitly stating the connection to your long-term goals."],
            "narrative_impact_observations": ["The personal anecdote is compelling."],
            "narrative_impact_next_steps": ["Expand slightly on the 'aha!' moment to deepen its impact."],
            "language_structure_observations": ["The essay is generally well-organized."],
            "language_structure_next_steps": ["Proofread carefully for minor agreement errors."],
            "grammar_issues": [
                {{"text": "pasion", "type": "spelling", "issue": "Misspelled word.", "suggestion": "passion"}},
                {{"text": "However this", "type": "add_candidate", "issue": "Missing comma after introductory element.", "suggestion": ","}},
                {{"text": "me and my friends", "type": "grammar", "issue": "Incorrect pronoun order.", "suggestion": "my friends and I"}}
            ],
            "admissions_perspective": "This essay shows strong potential. To enhance its competitiveness for {college_degree}, the applicant should focus on refining language precision."
        }}
        ```
        **IMPORTANT**: Be honest but constructive. Adhere strictly to the JSON output format.
        """
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            json_text = None
            if "```json" in response_text:
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match: json_text = match.group(1)
            
            if not json_text and "{" in response_text:
                first_brace = response_text.find("{"); last_brace = response_text.rfind("}")
                if first_brace != -1 and last_brace > first_brace:
                    potential_json = response_text[first_brace : last_brace + 1]
                    try:
                        json.loads(potential_json)
                        json_text = potential_json
                    except json.JSONDecodeError:
                        pass
            
            if not json_text: raise ValueError("No valid JSON found in AI response")
            feedback_data = json.loads(json_text)
            processing_time_val = (datetime.utcnow() - start_time).total_seconds()
            return self._process_ai_response(feedback_data, processing_time_val)

        except Exception as e:
            print(f"❌ Gemini API/JSON processing error: {str(e)}")
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)

    def _process_ai_response(self, feedback_data: Dict[str, Any], processing_time_val: float) -> Dict[str, Any]:
        """Process AI response into the application's expected format."""
        analysis = AnalysisData(
            overall_score=min(10.0, max(0.0, float(feedback_data.get("overall_score", 7.0)))),
            alignment_with_topic=AnalysisSection(
                key_observations=feedback_data.get("alignment_topic_observations", []),
                next_steps=feedback_data.get("alignment_topic_next_steps", [])),
            essay_narrative_impact=AnalysisSection(
                key_observations=feedback_data.get("narrative_impact_observations", []),
                next_steps=feedback_data.get("narrative_impact_next_steps", [])),
            language_and_structure=AnalysisSection(
                key_observations=feedback_data.get("language_structure_observations", []),
                next_steps=feedback_data.get("language_structure_next_steps", [])),
            content_breakdown=feedback_data.get("content_breakdown"),
            admissions_perspective=feedback_data.get("admissions_perspective")
        )
        highlights = [Highlight(**issue) for issue in feedback_data.get("grammar_issues", [])]
        return {"analysis": analysis, "highlights": highlights, "processing_time": processing_time_val}

    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive demo analysis when AI is not available or fails."""
        processing_time_val = (datetime.utcnow() - start_time).total_seconds()
        
        demo_highlights = []
        if "teh" in content.lower(): demo_highlights.append(Highlight(text="teh", type="spelling", issue="Misspelled word.", suggestion="the"))
        if "in actual fact" in content.lower(): demo_highlights.append(Highlight(text="in actual fact", type="remove_candidate", issue="Redundant phrase.", suggestion=""))
        if "utilize" in content.lower(): demo_highlights.append(Highlight(text="utilize", type="replace_candidate", issue="Consider simpler word.", suggestion="use"))
        if "However " in content and not "However," in content: demo_highlights.append(Highlight(text="However", type="add_candidate", issue="Missing comma after introductory element.", suggestion=","))
        if len(demo_highlights) < 2 and len(content.split()) > 30: demo_highlights.append(Highlight(text=content.split()[5], type="grammar", issue="General grammar review needed.", suggestion="Check this section for clarity."))

        final_score = min(10.0, max(1.0, round(6.5 + (len(content.split()) / 500.0) - (len(demo_highlights) * 0.3),1)))
        demo_analysis = AnalysisData(
            overall_score=final_score,
            alignment_with_topic=AnalysisSection(key_observations=["Essay generally addresses the prompt (Demo)."], next_steps=["Elaborate more with specific examples (Demo)."]),
            essay_narrative_impact=AnalysisSection(key_observations=["Personal voice is somewhat present (Demo)."], next_steps=["Develop a more compelling narrative arc (Demo)."]),
            language_and_structure=AnalysisSection(key_observations=["Basic sentence structure is clear (Demo)."], next_steps=["Improve sentence variety and complexity (Demo)."]),
            content_breakdown={"content_ideas": round(final_score * 0.9, 1), "structure_organization": round(final_score * 0.85, 1), "language_writing": round(final_score * 0.8, 1), "prompt_alignment": round(final_score * 0.95, 1)},
            admissions_perspective=f"This demo analysis suggests the essay has a foundational structure. For {college_degree or 'a competitive program'}, further development in depth and polish is recommended."
        )
        return {"analysis": demo_analysis, "highlights": demo_highlights[:4], "processing_time": processing_time_val}

evaluator = GeminiEssayEvaluator()

# ==============================================================================
# 8. FastAPI APPLICATION AND ENDPOINTS
# ==============================================================================
app = FastAPI(
    title="HelloIvy Essay Evaluator API",
    version="3.0.3 (Single-File)",
    description="Professional Essay Analysis Platform with User Authentication"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- AUTHENTICATION ENDPOINTS ---
@app.post("/api/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(db, email=form_data.username) # form-data 'username' field is used for email
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/users/register", response_model=UserSchema, status_code=201, tags=["Authentication"])
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, email=user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db=db, user=user)

@app.get("/api/users/me", response_model=UserSchema, tags=["Users"])
def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# --- ESSAY ENDPOINTS ---
@app.post("/api/analyze-essay", response_model=AnalysisResponse, tags=["Essays"])
async def analyze_essay_endpoint(
    submission: EssaySubmission,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.credits <= 0:
        raise HTTPException(status_code=403, detail="Insufficient credits. Please contact support.")
    
    # Decrement credits before making the AI call
    current_user.credits -= 1
    db.commit()

    try:
        ai_result = await evaluator.evaluate_essay(
            submission.content, submission.title, submission.question_type, submission.college_degree
        )
        
        # Save the successful analysis
        create_user_essay(db=db, essay=submission, user_id=current_user.id, analysis_details=ai_result)
        
        return AnalysisResponse(
            status="success",
            analysis=ai_result["analysis"],
            ai_provider="Gemini AI Enhanced" if evaluator.is_active() else "Demo Analysis Engine",
            highlights=ai_result["highlights"],
            processing_time=ai_result["processing_time"]
        )
    except Exception as e:
        # If any error occurs during AI call or processing, refund the credit
        current_user.credits += 1
        db.commit()
        print(f"❌ Analysis failed, credit refunded for user {current_user.email}. Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during essay analysis. Your credit has been refunded.")


@app.get("/api/essays/history", response_model=List[EssayResponseSchema], tags=["Essays"])
def get_essay_history(
    skip: int = 0, limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    return get_essays_by_user(db, user_id=current_user.id, skip=skip, limit=limit)

# --- HEALTH & ROOT ENDPOINTS ---
@app.get("/api/health", tags=["System"])
async def health_check():
    db_status = "Disconnected"
    try:
        with SessionLocal() as db:
            db.execute('SELECT 1')
        db_status = "Connected"
    except Exception as e:
        db_status = f"Error: {e}"

    return {
        "status": "healthy", 
        "version": app.version,
        "ai_engine_status": "Active" if evaluator.is_active() else "Demo Mode",
        "database_status": db_status
    }

@app.get("/", include_in_schema=False)
async def serve_root():
    # This endpoint can serve the main HTML file if you place it in the same directory
    html_file_path = 'essay_evaluator.html'
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    return HTMLResponse(content="<h1>Welcome to HelloIvy API</h1><p>See /docs for API documentation.</p>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    print("=" * 70)
    print(f"🚀 Starting HelloIvy Essay Evaluator Backend v{app.version}...")
    ai_status = "Gemini AI Enhanced" if evaluator.is_active() else "Demo Mode"
    print(f"📝 AI Engine: {ai_status}")
    print(f"💾 Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    print(f"🔑 Auth Secret Key Loaded: {'Yes' if SECRET_KEY != '09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7' else 'No (Using default dev key)'}")
    print(f"🌐 Server running on: http://localhost:{port}")
    print(f"📖 API Docs available at: http://localhost:{port}/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)