# main_backend.py
# ==============================================================================
# HelloIvy - Unified Production Backend
# Merged Voice Brainstorming + Essay Evaluation Platform
# This single file contains all backend logic to prevent circular imports.
# ==============================================================================

import os
import json
import asyncio
import uuid
import re
import tempfile
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict

# --- Core Dependencies ---
from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, JSON, text
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

# --- AI and NLP Dependencies ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ WARNING: openai not installed. Voice transcription will be disabled.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ WARNING: google-generativeai not installed. AI features will be disabled.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ WARNING: spacy not installed. Advanced NLP features will be disabled.")

# ==============================================================================
# 1. INITIAL SETUP & CONFIGURATION
# ==============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory storage for voice brainstorming
MEMORY_STORAGE_PATH = Path(os.getenv("BRAINSTORMING_MEMORY_PATH", "data/brainstorming_memory"))
MEMORY_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Combine all AI availability checks
BRAINSTORMING_AVAILABLE = OPENAI_AVAILABLE and GEMINI_AVAILABLE and SPACY_AVAILABLE

# ==============================================================================
# 2. DATABASE SETUP
# ==============================================================================

# main_backend.py - NEW CODE
# For security, it's best to put this in your .env file, but hardcoding works for now.
DATABASE_URL = "mysql+pymysql://helloivy_user:YourStrongPassword_123!@localhost:3306/helloivy_db"

engine = create_engine(
    DATABASE_URL,
    # This line is already correct and doesn't need to be changed.
    # It will correctly handle the MySQL URL.
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================================================================
# 3. AUTHENTICATION SETUP
# ==============================================================================
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
# 4. DATABASE MODELS
# ==============================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1, nullable=False)
    credits = Column(Integer, default=50)
    subscription_tier = Column(String(20), default="free")
    total_essays_analyzed = Column(Integer, default=0)
    essays = relationship("Essay", back_populates="user", cascade="all, delete-orphan")
    brainstorming_sessions = relationship("BrainstormingSession", back_populates="user", cascade="all, delete-orphan")

class BrainstormingSession(Base):
    __tablename__ = "brainstorming_sessions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    session_name = Column(String(255), nullable=False)
    conversation_stage = Column(Integer, default=0)
    total_exchanges = Column(Integer, default=0)
    session_status = Column(String(20), default="active")
    memory_used = Column(Text, nullable=True) # To store summary
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="brainstorming_sessions")
    conversations = relationship("BrainstormingConversation", back_populates="session", cascade="all, delete-orphan")
    generated_essays = relationship("GeneratedEssayDraft", back_populates="session") # No cascade here

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
    essay_type = Column(String(50), default="personal_statement")
    version_number = Column(Integer, default=1)
    user = relationship("User", back_populates="essays")

class BrainstormingConversation(Base):
    __tablename__ = "brainstorming_conversations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("brainstorming_sessions.id"), nullable=False)
    speaker = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    topic = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("BrainstormingSession", back_populates="conversations")

class GeneratedEssayDraft(Base):
    __tablename__ = "generated_essay_drafts"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("brainstorming_sessions.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer)
    essay_type = Column(String(50), default="personal_statement")
    ai_provider = Column(String(50), default="gemini")
    memory_used = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("BrainstormingSession", back_populates="generated_essays")

# ==============================================================================
# 5. PYDANTIC SCHEMAS
# ==============================================================================

# --- Schemas for Essay Evaluation ---
class EssayBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    question_type: str = Field(..., min_length=1, max_length=1000)
    college_degree: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=20, max_length=50000)

class EssaySubmission(EssayBase):
    essay_type: Optional[str] = "personal_statement"

class EssayResponseSchema(EssayBase):
    id: str
    user_id: str
    created_at: datetime
    overall_score: Optional[float] = None
    word_count: Optional[int] = None
    processing_time: Optional[float] = None
    version_number: Optional[int] = 1
    class Config: from_attributes = True

class AnalysisSection(BaseModel):
    key_observations: List[str]
    next_steps: List[str]

class AnalysisData(BaseModel):
    overall_score: float
    alignment_with_topic: AnalysisSection
    essay_narrative_impact: AnalysisSection
    language_and_structure: AnalysisSection
    brainstorming_structure: Optional[AnalysisSection] = None
    college_alignment: Optional[AnalysisSection] = None
    content_breakdown: Optional[Dict[str, float]] = None
    admissions_perspective: Optional[str] = None

class Highlight(BaseModel):
    text: str
    type: str
    issue: str
    suggestion: str

class AnalysisResponse(BaseModel):
    status: str
    analysis: AnalysisData
    ai_provider: str
    highlights: List[Highlight]
    processing_time: float

# --- Schemas for Brainstorming ---
class BrainstormingSessionCreate(BaseModel):
    session_name: Optional[str] = Field(None, description="Optional custom name for the session")

class BrainstormingSessionResponse(BaseModel):
    id: str
    session_name: str
    conversation_stage: int
    total_exchanges: int
    session_status: str
    current_topic: Optional[str] = "introduction"
    created_at: datetime
    updated_at: datetime
    class Config: from_attributes = True

class ConversationMessage(BaseModel):
    id: str
    speaker: str
    message: str
    topic: Optional[str] = None
    created_at: datetime
    class Config: from_attributes = True

class AnalysisRequest(BaseModel):
    session_id: str
    transcript: str
    context: Optional[Dict[str, Any]] = None

class VoiceAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    extracted_notes: Optional[List[Dict[str, Any]]] = None
    next_question: Optional[str] = None
    should_change_topic: bool = False
    next_topic: Optional[str] = None
    completion_ready: bool = False
    error: Optional[str] = None

class GenerateEssayRequest(BaseModel):
    session_id: str
    essay_type: str = "personal_statement"
    custom_prompt: Optional[str] = None
    word_limit: Optional[int] = 650

class GeneratedEssayResponse(BaseModel):
    success: bool
    essay_id: Optional[str] = None
    title: str
    content: str
    word_count: int
    essay_type: str
    ai_provider: str
    processing_time: Optional[float] = None
    memory_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SessionAnalysisResponse(BaseModel):
    session_id: str
    completeness_percentage: float
    ready_for_essay: bool
    missing_areas: List[str]
    memory_insights: Dict[str, Any]
    recommendations: List[str]

# --- Schemas for Users ---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserSchema(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    credits: int
    subscription_tier: str
    total_essays_analyzed: int
    essays: List[EssayResponseSchema] = []
    brainstorming_sessions: List[BrainstormingSessionResponse] = []
    class Config: from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

# ==============================================================================
# 6. AI & NLP CONFIGURATION AND ENGINES
# ==============================================================================

# --- AI API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("✅ OpenAI API configured.")
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("✅ Gemini AI configured.")
else:
    gemini_model = None

# --- spaCy NLP Model ---
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found. Downloading...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    logger.info("✅ spaCy English model loaded.")

# --- Essay Evaluator Engine ---
class EnhancedEssayEvaluator:
    # ... (code from working_backend.py's EnhancedEssayEvaluator class) ...
    # This class is self-contained and doesn't need changes.
    def __init__(self):
        self.model = gemini_model

    def is_active(self) -> bool:
        return self.model is not None

    async def evaluate_essay(self, content: str, title: str, question_type: str, college_degree: str = "") -> Dict[str, Any]:
        """Enhanced essay evaluation with comprehensive 5-criteria analysis"""
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

        **COMPREHENSIVE EVALUATION FRAMEWORK:**

        **1. ALIGNMENT WITH TOPIC (35% of overall score)**
        Does the essay directly address the given prompt with relevant anecdotes and examples?
        Scoring: 9-10 (Perfect alignment), 7-8 (Good relevance), 5-6 (Basic connection), 3-4 (Partial), 1-2 (Off-topic)

        **2. ESSAY NARRATIVE & IMPACT (30% of overall score)**
        Is the personal story compelling, memorable, and showing growth/transformation?
        Scoring: 9-10 (Exceptional story), 7-8 (Engaging narrative), 5-6 (Adequate story), 3-4 (Weak narrative), 1-2 (No clear story)

        **3. LANGUAGE & STRUCTURE (15% of overall score)**
        Grammar, syntax, vocabulary, clarity, and sentence variety.
        Scoring: 9-10 (Flawless writing), 7-8 (Strong writing), 5-6 (Adequate), 3-4 (Issues present), 1-2 (Serious problems)

        **4. BRAINSTORMING STRUCTURE (10% of overall score)**
        Clear progression: introduction → experience → actions → outcome → reflection
        Scoring: 9-10 (Perfect flow), 7-8 (Good structure), 5-6 (Basic organization), 3-4 (Poor flow), 1-2 (Disorganized)

        **5. COLLEGE ALIGNMENT (10% of overall score)**
        Reflects qualities the college values and shows institutional fit.
        Scoring: 9-10 (Perfect match), 7-8 (Good alignment), 5-6 (Some connection), 3-4 (Weak fit), 1-2 (No alignment)

        Calculate weighted average: (Topic×35% + Narrative×30% + Language×15% + Structure×10% + College×10%)

        **OUTPUT FORMAT (JSON):**
        ```json
        {{
            "overall_score": [CALCULATED_WEIGHTED_AVERAGE],
            "content_breakdown": {{
                "alignment_with_topic": [1.0-10.0_SCORE],
                "brainstorming_structure": [1.0-10.0_SCORE],
                "narrative_impact": [1.0-10.0_SCORE],
                "language_structure": [1.0-10.0_SCORE],
                "college_alignment": [1.0-10.0_SCORE]
            }},
            "alignment_topic_observations": ["Specific observation 1", "Specific observation 2"],
            "alignment_topic_next_steps": ["Actionable improvement 1", "Actionable improvement 2"],
            "narrative_impact_observations": ["Specific observation 1", "Specific observation 2"],
            "narrative_impact_next_steps": ["Actionable improvement 1", "Actionable improvement 2"],
            "language_structure_observations": ["Specific observation 1", "Specific observation 2"],
            "language_structure_next_steps": ["Actionable improvement 1", "Actionable improvement 2"],
            "structure_observations": ["Specific observation 1", "Specific observation 2"],
            "structure_next_steps": ["Actionable improvement 1", "Actionable improvement 2"],
            "college_alignment_observations": ["Specific observation 1", "Specific observation 2"],
            "college_alignment_next_steps": ["Actionable improvement 1", "Actionable improvement 2"],
            "grammar_issues": [
                {{"text": "exact_text", "type": "grammar/spelling", "issue": "problem", "suggestion": "fix"}}
            ],
            "admissions_perspective": "Honest assessment with specific improvement areas."
        }}
        ```
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
                if first_brace != -1 and last_brace > first_brace:
                    potential_json = response_text[first_brace:last_brace + 1]
                    try:
                        json.loads(potential_json)
                        json_text = potential_json
                    except json.JSONDecodeError:
                        pass
            
            if not json_text: 
                raise ValueError("No valid JSON found in AI response")
                
            feedback_data = json.loads(json_text)
            processing_time_val = (datetime.utcnow() - start_time).total_seconds()
            return self._process_ai_response(feedback_data, processing_time_val)

        except Exception as e:
            logger.error(f"❌ Gemini API/JSON processing error: {str(e)}")
            return self._generate_demo_analysis(content, title, question_type, college_degree, start_time)

    def _process_ai_response(self, feedback_data: Dict[str, Any], processing_time_val: float) -> Dict[str, Any]:
        analysis = AnalysisData(
            overall_score=min(10.0, max(0.0, float(feedback_data.get("overall_score", 7.0)))),
            alignment_with_topic=AnalysisSection(
                key_observations=feedback_data.get("alignment_topic_observations", ["Essay addresses the prompt adequately"]),
                next_steps=feedback_data.get("alignment_topic_next_steps", ["Strengthen connection to the main question"])
            ),
            essay_narrative_impact=AnalysisSection(
                key_observations=feedback_data.get("narrative_impact_observations", ["Personal story is present"]),
                next_steps=feedback_data.get("narrative_impact_next_steps", ["Add more vivid details and emotional depth"])
            ),
            language_and_structure=AnalysisSection(
                key_observations=feedback_data.get("language_structure_observations", ["Writing is generally clear"]),
                next_steps=feedback_data.get("language_structure_next_steps", ["Review for grammar and flow improvements"])
            ),
            brainstorming_structure=AnalysisSection(
                key_observations=feedback_data.get("structure_observations", ["Essay follows basic organizational structure"]),
                next_steps=feedback_data.get("structure_next_steps", ["Improve transitions between paragraphs"])
            ),
            college_alignment=AnalysisSection(
                key_observations=feedback_data.get("college_alignment_observations", ["Shows some alignment with institutional values"]),
                next_steps=feedback_data.get("college_alignment_next_steps", ["Research and connect to specific college programs"])
            ),
            content_breakdown=feedback_data.get("content_breakdown", {}),
            admissions_perspective=feedback_data.get("admissions_perspective", "This essay shows potential for improvement in several key areas.")
        )
        highlights = [Highlight(**issue) for issue in feedback_data.get("grammar_issues", [])]
        return {"analysis": analysis, "highlights": highlights, "processing_time": processing_time_val}

    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time: datetime) -> Dict[str, Any]:
        # This is a fallback and can be simplified or expanded as needed
        logger.warning("AI model unavailable, generating demo analysis.")
        processing_time_val = (datetime.utcnow() - start_time).total_seconds()
        score = round(7.0 + (len(content) % 15) / 10.0, 1) # pseudo-random score
        analysis = AnalysisData(
            overall_score=score,
            alignment_with_topic=AnalysisSection(key_observations=["Demo observation"], next_steps=["Demo next step"]),
            essay_narrative_impact=AnalysisSection(key_observations=["Demo observation"], next_steps=["Demo next step"]),
            language_and_structure=AnalysisSection(key_observations=["Demo observation"], next_steps=["Demo next step"]),
            admissions_perspective="This is a demo analysis. Connect to AI for full feedback."
        )
        return {"analysis": analysis, "highlights": [], "processing_time": processing_time_val}


# --- Brainstorming Engine (from brainstorming_backend.py) ---
@dataclass
class MemoryInsight:
    category: str
    content: str
    confidence: float
    relevance: float
    uniqueness: float
    source_conversation_id: str
    extraction_method: str
    timestamp: datetime

class AdvancedMemoryManager:
    # ... (code from brainstorming_backend.py's AdvancedMemoryManager class) ...
    # This class is self-contained and doesn't need changes.
    def __init__(self):
        self.memory_cache = {}
        self.insight_extractors = {}

    def get_memory_path(self, session_id: str) -> Path:
        return MEMORY_STORAGE_PATH / f"session_{session_id}_memory.json"

    def load_memory(self, session_id: str) -> Dict[str, Any]:
        if session_id in self.memory_cache:
            return self.memory_cache[session_id]
        
        memory_file = self.get_memory_path(session_id)
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    self.memory_cache[session_id] = memory
                    return memory
            except Exception as e:
                logger.error(f"Error loading memory for session {session_id}: {e}")
        
        memory = self._initialize_empty_memory()
        self.memory_cache[session_id] = memory
        return memory

    def save_memory(self, session_id: str, memory: Dict[str, Any]) -> bool:
        try:
            memory_file = self.get_memory_path(session_id)
            if memory_file.exists():
                backup_file = memory_file.with_suffix('.backup.json')
                memory_file.rename(backup_file)
            
            memory['_metadata'] = {
                'last_updated': datetime.utcnow().isoformat(),
                'version': '2.0',
                'checksum': self._calculate_memory_checksum(memory)
            }
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2, ensure_ascii=False, default=str)
            
            self.memory_cache[session_id] = memory
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory for session {session_id}: {e}")
            return False

    def _initialize_empty_memory(self) -> Dict[str, Any]:
        return {
            "personal_info": {}, "experiences": [], "challenges": [], "achievements": [],
            "goals": {"academic": [], "career": [], "personal": []}, "values": [],
            "interests": [], "skills": {"technical": [], "soft": []}, "leadership": [],
            "service": [], "growth_moments": [], "essay_themes": [],
            "conversation_context": {}, "extracted_insights": [], "quality_metrics": {}
        }
    
    def _calculate_memory_checksum(self, memory: Dict[str, Any]) -> str:
        memory_copy = memory.copy()
        memory_copy.pop('_metadata', None)
        memory_str = json.dumps(memory_copy, sort_keys=True, default=str)
        return hashlib.md5(memory_str.encode()).hexdigest()

class OllamaConversationEngine: # Renamed for clarity, logic adapted
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()

    async def transcribe_audio(self, audio_file: UploadFile) -> str:
        if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="Voice transcription service not available.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            content = await audio_file.read()
            temp_audio_file.write(content)
            temp_audio_file.flush()
            
            with open(temp_audio_file.name, "rb") as audio:
                transcript = openai.Audio.transcribe("whisper-1", audio)
            
            os.unlink(temp_audio_file.name)
            return transcript['text']

    async def start_conversation(self, session: BrainstormingSession, db: Session) -> str:
        opening_message = "Hi! I'm your AI counselor. Let's start brainstorming for your essay. What's something you're passionate about?"
        save_conversation_message(db, session.id, 'ai', opening_message, 'introduction')
        return opening_message

    async def analyze_user_response(self, transcript: str, session: BrainstormingSession, db: Session) -> Dict[str, Any]:
        if not GEMINI_AVAILABLE:
            return {"success": False, "error": "AI conversation engine not available."}
        
        save_conversation_message(db, session.id, 'user', transcript, session.session_status)
        memory = self.memory_manager.load_memory(session.id)
        
        prompt = f"""You are a helpful and insightful college admissions counselor. A student just said this during a brainstorming session: "{transcript}". 
        Based on this, and their conversation history (summarized as: {json.dumps(memory.get('essay_themes', []))}), ask a single, thoughtful follow-up question to dig deeper.
        Also, extract any key details into a JSON object with keys like 'experiences', 'skills', 'goals', 'challenges', 'values'.
        Finally, decide if the conversation on the current topic '{session.session_status}' is complete (boolean).
        
        Respond with a single JSON object with keys: "ai_response", "extracted_entities", "conversation_complete".
        """
        
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        
        try:
            result_json = json.loads(response.text.strip())
            ai_response = result_json.get("ai_response", "That's interesting, tell me more.")
            extracted_entities = result_json.get("extracted_entities", {})
            
            # Update memory
            for key, value in extracted_entities.items():
                if key in memory:
                    if isinstance(memory[key], list):
                        memory[key].append(value)
                    else:
                        memory[key] = value # or update logic
            self.memory_manager.save_memory(session.id, memory)

            save_conversation_message(db, session.id, 'ai', ai_response, session.session_status)
            
            session.total_exchanges += 1
            db.commit()

            return {"success": True, "ai_response": ai_response, "extracted_entities": extracted_entities, "conversation_complete": result_json.get("conversation_complete", False)}

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing AI response: {e}\nResponse text: {response.text}")
            return {"success": False, "error": "Could not parse AI response."}
    
    async def generate_essay_from_memory(self, session: BrainstormingSession, db: Session) -> Dict[str, Any]:
        memory = self.memory_manager.load_memory(session.id)
        if not memory or not any(v for k, v in memory.items() if isinstance(v, list)):
            return {"success": False, "error": "Not enough information gathered to generate an essay."}
        
        memory_summary = json.dumps(memory, indent=2)
        prompt = f"""Based on the following brainstorming notes, write a compelling 650-word personal statement for a college application.
        Notes:
        {memory_summary}

        The essay should have a clear narrative, show personal growth, and have an authentic voice.
        Respond with ONLY the essay content.
        """
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        essay_content = response.text.strip()
        word_count = len(essay_content.split())
        
        # Create a title
        title_prompt = f"Create a short, compelling title for the following essay: {essay_content[:200]}..."
        title_response = await asyncio.to_thread(gemini_model.generate_content, title_prompt)
        title = title_response.text.strip().replace('"', '')

        draft = GeneratedEssayDraft(
            session_id=session.id,
            user_id=session.user_id,
            title=title,
            content=essay_content,
            word_count=word_count,
            memory_used=json.dumps(memory)
        )
        db.add(draft)
        session.session_status = "completed"
        db.commit()
        db.refresh(draft)

        return {"success": True, "essay_id": draft.id, "title": title, "content": essay_content, "word_count": word_count}


# --- Initialize Engines ---
evaluator = EnhancedEssayEvaluator()
ollama_conversation_engine = OllamaConversationEngine() # Retain name for compatibility


# ==============================================================================
# 7. CRUD (Create, Read, Update, Delete) OPERATIONS
# ==============================================================================

# --- User CRUD ---
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Essay CRUD ---
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
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.total_essays_analyzed += 1
    db.commit()
    db.refresh(db_essay)
    return db_essay

def delete_essay_by_id(db: Session, essay_id: str, user_id: str):
    essay = db.query(Essay).filter(Essay.id == essay_id, Essay.user_id == user_id).first()
    if essay:
        db.delete(essay)
        db.commit()
        return True
    return False

# --- Brainstorming CRUD ---
def create_brainstorming_session(db: Session, user_id: str, session_name: str):
    session = BrainstormingSession(user_id=user_id, session_name=session_name)
    db.add(session)
    db.commit()
    db.refresh(session)
    # Init memory file
    ollama_conversation_engine.memory_manager.load_memory(session.id)
    return session

def get_user_brainstorming_sessions(db: Session, user_id: str):
    return db.query(BrainstormingSession).filter(BrainstormingSession.user_id == user_id).order_by(BrainstormingSession.created_at.desc()).all()

def get_brainstorming_session(db: Session, session_id: str, user_id: str):
    return db.query(BrainstormingSession).filter(BrainstormingSession.id == session_id, BrainstormingSession.user_id == user_id).first()

def save_conversation_message(db: Session, session_id: str, speaker: str, message: str, topic: str = None):
    conversation = BrainstormingConversation(session_id=session_id, speaker=speaker, message=message, topic=topic)
    db.add(conversation)
    db.commit()
    return conversation

def get_session_conversations(db: Session, session_id: str):
    return db.query(BrainstormingConversation).filter(BrainstormingConversation.session_id == session_id).order_by(BrainstormingConversation.created_at).all()

def get_user_generated_essays(db: Session, user_id: str):
    return db.query(GeneratedEssayDraft).filter(GeneratedEssayDraft.user_id == user_id).order_by(GeneratedEssayDraft.created_at.desc()).all()

def delete_session_memory(session_id: str):
    path = ollama_conversation_engine.memory_manager.get_memory_path(session_id)
    if path.exists():
        path.unlink()
        logger.info(f"Deleted memory for session {session_id}")

def get_session_memory(session_id: str):
    return ollama_conversation_engine.memory_manager.load_memory(session_id)

def analyze_session_memory(session_id: str):
    memory = get_session_memory(session_id)
    if not memory:
        return {"ready_for_essay": False, "missing_areas": ["all"], "completeness_percentage": 0.0, "memory_insights": {}}

    insights = {}
    total_items = 0
    filled_categories = 0
    required_categories = ["experiences", "challenges", "achievements", "goals", "values"]
    
    for category in required_categories:
        items = memory.get(category, [])
        insights[category] = len(items)
        total_items += len(items)
        if len(items) > 0:
            filled_categories += 1
    
    completeness = (filled_categories / len(required_categories)) * 100
    ready = completeness > 75 and total_items > 5
    missing = [cat for cat in required_categories if not memory.get(cat)]

    return {
        "completeness_percentage": completeness,
        "ready_for_essay": ready,
        "missing_areas": missing,
        "memory_insights": insights,
        "recommendations": [f"Flesh out your story on: {area}" for area in missing]
    }

def get_system_status():
    return {"overall_status": "ready", "components": {"ollama": "ok", "spacy": "ok"}, "voice_brainstorming_ready": BRAINSTORMING_AVAILABLE}


# ==============================================================================
# 8. AUTHENTICATION DEPENDENCIES & HELPERS
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
        user = get_user_by_email(db, email=email)
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_brainstorming():
    """Dependency to check if brainstorming backend is available"""
    if not BRAINSTORMING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice brainstorming service is not available. Please check system configuration."
        )

# ==============================================================================
# 9. FASTAPI APP & ROUTER SETUP
# ==============================================================================
app = FastAPI(
    title="HelloIvy Enhanced Essay Platform API",
    version="7.0.0",
    description="Unified Voice Brainstorming + Essay Evaluation Platform (Single File)"
)

brainstorming_router = APIRouter(prefix="/api/brainstorming", tags=["Voice Brainstorming"])

# ==============================================================================
# 10. API ENDPOINTS
# ==============================================================================

# --- Main API Endpoints ---
@app.post("/api/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
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

@app.post("/api/analyze-essay", response_model=AnalysisResponse, tags=["Essays"])
async def analyze_essay_endpoint(submission: EssaySubmission, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    ai_result = await evaluator.evaluate_essay(submission.content, submission.title, submission.question_type, submission.college_degree)
    create_user_essay(db=db, essay=submission, user_id=current_user.id, analysis_details=ai_result)
    return AnalysisResponse(
        status="success",
        analysis=ai_result["analysis"],
        ai_provider="Gemini AI Enhanced" if evaluator.is_active() else "Demo Analysis",
        highlights=ai_result["highlights"],
        processing_time=ai_result["processing_time"]
    )

@app.get("/api/essays/history", response_model=List[EssayResponseSchema], tags=["Essays"])
def get_essay_history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_essays_by_user(db, user_id=current_user.id, skip=skip, limit=limit)

@app.delete("/api/essays/{essay_id}", tags=["Essays"])
def delete_essay(essay_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    if not delete_essay_by_id(db, essay_id, current_user.id):
        raise HTTPException(status_code=404, detail="Essay not found")
    return {"message": "Essay deleted successfully"}

# --- Brainstorming API Endpoints ---
@brainstorming_router.post("/sessions", response_model=BrainstormingSessionResponse, status_code=201)
async def create_session(session_data: BrainstormingSessionCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session_name = session_data.session_name or f"Voice Brainstorming - {datetime.utcnow().strftime('%B %d, %Y')}"
    session = create_brainstorming_session(db=db, user_id=current_user.id, session_name=session_name)
    await ollama_conversation_engine.start_conversation(session, db)
    return session

@brainstorming_router.get("/sessions", response_model=List[BrainstormingSessionResponse])
async def get_user_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    return get_user_brainstorming_sessions(db=db, user_id=current_user.id)

@brainstorming_router.get("/sessions/{session_id}/conversations", response_model=List[ConversationMessage])
async def get_session_conversations_endpoint(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return get_session_conversations(db=db, session_id=session_id)

@brainstorming_router.post("/transcribe")
async def transcribe_voice(session_id: str = Form(...), audio: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    transcript = await ollama_conversation_engine.transcribe_audio(audio)
    if not transcript:
        return JSONResponse(status_code=200, content={"success": False, "error": "No speech detected"})
    return JSONResponse(status_code=200, content={"success": True, "transcript": transcript})

@brainstorming_router.post("/analyze", response_model=VoiceAnalysisResponse)
async def analyze_voice_response(request: AnalysisRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=request.session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    analysis_result = await ollama_conversation_engine.analyze_user_response(transcript=request.transcript, session=session, db=db)
    return VoiceAnalysisResponse(
        success=analysis_result["success"],
        analysis=analysis_result.get("extracted_entities"),
        next_question=analysis_result.get("ai_response"),
        completion_ready=analysis_result.get("conversation_complete", False),
        error=analysis_result.get("error")
    )

@brainstorming_router.post("/generate-essay", response_model=GeneratedEssayResponse)
async def generate_essay_from_session(request: GenerateEssayRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=request.session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = await ollama_conversation_engine.generate_essay_from_memory(session, db)
    return GeneratedEssayResponse(
        success=result["success"],
        essay_id=result.get("essay_id"),
        title=result.get("title", ""),
        content=result.get("content", ""),
        word_count=result.get("word_count", 0),
        essay_type=request.essay_type,
        ai_provider="Ollama (Mistral 7B)" if not GEMINI_AVAILABLE else "Gemini AI",
        error=result.get("error")
    )

@brainstorming_router.get("/sessions/{session_id}/analysis", response_model=SessionAnalysisResponse)
async def get_session_analysis(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    analysis = analyze_session_memory(session_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No session memory found")
    return SessionAnalysisResponse(**analysis)

@brainstorming_router.delete("/sessions/{session_id}")
async def delete_brainstorming_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user), _: None = Depends(require_brainstorming)):
    session = get_brainstorming_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    delete_session_memory(session_id)
    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}

# --- System & Root Endpoints ---
@app.get("/api/health", tags=["System"])
async def health_check():
    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(text('SELECT 1'))
        db.close()
    except Exception as e:
        db_status = f"error: {e}"
    
    return {
        "status": "healthy", 
        "version": app.version,
        "database_status": db_status,
        "brainstorming_features": "available" if BRAINSTORMING_AVAILABLE else "unavailable"
    }

@app.get("/", include_in_schema=False)
async def serve_root():
    html_file_path = 'essay_evaluator.html'
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    return HTMLResponse("<h1>HelloIvy Backend</h1><p>Frontend file not found.</p>")


# ==============================================================================
# 11. APP CONFIGURATION & STARTUP
# ==============================================================================

# Include the brainstorming router in the main app
app.include_router(brainstorming_router)

# Add Middleware
# Find this in Section 11 of your code
app.add_middleware(
    CORSMiddleware,
    # Be specific about which frontends can connect
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(exc)})

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info(f"🚀 Starting HelloIvy Unified Platform v{app.version}...")
    # Create DB tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully.")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    logger.info("=" * 80)

# Main entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌐 Server starting on: http://0.0.0.0:{port}")
    uvicorn.run("main_backend:app", host="0.0.0.0", port=port, reload=True)