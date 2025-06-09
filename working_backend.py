# ==============================================================================
# HelloIvy Essay Evaluator - All-in-One Backend with Comprehensive Evaluation
# Version: 4.0.0 (Enhanced 5-Criteria Evaluation System)
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
    class Config: 
        from_attributes = True

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

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserSchema(UserBase):
    id: str
    is_active: bool
    credits: int
    essays: List[EssayResponseSchema] = []
    class Config: 
        from_attributes = True

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

def delete_essay_by_id(db: Session, essay_id: str, user_id: str):
    essay = db.query(Essay).filter(Essay.id == essay_id, Essay.user_id == user_id).first()
    if essay:
        db.delete(essay)
        db.commit()
        return True
    return False

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
# 7. AI EVALUATOR CLASS WITH COMPREHENSIVE 5-CRITERIA SYSTEM
# ==============================================================================
gemini_model = None
if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(temperature=0.6, top_p=0.85, top_k=40, max_output_tokens=4096),
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
        """Enhanced Gemini AI essay evaluation with comprehensive 5-criteria analysis."""
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

        **CRITICAL EVALUATION INSTRUCTIONS:**
        You must provide HONEST, DIFFERENTIATED scoring. Most essays should score between 5.0-8.5, with truly exceptional essays scoring 9.0+. Be STRICT and REALISTIC in your assessment.

        **COMPREHENSIVE EVALUATION FRAMEWORK:**

        **1. ALIGNMENT WITH TOPIC (35% of overall score)**
        What to check:
        • Does the essay directly address the given prompt or question?
        • Is the central idea relevant and clearly developed?
        • Are the anecdotes and examples used appropriate to the topic?
        
        Scoring guide (BE STRICT):
        • 9–10: Perfectly addresses prompt with exceptional depth; every word serves the core idea.
        • 7–8: Clearly addresses prompt with good relevance; minor tangents.
        • 5–6: Addresses prompt but lacks focus; some irrelevant content.
        • 3–4: Partially addresses prompt; significant irrelevant sections.
        • 1–2: Barely addresses or completely misses the prompt.

        **2. ALIGNMENT WITH ESSAY BRAINSTORMING STRUCTURE (10% of overall score)**
        What to check:
        • Does the essay follow a clear structure (introduction → challenge/experience → actions taken → outcome → reflection)?
        • Is there a clear progression of ideas?
        
        Scoring guide (BE STRICT):
        • 9–10: Perfect structure with seamless flow and powerful transitions.
        • 7–8: Good structure with clear progression; minor transition issues.
        • 5–6: Basic structure present but choppy or unclear transitions.
        • 3–4: Poor structure; ideas jump around without clear flow.
        • 1–2: No clear structure; completely disorganized.

        **3. ESSAY NARRATIVE AND IMPACT (30% of overall score)**
        What to check:
        • Is the personal story compelling and memorable?
        • Does the essay show growth, insight, or transformation?
        • Does it evoke emotion, curiosity, or admiration?
        
        Scoring guide (BE STRICT):
        • 9–10: Absolutely compelling; unforgettable story with profound insight.
        • 7–8: Engaging story with clear growth; good emotional connection.
        • 5–6: Decent story but predictable; limited emotional impact.
        • 3–4: Weak story; generic experiences with little insight.
        • 1–2: No clear story or extremely boring/confusing narrative.

        **4. LANGUAGE & STRUCTURE (15% of overall score)**
        What to check:
        • Grammar, syntax, vocabulary, spelling
        • Clarity and fluency of writing
        • Variety in sentence structure and word choice
        
        Scoring guide (BE STRICT):
        • 9–10: Flawless writing; sophisticated vocabulary and perfect grammar.
        • 7–8: Strong writing with 1-2 minor errors; good vocabulary.
        • 5–6: Adequate writing with several errors; repetitive language.
        • 3–4: Many grammar/clarity issues; basic vocabulary.
        • 1–2: Serious writing problems that impede understanding.

        **5. ALIGNMENT WITH COLLEGE VALUES (10% of overall score)**
        What to check:
        • Does the essay reflect qualities the college values (curiosity, community impact, leadership, resilience)?
        • Is there a match between the student's values and the institution's ethos?
        
        Scoring guide (BE STRICT):
        • 9–10: Perfectly embodies multiple college values with specific examples.
        • 7–8: Shows good alignment with clear examples of valued qualities.
        • 5–6: Some alignment but vague or generic examples.
        • 3–4: Weak connection to college values; unclear fit.
        • 1–2: No evidence of college value alignment.

        **SCORING CALIBRATION GUIDELINES:**
        - Average essays should score 5.0-6.5 overall
        - Good essays should score 6.5-7.5 overall  
        - Strong essays should score 7.5-8.5 overall
        - Exceptional essays should score 8.5-9.5 overall
        - Perfect essays (rare) should score 9.5-10.0 overall

        **COMMON SCORING ERRORS TO AVOID:**
        - Don't give high scores just because the essay "sounds nice"
        - Penalize grammar errors more heavily (each error should reduce language score)
        - Generic stories about sports/volunteering should score lower
        - Cliché endings should reduce narrative impact
        - Off-topic content should significantly hurt topic alignment
        - Weak college connections should lower college alignment score

        **DETAILED FEEDBACK REQUIREMENTS:**

        Calculate each criterion score INDIVIDUALLY based on the specific content, then compute weighted average:
        Overall Score = (Topic×35% + Structure×10% + Narrative×30% + Language×15% + College×10%)

        Provide realistic, differentiated scores that reflect actual essay quality. If the essay has multiple grammar errors, language score should be 4.0-6.0. If the story is generic, narrative should be 4.0-6.5. Be honest and helpful.

        **OUTPUT FORMAT (JSON):**
        ```json
        {{
            "overall_score": [CALCULATED_WEIGHTED_AVERAGE],
            "content_breakdown": {{
                "alignment_with_topic": [1.0-10.0_REALISTIC_SCORE],
                "brainstorming_structure": [1.0-10.0_REALISTIC_SCORE],
                "narrative_impact": [1.0-10.0_REALISTIC_SCORE],
                "language_structure": [1.0-10.0_REALISTIC_SCORE],
                "college_alignment": [1.0-10.0_REALISTIC_SCORE]
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
                {{"text": "exact_text_from_essay", "type": "grammar/spelling/style", "issue": "specific_problem", "suggestion": "specific_fix"}}
            ],
            "admissions_perspective": "Honest assessment of competitiveness for {college_degree} with specific areas for improvement."
        }}
        ```

        **REMEMBER: BE REALISTIC AND DIFFERENTIATED IN YOUR SCORING. Most essays have room for improvement and should not score above 8.0 overall.**
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
        
        # Create comprehensive analysis sections
        analysis = AnalysisData(
            overall_score=min(10.0, max(0.0, float(feedback_data.get("overall_score", 7.0)))),
            
            # Topic Alignment Section
            alignment_with_topic=AnalysisSection(
                key_observations=feedback_data.get("alignment_topic_observations", ["Essay addresses the prompt adequately"]),
                next_steps=feedback_data.get("alignment_topic_next_steps", ["Strengthen connection to the main question"])
            ),
            
            # Narrative Impact Section  
            essay_narrative_impact=AnalysisSection(
                key_observations=feedback_data.get("narrative_impact_observations", ["Personal story is present"]),
                next_steps=feedback_data.get("narrative_impact_next_steps", ["Add more vivid details and emotional depth"])
            ),
            
            # Language & Structure Section
            language_and_structure=AnalysisSection(
                key_observations=feedback_data.get("language_structure_observations", ["Writing is generally clear"]),
                next_steps=feedback_data.get("language_structure_next_steps", ["Review for grammar and flow improvements"])
            ),
            
            # Brainstorming Structure Section
            brainstorming_structure=AnalysisSection(
                key_observations=feedback_data.get("structure_observations", ["Essay follows basic organizational structure"]),
                next_steps=feedback_data.get("structure_next_steps", ["Improve transitions between paragraphs"])
            ),
            
            # College Alignment Section
            college_alignment=AnalysisSection(
                key_observations=feedback_data.get("college_alignment_observations", ["Shows some alignment with institutional values"]),
                next_steps=feedback_data.get("college_alignment_next_steps", ["Research and connect to specific college programs"])
            ),
            
            # Content breakdown with 5 criteria
            content_breakdown=feedback_data.get("content_breakdown", {
                "alignment_with_topic": 7.0,
                "brainstorming_structure": 7.0, 
                "narrative_impact": 7.0,
                "language_structure": 7.0,
                "college_alignment": 7.0
            }),
            
            admissions_perspective=feedback_data.get("admissions_perspective", "This essay shows potential for improvement in several key areas.")
        )
        
        # Process grammar/editorial suggestions
        highlights = []
        for issue in feedback_data.get("grammar_issues", []):
            try:
                highlights.append(Highlight(**issue))
            except Exception as e:
                print(f"Warning: Could not process highlight: {issue}, error: {e}")
        
        return {
            "analysis": analysis, 
            "highlights": highlights, 
            "processing_time": processing_time_val
        }

    def _generate_demo_analysis(self, content: str, title: str, question_type: str, college_degree: str, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive demo analysis with realistic, variable scoring."""
        processing_time_val = (datetime.utcnow() - start_time).total_seconds()
        
        # Analyze content quality dynamically
        word_count = len(content.split())
        sentences = content.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Check for various quality indicators
        grammar_issues = []
        content_lower = content.lower()
        
        # Grammar and spelling checks
        if "teh" in content_lower: 
            grammar_issues.append(Highlight(text="teh", type="spelling", issue="Misspelled word.", suggestion="the"))
        if "However " in content and not "However," in content: 
            grammar_issues.append(Highlight(text="However", type="add_candidate", issue="Missing comma after introductory element.", suggestion=","))
        if "alot" in content_lower:
            grammar_issues.append(Highlight(text="alot", type="spelling", issue="Incorrect spelling.", suggestion="a lot"))
        if "utilize" in content_lower:
            grammar_issues.append(Highlight(text="utilize", type="replace_candidate", issue="Consider simpler word.", suggestion="use"))
        if "definately" in content_lower:
            grammar_issues.append(Highlight(text="definately", type="spelling", issue="Misspelled word.", suggestion="definitely"))
        if "recieve" in content_lower:
            grammar_issues.append(Highlight(text="recieve", type="spelling", issue="Misspelled word.", suggestion="receive"))
        
        # Content quality checks
        cliche_phrases = ["changed my life", "learned a valuable lesson", "made me who i am", "hard work pays off"]
        cliche_count = sum(1 for phrase in cliche_phrases if phrase in content_lower)
        
        # Check for specific, detailed examples vs. generic statements
        specific_details = sum(1 for indicator in ["when", "where", "how", "exactly", "specifically"] if indicator in content_lower)
        
        # Calculate dynamic scores based on actual content analysis
        
        # Topic Alignment (35% weight)
        topic_score = 7.0
        if word_count < 200:
            topic_score -= 1.5  # Too short
        elif word_count > 800:
            topic_score -= 0.5  # Might be too long
        if cliche_count > 2:
            topic_score -= 1.0  # Too many clichés
        if specific_details > 5:
            topic_score += 0.5  # Good specificity
        topic_score = max(3.0, min(9.0, topic_score))
        
        # Brainstorming Structure (10% weight)
        structure_score = 6.5
        if sentence_count < 8:
            structure_score -= 1.0  # Too few sentences
        elif sentence_count > 25:
            structure_score += 0.5  # Good development
        avg_sentence_length = word_count / max(sentence_count, 1)
        if avg_sentence_length < 8:
            structure_score -= 0.5  # Choppy sentences
        elif avg_sentence_length > 25:
            structure_score -= 0.3  # Too complex
        structure_score = max(3.0, min(8.5, structure_score))
        
        # Narrative Impact (30% weight)
        narrative_score = 6.8
        if cliche_count > 1:
            narrative_score -= cliche_count * 0.7  # Heavily penalize clichés
        if specific_details > 3:
            narrative_score += 0.8  # Reward specificity
        if "i learned" in content_lower or "i realized" in content_lower:
            narrative_score += 0.3  # Shows reflection
        if word_count > 400:
            narrative_score += 0.4  # Sufficient development
        narrative_score = max(3.5, min(8.8, narrative_score))
        
        # Language Structure (15% weight)
        language_score = 7.2 - (len(grammar_issues) * 0.8)  # Heavy penalty for errors
        if word_count < 150:
            language_score -= 1.0  # Too brief
        if avg_sentence_length > 30:
            language_score -= 0.5  # Overly complex
        # Check for repeated words
        words = content_lower.split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        repeated_words = [word for word, count in word_freq.items() if count > 3]
        language_score -= len(repeated_words) * 0.3
        language_score = max(2.5, min(8.5, language_score))
        
        # College Alignment (10% weight)
        college_score = 6.0
        value_words = ["leadership", "community", "service", "growth", "challenge", "innovation", "creativity", "resilience"]
        value_mentions = sum(1 for word in value_words if word in content_lower)
        college_score += value_mentions * 0.4
        if college_degree and len(college_degree) > 10:  # Has specific college info
            college_score += 0.5
        college_score = max(3.0, min(8.2, college_score))
        
        # Calculate weighted overall score
        overall_score = round(
            topic_score * 0.35 +
            structure_score * 0.10 +
            narrative_score * 0.30 +
            language_score * 0.15 +
            college_score * 0.10, 1
        )
        
        # Add some randomization to avoid identical scores
        import random
        random.seed(hash(content[:50]))  # Consistent randomization based on content
        score_variation = random.uniform(-0.3, 0.3)
        overall_score = round(max(3.0, min(9.5, overall_score + score_variation)), 1)
        
        # Generate contextual feedback based on actual scores
        topic_obs = ["Essay addresses the general theme adequately (Demo)"]
        topic_next = ["Strengthen direct connection to the prompt"]
        
        if topic_score < 6.0:
            topic_obs = ["Essay partially addresses the prompt but lacks focus (Demo)"]
            topic_next = ["Ensure every paragraph directly relates to the central question", "Remove tangential content that doesn't serve the main theme"]
        elif topic_score > 7.5:
            topic_obs = ["Essay maintains strong focus on the prompt throughout (Demo)", "Clear connection between personal experience and the question"]
            
        narrative_obs = ["Personal story provides some insight (Demo)"]
        narrative_next = ["Add more specific details and emotional depth"]
        
        if narrative_score < 6.0:
            narrative_obs = ["Story feels generic and lacks specific details (Demo)"]
            narrative_next = ["Replace clichés with unique, personal details", "Show rather than tell your growth and insights"]
        elif narrative_score > 7.5:
            narrative_obs = ["Compelling personal narrative with good specificity (Demo)", "Shows meaningful growth and self-reflection"]
            narrative_next = ["Consider adding one more vivid detail to strengthen impact"]
            
        language_obs = ["Writing demonstrates adequate command of English (Demo)"]
        language_next = ["Review for grammar and clarity improvements"]
        
        if len(grammar_issues) > 2:
            language_obs = ["Multiple grammar and spelling errors need attention (Demo)"]
            language_next = ["Proofread carefully for grammar and spelling errors", "Consider using grammar checking tools"]
        elif len(grammar_issues) == 0 and language_score > 7.0:
            language_obs = ["Clean, error-free writing with good flow (Demo)"]
            language_next = ["Consider varying sentence structure for enhanced rhythm"]

        final_scores = {
            "alignment_with_topic": round(topic_score, 1),
            "brainstorming_structure": round(structure_score, 1), 
            "narrative_impact": round(narrative_score, 1),
            "language_structure": round(language_score, 1),
            "college_alignment": round(college_score, 1)
        }

        demo_analysis = AnalysisData(
            overall_score=overall_score,
            
            alignment_with_topic=AnalysisSection(
                key_observations=topic_obs,
                next_steps=topic_next
            ),
            
            essay_narrative_impact=AnalysisSection(
                key_observations=narrative_obs,
                next_steps=narrative_next
            ),
            
            language_and_structure=AnalysisSection(
                key_observations=language_obs,
                next_steps=language_next
            ),
            
            brainstorming_structure=AnalysisSection(
                key_observations=[
                    f"Essay has {sentence_count} sentences with adequate structure (Demo)",
                    "Basic organizational flow is present"
                ],
                next_steps=[
                    "Strengthen transitions between major sections",
                    "Consider more impactful opening and closing"
                ]
            ),
            
            college_alignment=AnalysisSection(
                key_observations=[
                    f"Shows {value_mentions} mentions of valued qualities (Demo)",
                    "Demonstrates some personal growth valued by admissions"
                ],
                next_steps=[
                    f"Research specific values and programs at {college_degree or 'your target institution'}",
                    "Connect personal experiences more explicitly to future academic goals"
                ]
            ),
            
            content_breakdown=final_scores,
            
            admissions_perspective=f"This demo analysis suggests the essay scores {overall_score}/10 for {college_degree or 'a competitive program'}. {'Strong foundation with room for refinement' if overall_score > 7.0 else 'Significant improvement needed in multiple areas' if overall_score < 6.0 else 'Solid base requiring targeted improvements'} to maximize admission potential."
        )
        
        return {
            "analysis": demo_analysis, 
            "highlights": grammar_issues[:6], 
            "processing_time": processing_time_val
        }

evaluator = GeminiEssayEvaluator()

# ==============================================================================
# 8. FastAPI APPLICATION AND ENDPOINTS
# ==============================================================================
app = FastAPI(
    title="HelloIvy Essay Evaluator API",
    version="4.0.0",
    description="Professional Essay Analysis Platform with Comprehensive 5-Criteria Evaluation System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- AUTHENTICATION ENDPOINTS ---
@app.post("/api/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(db, email=form_data.username)
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
            ai_provider="Gemini AI Enhanced (5-Criteria System)" if evaluator.is_active() else "Demo Analysis Engine (5-Criteria System)",
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

@app.delete("/api/essays/{essay_id}", tags=["Essays"])
def delete_essay(
    essay_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a specific essay by ID (only if it belongs to the current user)"""
    success = delete_essay_by_id(db, essay_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Essay not found or access denied")
    return {"message": "Essay deleted successfully"}

@app.get("/api/essays/{essay_id}", tags=["Essays"])
def get_essay_by_id(
    essay_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific essay by ID with its analysis results"""
    essay = db.query(Essay).filter(Essay.id == essay_id, Essay.user_id == current_user.id).first()
    if not essay:
        raise HTTPException(status_code=404, detail="Essay not found")
    
    # Parse analysis result if available
    analysis_data = None
    if essay.analysis_result:
        try:
            analysis_data = json.loads(essay.analysis_result)
        except json.JSONDecodeError:
            pass
    
    return {
        "essay": essay,
        "analysis": analysis_data
    }

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
        "ai_engine_status": "Active (5-Criteria System)" if evaluator.is_active() else "Demo Mode (5-Criteria System)",
        "database_status": db_status,
        "evaluation_criteria": [
            "Alignment with Topic (35%)",
            "Essay Narrative & Impact (30%)", 
            "Language & Structure (15%)",
            "Brainstorming Structure (10%)",
            "College Alignment (10%)"
        ]
    }

@app.get("/api/evaluation-criteria", tags=["System"])
async def get_evaluation_criteria():
    """Get detailed information about the 5-criteria evaluation system"""
    return {
        "evaluation_system": "5-Criteria Comprehensive Assessment",
        "version": "4.0.0",
        "criteria": [
            {
                "name": "Alignment with Topic",
                "weight": 35,
                "description": "Does the essay directly address the given prompt with relevant anecdotes and examples?",
                "scoring_guide": {
                    "9-10": "Fully addresses the prompt with depth and nuance; every paragraph relates to the core idea.",
                    "7-8": "Mostly on-topic; some minor digressions.",
                    "5-6": "General relevance, but parts feel disconnected or off-track.",
                    "<5": "Vague, off-topic, or unclear response to the prompt."
                }
            },
            {
                "name": "Essay Narrative & Impact", 
                "weight": 30,
                "description": "Is the personal story compelling, memorable, and showing growth or transformation?",
                "scoring_guide": {
                    "9-10": "Highly compelling narrative with emotional or intellectual resonance.",
                    "7-8": "Solid story but lacks punch or vividness.",
                    "5-6": "Adequate but generic or forgettable.",
                    "<5": "Weak or confusing narrative with little impact."
                }
            },
            {
                "name": "Language & Structure",
                "weight": 15,
                "description": "Grammar, syntax, vocabulary, clarity, and sentence variety.",
                "scoring_guide": {
                    "9-10": "Polished, error-free writing with strong vocabulary and flow.",
                    "7-8": "Minor errors, generally clear.",
                    "5-6": "Noticeable issues in grammar or expression, some awkward phrasing.",
                    "<5": "Distracting errors, difficult to understand."
                }
            },
            {
                "name": "Brainstorming Structure",
                "weight": 10,
                "description": "Does the essay follow a clear progression of ideas with smooth transitions?",
                "scoring_guide": {
                    "9-10": "Perfect structural alignment, smooth transitions between ideas.",
                    "7-8": "Structure followed with slight deviation or abrupt transitions.",
                    "5-6": "Some structural elements missing or jumbled.",
                    "<5": "Disorganized or structure not followed."
                }
            },
            {
                "name": "College Alignment",
                "weight": 10,
                "description": "Does the essay reflect qualities the college values and show institutional fit?",
                "scoring_guide": {
                    "9-10": "Clearly embodies multiple college-aligned values.",
                    "7-8": "Values implied or present but not strongly emphasized.",
                    "5-6": "Limited evidence of alignment.",
                    "<5": "No clear connection to college values."
                }
            }
        ],
        "calculation": "Overall Score = (Topic×35% + Narrative×30% + Language×15% + Structure×10% + College×10%)"
    }

@app.get("/", include_in_schema=False)
async def serve_root():
    # This endpoint can serve the main HTML file if you place it in the same directory
    html_file_path = 'essay_evaluator.html'
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    return HTMLResponse(content="""
    <h1>Welcome to HelloIvy API v4.0.0</h1>
    <h2>🚀 Enhanced 5-Criteria Essay Evaluation System</h2>
    <p><strong>New Features:</strong></p>
    <ul>
        <li>✅ Comprehensive 5-criteria evaluation framework</li>
        <li>✅ Weighted scoring system (Topic 35%, Narrative 30%, Language 15%, Structure 10%, College 10%)</li>
        <li>✅ Detailed feedback for each criterion</li>
        <li>✅ College-specific value alignment assessment</li>
        <li>✅ Enhanced editorial suggestions with categorization</li>
        <li>✅ Professional admissions counselor perspective</li>
    </ul>
    <p>📖 <a href="/docs">View API Documentation</a></p>
    <p>🔍 <a href="/api/evaluation-criteria">View Evaluation Criteria Details</a></p>
    """)

# --- ADMIN/DEBUG ENDPOINTS (Optional) ---
@app.get("/api/admin/stats", tags=["Admin"])
def get_admin_stats(current_user: User = Depends(get_current_active_user)):
    """Get system statistics (you can add admin role checking here)"""
    with SessionLocal() as db:
        total_users = db.query(User).count()
        total_essays = db.query(Essay).count()
        avg_score = db.query(Essay.overall_score).filter(Essay.overall_score.isnot(None)).all()
        
        avg_score_value = None
        if avg_score:
            scores = [score[0] for score in avg_score if score[0] is not None]
            avg_score_value = sum(scores) / len(scores) if scores else None
        
        return {
            "total_users": total_users,
            "total_essays": total_essays,
            "average_essay_score": round(avg_score_value, 2) if avg_score_value else None,
            "ai_engine_active": evaluator.is_active(),
            "evaluation_system": "5-Criteria Comprehensive Assessment v4.0.0"
        }

# --- ERROR HANDLERS ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat(),
        "system_version": "4.0.0"
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat(),
        "system_version": "4.0.0"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    print("=" * 80)
    print(f"🚀 Starting HelloIvy Essay Evaluator Backend v{app.version}...")
    print("📊 NEW: 5-Criteria Comprehensive Evaluation System")
    print("   • Alignment with Topic (35%)")
    print("   • Essay Narrative & Impact (30%)")  
    print("   • Language & Structure (15%)")
    print("   • Brainstorming Structure (10%)")
    print("   • College Alignment (10%)")
    ai_status = "Gemini AI Enhanced (5-Criteria)" if evaluator.is_active() else "Demo Mode (5-Criteria)"
    print(f"🤖 AI Engine: {ai_status}")
    print(f"💾 Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    print(f"🔑 Auth Secret Key Loaded: {'Yes' if SECRET_KEY != '09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7' else 'No (Using default dev key)'}")
    print(f"🌐 Server running on: http://localhost:{port}")
    print(f"📖 API Docs available at: http://localhost:{port}/docs")
    print(f"🔍 Evaluation Criteria: http://localhost:{port}/api/evaluation-criteria")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)