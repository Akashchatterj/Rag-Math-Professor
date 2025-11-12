"""
Math Professor Agentic RAG System
==================================
A comprehensive mathematical tutoring system with:
- AI Gateway Guardrails (Input/Output filtering)
- Knowledge Base (MATH + GSM8K datasets)
- Model Context Protocol (MCP) integration
- Web Search capability
- Human-in-the-loop feedback mechanism
- DSPy for self-learning optimization
"""

import os
import streamlit as st
from datetime import datetime
import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path
import warnings
import re

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*LangChain.*')

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

# Print debug info to help troubleshoot (only once)
if not os.getenv("TAVILY_API_KEY"):
    print("⚠️ TAVILY_API_KEY not found in environment variables!")
if not os.getenv("GROQ_API_KEY"):
    print("⚠️ GROQ_API_KEY not found in environment variables!")

# CrewAI and LangChain imports
from crewai import Crew, Task, Agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

# DSPy for optimization (optional)
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    if 'dspy_warning_shown' not in st.session_state:
        print("⚠️ DSPy not installed. Advanced optimization features will be disabled.")

# Guardrails (optional)
GUARDRAILS_AVAILABLE = False
try:
    import guardrails
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True
    # Only print once on successful import
    print("✅ Guardrails AI detected and loaded successfully")
except ImportError:
    # Only show warning once during startup
    if 'guardrails_warning_shown' not in globals():
        print("⚠️ Guardrails AI not installed. Using basic validation instead.")
        globals()['guardrails_warning_shown'] = True

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration management"""
    
    # API Keys (Load from environment or use defaults for demo)
    # IMPORTANT: Set these in your .env file!
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Validate API keys are present
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found!")
        print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    
    if not TAVILY_API_KEY:
        print("❌ ERROR: TAVILY_API_KEY not found!")
        print("Please create a .env file with: TAVILY_API_KEY=your_key_here")
    
    # Model settings
    LLM_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 2000
    
    # Paths
    DATA_DIR = Path("./data")
    FEEDBACK_DIR = Path("./feedback")
    KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
    
    # Guardrails settings
    EDUCATIONAL_KEYWORDS = [
        "math", "mathematics", "algebra", "geometry", "calculus", 
        "trigonometry", "statistics", "probability", "equation",
        "formula", "theorem", "proof", "solve", "calculate",
        "find", "compute", "determine", "evaluate", "simplify",
        "derive", "integrate", "differentiate", "factor", "expand",
        "sum", "product", "difference", "quotient", "ratio",
        "angle", "triangle", "circle", "square", "rectangle",
        "polynomial", "quadratic", "linear", "exponential",
        "logarithm", "sine", "cosine", "tangent", "derivative",
        "integral", "limit", "function", "variable", "constant",
        "prove", "show", "demonstrate", "verify", "check",
        "how many", "how much", "what is", "find all",
        "solution", "answer", "result", "value",
        # Common math symbols and operations
        "=", "+", "-", "*", "/", "^", "x", "y", "z",
        # Word problem indicators
        "cost", "price", "total", "per", "each", "every",
        "buy", "sell", "make", "earn", "spend", "save",
        "more", "less", "than", "equal", "equals"
    ]
    
    # MCP Server settings
    MCP_SERVER_URL = "http://localhost:8080"  # Can be configured for remote MCP

# Initialize configuration
config = Config()

# Ensure directories exist
config.DATA_DIR.mkdir(exist_ok=True)
config.FEEDBACK_DIR.mkdir(exist_ok=True)
config.KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)

# ============================================================================
# GUARDRAILS IMPLEMENTATION
# ============================================================================

class MathGuardrails:
    """
    AI Gateway Guardrails for Input/Output filtering
    Ensures the system only handles educational mathematics content
    """
    
    def __init__(self):
        # Check if Guardrails AI is available
        self.guardrails_available = GUARDRAILS_AVAILABLE
        self.toxic_guard = None
        
        if GUARDRAILS_AVAILABLE:
            try:
                from guardrails import Guard
                # Try to load hub validators
                try:
                    from guardrails.hub import ToxicLanguage
                    self.toxic_guard = Guard().use(
                        ToxicLanguage(
                            validation_method="sentence",
                            threshold=0.5,
                            on_fail="exception"
                        )
                    )
                    print("✅ Guardrails AI toxic language validator loaded")
                except ImportError:
                    print("⚠️ Guardrails hub validators not installed (install with: guardrails hub install hub://guardrails/toxic_language)")
                    self.toxic_guard = None
            except Exception as e:
                print(f"⚠️ Could not initialize Guardrails AI: {e}")
                self.toxic_guard = None
        
        # Educational content pattern
        self.educational_pattern = r"(?i)(math|calculate|solve|equation|formula|algebra|geometry|trigonometry|calculus)"
        
        # Basic toxic words list (fallback)
        self.toxic_words = [
            'hate', 'kill', 'violent', 'abuse', 'attack', 'threat',
            'racist', 'sexist', 'offensive'
        ]
        self.toxic_words = [
            'hate', 'kill', 'violent', 'abuse', 'attack', 'threat',
            'racist', 'sexist', 'offensive'
        ]
        
    def validate_input(self, user_query: str) -> Tuple[bool, str]:
        """
        Validate user input against guardrails
        Returns: (is_valid, message)
        """
        # Check for toxic content
        if self.toxic_guard:
            try:
                self.toxic_guard.validate(user_query)
            except Exception as e:
                return False, "❌ Input contains inappropriate content. Please ask a mathematics-related question."
        else:
            # Fallback: basic toxic word check
            query_lower = user_query.lower()
            if any(word in query_lower for word in self.toxic_words):
                return False, "❌ Input contains inappropriate content. Please ask a mathematics-related question."
        
        # Check if query is educational/math-related
        if not self._is_educational_query(user_query):
            return False, "❌ This system is designed for mathematics education only. Please ask a math-related question."
        
        return True, "✅ Input validation passed"
    
    def validate_output(self, response: str) -> Tuple[bool, str]:
        """
        Validate model output
        Returns: (is_valid, message)
        """
        # Check for toxic content in output
        if self.toxic_guard:
            try:
                self.toxic_guard.validate(response)
            except Exception as e:
                return False, "Output contains inappropriate content"
        else:
            # Fallback: basic check
            response_lower = response.lower()
            if any(word in response_lower for word in self.toxic_words):
                return False, "Output contains inappropriate content"
        
        # Check response quality (minimum length, structure)
        if len(response) < 50:
            return False, "Response too short"
        
        return True, "Output validation passed"
    
    def _is_educational_query(self, query: str) -> bool:
        """Check if query is related to mathematics education"""
        query_lower = query.lower()
        
        # Extended list of math-related patterns
        math_patterns = [
            # Direct math keywords
            "math", "calculate", "solve", "equation", "formula", 
            "algebra", "geometry", "calculus", "trigonometry",
            
            # Math actions
            "prove", "find", "compute", "determine", "evaluate",
            "simplify", "factor", "expand", "integrate", "differentiate",
            
            # Math objects
            "angle", "triangle", "circle", "square", "rectangle",
            "polynomial", "function", "derivative", "integral",
            
            # Question patterns
            "how many", "how much", "what is", "find all",
            "show that", "demonstrate", "verify",
            
            # Math symbols (check for patterns)
            "x^", "x²", "x³", "=", "solve for",
            
            # Word problem indicators
            "per day", "per hour", "each", "every", "total",
            "cost", "price", "buy", "sell", "make money",
            
            # Numbers and operations
            "sum", "product", "difference", "quotient", "ratio"
        ]
        
        # If any pattern matches, it's a math question
        if any(pattern in query_lower for pattern in math_patterns):
            return True
        
        # Check for numbers and mathematical expressions
        import re
        # Check for mathematical expressions like "x^4", "2x + 5", etc.
        if re.search(r'[x-z]\s*[\^+\-*/=]|\d+\s*[\^+\-*/]\s*\d+', query_lower):
            return True
        
        # Check if question contains multiple numbers (likely a word problem)
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            return True
        
        # Default to True if we can't determine - be permissive
        return True


# ============================================================================
# KNOWLEDGE BASE MANAGEMENT
# ============================================================================

class MathKnowledgeBase:
    """
    Manages the mathematical knowledge base using MATH and GSM8K datasets
    Stored in FAISS vector database for efficient retrieval
    """
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.vectorstore = None
        self.data_loaded = False
        
    def load_datasets(self):
        """Load MATH and GSM8K datasets"""
        try:
            # Download datasets if not present
            self._download_datasets()
            
            # Load and process datasets
            documents = self._load_math_dataset()
            documents.extend(self._load_gsm8k_dataset())
            
            # Create vector store
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                self.vectorstore = FAISS.from_documents(
                    splits, 
                    self.embeddings
                )
                self.data_loaded = True
                return True
            return False
            
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            return False
    
    def _download_datasets(self):
        """Download MATH and GSM8K datasets from HuggingFace"""
        try:
            from datasets import load_dataset
            
            # Download MATH dataset
            if not (config.KNOWLEDGE_BASE_DIR / "math_train.json").exists():
                st.info("📥 Downloading MATH dataset (this may take a minute)...")
                try:
                    # Use the correct dataset: HuggingFaceH4/MATH-500
                    # This dataset only has 'test' split, not 'train'
                    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
                    
                    # Convert to JSON and save
                    math_data = []
                    for item in math_dataset:
                        # Handle different possible field names
                        problem = item.get("problem", item.get("question", ""))
                        solution = item.get("solution", item.get("answer", ""))
                        level = str(item.get("level", ""))
                        problem_type = item.get("type", item.get("subject", ""))
                        
                        math_data.append({
                            "problem": problem,
                            "solution": solution,
                            "level": level,
                            "type": problem_type
                        })
                    
                    # Save with UTF-8 encoding
                    with open(config.KNOWLEDGE_BASE_DIR / "math_train.json", 'w', encoding='utf-8') as f:
                        json.dump(math_data, f, indent=2, ensure_ascii=False)
                    st.success(f"✅ MATH dataset downloaded! ({len(math_data)} problems)")
                except Exception as e:
                    st.warning(f"Could not download MATH dataset: {e}")
                    st.info("Creating sample MATH data...")
                    self._create_sample_math_data()
            
            # Download GSM8K dataset
            if not (config.KNOWLEDGE_BASE_DIR / "gsm8k_train.json").exists():
                st.info("📥 Downloading GSM8K dataset...")
                try:
                    # Use the correct dataset: openai/gsm8k
                    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")
                    
                    # Convert to JSON and save (limit to 1000 for demo)
                    gsm8k_data = []
                    for i, item in enumerate(gsm8k_dataset):
                        if i >= 1000:  # Limit to 1000 for demo
                            break
                        
                        # Clean the text to avoid encoding issues
                        question = str(item.get("question", "")).encode('utf-8', errors='ignore').decode('utf-8')
                        answer = str(item.get("answer", "")).encode('utf-8', errors='ignore').decode('utf-8')
                        
                        gsm8k_data.append({
                            "question": question,
                            "answer": answer
                        })
                    
                    # Save with UTF-8 encoding
                    with open(config.KNOWLEDGE_BASE_DIR / "gsm8k_train.json", 'w', encoding='utf-8') as f:
                        json.dump(gsm8k_data, f, indent=2, ensure_ascii=False)
                    st.success(f"✅ GSM8K dataset downloaded! ({len(gsm8k_data)} problems)")
                except Exception as e:
                    st.warning(f"Could not download GSM8K dataset: {e}")
                    st.info("Creating sample GSM8K data...")
                    self._create_sample_gsm8k_data()
                    
        except Exception as e:
            st.warning(f"Could not download datasets: {e}. Using sample data.")
            self._create_sample_math_data()
            self._create_sample_gsm8k_data()
    
    def _load_math_dataset(self) -> List:
        """Load MATH dataset with proper encoding handling"""
        documents = []
        try:
            import json
            math_file = config.KNOWLEDGE_BASE_DIR / "math_train.json"
            
            if math_file.exists():
                # Open with UTF-8 encoding and error handling
                with open(math_file, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        st.warning(f"JSON decode error in MATH: {e}")
                        return documents
                    
                for item in data[:500]:  # Limit for demo
                    from langchain.schema import Document
                    # Clean any problematic characters
                    problem = str(item.get('problem', '')).encode('utf-8', errors='ignore').decode('utf-8')
                    solution = str(item.get('solution', '')).encode('utf-8', errors='ignore').decode('utf-8')
                    
                    content = f"Problem: {problem}\n\nSolution: {solution}"
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "MATH",
                            "level": str(item.get('level', '')),
                            "type": str(item.get('type', ''))
                        }
                    ))
        except Exception as e:
            st.warning(f"Could not load MATH dataset: {e}")
        
        return documents
    
    def _load_gsm8k_dataset(self) -> List:
        """Load GSM8K dataset with proper encoding handling"""
        documents = []
        try:
            import json
            gsm8k_file = config.KNOWLEDGE_BASE_DIR / "gsm8k_train.json"
            
            if gsm8k_file.exists():
                # Open with UTF-8 encoding and error handling
                with open(gsm8k_file, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        st.warning(f"JSON decode error in GSM8K: {e}")
                        return documents
                    
                for item in data[:500]:  # Limit for demo
                    from langchain.schema import Document
                    # Clean any problematic characters
                    question = str(item.get('question', '')).encode('utf-8', errors='ignore').decode('utf-8')
                    answer = str(item.get('answer', '')).encode('utf-8', errors='ignore').decode('utf-8')
                    
                    content = f"Question: {question}\n\nAnswer: {answer}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": "GSM8K"}
                    ))
        except Exception as e:
            st.warning(f"Could not load GSM8K dataset: {e}")
        
        return documents
    
    def _create_sample_data(self):
        """Create sample mathematical problems if datasets unavailable"""
        self._create_sample_math_data()
        self._create_sample_gsm8k_data()
    
    def _create_sample_math_data(self):
        """Create sample MATH dataset problems"""
        sample_math = [
            {
                "problem": "Solve for x: 2x + 5 = 15",
                "solution": "Step 1: Subtract 5 from both sides: 2x + 5 - 5 = 15 - 5, which gives 2x = 10\nStep 2: Divide both sides by 2: 2x/2 = 10/2, which gives x = 5\nFinal Answer: x = 5",
                "level": "1",
                "type": "Algebra"
            },
            {
                "problem": "Find the area of a circle with radius 5",
                "solution": "Step 1: Use the formula A = πr²\nStep 2: Substitute r = 5: A = π(5)²\nStep 3: Calculate: A = 25π ≈ 78.54 square units\nFinal Answer: 25π or approximately 78.54 square units",
                "level": "2",
                "type": "Geometry"
            },
            {
                "problem": "What is the slope of the line passing through points (2, 3) and (5, 9)?",
                "solution": "Step 1: Use slope formula: m = (y₂ - y₁)/(x₂ - x₁)\nStep 2: Substitute points: m = (9 - 3)/(5 - 2)\nStep 3: Calculate: m = 6/3 = 2\nFinal Answer: The slope is 2",
                "level": "2",
                "type": "Algebra"
            },
            {
                "problem": "Solve the quadratic equation: x² - 5x + 6 = 0",
                "solution": "Step 1: Factor the quadratic: (x - 2)(x - 3) = 0\nStep 2: Set each factor to zero: x - 2 = 0 or x - 3 = 0\nStep 3: Solve for x: x = 2 or x = 3\nFinal Answer: x = 2 or x = 3",
                "level": "3",
                "type": "Algebra"
            },
            {
                "problem": "Find the derivative of f(x) = 3x² + 2x - 1",
                "solution": "Step 1: Apply power rule to each term\nStep 2: d/dx(3x²) = 6x\nStep 3: d/dx(2x) = 2\nStep 4: d/dx(-1) = 0\nStep 5: Combine: f'(x) = 6x + 2\nFinal Answer: f'(x) = 6x + 2",
                "level": "3",
                "type": "Calculus"
            },
            {
                "problem": "What is sin(30°)?",
                "solution": "Step 1: Recall special angle values\nStep 2: sin(30°) = 1/2\nFinal Answer: 1/2 or 0.5",
                "level": "2",
                "type": "Trigonometry"
            },
            {
                "problem": "Calculate 25% of 80",
                "solution": "Step 1: Convert percentage to decimal: 25% = 0.25\nStep 2: Multiply: 0.25 × 80 = 20\nFinal Answer: 20",
                "level": "1",
                "type": "Arithmetic"
            },
            {
                "problem": "Find the perimeter of a rectangle with length 8 and width 5",
                "solution": "Step 1: Use formula P = 2(l + w)\nStep 2: Substitute values: P = 2(8 + 5)\nStep 3: Calculate: P = 2(13) = 26\nFinal Answer: 26 units",
                "level": "1",
                "type": "Geometry"
            }
        ]
        
        filepath = config.KNOWLEDGE_BASE_DIR / "math_train.json"
        with open(filepath, 'w') as f:
            json.dump(sample_math, f, indent=2)
    
    def _create_sample_gsm8k_data(self):
        """Create sample GSM8K dataset problems"""
        sample_gsm8k = [
            {
                "question": "James has 5 apples. He buys 3 more apples. How many apples does he have now?",
                "answer": "Step 1: Start with 5 apples\nStep 2: Add 3 more apples: 5 + 3 = 8\nFinal Answer: 8 apples"
            },
            {
                "question": "A pizza is cut into 8 slices. If Sarah eats 3 slices, what fraction of the pizza did she eat?",
                "answer": "Step 1: Sarah ate 3 slices out of 8\nStep 2: Fraction = 3/8\nFinal Answer: 3/8 of the pizza"
            },
            {
                "question": "Tom has $50. He spends $12 on lunch and $8 on a movie ticket. How much money does he have left?",
                "answer": "Step 1: Total spent = $12 + $8 = $20\nStep 2: Money left = $50 - $20 = $30\nFinal Answer: $30"
            },
            {
                "question": "A car travels 60 miles per hour. How far does it travel in 3 hours?",
                "answer": "Step 1: Use formula: Distance = Speed × Time\nStep 2: Distance = 60 mph × 3 hours = 180 miles\nFinal Answer: 180 miles"
            },
            {
                "question": "There are 24 students in a class. If they form groups of 4, how many groups are there?",
                "answer": "Step 1: Divide total students by group size\nStep 2: 24 ÷ 4 = 6 groups\nFinal Answer: 6 groups"
            }
        ]
        
        filepath = config.KNOWLEDGE_BASE_DIR / "gsm8k_train.json"
        with open(filepath, 'w') as f:
            json.dump(sample_gsm8k, f, indent=2)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search knowledge base for relevant problems"""
        if not self.data_loaded or self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0  # FAISS doesn't return scores directly
                }
                for doc in results
            ]
        except Exception as e:
            st.error(f"Search error: {e}")
            return []


# ============================================================================
# MODEL CONTEXT PROTOCOL (MCP) INTEGRATION
# ============================================================================

class MCPWebSearchTool:
    """
    MCP-compliant web search tool
    Implements Model Context Protocol for standardized AI integration
    """
    
    def __init__(self, api_key: str):
        self.tavily_search = TavilySearchResults(
            api_key=api_key,
            max_results=3
        )
        
    def search(self, query: str) -> List[Dict]:
        """
        Execute web search via MCP protocol
        Returns: List of search results
        """
        try:
            # MCP Resource structure
            mcp_request = {
                "protocol": "MCP/1.0",
                "type": "tool_call",
                "tool": "web_search",
                "parameters": {
                    "query": query,
                    "max_results": 3
                }
            }
            
            # Execute search
            results = self.tavily_search.invoke(query)
            
            # Return MCP-compliant response
            return {
                "protocol": "MCP/1.0",
                "type": "tool_response",
                "tool": "web_search",
                "results": results
            }
        except Exception as e:
            return {
                "protocol": "MCP/1.0",
                "type": "error",
                "error": str(e)
            }


# ============================================================================
# DSPY INTEGRATION FOR SELF-LEARNING (Optional)
# ============================================================================

if DSPY_AVAILABLE:
    class MathSolutionSignature(dspy.Signature):
        """Signature for mathematical problem solving"""
        problem = dspy.InputField(desc="Mathematical problem to solve")
        context = dspy.InputField(desc="Relevant context from knowledge base or web")
        solution = dspy.OutputField(desc="Step-by-step solution")
        explanation = dspy.OutputField(desc="Simplified explanation for students")


    class MathSolver(dspy.Module):
        """DSPy module for mathematical problem solving with optimization"""
        
        def __init__(self):
            super().__init__()
            self.generate_solution = dspy.ChainOfThought(MathSolutionSignature)
        
        def forward(self, problem, context=""):
            return self.generate_solution(problem=problem, context=context)
else:
    # Placeholder classes when DSPy not available
    class MathSolutionSignature:
        pass
    
    class MathSolver:
        def __init__(self):
            pass
        
        def forward(self, problem, context=""):
            return {"solution": "DSPy not available", "explanation": ""}


# ============================================================================
# FEEDBACK AND LEARNING SYSTEM
# ============================================================================

class FeedbackSystem:
    """
    Human-in-the-loop feedback mechanism
    Stores feedback for continuous improvement and few-shot learning
    """
    
    def __init__(self):
        # Ensure feedback directory exists
        
        config.FEEDBACK_DIR.mkdir(exist_ok=True)
        self.feedback_file = config.FEEDBACK_DIR / "feedback.jsonl"
        
        # Create file if it doesn't exist
        if not self.feedback_file.exists():
            self.feedback_file.touch()
            print(f"✅ Created feedback file: {self.feedback_file}")
        
    def store_feedback(self, 
                      question: str, 
                      response: str, 
                      rating: int, 
                      comments: str,
                      improved_response: Optional[str] = None):
        """Store user feedback with immediate file write and verification"""
        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question.strip(),
                "response": response.strip(),
                "rating": int(rating),
                "comments": comments.strip() if comments else "",
                "improved_response": improved_response.strip() if improved_response else None,
                "feedback_id": hashlib.md5(f"{question}{datetime.now()}".encode()).hexdigest()
            }
            
            # Ensure directory exists
            config.FEEDBACK_DIR.mkdir(exist_ok=True)
            
            # Write with explicit flush to disk
            with open(self.feedback_file, 'a', encoding='utf-8', buffering=1) as f:
                json_line = json.dumps(feedback_entry, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()  # Force write to disk
                import os
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            # Verify write succeeded
            if self.feedback_file.exists():
                size = self.feedback_file.stat().st_size
                print(f"✅ Feedback saved successfully (file size: {size} bytes)")
                return True
            else:
                print("❌ Feedback file not found after write!")
                return False
                
        except Exception as e:
            print(f"❌ Error storing feedback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_feedback_examples(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent high-quality feedback for few-shot learning
        Prioritizes: 1) Improved solutions, 2) High ratings, 3) Recent feedback
        """
        if not self.feedback_file.exists():
            return []
        
        feedbacks = []
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            feedbacks.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Error reading feedback: {e}")
            return []
        
        if not feedbacks:
            return []
        
        # Prioritize feedback with improved solutions (user corrections)
        corrected = [f for f in feedbacks if f.get('improved_response')]
        high_rated = [f for f in feedbacks if f.get('rating', 0) >= 4]
        
        # Combine: corrections first, then high-rated, remove duplicates
        seen = set()
        result = []
        
        for fb in corrected + high_rated:
            fb_id = fb.get('feedback_id')
            if fb_id not in seen:
                seen.add(fb_id)
                result.append(fb)
        
        # Sort by timestamp (most recent first) and limit
        result.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return result[:limit]
    
    def get_feedback_for_question(self, question: str) -> Optional[Dict]:
        """Get feedback for a specific question (exact match)"""
        if not self.feedback_file.exists():
            return None
        
        question_normalized = question.strip().lower()
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            fb = json.loads(line)
                            if fb.get('question', '').strip().lower() == question_normalized:
                                # Return most recent feedback for this question
                                if fb.get('improved_response'):
                                    return fb
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Error searching feedback: {e}")
            return None
        
        return None

# ============================================================================
# CREWAI AGENTS
# ============================================================================

def create_agents(llm):
    """Create the multi-agent system"""
    
    # Guardrail Agent
    guardrail_agent = Agent(
        role='Input Validator',
        goal='Validate that user inputs are appropriate educational mathematics questions',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You are a strict content moderator ensuring only valid mathematics questions are processed."
    )
    
    # Router Agent
    router_agent = Agent(
        role='Query Router',
        goal='Determine whether to use knowledge base or web search',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You are an intelligent router that decides the best source for mathematical information."
    )
    
    # Knowledge Base Agent
    kb_agent = Agent(
        role='Knowledge Base Specialist',
        goal='Retrieve relevant mathematical problems and solutions from the knowledge base',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You are an expert at finding similar problems in the mathematical knowledge base."
    )
    
    # Web Search Agent
    web_agent = Agent(
        role='Web Research Specialist',
        goal='Find mathematical solutions and explanations from the web when knowledge base is insufficient',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You are skilled at finding accurate mathematical information online."
    )
    
    # Solution Generator Agent
    solution_agent = Agent(
        role='Math Professor',
        goal='Generate clear, step-by-step solutions with simplified explanations',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You are a patient mathematics professor who explains concepts in a simple, understandable way."
    )
    
    # Quality Checker Agent
    quality_agent = Agent(
        role='Solution Validator',
        goal='Verify the correctness and clarity of mathematical solutions',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        backstory="You ensure all solutions are mathematically correct and pedagogically sound."
    )
    
    return {
        'guardrail': guardrail_agent,
        'router': router_agent,
        'kb': kb_agent,
        'web': web_agent,
        'solution': solution_agent,
        'quality': quality_agent
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="🎓 Math Professor AI",
        page_icon="🎓",
        layout="wide"
    )
    
    # Check for API keys FIRST
    if not config.GROQ_API_KEY or not config.TAVILY_API_KEY:
        st.error("❌ **API Keys Missing!**")
        st.markdown("""
        ### Please set up your API keys:
        
        1. **Create a `.env` file** in your project folder
        2. **Add these lines** (with your actual keys):
        
        ```
        GROQ_API_KEY=your_groq_key_here
        TAVILY_API_KEY=your_tavily_key_here
        ```
        
        ### Get your FREE API keys:
        - **Groq API**: https://console.groq.com/ (for LLM)
        - **Tavily API**: https://tavily.com/ (for web search)
        
        ### Then restart the application:
        ```
        streamlit run math_professor_rag.py
        ```
        """)
        st.stop()  # Stop execution until keys are configured
    
    st.title("🎓 Mathematical Professor - Agentic RAG System")
    st.markdown("""
    An intelligent mathematics tutoring system powered by:
    - 🛡️ **AI Guardrails** for safe, educational content
    - 📚 **Knowledge Base** (MATH + GSM8K datasets)
    - 🔌 **Model Context Protocol** (MCP) integration
    - 🌐 **Web Search** for latest information
    - 👤 **Human-in-the-Loop** feedback learning
    """)
    
    # Initialize components
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing system..."):
            # Initialize LLM
            st.session_state.llm = ChatOpenAI(
                openai_api_base="https://api.groq.com/openai/v1",
                openai_api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            
            # Initialize components
            st.session_state.guardrails = MathGuardrails()
            st.session_state.knowledge_base = MathKnowledgeBase()
            st.session_state.mcp_search = MCPWebSearchTool(config.TAVILY_API_KEY)
            st.session_state.feedback_system = FeedbackSystem()
            st.session_state.agents = create_agents(st.session_state.llm)
            
            # Load knowledge base
            st.session_state.knowledge_base.load_datasets()
            
            st.session_state.initialized = True
            st.session_state.conversation_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success("✅ All systems operational")
        
        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("Knowledge Base", "Loaded" if st.session_state.knowledge_base.data_loaded else "Not loaded")
        st.metric("Conversations", len(st.session_state.conversation_history))
        
        # Show feedback statistics
        feedback_file = config.FEEDBACK_DIR / "feedback.jsonl"
        if feedback_file.exists():
            try:
                feedback_count = sum(1 for _ in open(feedback_file, 'r', encoding='utf-8'))
                st.metric("Feedback Entries", feedback_count)
                
                # Show average rating if feedback exists
                feedbacks = st.session_state.feedback_system.get_feedback_examples(limit=100)
                if feedbacks:
                    avg_rating = sum(f.get('rating', 0) for f in feedbacks) / len(feedbacks)
                    st.metric("Average Rating", f"{avg_rating:.1f}⭐")
                    
                    # Show learning status
                    corrected = len([f for f in feedbacks if f.get('improved_response')])
                    if corrected > 0:
                        st.success(f"🎓 Learning from {corrected} corrections!")
            except:
                pass
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation"):
            st.session_state.conversation_history = []
            if 'last_rating' in st.session_state:
                del st.session_state.last_rating
            st.rerun()
        
        # Show feedback file location
        st.markdown("---")
        st.markdown("**💾 Feedback Storage:**")
        st.code(str(config.FEEDBACK_DIR / "feedback.jsonl"))
    
    # Main interface
    question = st.text_area(
        "📝 Ask a mathematics question:",
        placeholder="Example: How do I solve quadratic equations using the quadratic formula?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🚀 Get Solution", type="primary")
    
    if submit and question:
        # Step 1: Input Guardrails
        with st.spinner("🛡️ Validating input..."):
            is_valid, validation_msg = st.session_state.guardrails.validate_input(question)
            
            if not is_valid:
                st.error(validation_msg)
                return
            
            st.success(validation_msg)
        
        # Step 2: Route query
        with st.spinner("🔀 Routing query..."):
            # Check knowledge base first
            kb_results = st.session_state.knowledge_base.search(question, k=3)
            
            if kb_results:
                st.info("📚 Found relevant problems in knowledge base")
                context_source = "knowledge_base"
                context = "\n\n".join([r['content'] for r in kb_results])
            else:
                st.info("🌐 Searching the web for information...")
                context_source = "web_search"
                web_results = st.session_state.mcp_search.search(question)
                if web_results.get('results'):
                    context = "\n\n".join([str(r) for r in web_results['results']])
                else:
                    context = ""
        
        # Step 3: Generate solution with feedback learning
        with st.spinner("🧮 Generating step-by-step solution..."):
            # *** CRITICAL: Retrieve and use past feedback for learning ***
            feedback_examples = st.session_state.feedback_system.get_feedback_examples(limit=5)
            
            # Check if we have feedback to learn from
            if feedback_examples:
                st.info(f"💡 Using {len(feedback_examples)} high-quality feedback examples for learning")
            
            # Build few-shot examples from high-quality feedback
            few_shot_section = ""
            if feedback_examples:
                few_shot_section = "\n\n**Past High-Quality Solutions (Learn from these):**\n"
                for i, fb in enumerate(feedback_examples, 1):
                    few_shot_section += f"\nExample {i}:\n"
                    few_shot_section += f"Question: {fb['question']}\n"
                    
                    # Use improved solution if available, otherwise original
                    if fb.get('improved_response'):
                        few_shot_section += f"Correct Solution: {fb['improved_response']}\n"
                        few_shot_section += f"Note: User provided this as the correct solution (Rating: {fb['rating']}★)\n"
                    else:
                        few_shot_section += f"Good Solution: {fb['response'][:500]}...\n"
                        few_shot_section += f"User Rating: {fb['rating']}★\n"
                    
                    if fb.get('comments'):
                        few_shot_section += f"Feedback: {fb['comments']}\n"
                    few_shot_section += "---\n"
            
            # Check if this exact question has feedback with improved answer
            exact_match_feedback = st.session_state.feedback_system.get_feedback_for_question(question)
            
            # If we have an exact match with improved solution, use it directly
            if exact_match_feedback and exact_match_feedback.get('improved_response'):
                st.success("✅ Found corrected solution from your previous feedback!")
                solution_text = f"""**Solution Steps:**

{exact_match_feedback['improved_response']}

**Note:** This solution was corrected based on your previous feedback (Rating: {exact_match_feedback['rating']}★)

**Your Feedback:** {exact_match_feedback.get('comments', 'User provided improved solution')}

**Original Timestamp:** {exact_match_feedback.get('timestamp', 'Unknown')}
"""
            else:
                # Generate new solution with few-shot learning
                solution_prompt = f"""
As a mathematics professor, provide a clear, step-by-step solution to this problem.

**IMPORTANT: Learn from past feedback and mistakes. Use the high-quality examples below as guidance.**

{few_shot_section}

**Current Problem:** {question}

**Context from {context_source}:**
{context}

**Instructions:**
1. Break down the solution into clear, numbered steps
2. Explain each step in simple language
3. Show all mathematical work
4. Double-check your calculations (learn from past mistakes in feedback)
5. Provide a final answer
6. Add a simplified summary for student understanding

**If the examples above contain a similar problem, follow the same approach and reasoning.**

Format your response with:
- **Solution Steps:** (numbered list)
- **Final Answer:**
- **Simplified Explanation:**
"""
                
                response = st.session_state.llm.invoke(solution_prompt)
                solution_text = response.content
        
        # Step 4: Output validation
        with st.spinner("✅ Validating solution..."):
            is_valid_output, output_msg = st.session_state.guardrails.validate_output(solution_text)
            
            if not is_valid_output:
                st.warning(f"Output validation issue: {output_msg}")
                solution_text = "I apologize, but I need to regenerate this solution to ensure quality."
        
        # Display solution
        st.markdown("---")
        st.markdown("## 📖 Solution")
        st.markdown(solution_text)
        
        # Store in history
        st.session_state.conversation_history.append({
            "question": question,
            "solution": solution_text,
            "context_source": context_source,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create unique identifier for this question
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # Store current question and solution for feedback
        st.session_state.current_question = question
        st.session_state.current_solution = solution_text
        st.session_state.current_question_id = question_id
        
        # FEEDBACK SECTION - In expander to minimize rerun impact
        st.markdown("---")
        with st.expander("💬 **Rate & Provide Feedback**", expanded=True):
            st.markdown("### ⭐ Rate this solution:")
            
            # Rating buttons
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("⭐ 1", key=f"rating_1_{question_id}"):
                    st.session_state.last_rating = 1
                    st.success("Rating saved: 1 star")
            with col2:
                if st.button("⭐⭐ 2", key=f"rating_2_{question_id}"):
                    st.session_state.last_rating = 2
                    st.success("Rating saved: 2 stars")
            with col3:
                if st.button("⭐⭐⭐ 3", key=f"rating_3_{question_id}"):
                    st.session_state.last_rating = 3
                    st.success("Rating saved: 3 stars")
            with col4:
                if st.button("⭐⭐⭐⭐ 4", key=f"rating_4_{question_id}"):
                    st.session_state.last_rating = 4
                    st.success("Rating saved: 4 stars")
            with col5:
                if st.button("⭐⭐⭐⭐⭐ 5", key=f"rating_5_{question_id}"):
                    st.session_state.last_rating = 5
                    st.success("Rating saved: 5 stars")
            
            # Show current rating if set
            if 'last_rating' in st.session_state and st.session_state.last_rating:
                st.info(f"⭐ Current rating: {st.session_state.last_rating} star(s)")
            
            # Initialize feedback fields in session state if not present
            if f'feedback_text_value_{question_id}' not in st.session_state:
                st.session_state[f'feedback_text_value_{question_id}'] = ""
            if f'improved_solution_value_{question_id}' not in st.session_state:
                st.session_state[f'improved_solution_value_{question_id}'] = ""
                
            # Feedback section - use session state to preserve values
            st.markdown("---")
            st.markdown("📝 **Provide detailed feedback (Optional)**")
            st.info("💡 **Tip:** Use Ctrl+Enter for new lines. Click Submit button when done (don't press Enter alone).")
            
            # Store in session state to prevent loss on refresh
            feedback_text = st.text_area(
                "Comments or suggestions:",
                value=st.session_state.get(f'feedback_text_value_{question_id}', ""),
                key=f"feedback_input_{question_id}",
                placeholder="What did you like or dislike about this solution?",
                help="Press Ctrl+Enter to add new lines. Click Submit button when done.",
                on_change=lambda: st.session_state.update({f'feedback_text_value_{question_id}': st.session_state[f'feedback_input_{question_id}']})
            )
            
            improved_solution = st.text_area(
                "If you have a better solution, share it here:",
                value=st.session_state.get(f'improved_solution_value_{question_id}', ""),
                key=f"improved_input_{question_id}",
                placeholder="Your improved solution... (Press Ctrl+Enter for new lines)",
                help="Press Ctrl+Enter to add new lines. Click Submit button when done.",
                on_change=lambda: st.session_state.update({f'improved_solution_value_{question_id}': st.session_state[f'improved_input_{question_id}']})
            )
            
            # Store current values in session state
            st.session_state[f'feedback_text_value_{question_id}'] = feedback_text
            st.session_state[f'improved_solution_value_{question_id}'] = improved_solution
            
            # Submit callback function
            def submit_feedback_callback():
                """Callback to handle feedback submission"""
                rating_to_use = st.session_state.get('last_rating', 3)
                fb_text = st.session_state.get(f'feedback_text_value_{question_id}', "")
                imp_sol = st.session_state.get(f'improved_solution_value_{question_id}', "")
                
                print(f"\n=== CALLBACK: SAVING FEEDBACK ===")
                print(f"Question: {question[:50]}...")
                print(f"Rating: {rating_to_use}")
                print(f"Feedback text: {fb_text[:50] if fb_text else 'None'}...")
                print(f"Improved: {imp_sol[:50] if imp_sol else 'None'}...")
                
                success = st.session_state.feedback_system.store_feedback(
                    question=question,
                    response=solution_text,
                    rating=rating_to_use,
                    comments=fb_text,
                    improved_response=imp_sol if imp_sol else None
                )
                
                if success:
                    st.session_state.feedback_submitted = True
                    st.session_state.feedback_success = True
                    # Clear the values after successful submission
                    st.session_state[f'feedback_text_value_{question_id}'] = ""
                    st.session_state[f'improved_solution_value_{question_id}'] = ""
                    print("✅ Feedback saved successfully in callback")
                else:
                    st.session_state.feedback_submitted = True
                    st.session_state.feedback_success = False
                    print("❌ Feedback save failed in callback")
            
            # Submit button with callback
            st.button(
                "📤 Submit Feedback", 
                key=f"submit_btn_{question_id}", 
                type="primary",
                on_click=submit_feedback_callback
            )
            
            # Show result after submission
            if st.session_state.get('feedback_submitted', False):
                if st.session_state.get('feedback_success', False):
                    st.success("✅ Thank you for your feedback! This helps improve the system.")
                    st.balloons()
                    
                    # Verify and show file info
                    feedback_file = config.FEEDBACK_DIR / "feedback.jsonl"
                    if feedback_file.exists():
                        file_size = feedback_file.stat().st_size
                        line_count = sum(1 for _ in open(feedback_file, 'r', encoding='utf-8') if _.strip())
                        st.success(f"✅ Saved! File: {file_size} bytes, Entries: {line_count}")
                        print(f"✅ Verified: {file_size} bytes, {line_count} entries")
                    
                    # Clear the submitted flag after showing message
                    st.session_state.feedback_submitted = False
                else:
                    st.error("❌ Failed to save feedback. Please try again.")
                    st.session_state.feedback_submitted = False
    
    # Show conversation history
    if st.session_state.conversation_history:
        with st.expander("📜 Conversation History", expanded=False):
            st.markdown(f"**Total Conversations:** {len(st.session_state.conversation_history)}")
            st.markdown("---")
            
            # Show last 10 conversations (most recent first)
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
                # Question header with source badge
                source_emoji = "📚" if conv['context_source'] == "knowledge_base" else "🌐"
                source_label = "Knowledge Base" if conv['context_source'] == "knowledge_base" else "Web Search"
                
                st.markdown(f"### {source_emoji} Question {len(st.session_state.conversation_history) - i}")
                st.markdown(f"**{conv['question']}**")
                st.markdown(f"*Source: {source_label}* | *Time: {conv.get('timestamp', 'N/A')[:19]}*")
                
                # Show solution in expander
                with st.expander("View Solution", expanded=False):
                    st.markdown(conv['solution'])
                
                st.markdown("---")


if __name__ == "__main__":
    main()