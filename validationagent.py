#!/usr/bin/env python3
"""
LangGraph Validation Agent - Multi-Model Support
"""

import os
import sys
import re
import json
import time
import requests
from typing import List, Dict, Any, Tuple, Optional, TypedDict, Union
from typing_extensions import Annotated
from datetime import datetime
from urllib.parse import quote
from pathlib import Path

# LangChain and LangGraph imports - Updated for multi-model support
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # New import for Claude
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# New imports for vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache

# Load environment variables
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)
else:
    pass  # Removed debug print for production

# =======================
# Configuration - Updated for multi-model support
# =======================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # New for Claude
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()  # New: Choose "gemini", "claude", or "openai"
TARGET_SECONDS_MIN = int(os.getenv("TARGET_SECONDS_MIN", "30"))
TARGET_SECONDS_MAX = int(os.getenv("TARGET_SECONDS_MAX", "60"))
TARGET_SECONDS = (TARGET_SECONDS_MIN, TARGET_SECONDS_MAX)
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
# New: enforce a minimum number of iterations before completion (even if all checks pass)
MIN_ITERATIONS = int(os.getenv("MIN_ITERATIONS", "2"))
WPM = int(os.getenv("WPM", "140"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "3500"))
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "8"))

# Scientific term extraction method configuration
USE_AI_EXTRACTION = os.getenv("USE_AI_EXTRACTION", "true").lower() == "true"

extraction_method = "AI-powered" if USE_AI_EXTRACTION else "Regex-based"

# =======================
# Multi-Model LLM Factory - New function for dynamic model selection
# =======================
def get_llm(provider: str, model_name: str) -> Any:
    """Factory function to initialize LLM based on provider.
    
    Reasoning: Centralizes model setup for flexibility, ensuring reproducibility by using env vars.
    """
    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required for Gemini")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, max_tokens=8000)
    elif provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY required for Claude")
        return ChatAnthropic(model_name=model_name, temperature=0.2, timeout=None, stop=None)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI")
        return ChatOpenAI(model=model_name, temperature=0.2)
    else:
        # Default to Gemini with warning for unsupported providers
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_tokens=8000)

# =======================
# Vector Store and Embeddings Setup with Fallback Strategy
# =======================

# Set up LLM caching for efficiency
# Reasoning: Caching reduces redundant embedding computations, maintaining accuracy 
# while optimizing for iterative validation workflows.
set_llm_cache(InMemoryCache())

def create_local_embeddings():
    """Create local embeddings using HuggingFace sentence-transformers"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Use a lightweight scientific model optimized for short texts
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings, "local"
    except Exception as e:
        return None, "none"

# Initialize local embeddings only
embeddings, embedding_type = create_local_embeddings()

# Load chemistry keywords and create vector store
# Reasoning: Vector store enables efficient semantic search, reducing false positives 
# in keyword extraction for scientific accuracy.
def load_chemistry_vector_store():
    """Load chemistry keywords and create FAISS vector store"""
    try:
        if not os.path.exists("chemistry_keywords.txt"):
            return None
        
        if embeddings is None:
            return None
        
        with open("chemistry_keywords.txt", "r") as f:
            keywords = [line.strip() for line in f if line.strip()]
        
        if not keywords:
            return None
            
        docs = [Document(page_content=kw) for kw in keywords]
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        return None

# Initialize vector store (load once at startup for efficiency)
chemistry_vector_store = load_chemistry_vector_store()

# =======================
# State Definition with Memory (UNCHANGED - preserving exact behavior)
# =======================

class ValidationAgentState(TypedDict):
    """Enhanced state for the validation agent with memory and cross-agent communication"""
    
    # Core input data
    messages: Annotated[List[BaseMessage], add_messages]
    draft_content: str
    original_prompt: str
    pdf_context: str
    target_seconds: Tuple[int, int]
    
    # Processing state
    current_iteration: int
    max_iterations: int
    is_json_format: bool
    
    # Evidence and research
    evidence_sources: List[Dict[str, Any]]
    search_queries_used: List[str]
    scientific_keywords: List[str]
    
    # Validation results
    validation_results: Dict[str, Any]
    changes_made: List[Dict[str, Any]]
    used_citations: List[int]
    
    # Memory and persistence
    conversation_history: List[Dict[str, Any]]
    agent_memory: Dict[str, Any]  # Persistent memory for cross-agent communication
    processing_metadata: Dict[str, Any]
    
    # Control flow
    next_action: str
    is_complete: bool
    error_message: Optional[str]
    
    # Output
    final_output: Dict[str, Any]

# =======================
# Tools for the Agent (UNCHANGED - preserving exact behavior)
# =======================

@tool
def estimate_reading_time(text: str, wpm: int = 140) -> Dict[str, Any]:
    """Estimate reading/narration time for text content"""
    if not text.strip():
        return {"seconds": 0, "words": 0, "wpm": wpm}
    
    words = len(text.split())
    seconds = (words / wpm) * 60
    
    return {
        "seconds": round(seconds, 1),
        "words": words,
        "wpm": wpm,
        "formatted": f"{seconds:.1f}s ({words} words at {wpm} wpm)"
    }

@tool
def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search Wikipedia for scientific information"""
    headers = {
        'User-Agent': 'LangGraphValidationAgent/1.0 (Educational Research)'
    }
    
    try:
        # Search for articles
        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search", 
                "srsearch": query,
                "format": "json",
                "srlimit": max_results
            },
            headers=headers,
            timeout=20
        )
        
        if search_response.status_code != 200:
            return []
            
        hits = search_response.json().get("query", {}).get("search", [])
        results = []
        
        for hit in hits[:max_results]:
            title = hit.get("title", "")
            if not title:
                continue
                
            # Get article summary
            try:
                summary_response = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                    headers=headers,
                    timeout=20
                )
                
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    results.append({
                        "title": summary_data.get("title", title),
                        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "summary": summary_data.get("extract", "")[:500],
                        "source": "wikipedia",
                        "relevance_score": hit.get("score", 0)
                    })
                    
            except Exception as e:
                print(f"Error getting summary for {title}: {e}")
                continue
                
        return results
        
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return []

@tool
def search_tavily(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """Search using Tavily API for web content"""
    if not TAVILY_API_KEY:
        print("Warning: Tavily API key not found")
        return []
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": []
            },
            timeout=20
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        for result in data.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "summary": result.get("content", "")[:500],
                "source": "tavily",
                "relevance_score": 1.0  # Tavily doesn't provide scores
            })
        
        return results
        
    except requests.RequestException as e:
        print(f"Tavily search error: {e}")
        return []

def _tfidf_keyword_match(text: str, max_terms: int = 10) -> List[str]:
    """Simple TF-IDF based keyword matching as final fallback"""
    try:
        if not os.path.exists("chemistry_keywords.txt"):
            return []
        
        with open("chemistry_keywords.txt", "r") as f:
            keywords = [line.strip().lower() for line in f if line.strip()]
        
        text_lower = text.lower()
        matched_keywords = []
        
        # Simple frequency-based matching
        for keyword in keywords:
            if keyword in text_lower:
                # Count occurrences as relevance score
                count = text_lower.count(keyword)
                matched_keywords.append((keyword, count))
        
        # Sort by frequency and return top matches
        matched_keywords.sort(key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in matched_keywords[:max_terms]]
        
    except Exception as e:
        print(f"TF-IDF fallback error: {e}")
        return []

@tool 
def extract_scientific_terms(text: str, max_terms: int = 10) -> List[str]:
    """Extract scientific keywords and terms from text using hybrid regex + semantic matching"""
    
    # Existing regex extraction (keep for fallback)
    patterns = [
        # Physics/Chemistry terms
        r'\b(?:electron|proton|neutron|atom|molecule|crystal|lattice|quantum|magnetic|electric|thermal|mechanical|optical)\w*\b',
        # Material science terms  
        r'\b(?:flexoelectricity|piezoelectric|ferroelectric|dielectric|conductor|semiconductor|insulator)\b',
        # Scientific principles/effects
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:principle|effect|law|theory|equation|model|method)\b',
        # Chemical formulas (simple)
        r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b',
        # Scientific units
        r'\b\d+\.?\d*\s*(?:nm|μm|mm|cm|m|km|MHz|GHz|eV|keV|MeV|GeV)\b'
    ]
    
    regex_terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        regex_terms.update(matches)
    
    # New: Semantic matching with vector store + TF-IDF fallback
    # Reasoning: Vector similarity captures contextual relationships (e.g., cosine distance for term relevance), 
    # enhancing accuracy for complex scientific narratives.
    semantic_terms = []
    if chemistry_vector_store is not None:
        try:
            # Use similarity search to find relevant chemistry keywords
            similar_docs = chemistry_vector_store.similarity_search(text, k=max_terms)
            semantic_terms = [doc.page_content for doc in similar_docs]
        except Exception as e:
            print(f"Vector store search error: {e}. Using TF-IDF fallback.")
            semantic_terms = _tfidf_keyword_match(text, max_terms)
    else:
        # Fallback to TF-IDF when no vector store available
        semantic_terms = _tfidf_keyword_match(text, max_terms)
    
    # Combine and deduplicate
    # Reasoning: Union of methods ensures no terms are missed, with filtering for scientific relevance.
    all_terms = list(regex_terms.union(set(semantic_terms)))
    filtered_terms = [term for term in all_terms 
                     if len(term) > 2 and term.lower() not in {'the', 'and', 'for', 'with', 'show'}]
    
    return sorted(filtered_terms, key=lambda x: len(x), reverse=True)[:max_terms]

@tool
def validate_json_structure(content: str) -> Dict[str, Any]:
    """Validate and analyze JSON storyboard structure"""
    try:
        data = json.loads(content)
        
        # Check for required storyboard fields
        required_fields = ['prompt', 'title', 'scenes']
        has_required = all(field in data for field in required_fields)
        
        # Analyze scenes
        scenes = data.get('scenes', [])
        scene_analysis = {
            'count': len(scenes),
            'has_narration': sum(1 for scene in scenes if 'narration' in scene),
            'has_visuals': sum(1 for scene in scenes if 'visuals' in scene),
            'total_narration_length': sum(len(scene.get('narration', '')) for scene in scenes)
        }
        
        return {
            'is_valid_json': True,
            'is_storyboard': has_required,
            'required_fields_present': has_required,
            'scene_analysis': scene_analysis,
            'structure_score': 1.0 if has_required else 0.5
        }
        
    except json.JSONDecodeError as e:
        return {
            'is_valid_json': False,
            'is_storyboard': False,
            'error': str(e),
            'structure_score': 0.0
        }

# =======================
# Agent Nodes with Memory Integration (UNCHANGED behavior, updated LLM)
# =======================

def initialize_agent(state: ValidationAgentState) -> ValidationAgentState:
    """Initialize the validation agent with memory setup"""
    
    # Initialize agent memory if not exists
    if "agent_memory" not in state:
        state["agent_memory"] = {}
    
    memory = state["agent_memory"]
    
    # Set up memory structure if not exists
    if "validation_history" not in memory:
        memory["validation_history"] = []
    if "learned_patterns" not in memory:
        memory["learned_patterns"] = []
    if "successful_strategies" not in memory:
        memory["successful_strategies"] = []
    if "failed_approaches" not in memory:
        memory["failed_approaches"] = []
    if "cached_evidence" not in memory:
        memory["cached_evidence"] = {}
    
    # Initialize processing metadata
    state["processing_metadata"] = {
        "session_start": datetime.now().isoformat(),
        "agent_version": "langgraph_multi_model_1.0",
        "model_used": MODEL,
        "provider_used": MODEL_PROVIDER,
        "memory_size": len(memory["validation_history"])
    }
    
    # Initialize state fields (start at 1 like original)
    state["current_iteration"] = 1
    state["max_iterations"] = MAX_ROUNDS
    state["evidence_sources"] = []
    state["search_queries_used"] = []
    state["scientific_keywords"] = []
    state["validation_results"] = {}
    state["changes_made"] = []
    state["used_citations"] = []
    state["conversation_history"] = []
    state["is_complete"] = False
    state["error_message"] = None
    state["next_action"] = "analyze_content"
    
    return state

def analyze_content(state: ValidationAgentState) -> ValidationAgentState:
    """Analyze the content structure and extract key information"""
    print("\n🔎 Analyzing content structure and extracting keywords...")
    
    # Analyze JSON structure
    structure_analysis = validate_json_structure.invoke({"content": state["draft_content"]})
    state["is_json_format"] = structure_analysis.get("is_storyboard", False)
    print(f"   - Format detected: {'JSON storyboard' if state['is_json_format'] else 'Plain text'}")
    
    # Extract scientific keywords
    keywords = extract_scientific_terms.invoke({"text": state["draft_content"]})
    state["scientific_keywords"] = keywords
    print(f"   - Extracted {len(keywords)} keywords: {keywords[:5]}...")
    
    # Estimate reading time
    time_analysis = estimate_reading_time.invoke({"text": state["draft_content"], "wpm": WPM})
    print(f"   - Estimated reading time: {time_analysis.get('formatted', 'N/A')}")
    
    # Update processing metadata
    state["processing_metadata"]["analysis_results"] = {
        "format_detected": "JSON storyboard" if state["is_json_format"] else "text",
        "keywords_found": len(keywords),
        "estimated_time": time_analysis.get("seconds", 0),
        "structure_score": structure_analysis.get("structure_score", 0)
    }
    
    # Add analysis to conversation history
    analysis_summary = (
        f"Content Analysis:\n"
        f"- Format: {'JSON storyboard' if state['is_json_format'] else 'Text'}\n"
        f"- Keywords found: {keywords[:5]}\n"  # Show first 5
        f"- Estimated time: {time_analysis.get('formatted', 'Unknown')}\n"
        f"- Structure score: {structure_analysis.get('structure_score', 0)}"
    )
    
    state["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "action": "analyze_content",
        "summary": analysis_summary
    })
    
    state["next_action"] = "gather_evidence"
    return state

def extract_scientific_terms_ai(content: str) -> List[str]:
    """AI-powered scientific term extraction using Gemini"""
    
    try:
        # Use dynamic LLM for extraction
        llm = get_llm(MODEL_PROVIDER, MODEL)
        
        # Use Gemini to intelligently extract scientific terms
        extraction_prompt = f"""
Extract 5-8 key scientific terms from this content that would be most useful for web searches to validate scientific accuracy.

CONTENT:
{content[:2000]}  

REQUIREMENTS:
- Focus on specific scientific concepts, not general words
- Include both compound terms (e.g. "spin qubit", "NV center") and important single terms
- Prefer technical terms that experts would search for
- Avoid common English words, educational terms, or narrative language
- Return only the terms, one per line, no explanations

EXAMPLE OUTPUT:
nitrogen vacancy center
spin qubit
quantum decoherence
diamond lattice
electron spin resonance
"""

        response = llm.invoke(extraction_prompt)
        content_response = response.content
        content_str = content_response if isinstance(content_response, str) else str(content_response)
        ai_terms = [term.strip() for term in content_str.split('\n') if term.strip()]
        
        # Filter out any remaining non-scientific terms and artifacts
        scientific_terms = []
        for term in ai_terms:
            # Basic filters for obvious non-scientific terms and code artifacts
            if (len(term) > 2 and 
                not term.lower() in ['the', 'and', 'but', 'for', 'you', 'can', 'show', 'example'] and
                not term.startswith(('How', 'What', 'When', 'Where', 'Why')) and
                not term in ['```', '---', '***'] and  # Filter markdown artifacts
                not term.startswith(('#', '*', '-', '`'))):  # Filter markdown formatting
                scientific_terms.append(term)
        
        return scientific_terms
        
    except Exception as e:
        print(f"⚠️ AI extraction failed ({e}), falling back to regex method...")
        return extract_scientific_terms_regex(content)

def extract_scientific_terms_regex(content: str) -> List[str]:
    """Regex-based scientific term extraction (fallback method)"""
    
    # First, extract compound scientific terms (2-3 words)
    compound_terms = re.findall(
        r"\b(?:NV[- ]center|spin[- ]qubit|quantum[- ](?:state|system|mechanics)|nitrogen[- ]vacancy|"
        r"diamond[- ]lattice|energy[- ]level|ground[- ]state|excited[- ]state)\b", 
        content, re.IGNORECASE
    )
    
    # Then extract single scientific terms with better filtering
    single_terms = re.findall(
        r"\b(?:diamond|qubit|spin|quantum|electron|nitrogen|vacancy|coherence|decoherence|"
        r"photon|phonon|lattice|crystal|fluorescence|microwave|laser)\b", 
        content, re.IGNORECASE
    )
    
    # Combine and clean up terms
    scientific_terms = list(dict.fromkeys(compound_terms + single_terms))
    
    return scientific_terms

def gather_evidence(state: ValidationAgentState) -> ValidationAgentState:
    """Gather external evidence following original validation-agent.py approach"""
    print("\n📚 Gathering evidence from external sources...")
    
    # Check cached evidence first
    memory = state["agent_memory"]
    cached_evidence = memory.get("cached_evidence", {})
    
    evidence_sources = []
    seen_urls = set()
    
    # Original approach: Use prompt first, then add scientific terms from draft
    original_prompt = state["original_prompt"]
    current_draft = state["draft_content"]
    
    # Scientific term extraction - AI or regex based on configuration  
    if USE_AI_EXTRACTION:
        scientific_terms = extract_scientific_terms_ai(current_draft)
    else:
        scientific_terms = extract_scientific_terms_regex(current_draft)
    
    # Build query list: original prompt first, then top scientific terms
    queries = [original_prompt] + scientific_terms[:5]  # Limit to top 5 scientific terms
    print(f"   - Using {len(queries)} search queries: {queries}")
    
    # Search with each query (prompt + scientific terms)
    for i, query in enumerate(queries):
        if len(evidence_sources) >= MAX_SOURCES:
            break
            
        cache_key = f"combined_{query[:50].lower()}"
        
        if cache_key in cached_evidence:
            cached_results = cached_evidence[cache_key]
            print(f"   - Found {len(cached_results)} cached results for query: '{query}'")
            for result in cached_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    evidence_sources.append(result)
                    seen_urls.add(url)
        else:
            print(f"   - Searching online for: '{query}'")
            # Search both Tavily and Wikipedia for each query (original approach)
            tavily_results = search_tavily.invoke({"query": query, "max_results": 1})
            wiki_results = search_wikipedia.invoke({"query": query, "max_results": 1})
            
            valid_results = []
            
            # Combine results from both sources
            for result in tavily_results + wiki_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    valid_results.append(result)
                    evidence_sources.append(result)
                    seen_urls.add(url)
                    
                if len(evidence_sources) >= MAX_SOURCES:
                    break
            
            print(f"   - Found {len(valid_results)} new sources for query: '{query}'")
            cached_evidence[cache_key] = valid_results
            state["search_queries_used"].append(query)
        
        # Brief delay between searches
        time.sleep(0.15)
    
    state["evidence_sources"] = evidence_sources[:MAX_SOURCES]  # Limit total sources
    print(f"   - Total evidence sources gathered: {len(evidence_sources)}")
    
    # Update memory cache
    memory["cached_evidence"] = cached_evidence
    
    # Add to conversation history
    state["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "action": "gather_evidence", 
        "summary": f"Gathered {len(evidence_sources)} evidence sources for validation"
    })
    
    state["next_action"] = "validate_content"
    return state



def validate_content(state: ValidationAgentState) -> ValidationAgentState:
    """Perform LLM-based validation matching original agent behavior exactly"""
    print(f"\n🔬 Validating content with {MODEL_PROVIDER.upper()}/{MODEL}...")
    
    # Use dynamic LLM for validation
    llm = get_llm(MODEL_PROVIDER, MODEL)
    
    # Prepare evidence block exactly like original
    evidence_block = "\n\n".join([
        f"[{i+1}] {evidence.get('title', '')}\nURL: {evidence.get('url', '')}\nSummary: {evidence.get('summary', '')}"
        for i, evidence in enumerate(state["evidence_sources"])
    ]) if state["evidence_sources"] else "None available."
    
    # Check if content is JSON format storyboard (like original)
    is_json_format = False
    try:
        draft_obj = json.loads(state["draft_content"])
        if "prompt" in draft_obj and "scenes" in draft_obj:
            is_json_format = True
    except json.JSONDecodeError:
        is_json_format = False
    
    # Use EXACT original system prompt and formatting (PERFECTLY IDENTICAL)
    if is_json_format:
        system_content = (
            "You are a rigorous science editor for materials & chemistry explainers. "
            "The input is a structured JSON storyboard with prompt, title, age_level, video_length, and scenes array. "
            "Your job: (1) FACT-CHECK and CORRECT the content; (2) maintain JSON STRUCTURE; "
            "(3) fit total narration LENGTH to target time; (4) ensure CONSISTENCY with the original prompt such as the age level, title and relevance to prompt given. Make sure to audit and improve the visuals as wells using scenes that can be rendered by manim or its manim chemistry or manim physics plugins and being as discriptive as possible."
            "Use ONLY the EVIDENCE block for citations in narration text; add inline numeric citations like [1], [2]. "
            "If evidence is insufficient, keep cautious language (≈, ~, 'approximately'). "
            "\n"
            "When validating scientific accuracy:\n"
            "- First, extract *scientific keywords and concepts* from the storyboard JSON.\n"
            "  Examples: 'electron cloud', 'flexoelectricity', 'Heisenberg uncertainty principle', 'atomic nucleus'.\n"
            "- Ignore narrative or decorative language (e.g. 'sparkling kitchen', 'favorite book').\n"
            "- Use only these keywords for external web lookups.\n"
            "\n"
            "Output strict JSON with keys: revised (complete JSON storyboard), validation, changes, used_citations."
        )
        
        format_instructions = (
            "REQUIREMENTS:\n"
            "• Accuracy: Correct factual errors in narration and visuals; add citations [n] that map to EVIDENCE. Also add details about news articles and publications if found.\n"
            "• Structure: Maintain JSON format with prompt, title, age_level, video_length, and scenes array.\n"
            "• Each scene must have: scene_id, title, narration, visuals array.\n"
            "• Length: Adjust total narration across all scenes to meet target seconds (assume ~140 wpm).\n"
            "• Consistency: Ensure content matches the ORIGINAL PROMPT and fits the age_level.\n"
            "• Output JSON only with this schema:\n"
            "{\n"
            '  "revised": {complete JSON storyboard object with all corrections},\n'
            '  "validation": {"accuracy":"pass|warn","structure":"pass|warn","length":"pass|warn","consistency":"pass|warn","reasons":{...}},\n'
            '  "changes": [{"type":"fact_correction|restructure|length_adjust|consistency_fix","scene":"1","before":"...","after":"...","rationale":"..."}],\n'
            '  "used_citations": [1,2, ...]\n'
            "}\n"
            "IMPORTANT: For 'scene' field in changes, use ONLY the scene number (1, 2, 3, etc.), NOT 'Scene 1' or 'scene 1'.\n"
        )
    else:
        system_content = (
            "You are a rigorous science editor for materials & chemistry explainers. "
            "Your job: (1) FACT-CHECK and CORRECT the draft; (2) enforce proper STORYBOARD STRUCTURE; "
            "(3) fit LENGTH to a target time; (4) ensure CONSISTENCY with the original prompt and provided PDF excerpts. "
            "Use ONLY the EVIDENCE block for citations when possible; add inline numeric citations like [1], [2]. "
            "If evidence is insufficient, keep cautious language (≈, ~, 'approximately'). "
            "\n"
            "When validating scientific accuracy:\n"
            "- First, extract *scientific keywords and concepts* from the storyboard.\n"
            "  Examples: 'electron cloud', 'flexoelectricity', 'Heisenberg uncertainty principle', 'atomic nucleus'.\n"
            "- Ignore narrative or decorative language (e.g. 'sparkling kitchen', 'favorite book').\n"
            "- Use only these keywords for external web lookups.\n"
            "\n"
            "Output strict JSON with keys: revised, validation, changes, used_citations."
        )
        
        format_instructions = (
            "REQUIREMENTS:\n"
            "• Accuracy: Correct factual errors; add citations [n] that map to the EVIDENCE list.\n"
            "• Structure: Include a clear Title, 3–5 Learning Objectives (measurable verbs), and sections in this order:\n"
            "  Motivation → Core Concept(s) → Worked Example → Recap. Keep concise.\n"
            "• Length: Adjust narration to meet the target seconds (assume ~140 wpm).\n"
            "• Consistency: Ensure content matches the ORIGINAL PROMPT and does not contradict the PDF EXCERPTS.\n"
            "• Output JSON only with this schema:\n"
            "{\n"
            '  "revised": "<final storyboard with inline numeric citations>",\n'
            '  "validation": {"accuracy":"pass|warn","structure":"pass|warn","length":"pass|warn","consistency":"pass|warn","reasons":{...}},\n'
            '  "changes": [{"type":"fact_correction|restructure|length_adjust|consistency_fix","before":"...","after":"...","rationale":"..."}],\n'
            '  "used_citations": [1,2, ...]\n'
            "}\n"
        )
    
    # EXACT original message format with system + user
    user_content = (
        f"ORIGINAL PROMPT:\n{state['original_prompt']}\n\n"
        f"PDF EXCERPTS (evidence to preserve alignment; do not copy verbatim):\n\n\n"  # Empty PDF blob like original
        f"EVIDENCE (numbered for citations):\n{evidence_block}\n\n"
        f"TARGET LENGTH (seconds): {state['target_seconds'][0]}–{state['target_seconds'][1]}\n\n"
        f"DRAFT STORYBOARD (to validate and correct):\n"
        f"{state['draft_content']}\n\n"
        f"{format_instructions}"
    )
    
    try:
        # Get LLM response using EXACT original two-message format 
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]
        
        response = llm.invoke(messages)
        content = response.content

        # Helper: normalize AI content to string and extract a JSON object/array if present
        def _normalize_content_to_text(c) -> str:
            if isinstance(c, str):
                return c
            # LangChain AIMessage.content can be a list of parts; join their text
            if isinstance(c, list):
                parts = []
                for p in c:
                    try:
                        # Try common shapes from providers
                        if isinstance(p, dict) and "text" in p:
                            parts.append(p.get("text", ""))
                        else:
                            parts.append(str(p))
                    except Exception:
                        parts.append(str(p))
                return "\n".join(parts)
            try:
                return str(c)
            except Exception:
                return ""

        def _extract_json_block(s: str) -> Optional[str]:
            t = (s or "").strip()
            if not t:
                return None
            # Prefer fenced ```json blocks
            if t.startswith('```'):
                lines = t.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        if not in_block:
                            in_block = True
                            continue
                        else:
                            break
                    if in_block:
                        json_lines.append(line)
                candidate = "\n".join(json_lines).strip()
                if candidate:
                    return candidate
            # Scan for balanced JSON object or array
            def scan_balanced(open_ch: str, close_ch: str) -> Optional[str]:
                i = 0
                n = len(t)
                while i < n:
                    if t[i] == open_ch:
                        depth = 0
                        j = i
                        in_str = False
                        esc = False
                        while j < n:
                            ch = t[j]
                            if in_str:
                                if esc:
                                    esc = False
                                elif ch == '\\':
                                    esc = True
                                elif ch == '"':
                                    in_str = False
                            else:
                                if ch == '"':
                                    in_str = True
                                elif ch == open_ch:
                                    depth += 1
                                elif ch == close_ch:
                                    depth -= 1
                                    if depth == 0:
                                        return t[i:j+1]
                            j += 1
                    i += 1
                return None
            candidate = scan_balanced('{', '}') or scan_balanced('[', ']')
            return candidate

        # Parse response for original format - handle provider quirks robustly
        try:
            content_str = _normalize_content_to_text(content)
            json_candidate = _extract_json_block(content_str)
            if json_candidate is None:
                # Nothing parseable; force failure path
                raise json.JSONDecodeError("No JSON found in response", content_str, 0)
            result = json.loads(json_candidate)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   - ⚠️ Failed to parse LLM response as JSON: {e}")
            # Fallback to original format structure
            result = {
                "revised": state["draft_content"],
                "validation": {
                    "accuracy": "warn",
                    "structure": "warn", 
                    "length": "warn",
                    "consistency": "warn",
                    "reasons": {"error": f"Unable to parse detailed validation response: {str(e)}"}
                },
                "changes": [],
                "used_citations": []
            }
        
        # Store results in original format and clean up scene numbers
        validation_results = result.get("validation", {})
        
        # Calculate overall score based on pass/warn status (like original agent)
        scores = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
        individual_scores = [
            scores.get(validation_results.get("accuracy", "warn"), 0.0),
            scores.get(validation_results.get("structure", "warn"), 0.0),
            scores.get(validation_results.get("length", "warn"), 0.0),
            scores.get(validation_results.get("consistency", "warn"), 0.0)
        ]
        overall_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        validation_results["overall"] = overall_score
        
        state["validation_results"] = validation_results
        print(f"   - Validation complete. Overall score: {overall_score:.2f}")
        
        # Fix scene numbers to match original format (just numbers, not "Scene N")
        changes_made = result.get("changes", [])
        for change in changes_made:
            if "scene" in change:
                scene_value = str(change["scene"])
                # Convert "Scene 1", "Scene 2" etc. to just "1", "2"
                if scene_value.lower().startswith("scene"):
                    # Extract just the number
                    import re
                    match = re.search(r'scene\s*(\d+)', scene_value, re.IGNORECASE)
                    if match:
                        change["scene"] = match.group(1)
        
        state["changes_made"] = changes_made
        state["used_citations"] = result.get("used_citations", [])
        print(f"   - {len(changes_made)} changes suggested by the LLM.")
        
        # Update the draft content if revised
        if "revised" in result:
            state["draft_content"] = json.dumps(result["revised"]) if isinstance(result["revised"], dict) else str(result["revised"])
        
        # Update memory with validation outcome
        memory = state["agent_memory"]
        memory["validation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "prompt": state["original_prompt"],
            "validation_status": result.get("validation", {}),
            "changes_count": len(result.get("changes", [])),
            "model_used": MODEL
        })
        
        # Learn patterns based on validation status
        validation_status = result.get("validation", {})
        statuses = [validation_status.get("accuracy"), validation_status.get("structure"), 
                   validation_status.get("length"), validation_status.get("consistency")]
        
        if all(s == "pass" for s in statuses):
            memory["successful_strategies"].append(f"All pass with {len(state['evidence_sources'])} evidence sources")
        
        # Add to conversation history
        state["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "validate_content",
            "summary": f"Validation complete - Status: {statuses}"
        })
        
        state["next_action"] = "check_completion"
        
    except Exception as e:
        state["error_message"] = f"Validation error: {str(e)}"
        state["next_action"] = "finalize_results"
        print(f"   - ❌ An unexpected error occurred during validation: {e}")
    
    return state

def check_completion(state: ValidationAgentState) -> ValidationAgentState:
    """Check if validation meets completion criteria using original format"""
    print("\n🤔 Checking completion criteria...")
    
    results = state["validation_results"]
    
    # Check completion based on pass/warn status like original
    statuses = [results.get("accuracy"), results.get("structure"), 
               results.get("length"), results.get("consistency")]
    
    # Completion criteria: all pass AND at least MIN_ITERATIONS reached, OR max iterations reached
    all_pass = all(s == "pass" for s in statuses)
    max_iterations_reached = state["current_iteration"] >= state["max_iterations"]
    min_iterations_reached = state["current_iteration"] >= MIN_ITERATIONS

    if (all_pass and min_iterations_reached) or max_iterations_reached:
        state["is_complete"] = True
        state["next_action"] = "finalize_results"
        
        completion_reason = (
            "All validation criteria passed" if all_pass and min_iterations_reached
            else "Maximum iterations reached"
        )
        print(f"   - ✅ Completion criteria met: {completion_reason}")
        
        state["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "check_completion",
            "summary": f"Validation complete: {completion_reason}"
        })
    else:
        # Continue with another iteration
        state["current_iteration"] += 1
        state["next_action"] = "gather_evidence"
        print(f"   - ❌ Completion criteria not met. Continuing to iteration {state['current_iteration']}.")
        
        state["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "check_completion",
            "summary": f"Continuing validation - Iteration {state['current_iteration']} (Status: {statuses})"
        })
    
    return state

def finalize_results(state: ValidationAgentState) -> ValidationAgentState:
    """Prepare final output and update persistent memory (UNCHANGED)"""
    print("\n🏁 Finalizing results and updating agent memory...")
    
    # Create final output (UNCHANGED)
    state["final_output"] = {
        "validation_complete": state["is_complete"],
        "validation_results": state["validation_results"],
        "evidence_sources": state["evidence_sources"],
        "processing_metadata": state["processing_metadata"],
        "conversation_summary": state["conversation_history"],
        "iterations_used": state["current_iteration"],
        "error_message": state["error_message"]
    }
    
    # Update agent memory with session learnings (UNCHANGED)
    memory = state["agent_memory"]
    
    # Add learned patterns (UNCHANGED)
    if state["validation_results"]:
        pattern = f"Session completed with {len(state['evidence_sources'])} sources, score: {state['validation_results'].get('overall', 0):.2f}"
        memory["learned_patterns"].append(pattern)
        print(f"   - Learned new pattern: {pattern}")
    
    # Limit memory size to prevent unbounded growth (UNCHANGED)
    for key in ["validation_history", "learned_patterns", "successful_strategies"]:
        if len(memory.get(key, [])) > 50:  # Keep last 50 items
            memory[key] = memory[key][-50:]
    
    # Final conversation entry (UNCHANGED)
    state["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "action": "finalize_results",
        "summary": "Validation session completed and memory updated"
    })
    
    print(f"   - ✅ Validation session concluded. Overall score: {state['validation_results'].get('overall', 0):.2f}")
    return state

def save_validation_report(result: Dict[str, Any], thread_id: str):
    """Save detailed validation report to JSON and Markdown files."""
    
    # Create a unique filename for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"validation_report_{thread_id}_{timestamp}"
    
    # 1. Save full JSON report
    json_filename = f"{filename_base}.json"
    try:
        with open(json_filename, "w") as f:
            # Use a custom encoder to handle non-serializable types if needed
            json.dump(result, f, indent=4, default=str)
        print(f"✅ Saved detailed JSON report to: {json_filename}")
    except Exception as e:
        print(f"❌ Error saving JSON report: {e}")
        
    # 2. Save human-readable Markdown report
    md_filename = f"{filename_base}.md"
    try:
        with open(md_filename, "w") as f:
            f.write(f"# Validation Report - {thread_id}\n\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
            f.write(f"**Model Used:** {result.get('metadata', {}).get('provider_used', 'N/A')}/{result.get('metadata', {}).get('model_used', 'N/A')}\n\n")
            
            # Validation summary
            validation = result.get("validation_results", {})
            f.write("## 📊 Validation Summary\n\n")
            f.write(f"| Category      | Status                               |\n")
            f.write(f"|---------------|--------------------------------------|\n")
            f.write(f"| Accuracy      | {validation.get('accuracy', 'N/A')}      |\n")
            f.write(f"| Structure     | {validation.get('structure', 'N/A')}    |\n")
            f.write(f"| Length        | {validation.get('length', 'N/A')}       |\n")
            f.write(f"| Consistency   | {validation.get('consistency', 'N/A')}  |\n")
            f.write(f"| **Overall Score** | **{validation.get('overall', 0):.2f}** |\n\n")
            
            # Validation reasons
            reasons = validation.get("reasons", {})
            if reasons:
                f.write("### 📝 Validation Reasons\n\n")
                for key, value in reasons.items():
                    f.write(f"- **{key.capitalize()}:** {value}\n")
                f.write("\n")

            # Changes made
            changes = result.get("state", {}).get("changes_made", [])
            if changes:
                f.write("## 🔄 Changes Made\n\n")
                for i, change in enumerate(changes):
                    f.write(f"### Change {i+1}: {change.get('type', 'N/A')}\n")
                    f.write(f"- **Scene:** {change.get('scene', 'N/A')}\n")
                    f.write(f"- **Rationale:** {change.get('rationale', 'N/A')}\n")
                    f.write("```diff\n")
                    f.write(f"- {change.get('before', '')}\n")
                    f.write(f"+ {change.get('after', '')}\n")
                    f.write("```\n\n")
            
            # Evidence sources
            evidence = result.get("evidence", [])
            if evidence:
                f.write("## 📚 Evidence Sources\n\n")
                for i, source in enumerate(evidence):
                    f.write(f"{i+1}. **{source.get('title', 'N/A')}**\n")
                    f.write(f"   - URL: {source.get('url', '#')}\n")
                    f.write(f"   - Source: {source.get('source', 'N/A')}\n\n")
            
            # Revised content
            revised_content = result.get("state", {}).get("draft_content", "")
            if revised_content:
                f.write("## 📄 Revised Content\n\n")
                try:
                    # Try to format it as pretty-printed JSON
                    revised_json = json.loads(revised_content)
                    f.write("```json\n")
                    f.write(json.dumps(revised_json, indent=2))
                    f.write("\n```\n")
                except (json.JSONDecodeError, TypeError):
                    # Fallback to just printing the string
                    f.write("```\n")
                    f.write(revised_content)
                    f.write("\n```\n")

        print(f"✅ Saved readable Markdown report to: {md_filename}")
    except Exception as e:
        print(f"❌ Error saving Markdown report: {e}")

# =======================
# LangGraph Workflow Creation (UNCHANGED)
# =======================

def create_validation_agent_graph():
    """Create the validation agent LangGraph workflow"""
    
    # Create the graph (UNCHANGED)
    workflow = StateGraph(ValidationAgentState)
    
    # Add nodes (UNCHANGED)
    workflow.add_node("initialize_agent", initialize_agent)
    workflow.add_node("analyze_content", analyze_content)
    workflow.add_node("gather_evidence", gather_evidence)  
    workflow.add_node("validate_content", validate_content)
    workflow.add_node("check_completion", check_completion)
    workflow.add_node("finalize_results", finalize_results)
    
    # Define edges (UNCHANGED)
    workflow.add_edge(START, "initialize_agent")
    workflow.add_edge("initialize_agent", "analyze_content")
    workflow.add_edge("analyze_content", "gather_evidence")
    workflow.add_edge("gather_evidence", "validate_content")
    workflow.add_edge("validate_content", "check_completion")
    
    # Conditional edges (UNCHANGED)
    def route_after_completion_check(state: ValidationAgentState) -> str:
        if state["is_complete"]:
            return "finalize_results"
        else:
            return "gather_evidence"  # Continue iteration
    
    workflow.add_conditional_edges(
        "check_completion",
        route_after_completion_check,
        {
            "finalize_results": "finalize_results",
            "gather_evidence": "gather_evidence"
        }
    )
    
    workflow.add_edge("finalize_results", END)
    
    # Compile the graph (UNCHANGED)
    return workflow.compile()

# =======================
# Main Execution Function (UNCHANGED interface, Gemini backend)
# =======================

def run_validation_agent(draft_content: str, original_prompt: str, 
                        pdf_context: str = "", thread_id: str = "default",
                        target_seconds: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Run the validation agent with persistent memory
    
    BEHAVIOR UNCHANGED - Same interface, same outputs, just using Gemini instead of OpenAI
    """
    
    if target_seconds is None:
        target_seconds = TARGET_SECONDS
        
    # Create agent (UNCHANGED)
    agent = create_validation_agent_graph()
    
    # Initial state (UNCHANGED)
    initial_state: ValidationAgentState = {
        "messages": [HumanMessage(content=f"Validate: {original_prompt}")],
        "draft_content": draft_content,
        "original_prompt": original_prompt, 
        "pdf_context": pdf_context,
        "target_seconds": target_seconds,
        "current_iteration": 0,
        "max_iterations": MAX_ROUNDS,
        "is_json_format": False,
        "evidence_sources": [],
        "search_queries_used": [],
        "scientific_keywords": [],
        "validation_results": {},
        "changes_made": [],
        "used_citations": [],
        "conversation_history": [],
        "agent_memory": {},  # Will be initialized by agent
        "processing_metadata": {},
        "next_action": "",
        "is_complete": False,
        "error_message": None,
        "final_output": {}
    }
    
    try:
        # Run the agent (UNCHANGED)
        result = agent.invoke(initial_state)
        
        # Return format (UNCHANGED)
        return {
            "success": True,
            "validation_results": result["validation_results"],
            "agent_memory": result["agent_memory"], 
            "conversation": result["conversation_history"],
            "evidence": result["evidence_sources"],
            "metadata": result["processing_metadata"],
            "state": result,  # Full state for other agents
            "error": result["error_message"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "agent_memory": {},
            "conversation": [],
            "evidence": [],
            "metadata": {},
            "state": initial_state
        }

# =======================
# JSON File Processing Functions (NEW - from run_gemini_validation.py)
# =======================

def generate_output_filenames(input_file: str) -> Dict[str, str]:
    """Generate output filenames based on input file"""
    base_name = Path(input_file).stem  # Remove .json extension
    
    return {
        "json": f"{base_name}_validated.json",
        "md": f"{base_name}_validated.md", 
        "report": f"{base_name}_validation_report.json"
    }

def load_storyboard_json(filepath: str) -> Tuple[Dict[str, Any], str]:
    """Load JSON storyboard file and extract content and prompt"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats (conversation vs direct storyboard)
    if isinstance(data, list):
        # Conversation format - extract storyboard from assistant messages
        storyboard_content = ""
        original_prompt = ""
        
        for msg in data:
            if msg.get("role") == "assistant" and "content" in msg:
                storyboard_content = msg["content"]
            elif msg.get("role") == "user" and "content" in msg:
                original_prompt = msg["content"]
        
        return {"format": "conversation", "content": storyboard_content}, original_prompt
        
    elif isinstance(data, dict) and "scenes" in data:
        # Direct JSON storyboard format
        original_prompt = data.get("prompt", "No prompt provided")
        storyboard_json = json.dumps(data, indent=2, ensure_ascii=False)
        
        return {"format": "direct_json", "content": storyboard_json, "data": data}, original_prompt
    
    else:
        raise ValueError(f"Unrecognized JSON format in {filepath}")

def save_validation_outputs(result: Dict[str, Any], output_files: Dict[str, str], 
                          input_file: str, original_data: Dict[str, Any]) -> None:
    """Save validation outputs in JSON, Markdown, and report formats"""
    
    validation_results = result.get("validation_results", {})
    
    # Save validated storyboard as JSON
    try:
        should_save_json = True
        
        # Look for final content in state
        final_content = None
        if "state" in result and "draft_content" in result["state"]:
            final_content = result["state"]["draft_content"]
        
        if final_content and should_save_json:
            # Try to parse as JSON
            if isinstance(final_content, str):
                try:
                    parsed_storyboard = json.loads(final_content)
                    
                    # Save as JSON if it has storyboard structure
                    if isinstance(parsed_storyboard, dict) and ("scenes" in parsed_storyboard or "prompt" in parsed_storyboard):
                        with open(output_files["json"], "w", encoding="utf-8") as f:
                            json.dump(parsed_storyboard, f, ensure_ascii=False, indent=2)
                        print(f"✅ Saved validated storyboard: {output_files['json']}")
                    else:
                        raise ValueError("Not a structured storyboard")
                        
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, save original data (minimally modified)
                    if "data" in original_data:
                        with open(output_files["json"], "w", encoding="utf-8") as f:
                            json.dump(original_data["data"], f, ensure_ascii=False, indent=2)
                        print(f"✅ Saved original structure: {output_files['json']}")
            else:
                # Save original data structure if final_content is not string
                if "data" in original_data:
                    with open(output_files["json"], "w", encoding="utf-8") as f:
                        json.dump(original_data["data"], f, ensure_ascii=False, indent=2)
                    print(f"✅ Saved original structure: {output_files['json']}")
        else:
            # No final content found, save original data 
            if "data" in original_data:
                with open(output_files["json"], "w", encoding="utf-8") as f:
                    json.dump(original_data["data"], f, ensure_ascii=False, indent=2)
                print(f"✅ Saved original structure: {output_files['json']}")
                    
    except Exception as e:
        print(f"⚠️ JSON save error: {e}")
    
    # Save markdown version
    try:
        with open(output_files["md"], "w", encoding="utf-8") as f:
            if "data" in original_data and isinstance(original_data["data"], dict):
                data = original_data["data"]
                f.write(f"# {data.get('title', 'Storyboard')}\n\n")
                f.write(f"**Prompt:** {data.get('prompt', '')}\n\n")
                f.write(f"**Age Level:** {data.get('age_level', '')}\n\n")
                f.write(f"**Video Length:** {data.get('video_length', '')}\n\n")
                
                scenes = data.get('scenes', [])
                for i, scene in enumerate(scenes, 1):
                    f.write(f"## Scene {i}: {scene.get('title', f'Scene {i}')}\n\n")
                    f.write(f"**Narration:** {scene.get('narration', '')}\n\n")
                    
                    visuals = scene.get('visuals', [])
                    if visuals:
                        f.write("**Visuals:**\n")
                        for visual in visuals:
                            if isinstance(visual, dict):
                                f.write(f"- {visual.get('description', str(visual))}\n")
                            else:
                                f.write(f"- {visual}\n")
                        f.write("\n")
            else:
                f.write(original_data.get("content", "No content available"))
                
        print(f"✅ Saved markdown version: {output_files['md']}")
        
    except Exception as e:
        print(f"⚠️ Markdown save error: {e}")
    
    # Save validation report
    try:
        # Format citations
        citations_formatted = []
        if result.get("evidence"):
            for i, evidence in enumerate(result["evidence"], 1):
                citations_formatted.append({
                    "index": i,
                    "title": evidence.get("title", ""),
                    "url": evidence.get("url", ""),
                    "summary": evidence.get("summary", "")
                })
        
        # Format processing history
        processing_history = []
        if result.get("success") and "state" in result:
            final_state = result["state"]
            processing_history.append({
                "revised": json.loads(final_state.get("draft_content", "{}")) if final_state.get("draft_content", "").startswith("{") else final_state.get("draft_content", ""),
                "validation": final_state.get("validation_results", {}),
                "changes": final_state.get("changes_made", []),
                "used_citations": final_state.get("used_citations", [])
            })
        
        report = {
            "input_file": input_file,
            "model_provider": MODEL_PROVIDER,
            "model_used": MODEL,
            "iterations": result["state"].get("current_iteration", 1) if "state" in result else 1,
            "validation_results": validation_results,
            "changes_made": result["state"].get("changes_made", []) if "state" in result else [],
            "citations_used": citations_formatted,
            "processing_history": processing_history
        }
        
        with open(output_files["report"], "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved validation report: {output_files['report']}")
        
    except Exception as e:
        print(f"⚠️ Report save error: {e}")

def process_storyboard_file(input_file: str, target_seconds: Tuple[int, int] = (30, 60)) -> Dict[str, Any]:
    """Process a JSON storyboard file with validation"""
    
    print("=" * 70)
    print(f"🚀 STARTING {MODEL_PROVIDER.upper()} VALIDATION")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return {"success": False, "error": f"File not found: {input_file}"}
    
    # Generate output filenames
    output_files = generate_output_filenames(input_file)
    
    print(f"📊 Configuration:")
    print(f"   - Input: {input_file}")
    print(f"   - Model: {MODEL_PROVIDER} - {MODEL}")
    print(f"   - Target: {target_seconds} seconds")
    print(f"   - Output files: {output_files['json']}, {output_files['md']}")
    
    try:
        print(f"\n🔄 STEP 1: LOADING INPUT DATA")
        print("-" * 50)
        
        # Load storyboard
        storyboard_data, original_prompt = load_storyboard_json(input_file)
        
        print(f"✅ Successfully loaded {storyboard_data['format']} format")
        print(f"   📊 Original prompt: {original_prompt[:100]}...")
        
        print(f"\n🔄 STEP 2: RUNNING {MODEL_PROVIDER.upper()} VALIDATION")
        print("-" * 50)
        
        # Add timing for performance analysis
        start_time = time.time()
        
        # Run validation with multi-model agent
        result = run_validation_agent(
            draft_content=storyboard_data["content"],
            original_prompt=original_prompt,
            target_seconds=target_seconds,
            thread_id=f"file_validation_{Path(input_file).stem}"
        )
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        if result["success"]:
            overall_score = result["validation_results"].get("overall", 0.0)
            print(f"✅ Validation completed - Score: {overall_score:.2f}")
            
            # Display evidence sources in detail
            evidence = result.get('evidence', [])
            print(f"   📚 Evidence sources: {len(evidence)}")
            
            if evidence:
                print(f"\n🔍 Evidence Sources Used:")
                for i, source in enumerate(evidence, 1):
                    source_type = source.get('source', 'unknown')
                    title = source.get('title', 'No title')
                    url = source.get('url', 'No URL')
                    summary = source.get('summary', 'No summary')
                    
                    print(f"   {i}. [{source_type.upper()}] {title}")
                    print(f"      URL: {url}")
                    print(f"      Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}")
                    print()
            
            print(f"   🧠 Memory entries: {len(result.get('agent_memory', {}).get('validation_history', []))}")
            print(f"   ⏱️ Processing time: {elapsed:.2f}s")
        else:
            print(f"❌ Validation failed: {result.get('error', 'Unknown error')}")
            return result
        
        print(f"\n🔄 STEP 3: SAVING OUTPUT FILES")
        print("-" * 50)
        
        # Save outputs
        save_validation_outputs(result, output_files, input_file, storyboard_data)
        
        print(f"\n" + "=" * 50)
        print(f"🎉 VALIDATION COMPLETE")
        print(f"=" * 50)
        print(f"✅ Completed successfully with {MODEL_PROVIDER}")
        print(f"📝 Overall score: {result['validation_results'].get('overall', 0):.2f}")
        print(f"📚 Evidence sources: {len(result.get('evidence', []))}")
        print(f"🔄 Memory updated with new learnings")
        
        print(f"\n📁 Output Files Created:")
        for file_type, filepath in output_files.items():
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   ✅ {filepath} ({size} bytes)")
            else:
                print(f"   ❌ {filepath} (not created)")
        
        return result
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

# =======================
# CLI and Testing
# =======================

if __name__ == "__main__":
    # Check for required API keys based on model provider
    if MODEL_PROVIDER == "gemini" and (not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here"):
        print("❌ GOOGLE_API_KEY required for Gemini")
        exit(1)
    if MODEL_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY required for OpenAI")
        exit(1)
    if MODEL_PROVIDER == "claude" and not ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY required for Claude")
        exit(1)
    
    # Check for chemistry keywords file
    if not os.path.exists("chemistry_keywords.txt"):
        print("⚠️ chemistry_keywords.txt not found. Semantic keyword matching will be limited.")
    
    # CLI interface for processing storyboard files
    if len(sys.argv) < 2:
        print("🤖 LangGraph Multi-Model Validation Agent")
        print("=" * 60)
        print(f"📡 Current provider: {MODEL_PROVIDER} ({MODEL})")
        print("\nUsage:")
        print(f"   python {sys.argv[0]} <input_file.json> [target_seconds_min] [target_seconds_max]")
        print("\nExample:")
        print(f"   python {sys.argv[0]} storyboard_atom.json 30 60")
        print(f"   python {sys.argv[0]} storyboardv2.json")
        
        print("\n💡 To change model provider, edit MODEL_PROVIDER in your .env file:")
        print("   MODEL_PROVIDER=gemini   # Uses Google Gemini")
        print("   MODEL_PROVIDER=claude   # Uses Anthropic Claude")
        print("   MODEL_PROVIDER=openai   # Uses OpenAI GPT")
        
        print("\nAvailable JSON files in current directory:")
        json_files = [f for f in os.listdir('.') if f.endswith('.json') and not f.endswith('_validated.json') and not f.endswith('_report.json')]
        for f in sorted(json_files):
            print(f"   📄 {f}")
        
        # If no arguments, run built-in test
        print(f"\n🧪 Running built-in test with {MODEL_PROVIDER}...")
        test_content = """{
            "prompt": "Explain flexoelectricity in materials science",
            "title": "Flexoelectricity Fundamentals", 
            "scenes": [
                {
                    "narration": "Flexoelectricity is an electromechanical coupling between strain gradients and electric polarization in materials.",
                    "visuals": "Show crystal lattice deformation"
                }
            ]
        }"""
        
        result = run_validation_agent(
            draft_content=test_content,
            original_prompt="Explain flexoelectricity in materials science",
            thread_id="test_session"
        )
        
        if result["success"]:
            print(f"✅ Test successful with {MODEL_PROVIDER}!")
            print(f"   Overall score: {result['validation_results'].get('overall', 0):.2f}")
            print(f"   Evidence sources: {len(result['evidence'])}")
            print(f"   Memory entries: {len(result['agent_memory'].get('validation_history', []))}")
        else:
            print(f"❌ Test failed: {result['error']}")
        
        exit(0)
    
    # Parse command line arguments for file processing
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 4:
        try:
            target_min = int(sys.argv[2])
            target_max = int(sys.argv[3])
            target_seconds = (target_min, target_max)
        except ValueError:
            print("❌ Invalid target seconds. Using default (30, 60)")
            target_seconds = (30, 60)
    else:
        target_seconds = (30, 60)
    
    # Process the JSON storyboard file
    result = process_storyboard_file(input_file, target_seconds)
    
    if result["success"]:
        print(f"\n🎉 SUCCESS! Your JSON storyboard has been validated with {MODEL_PROVIDER}.")
        print(f"✨ Multi-model support: switch providers anytime in your .env file!")
    else:
        print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)