#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StoryboardAgent - LangGraph
Generates a storyboard JSON and markdown response using LangChain, with multi-model support
(Gemini/OpenAI/Claude) selected via environment variables to match validation agent style.
"""

# You may need to run (once): pip install --upgrade langchain-openai langchain-core langgraph pypdf python-dotenv

import os
import re
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- Configuration & .env loading (multi-model support) ---
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Allow storyboard-specific overrides, fall back to global MODEL/MODEL_PROVIDER
MODEL = os.getenv("STORYBOARD_MODEL", os.getenv("MODEL", "gemini-2.5-flash"))
MODEL_PROVIDER = os.getenv("STORYBOARD_MODEL_PROVIDER", os.getenv("MODEL_PROVIDER", "gemini")).lower()

def get_llm(provider: str, model_name: str):
    """Return a chat LLM based on provider selection.
    Supported providers: gemini, openai, claude
    """
    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini (MODEL_PROVIDER=gemini)")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0, max_tokens=8000)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI (MODEL_PROVIDER=openai)")
        return ChatOpenAI(model=model_name, temperature=0.0)
    elif provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude (MODEL_PROVIDER=claude)")
        # Include optional params to satisfy stricter type checkers
        return ChatAnthropic(model_name=model_name, temperature=0.0, timeout=None, stop=None)
    else:
        # Default to Gemini
        if not GOOGLE_API_KEY:
            raise ValueError("Unsupported MODEL_PROVIDER and GOOGLE_API_KEY missing for default Gemini")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_tokens=8000)

# ======================================================
# Helper: load files (PDFs and images)
# ======================================================

def load_files(file_paths: List[str]):
    """Load files from local paths. Supports PDFs and images.
    Returns (file_contexts, image_refs).
    """
    file_contexts: List[str] = []
    image_refs: List[Dict[str, Any]] = []

    for path in file_paths:
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages[:5]:  # limit to first 5 pages
                text += page.extract_text() or ""
            file_contexts.append(f"From uploaded material {path}:\n{text}")

        elif path.lower().endswith((".png", ".jpg", ".jpeg")):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_refs.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64
                }
            })
        else:
            file_contexts.append(f"{path}: (unsupported format)")

    return file_contexts, image_refs

# ======================================================
# Storyboard prompt template
# ======================================================

storyboard_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a storyboard generator specialized in **materials science and chemistry education**.
Your job is to create clear, accurate, and engaging storyboards for explainer videos.

Guidelines:
- Use correct terminology (e.g., crystal lattice, dielectric constant, polarization, phonons).
- Provide age-appropriate analogies when asked (e.g., explain atoms as Lego blocks for younger audiences).
- Correct misconceptions.
- If user uploads PDFs or images, integrate their content as authoritative references and cite them as “(from uploaded material)”.
- Always structure your reply as a numbered list of **scenes**, each with:
          1. Scene Title and Scene duration
          2. Visual Description (be descriptive- for example, include colors, shapes, flow, and animations to use)
          3. Narration (clear, concise, age-appropriate, scientifically accurate)
        - **Do NOT include people, faces, animals, or any visuals that cannot be rendered in Manim, ASE, RDKit, or simple 2D graphics.**
- The storyboard elements must be ultimately built by valid objects in **Manim Community**, so create the script accordingly.

For every reply, produce TWO sections:

1. Human-readable storyboard (Markdown).
2. Machine-readable storyboard (JSON), in this schema:

{{
  "prompt": "string", <-- include the original user prompt here
  "title": "string",
  "age_level": "string",
  "video_length": "string",
  "scenes": [
    {{
      "scene_id": number,
      "title": "string",
      "narration": "string",
      "visuals": [{{"description": "string"}}]
    }}
  ]
}}

Copy the user's input prompt exactly into the 'prompt' field.
The JSON block must be enclosed between lines:
===JSON START===
... JSON here ...
===JSON END===
Do not add any explanations outside these two sections.
"""),
    ("user", "{prompt}")
])

# Chain + reply splitter
storyboard_llm = get_llm(MODEL_PROVIDER, MODEL)
storyboard_chain = storyboard_prompt | storyboard_llm | StrOutputParser()

def split_reply(reply: str):
    m = re.search(r'===JSON START===\s*(\{.*\})\s*===JSON END===', reply, re.S)
    json_block = None
    if m:
        try:
            json_block = json.loads(m.group(1))
        except Exception as e:
            print("⚠️ JSON parsing error:", e)
    md_block = reply.split("===JSON START===")[0].strip()
    return md_block, json_block

# ======================================================
# LangGraph state and node
# ======================================================

class StoryboardAgentState(TypedDict):
    prompt: str
    files: List[str]
    conversation: List[Dict[str, Any]]
    storyboard: Dict[str, Any]
    md_block: str
    reply: str
    image_refs: List[Dict[str, Any]]

def storyboard_agent(state: StoryboardAgentState) -> StoryboardAgentState:
    conversation = state.get("conversation", [])
    user_prompt = state["prompt"]

    # If files are passed, load and merge them
    file_contexts: List[str] = []
    image_refs: List[Dict[str, Any]] = []
    if state.get("files"):
        file_contexts, image_refs = load_files(state["files"])
        if file_contexts:
            user_prompt += f"""

=== Uploaded Material ===
{chr(10).join(file_contexts)}
=== End Uploaded Material ===
"""

    conversation.append({"role": "user", "content": user_prompt})

    # Run LLM chain
    conv_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation])
    reply = storyboard_chain.invoke({"prompt": conv_text})

    conversation.append({"role": "assistant", "content": reply})
    md_block, storyboard_json = split_reply(reply)

    state["reply"] = reply
    state["md_block"] = md_block
    state["storyboard"] = storyboard_json or {}
    state["conversation"] = conversation
    state["image_refs"] = image_refs  # keep images around if needed
    return state

# ======================================================
# LangGraph workflow
# ======================================================
workflow = StateGraph(StoryboardAgentState)
workflow.add_node("Storyboard", storyboard_agent)
workflow.set_entry_point("Storyboard")
workflow.add_edge("Storyboard", END)
app = workflow.compile()
