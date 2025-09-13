#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coding Agent
Generates Manim Community Python code from storyboard JSON using model-agnostic LLMs
and optionally renders a video. This module is UI-agnostic and can be used from
CLI, other scripts, or a Gradio app.
"""

import os
import re
import json
import threading
import subprocess
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re as _re

# Optional TTS for narration merge
try:
    from gtts import gTTS  # type: ignore
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

load_dotenv(override=True)

# Model config (separate namespace with fallbacks)
CODING_MODEL = os.getenv("CODING_MODEL", os.getenv("MODEL", "gemini-2.5-flash"))
CODING_MODEL_PROVIDER = os.getenv("CODING_MODEL_PROVIDER", os.getenv("MODEL_PROVIDER", "gemini")).lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_llm(provider: str, model_name: str):
    """Return a chat LLM based on provider selection (gemini|openai|claude)."""
    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini (CODING_MODEL_PROVIDER=gemini)")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0, max_tokens=8000)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI (CODING_MODEL_PROVIDER=openai)")
        return ChatOpenAI(model=model_name, temperature=0.0)
    elif provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude (CODING_MODEL_PROVIDER=claude)")
        return ChatAnthropic(model_name=model_name, temperature=0.0, timeout=None, stop=None)
    else:
        if not GOOGLE_API_KEY:
            raise ValueError("Unsupported provider and GOOGLE_API_KEY missing for default Gemini")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_tokens=8000)


class Prompts:
    SYSTEM_PROMPT = """
You convert structured storyboard JSON into a single Manim Community v0.19.x Python file.
You are a strict Manim developer for physics, chemistry, and material science visualizations.
Your goal is to use description of each scene in the storyboard and generate relevant visualizations for them. It should be of highest quality!
Each scene will be accompanied eventually by the narration as an audio, so worth paying attention to it.
The total length of the video will be 35 or so seconds, so divide it accordingly. Use smaller fonts for clarity.

Performance-first Rules
- Always generate code that renders quickly (optimize for runtime).
- Prefer 2D scenes only. Use class StoryboardScene(MovingCameraScene):
- Avoid 3D objects and heavy effects; prefer simple fades/transforms; group with VGroup.
- Keep equations and molecules minimal; show the concept.

Generic Requirements
- Exactly one Scene class: StoryboardScene.
- Imports: from manim import * ; import numpy as np
- Allowed primitives: VGroup, Dot, Line, Arrow, Polygon, Circle, Square, Rectangle, Arc, MathTex, Text.
- No external assets or web requests. No invented objects.
- Use Text/MathTex for labels; animate submobjects properly.

Style & Output
- Palette: BLUE_E, TEAL_C, YELLOW_B, RED_B, GREY_B.
- Use MovingCameraScene only if needed.
- Mark sections clearly with comments (e.g., # Scene 1: Intro).
- Code must be runnable with: manim -ql manim.py StoryboardScene -o video.mp4

Key Notes
- The storyboard is ground truth. Approximate complex visuals with simple animations.
- Prioritize speed + clarity > detail.
- Output only the Python code block, nothing else.
"""

    FIXER_PROMPT = """
You are a Manim Community v0.19+ code fixer. Take the input Python code and apply all necessary fixes to ensure it runs without errors.

Rules:
- Output only a Python code block, nothing else.
- If self.play is given an empty list, guard it with a condition.
- Replace deprecated animations (ShowCreation→Create, ShowPassingFlash→Indicate, ShowCreationThenFadeOut→CreateThenFadeOut).
- Use Write only with Text/MathTex; use Create for shapes.
- Ensure groups are non-empty before animating; fix invalid attributes.
- Ensure MathTex/Tex strings are valid.
- Must remain self-contained and runnable with: manim -ql manim.py StoryboardScene -o video.mp4
"""


def _extract_code_block(raw: str) -> str:
    # Prefer the first fenced code block regardless of language label
    m = _re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```", raw, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If there are stray backticks, remove them
    if '```' in raw:
        return raw.replace('```python', '').replace('```Python', '').replace('```py', '').replace('```', '').strip()
    return raw.strip()


def _sanitize_code(text: str) -> str:
    """Remove any markdown fences and return plain Python code."""
    code = _extract_code_block(text)
    # If still any backticks remain, strip them
    code = code.replace('```python', '').replace('```Python', '').replace('```py', '').replace('```', '')
    return code.strip()


def generate_manim_code(storyboard_json: Dict[str, Any], provider: Optional[str] = None, model: Optional[str] = None) -> str:
    provider = (provider or CODING_MODEL_PROVIDER).lower()
    model = model or CODING_MODEL
    llm = get_llm(provider, model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", Prompts.SYSTEM_PROMPT),
        ("user", "STORYBOARD_JSON:\n{storyboard}")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"storyboard": json.dumps(storyboard_json, indent=2)})
    return _sanitize_code(raw)


def fix_manim_code(code: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    provider = (provider or CODING_MODEL_PROVIDER).lower()
    model = model or CODING_MODEL
    llm = get_llm(provider, model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", Prompts.FIXER_PROMPT),
        ("user", "Here is the code to fix:\n```python\n{code}\n```")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"code": code})
    return _sanitize_code(raw)


def render_manim(code: str, scene_class: str = "StoryboardScene", quality: str = "-ql", output_video: str = "video.mp4") -> Tuple[Optional[str], str, str]:
    """Render Manim code to a video. Returns (video_path or None, status_text, manim_path)."""
    manim_path = "manim.py"
    with open(manim_path, "w") as f:
        f.write(_sanitize_code(code))

    try:
        proc = subprocess.run(
            ["manim", quality, manim_path, scene_class, "-o", output_video],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        return None, f"❌ Manim failed:\n{e.stderr or e.stdout}", manim_path

    return output_video, "✅ Rendered (video.mp4)", manim_path


def maybe_merge_narration_async(storyboard_json: Dict[str, Any], video_path: str, output_path: str = "final_output.mp4") -> None:
    if not HAS_GTTS:
        return
    narration = " ".join(scene.get("narration", "") for scene in storyboard_json.get("scenes", []))
    if not narration.strip():
        return

    def _merge():
        try:
            audio_file = "narration.mp3"
            gTTS(narration).save(audio_file)
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", output_path],
                check=True
            )
        except Exception:
            pass

    threading.Thread(target=_merge, daemon=True).start()


def run_coding_pipeline(storyboard_json: Dict[str, Any], provider: Optional[str] = None, model: Optional[str] = None, render: bool = True, merge_narration: bool = True) -> Dict[str, Any]:
    code = generate_manim_code(storyboard_json, provider, model)
    fixed = fix_manim_code(code, provider, model)
    result: Dict[str, Any] = {"code": fixed}

    if render:
        video_path, status, manim_path = render_manim(fixed)
        result.update({"video_path": video_path, "status": status, "manim_path": manim_path})
        if video_path and merge_narration:
            maybe_merge_narration_async(storyboard_json, video_path)
    return result
