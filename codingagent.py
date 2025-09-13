# -*- coding: utf-8 -*-
"""app.py - Storyboard → (Pseudo Validation) → Coding → Manim.py + Video"""

import gradio as gr
import subprocess, json, re, os, random, time, threading, traceback
from gtts import gTTS
from openai import OpenAI

# LangGraph / LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ===========================
# Facts Loader
# ===========================
FACTS_FILE = "facts.txt"

class FactsProvider:
    def __init__(self, filepath=FACTS_FILE):
        self.filepath = filepath
        self.facts = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.facts = [line.strip() for line in f if line.strip()]
        if not self.facts:
            self.facts = [
                "Honey never spoils.",
                "Bananas are technically berries, but strawberries are not.",
                "Sharks existed before trees!",
                "Octopuses have three hearts.",
                "Water can boil and freeze at the same time (triple point).",
            ]

    def random_fact(self):
        return random.choice(self.facts)

facts_provider = FactsProvider()

# ===========================
# Storyboard Agent (LangGraph)
# ===========================
storyboard_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a storyboard generator specialized in **materials science and chemistry education**.
Always produce TWO sections:
1. Human-readable storyboard (Markdown).
2. Machine-readable storyboard (JSON), schema:
{
  "prompt": "string",
  "title": "string",
  "age_level": "string",
  "video_length": "string",
  "scenes": [
    {
      "scene_id": number,
      "title": "string",
      "narration": "string",
      "visuals": [{"description": "string"}]
    }
  ]
}
Enclose JSON between:
===JSON START===
... JSON here ...
===JSON END===
"""),
    ("user", "{prompt}")
])

storyboard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
storyboard_chain = storyboard_prompt | storyboard_llm | StrOutputParser()

def split_reply(reply: str):
    m = re.search(r'===JSON START===\s*(\{.*\})\s*===JSON END===', reply, re.S)
    json_block = None
    if m:
        try:
            json_block = json.loads(m.group(1))
        except Exception:
            json_block = None
    md_block = reply.split("===JSON START===")[0].strip()
    return md_block, json_block

def storyboard_agent(state: dict):
    user_prompt = state["prompt"]
    reply = storyboard_chain.invoke({"prompt": user_prompt})
    md_block, storyboard_json = split_reply(reply)
    state["reply"] = reply
    state["md_block"] = md_block
    state["storyboard"] = storyboard_json
    return state

workflow = StateGraph(dict)
workflow.add_node("Storyboard", storyboard_agent)
workflow.set_entry_point("Storyboard")
workflow.add_edge("Storyboard", END)
app = workflow.compile()

def generate_storyboard(user_prompt: str):
    state = {"prompt": user_prompt}
    result = app.invoke(state)
    return result.get("md_block"), result.get("storyboard")

# ===========================
# Coding Agent Prompts
# ===========================
class Prompts:
    SYSTEM_PROMPT = """
    You convert structured storyboard JSON into a single **Manim Community v0.19.x** Python file.
    You are a strict Manim developer for physics, chemistry, and material science visualizations.
    Your goal is to use description of each scene in the storyboard and generate relevant visualizations for them. It should be of highest quality!
    Each scene will be accompanied eventually by the narration as an audio, so worth paying attention to it.
    The total length of the video will be 35 or so seconds, so divide it accordingly. Make it very clear, notice that your work will enable learning for millions of people across age groups. The font should be small so things are clearer.

    ====================
    Performance-first Rules
    ====================
    - Always generate code that renders **quickly** (optimize for runtime).
    - The video should be legible. Minimum text, avoid using things that do not render well. Use smaller fonts. Remove old items before layering new ones.
    - Prefer **2D scenes only**. Use `class StoryboardScene(MovingCameraScene):`.
    - Do NOT use `ThreeDScene`, `ThreeDCamera`, `Sphere`, or other heavy 3D objects.
    - Avoid ambient camera rotation or long lagged animations.
    -   Prefer simple fades and transforms for clarity.
    -   Bundle with VGroup for speed.
    - Target total render time < 1 minute.
    - Use fades, transforms, and light camera moves. Do not crowd the scene.
    - Keep equations and molecules minimal — show the concept.

    ====================
    Generic Requirements
    ====================
    - Exactly one Scene class: StoryboardScene.
    - Required imports:
      from manim import *
      import numpy as np
    - Allowed primitives:
      VGroup, Dot, Line, Arrow, Polygon, Circle, Square, Rectangle, Arc, MathTex, Text.
    - DO NOT invent objects (Sparkle, Glow, HandIcon, etc.).
    - DO NOT use external assets (SVG, PNG, file I/O, web requests).
    - Use Text or MathTex for labels/equations.
    - For equations, always animate submobjects (eq[0], eq[2]), not strings.
    - Ensure all .get_center(), .shift(), .rotate() calls are only on valid Manim objects.

    ====================
    Chemistry & Physics Plugins
    ====================
    - Allowed if storyboard explicitly requires:
      - manim-Chemistry → from manim_chemistry import *. Do not use Chanim though.
    - Default to simple Circles/Lines approximations if plugins may be missing.
    - Keep molecules small (few atoms/bonds).

    ====================
    Style & Output Rules
    ====================
    - Color palette: BLUE_E, TEAL_C, YELLOW_B, RED_B, GREY_B.
    - Use MovingCameraScene if camera motion is needed.
    - Mark sections clearly with comments:
      # Scene 1: Intro
      # Scene 2: Ice Structure
    - Code must be self-contained and runnable with:
      manim -ql storyboard.py StoryboardScene -o video.mp4
      (use -ql for faster render).

    ====================
    Key Notes
    ====================
    - The storyboard is ground truth.
    - Approximate complex visuals with simple animations if needed.
    - Prioritize speed + clarity > detail/accuracy.
    - Output only the Python code block, nothing else.
    """

    FIXER_PROMPT = """
    You are a Manim Community v0.19+ code fixer.
    Take the input Python code and apply **all necessary fixes** to ensure it runs without errors or deprecation warnings.

    Rules:
    - Always output **only a Python code block**, nothing else.
    - If `self.play` is called with an empty list (e.g. `self.play(*[])`), wrap it with a condition: only call if the list is non-empty.
    - Replace deprecated animations:
      - `ShowCreation` → `Create`
      - `ShowPassingFlash` → `Indicate`
      - `ShowCreationThenFadeOut` → `CreateThenFadeOut`
    - Ensure `Write` is only used with `Text` or `MathTex`; use `Create` for shapes/lines.
    - Replace `set_stroke(color=..., width=...)` with `set_color(...).set_stroke(width=...)` if color is the intent.
    - Remove or replace any invalid attributes (e.g. `self.camera.frame` in `ThreeDScene` → use `self.camera` or `self.camera.frame` in `MovingCameraScene`).
    - Always ensure groups (VGroup) are non-empty before animating or transforming them.
    - Ensure MathTex/Tex are used correctly (wrap strings properly, no raw LaTeX errors).
    - Output must remain self-contained, runnable with:
      `manim -ql storyboard.py StoryboardScene -o video.mp4`

    Check carefully: There is always at least one bad reference. Fix them all.
    """

# ===========================
# LLM Client
# ===========================
class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _extract_code_block(self, raw: str) -> str:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw)
        return (m.group(1) if m else raw).strip()

    def generate_manim(self, storyboard, model="gpt-5") -> str:
        user_payload = f"STORYBOARD_JSON:\n{json.dumps(storyboard, indent=2)}"
        resp = self.client.responses.create(
            model=model,
            input=[{"role": "system", "content": Prompts.SYSTEM_PROMPT},
                   {"role": "user", "content": user_payload}],
        )
        raw = resp.output_text or str(resp)
        return self._extract_code_block(raw)

    def fix_manim(self, code: str, model="gpt-5") -> str:
        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": Prompts.FIXER_PROMPT},
                {"role": "user", "content": f"Here is the code:\n```python\n{code}\n```"},
            ],
        )
        raw = resp.output_text or str(resp)
        m = re.search(r"```python\s*([\s\S]*?)```", raw)
        return m.group(1).strip() if m else raw.strip()

# ===========================
# Pipeline
# ===========================
class Pipeline:
    def __init__(self):
        self.llm = LLMClient()

    def run(self, storyboard_json):
        try:
            code = self.llm.generate_manim(storyboard_json)
            code = self.llm.fix_manim(code)
            manim_path = "manim.py"
            with open(manim_path, "w") as f:
                f.write(code)

            video_file = "video.mp4"
            try:
                subprocess.run(
                    ["manim", "-ql", manim_path, "StoryboardScene", "-o", video_file],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                return None, f"❌ Manim failed:\n{e.stderr or e.stdout}", code, manim_path

            narration = " ".join(scene["narration"] for scene in storyboard_json.get("scenes", []))
            if narration.strip():
                def _merge():
                    audio_file = "narration.mp3"
                    gTTS(narration).save(audio_file)
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", video_file, "-i", audio_file,
                         "-c:v", "copy", "-c:a", "aac", "final_output.mp4"],
                        check=True
                    )
                threading.Thread(target=_merge).start()

            return video_file, "✅ Rendered (merging narration in background…)", code, manim_path

        except Exception as e:
            tb = traceback.format_exc()
            return None, f"❌ Error: {e}\n\nTraceback:\n{tb}", "", ""

pipeline_runner = Pipeline()

# ===========================
# Gradio UI
# ===========================
with gr.Blocks(title="Storyboard → Validation → Coding → Video") as demo:
    gr.Markdown("## 🎬 Atomic Shorts: Storyboard → Validation → Coding → Narrated Video")

    with gr.Row():
        prompt_in = gr.Textbox(label="Enter your prompt", placeholder="e.g. Explain NV- center in diamond...")
        go_btn = gr.Button("Generate Storyboard")

    storyboard_box = gr.Textbox(label="📖 Storyboard Agent Output", interactive=False, lines=12)
    storyboard_json_box = gr.Textbox(visible=False)  # hidden JSON passthrough
    to_validation_btn = gr.Button("➡️ Send to Validation Agent", visible=False)

    validation_box = gr.Textbox(label="✅ Validation Output", interactive=False, lines=12)
    back_to_storyboard_btn = gr.Button("🔄 Back to Storyboard Agent", visible=False)
    to_coding_btn = gr.Button("➡️ Send to Coding Agent", visible=False)

    video_out = gr.Video(label="🎥 Final Video", visible=False)
    download_btn = gr.File(label="⬇️ Download manim.py", visible=False)
    logs = gr.Textbox(label="Logs / Errors")

    # Step 1: Storyboard
    def run_storyboard(prompt):
        md, json_block = generate_storyboard(prompt)
        return md, json.dumps(json_block, indent=2) if json_block else "{}"

    go_btn.click(run_storyboard, inputs=prompt_in, outputs=[storyboard_box, storyboard_json_box])
    go_btn.click(lambda _: gr.update(visible=True), inputs=storyboard_box, outputs=to_validation_btn)

    # Step 2: Validation (just echo storyboard since no agent exists)
    def run_validation(storyboard_json_str):
        try:
            storyboard_json = json.loads(storyboard_json_str)
        except:
            return "❌ Invalid storyboard JSON"
        return "⚠️ No Validation Agent implemented.\n\nPassing storyboard directly:\n\n" + json.dumps(storyboard_json, indent=2)

    to_validation_btn.click(run_validation, inputs=storyboard_json_box, outputs=validation_box)

    def show_validation_buttons(_):
        return gr.update(visible=True), gr.update(visible=True)
    to_validation_btn.click(run_validation, inputs=storyboard_json_box, outputs=validation_box).then(
        show_validation_buttons, inputs=validation_box, outputs=[back_to_storyboard_btn, to_coding_btn]
    )

    # Step 3: Coding
    def run_coding(storyboard_json_str):
        try:
            storyboard_json = json.loads(storyboard_json_str)
        except:
            return None, "❌ Invalid storyboard JSON", None, None
        video, status, code, manim_path = pipeline_runner.run(storyboard_json)
        return video, status, manim_path

    to_coding_btn.click(run_coding, inputs=storyboard_json_box, outputs=[video_out, logs, download_btn])

if __name__ == "__main__":
    demo.launch()
