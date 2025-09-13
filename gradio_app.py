#!/usr/bin/env python3
"""
Gradio UI matching the original coding agent flow:
- Generate Storyboard
- Send to Validation Agent
- Send to Coding Agent (Manim + Video)

Rewired to use our storyboardagent, validationagent, and codingagent modules.
"""

import os
import json
import re
import subprocess
from typing import cast, Optional

# Prefer Python-based video editing if available
try:
    import moviepy.editor as mpy  # type: ignore
    HAS_MOVIEPY = True
except Exception:
    HAS_MOVIEPY = False

import gradio as gr
from dotenv import load_dotenv

from storyboardagent import app as storyboard_app, StoryboardAgentState
from validationagent import run_validation_agent
from codingagent import run_coding_pipeline

load_dotenv(override=True)


def split_reply_for_md_and_json(reply: str):
    m = re.search(r'===JSON START===\s*(\{.*\})\s*===JSON END===', reply, re.S)
    json_block = None
    if m:
        try:
            json_block = json.loads(m.group(1))
        except Exception:
            json_block = None
    md_block = reply.split("===JSON START===")[0].strip()
    return md_block, json_block


def run_storyboard(user_prompt: str):
    state: StoryboardAgentState = cast(StoryboardAgentState, {
        "prompt": user_prompt,
        "files": [],
        "conversation": [],
        "storyboard": {},
        "md_block": "",
        "reply": "",
        "image_refs": [],
    })
    result = storyboard_app.invoke(state)
    md_block = result.get("md_block", "")
    storyboard_json = result.get("storyboard") or {}
    return md_block, json.dumps(storyboard_json, indent=2) if storyboard_json else "{}"


def run_validation(storyboard_json_str: str):
    try:
        storyboard_json = json.loads(storyboard_json_str)
    except Exception:
        return "❌ Invalid storyboard JSON"
    draft = json.dumps(storyboard_json, indent=2)
    original_prompt = storyboard_json.get("prompt", "")
    res = run_validation_agent(draft_content=draft, original_prompt=original_prompt, pdf_context="", thread_id="ui")
    vr = res.get("validation_results", {}) or {}
    overall = vr.get("overall")
    lines = ["✅ Validation complete:"]
    for k in ["accuracy", "structure", "length", "consistency"]:
        lines.append(f"- {k}: {vr.get(k, 'warn')}")
    lines.append(f"- overall: {overall if overall is not None else 'N/A'}")
    decision = "Approved" if all(vr.get(k) == "pass" for k in ["accuracy", "structure", "length", "consistency"]) else "Needs revisions"
    lines.append(f"- decision: {decision}")
    return "\n".join(lines)


def _clip_video(input_path: str, start: Optional[float], end: Optional[float]) -> Optional[str]:
    try:
        if start is None or start < 0:
            start = 0.0
        # If no valid end or end <= start, return original
        if end is None or end <= 0 or end <= start:
            return input_path

        out_path = f"video_clip_{int(start)}_{int(end)}.mp4"

        if HAS_MOVIEPY:
            # Use MoviePy (which bundles imageio-ffmpeg) to avoid system ffmpeg dependency
            clip = mpy.VideoFileClip(input_path)
            # Guard end to clip duration
            end_eff = min(end, clip.duration or end)
            sub = clip.subclip(start, end_eff)
            # Use source fps if available, else default 24
            fps = getattr(clip, "fps", None) or 24
            sub.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None)
            sub.close()
            clip.close()
            return out_path
        else:
            # Fallback to ffmpeg CLI if MoviePy is unavailable
            duration = max(0.1, end - start)
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
                "-t", str(duration), "-c", "copy", out_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                cmd2 = [
                    "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
                    "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", out_path
                ]
                res2 = subprocess.run(cmd2, capture_output=True, text=True)
                if res2.returncode != 0:
                    return input_path
            return out_path
    except Exception:
        return input_path


def run_coding(storyboard_json_str: str, start_time: Optional[float], end_time: Optional[float]):
    try:
        storyboard_json = json.loads(storyboard_json_str)
    except Exception:
        return None, "❌ Invalid storyboard JSON", gr.update(visible=False)
    out = run_coding_pipeline(storyboard_json)
    video_path = out.get("video_path")
    status = out.get("status", "")
    manim_path = out.get("manim_path")
    if video_path:
        clipped = _clip_video(video_path, start_time, end_time)
        video_path = clipped or video_path
    return video_path, status, gr.update(value=manim_path, visible=True)


with gr.Blocks(title="Storyboard → Validation → Coding → Video") as demo:
    gr.Markdown("## 🎬 Atomic Shorts: Storyboard → Validation → Coding → Narrated Video")

    with gr.Row():
        prompt_in = gr.Textbox(label="Enter your prompt", placeholder="e.g. Explain NV- center in diamond...")
        go_btn = gr.Button("Generate Storyboard")

    storyboard_box = gr.Textbox(label="📖 Storyboard Agent Output", interactive=False, lines=12)
    storyboard_json_box = gr.Textbox(visible=False)
    to_validation_btn = gr.Button("➡️ Send to Validation Agent", visible=False)

    validation_box = gr.Textbox(label="✅ Validation Output", interactive=False, lines=12)
    back_to_storyboard_btn = gr.Button("🔄 Back to Storyboard Agent", visible=False)
    to_coding_btn = gr.Button("➡️ Send to Coding Agent", visible=False)

    video_out = gr.Video(label="🎥 Final Video", visible=True)
    with gr.Row():
        start_time = gr.Number(label="Start time (s)", value=0)
        end_time = gr.Number(label="End time (s)", value=None)
    download_btn = gr.File(label="⬇️ Download manim.py", visible=False)
    logs = gr.Textbox(label="Logs / Errors")

    # Step 1: Storyboard
    go_btn.click(run_storyboard, inputs=prompt_in, outputs=[storyboard_box, storyboard_json_box])
    go_btn.click(lambda _: gr.update(visible=True), inputs=storyboard_box, outputs=to_validation_btn)

    # Step 2: Validation
    to_validation_btn.click(run_validation, inputs=storyboard_json_box, outputs=validation_box)

    def show_next_buttons(_):
        return gr.update(visible=True), gr.update(visible=True)

    to_validation_btn.click(run_validation, inputs=storyboard_json_box, outputs=validation_box).then(
        show_next_buttons, inputs=validation_box, outputs=[back_to_storyboard_btn, to_coding_btn]
    )

    # Step 3: Coding
    to_coding_btn.click(run_coding, inputs=[storyboard_json_box, start_time, end_time], outputs=[video_out, logs, download_btn])


if __name__ == "__main__":
    demo.launch()
