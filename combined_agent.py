#!/usr/bin/env python3
"""
Combined Agent: Storyboard Generation and Validation Workflow
"""
import os
import json
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple, cast

# Import agent functions
from storyboardagent import app as storyboard_app, StoryboardAgentState
from validationagent import run_validation_agent, generate_output_filenames, save_validation_outputs
from codingagent import run_coding_pipeline

# Load environment variables from .env file
load_dotenv(override=True)

def run_combined_workflow(prompt: str, file_paths: Optional[list] = None, 
                          target_seconds: Tuple[int, int] = (30, 60),
                          thread_id: str = "combined_run",
                          do_coding: bool = False,
                          coding_provider: Optional[str] = None,
                          coding_model: Optional[str] = None,
                          render_video: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs the full storyboard generation and validation workflow.
    
    1. Generates a storyboard using the storyboard agent.
    2. Validates the generated storyboard using the validation agent.
    """
    print("=======================================")
    print("🚀 Starting Storyboard Generation")
    print("=======================================")
    
    # --- Step 1: Run Storyboard Agent ---
    # Provide all expected keys for StoryboardAgentState to satisfy type checkers
    storyboard_input = cast(StoryboardAgentState, {
        "prompt": prompt,
        "files": file_paths or [],
        "conversation": [],
        "storyboard": {},
        "md_block": "",
        "reply": "",
        "image_refs": [],
    })
    storyboard_result = storyboard_app.invoke(storyboard_input)
    
    storyboard_json = storyboard_result.get("storyboard")
    
    if not storyboard_json:
        print("❌ Error: Storyboard generation failed. No JSON output.")
        return {"error": "Storyboard generation failed."}, {}
        
    print("✅ Storyboard Generated Successfully.")
    print(f"   - Title: {storyboard_json.get('title', 'N/A')}")
    print(f"   - Scenes: {len(storyboard_json.get('scenes', []))}")

    # --- Step 2: Run Validation Agent ---
    print("\n=======================================")
    print("🔬 Starting Content Validation")
    print("=======================================")
    
    draft_content = json.dumps(storyboard_json, indent=2)
    original_prompt = storyboard_json.get("prompt", prompt) # Use prompt from JSON if available
    
    validation_result = run_validation_agent(
        draft_content=draft_content,
        original_prompt=original_prompt,
        pdf_context="",  # PDF context is handled by storyboard agent now
        thread_id=thread_id,
        target_seconds=target_seconds
    )
    
    print("✅ Validation Complete.")

    # Extract criteria statuses and overall score from validation agent
    vr = validation_result.get("validation_results", {}) or {}
    accuracy = vr.get("accuracy", "warn")
    structure = vr.get("structure", "warn")
    length = vr.get("length", "warn")
    consistency = vr.get("consistency", "warn")
    overall = vr.get("overall", None)

    # Determine final decision: Approved iff all criteria passed
    all_pass = all(s == "pass" for s in [accuracy, structure, length, consistency])
    final_decision = "Approved" if all_pass else "Needs revisions"
    validation_result["final_decision"] = final_decision

    print("   - Criteria status:")
    print(f"     • Accuracy:   {accuracy}")
    print(f"     • Structure:  {structure}")
    print(f"     • Length:     {length}")
    print(f"     • Consistency:{' ' if consistency != 'pass' else ''}{consistency}")
    if overall is not None:
        print(f"   - Overall score: {overall:.2f}")
    print(f"   - Final Decision: {final_decision}")
    if final_decision == "Approved":
        print("   - 🎉 Storyboard is approved and ready!")
    else:
        print("   - 📝 Storyboard needs revisions based on feedback.")
        
    # Optional coding phase: generate Manim code and render
    if do_coding:
        print("\n=======================================")
        print("🛠️  Starting Coding (Manim generation)")
        print("=======================================")
        coding_result = run_coding_pipeline(
            storyboard_json,
            provider=coding_provider,
            model=coding_model,
            render=render_video,
            merge_narration=True,
        )
        validation_result["coding"] = {
            "status": coding_result.get("status", ""),
            "video_path": coding_result.get("video_path"),
            "manim_path": coding_result.get("manim_path"),
        }
        # Save code file for convenience
        code = coding_result.get("code")
        if code:
            with open("manim.py", "w") as f:
                f.write(code)
            print("✅ Saved generated Manim code to manim.py")

    return validation_result, storyboard_json

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="Run the combined storyboard → validation → (optional) coding workflow.")
    parser.add_argument("prompt", type=str, help="The main prompt for the storyboard.")
    parser.add_argument("-f", "--files", nargs="*", help="Optional paths to PDF or image files for context.")
    parser.add_argument("-o", "--output", type=str, help="Optional output file name for the validation report (without extension).")
    parser.add_argument("--coding", action="store_true", help="Run coding phase to generate Manim code and render video.")
    parser.add_argument("--coding-provider", type=str, default=os.getenv("CODING_MODEL_PROVIDER", os.getenv("MODEL_PROVIDER", "gemini")), help="Provider for coding agent (gemini|openai|claude)")
    parser.add_argument("--coding-model", type=str, default=os.getenv("CODING_MODEL", os.getenv("MODEL", "gemini-2.5-flash")), help="Model name for coding agent")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering the video; only generate code.")
    
    args = parser.parse_args()
    
    # Use a default output file name if not provided
    output_base = args.output or "storyboard_output"
    
    # Generate a unique thread_id for this run
    from datetime import datetime
    thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run the main workflow
    result, original_storyboard = run_combined_workflow(
        args.prompt,
        args.files,
        thread_id=thread_id,
        do_coding=args.coding,
        coding_provider=args.coding_provider,
        coding_model=args.coding_model,
        render_video=not args.no_render,
    )

    # --- Step 3: Save Outputs ---
    # Save only when validation ran successfully and we have a storyboard
    if result and result.get("success") and original_storyboard:
        print("\n=======================================")
        print(f"💾 Saving outputs to '{output_base}_...'")
        print("=======================================")
        
        # Create a dummy input file name for context
        input_file_for_naming = f"{output_base}.json"
        output_files = generate_output_filenames(input_file_for_naming)
        
        # Wrap the original storyboard in the structure expected by save_validation_outputs
        original_data_wrapped = {
            "format": "direct_json",
            "content": json.dumps(original_storyboard, indent=2),
            "data": original_storyboard,
        }
        
        save_validation_outputs(result, output_files, input_file_for_naming, original_data_wrapped)
        print(f"✅ Report and validated storyboard saved successfully.")
        # If coding ran, report outputs
        coding_meta = result.get("coding") if result else None
        if coding_meta:
            if coding_meta.get("video_path"):
                print(f"🎥 Video: {coding_meta['video_path']}")
            if coding_meta.get("manim_path"):
                print(f"🧾 Manim code: {coding_meta['manim_path']}")

if __name__ == "__main__":
    main()
