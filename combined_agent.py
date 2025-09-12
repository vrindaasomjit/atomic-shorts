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

# Load environment variables from .env file
load_dotenv()

def run_combined_workflow(prompt: str, file_paths: Optional[list] = None, 
                          target_seconds: Tuple[int, int] = (30, 60),
                          thread_id: str = "combined_run") -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    
    final_decision = validation_result.get("final_decision", "N/A")
    print(f"   - Final Decision: {final_decision}")
    if final_decision == "Approved":
        print("   - 🎉 Storyboard is approved and ready!")
    else:
        print("   - 📝 Storyboard needs revisions based on feedback.")
        
    return validation_result, storyboard_json

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="Run the combined storyboard generation and validation workflow.")
    parser.add_argument("prompt", type=str, help="The main prompt for the storyboard.")
    parser.add_argument("-f", "--files", nargs="*", help="Optional paths to PDF or image files for context.")
    parser.add_argument("-o", "--output", type=str, help="Optional output file name for the validation report (without extension).")
    
    args = parser.parse_args()
    
    # Use a default output file name if not provided
    output_base = args.output or "storyboard_output"
    
    # Generate a unique thread_id for this run
    from datetime import datetime
    thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run the main workflow
    result, original_storyboard = run_combined_workflow(args.prompt, args.files, thread_id=thread_id)

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

if __name__ == "__main__":
    main()
