#@title 🎬 Storyboard Agent (click ▶ to expand code)
##@markdown Enter your **video idea** and other details below, then click the buttons.

# Install libraries
import os

!pip install --upgrade openai pypdf

from openai import OpenAI
from google.colab import files, output
import ipywidgets as widgets
from IPython.display import display, HTML
import io, base64, json
from pypdf import PdfReader

import re, json

# ---- ENSURING CORRECT OUTPUT FORMAT ----
def split_reply(reply: str):
    """
    Split model reply into markdown (for display) and JSON (for saving).
    """
    m = re.search(r'===JSON START===\s*(\{.*\})\s*===JSON END===', reply, re.S)
    json_block = None
    if m:
        try:
            json_block = json.loads(m.group(1))
        except Exception as e:
            print("⚠️ JSON parsing error:", e)
    # Markdown is everything before JSON START
    md_block = reply.split("===JSON START===")[0].strip()
    return md_block, json_block

# ---- SETUP ----
# Your API key
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

conversation_history = [
    {"role": "system", "content": """
        You are a storyboard generator specialized in **materials science and chemistry education**.
        Your job is to create clear, accurate, and engaging storyboards for explainer videos.

        Domain rules:
        - Use **materials science and chemistry concepts** accurately.
        - Highlight the role of atomic structure, bonds, energy, and real-world applications.
        - For younger audiences and the general public, use analogies (e.g., comparing dipoles to tiny magnets) where appropriate.
        - For undergraduate and graduate audiences, display formulas where appropriate. 
        - Use correct scientific terminology (e.g., crystal lattice, dielectric constant, polarization, phonons).
        - Correct common misconceptions (e.g., piezoelectric ≠ ferroelectric).

        General rules:
        - If user uploads PDFs or images, integrate their content as authoritative references and cite them as “(from uploaded material)”.
        - Always structure your reply as a numbered list of **scenes**, each with:
          1. Scene Title and Scene duration
          2. Visual Description 
          3. Narration 
        - The **Video Description** should be extremely detailed and explicit so that another agent can directly generate Manim code without guessing. 
          Follow these rules:
          - Always specify shapes (e.g., circle, square, lattice grid, polygon, arrow, vector).
          - Always specify the positions and sizes of all objects. 
          - Always specify colors. Use hex code or common names for colors.
          - Mention opacity, shading, gradients, or glow effects where important.
          - Distinguish different categories by consistent color (e.g., “atoms are blue, vacancies are red, phonons are wavy yellow lines”).
          - Describe motion/animation (fade in, rotate, oscillate).
          - Specify timing relative to narration (e.g., “fade in as narrator says ‘diamond lattice’”).
          - Include all labels, text, and equations in LaTeX where appropriate. Define their positioning relative to the object on screen.
          - Indicate camera orientation (2D, 3D perspective, zoom).
          - Describe transitions from previous scene. Each scene should describe how it connects to the previous one: “Zoom into one atom from the lattice in Scene 1,” “Crossfade to band diagram in Scene 2.”
          - Maintain visual consistency (same colors/symbols for the same concepts across scenes).
          - If introducing a new element, re-describe it explicitly.
          - No vague language (“something appears,” “stuff moves around”).
          - Always specify what appears, how it appears, what color/shape, and how it moves.
          - **Do NOT include people, faces, animals, or any visuals that cannot be rendered in Manim, ASE, RDKit, or simple 2D graphics.**
          - The storyboard elements must be ultimately built by valid objects in **Manim Community**, so create the Visual Description accordingly.
        - For the 'Narration', be clear, concise, age-appropriate, scientifically accurate.
        

        For every reply, produce TWO sections in this exact order:

        1. Human-readable storyboard (Markdown).
        2. Machine-readable storyboard in JSON ONLY, following this schema:

        {
          "prompt": "string",    <-- include the original user prompt here
          "title": "string",
          "age_level": "string",
          "video_length": "string",
          "scenes": [
            {
              "scene_id": number,
              "title": "string",
              "narration": "string",
              "visuals": [
                {"description": "string"}
              ]
            }
          ]
        }

        Copy the user's input prompt exactly into the 'prompt' field.
        The JSON block must be enclosed between lines:
        ===JSON START===
        ... JSON here ...
        ===JSON END===

        Do not add any explanations outside these two sections.
        """}
]

uploaded_files = {}
file_contexts = []
image_refs = []

# ---- FILE UPLOAD BUTTON ----
upload_button = widgets.FileUpload(
    accept='.pdf,.png,.jpg,.jpeg',
    multiple=True,
    description="⬆️ Upload files",
    style={'button_color': '#0284c7'}  # make it blue
)

def handle_file_upload(change):
    """Triggered when files are uploaded via the blue button."""
    global uploaded_files, file_contexts, image_refs
    uploaded_files = change["new"]
    file_contexts.clear()
    image_refs.clear()

    for i, name in enumerate(uploaded_files.keys()):
        if i >= 5:  # max 5 files
            break
        data = uploaded_files[name]['content']
        if name.lower().endswith(".pdf"):
            # Extract text from PDF
            reader = PdfReader(io.BytesIO(data))
            text = ""
            for page in reader.pages[:5]:  # read first 5 pages for efficiency
                text += page.extract_text() or ""
            file_contexts.append(f"Content from {name}:\n{text}")
        elif name.lower().endswith((".png", ".jpg", ".jpeg")):
            # Encode image for potential use later
            b64 = base64.b64encode(data).decode("utf-8")
            image_refs.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
        else:
            file_contexts.append(f"{name}: (unsupported format)")

    # Push extracted content into conversation as a SYSTEM message
    if file_contexts:
        context_text = "\n\n".join(file_contexts)
        conversation_history.append({
            "role": "system",
            "content": f"Here are reference materials uploaded by the user. Use them as background knowledge when creating the storyboard:\n{context_text}"
        })

    with output_area:
        output_area.clear_output()
        print(f"✅ Uploaded {len(uploaded_files)} file(s): {list(uploaded_files.keys())[:5]}")

upload_button.observe(handle_file_upload, names="value")

# ---- CHAT FUNCTION ----
latest_storyboard = None  # put this at top-level (global)

def chat_with_storyboard_agent(prompt, age_level, video_length):
    """Send user input to ChatGPT and return the assistant’s reply."""
    user_message = f"""
    Create or refine a storyboard with the following inputs:

    - Prompt / Topic: {prompt}
    - Target Audience Age Level: {age_level}
    - Desired Video Length: {video_length}

    """

    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can switch to "gpt-4o"
        messages=conversation_history,
    )

    reply = response.choices[0].message.content
    conversation_history.insert(1, {"role": "assistant", "content": reply})  # put latest reply at top

    md_block, storyboard_json = split_reply(reply)

    # Store JSON for saving later
    global latest_storyboard
    latest_storyboard = storyboard_json

    # ---- Display so newest reply shows first ----
    output_area.clear_output()
    with output_area:
        display(HTML(f"""
        <div style="background-color:#e0f9ff; border-left: 4px solid #0284c7;
                    padding: 10px; margin: 10px 0; border-radius: 12px;
                    font-family: Arial, sans-serif;">
          <b>🎬 Storyboard Agent (latest):</b><br>{md_block.replace("\n", "<br>")}
        </div>
        """))

    return reply

# ---- INPUT WIDGETS ----
prompt_box = widgets.Textarea(
    placeholder='Describe your video idea here...',
    description='Prompt:',
    layout=widgets.Layout(width='100%', height='120px')
)

age_box = widgets.Text(
    placeholder='e.g., High school students, general audience, experts...',
    description='Age Level:',
    layout=widgets.Layout(width='100%')
)

length_box = widgets.Text(
    placeholder='e.g., 20 seconds, 30 seconds...',
    description='Video Length:',
    layout=widgets.Layout(width='100%')
)

send_button = widgets.Button(
    description="Send to Agent",
    style={'button_color': '#0284c7'}
)

output_area = widgets.Output()

def on_send_click(b):
    prompt = prompt_box.value.strip()
    age_level = age_box.value.strip()
    video_length = length_box.value.strip()

    if not prompt:
        with output_area:
            output_area.clear_output()
            print("⚠️ Please enter a prompt.")
        return

    chat_with_storyboard_agent(prompt, age_level, video_length)

send_button.on_click(on_send_click)

# ---- SAVE BUTTON ----
save_button = widgets.Button(
    description="💾 Save Storyboard",
    style={'button_color': '#22c55e'}  # green
)

def on_save_click(b):
    if latest_storyboard is None:
        print("⚠️ No storyboard JSON found yet.")
        return
    filename = "storyboard.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(latest_storyboard, f, ensure_ascii=False, indent=2)
    files.download(filename)

save_button.on_click(on_save_click)

# ---- DISPLAY UI ----
display(upload_button, prompt_box, age_box, length_box, send_button, save_button, output_area)