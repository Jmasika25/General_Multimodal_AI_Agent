# Load the necessary libraries

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Load API key
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Groq LLM (LLaMA-family model)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.2,
)

# BLIP Captioning Model (for images -> text)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()

def caption_image(image_path: str) -> str:
    """Generate a caption for the input image using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out_ids = blip_model.generate(**inputs, max_new_tokens=64)
    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    return caption

# ---- LangGraph State Definition ----
class AgentState(TypedDict):
    question: str
    image_path: Optional[str]
    text_answer: Optional[str]
    image_answer: Optional[str]

# ---- Workflow ----
workflow = StateGraph(AgentState)

def text_only_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["question"])
    return {"text_answer": response.content}

def image_qna_node(state: AgentState) -> AgentState:
    caption = caption_image(state["image_path"])
    prompt = f"Image description: {caption}\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"image_answer": response.content}

# Add nodes
workflow.add_node("text_only", text_only_node)
workflow.add_node("image_qna", image_qna_node)

# --- Router logic (pure function for edges, not a node) ---
def router(state: AgentState) -> str:
    if state.get("image_path"):
        return "image_qna"
    return "text_only"

# Entry point
workflow.set_entry_point("text_only")

# Use START -> router conditional edges
workflow.add_conditional_edges(
    START,   # from graph entrypoint
    router,  # function returning a string
    {
        "text_only": "text_only",
        "image_qna": "image_qna"
    }
)

# End edges
workflow.add_edge("text_only", END)
workflow.add_edge("image_qna", END)

# Compile
app = workflow.compile()


# ---- Test the usage ----
if __name__ == "__main__":
    # Text question
    result1 = app.invoke({"question": "What is the blockchain behind Ethereum coin?"})
    print("Text Answer:", result1.get("text_answer"))

    # Image-related Q&A
    result2 = app.invoke({
        "question": "Interpret the Image of a graph given",
        "image_path": "bnb.jpg"
    })
    print("Image Answer:", result2.get("image_answer"))


import gradio as gr

# ---- Gradio Interface ----

def gradio_agent(question, image):
    """
    Wrapper for Gradio UI.
    - question: text input from user
    - image: optional uploaded image
    """
    image_path = None
    if image is not None:
        image_path = "temp_uploaded.png"
        image.save(image_path)

    result = app.invoke({
        "question": question,
        "image_path": image_path
    })

    # Prefer image_answer if available, else text_answer
    if result.get("image_answer"):
        return result["image_answer"]
    elif result.get("text_answer"):
        return result["text_answer"]
    else:
        return "Sorry, I could not generate an answer."


# UI components
with gr.Blocks() as demo:
    gr.Markdown("## AI Q&A Agent with Image Understanding")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
            image = gr.Image(type="pil", label="Upload an Image (optional)")
            submit = gr.Button("Ask Agent")
        
        with gr.Column():
            output = gr.Textbox(label="Answer")

    # Connect button to function
    submit.click(fn=gradio_agent, inputs=[question, image], outputs=output)


# Launch Gradio
if __name__ == "__main__":
    demo.launch(share = True)
