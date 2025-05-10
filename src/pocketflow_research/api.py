import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import textwrap
from typing import List, cast, Tuple

from pocketflow_research.flow import create_research_agent_flow
from pocketflow_research.models import (
    SharedStore,
    RetrievedChunk,
    ChatMessage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI() # Define app here so it can be used by gr.mount_gradio_app

class ResearchRequest(BaseModel):
    topic: str
    source: str = "thaijo"

class ResearchResponse(BaseModel):
    answer: str

@app.post("/api/research", response_model=ResearchResponse)
async def run_research_api(request: ResearchRequest):
    logger.info(
        f"API request received: topic='{request.topic}', source='{request.source}'"
    )
    try:
        research_flow = create_research_agent_flow(source=request.source)
        shared_data_api = cast(SharedStore, {"topic": request.topic})
        research_flow.run(shared_data_api)
        final_answer = shared_data_api.get("final_answer", "No answer generated.")
        logger.info(f"API request processed. Answer length: {len(final_answer)}")
        return ResearchResponse(answer=final_answer)
    except ValueError as ve:
        logger.error(f"Value error during API request: {ve}")
        return ResearchResponse(answer=f"Error: {str(ve)}")
    except Exception as e:
        logger.exception(
            f"Unexpected error during API request for topic '{request.topic}': {e}"
        )
        return ResearchResponse(
            answer=f"An unexpected error occurred: {str(e)}"
        )

# --- Gradio Helper Functions ---

def format_chunks_for_display(chunks_with_source: List[RetrievedChunk]) -> str:
    """Formats the retrieved chunks (RetrievedChunk Pydantic models) into a Markdown string."""
    if not chunks_with_source:
        return "No relevant chunks were retrieved for display."

    formatted_string = "--- Retrieved Chunks ---\n" # Removed markdown backticks for gr.Markdown
    for i, item in enumerate(chunks_with_source):
        text = item.text
        source_info = item.original_source
        title = source_info.title
        source_type = source_info.source
        display_text = textwrap.shorten(
            text.replace("\n", " "), width=200, placeholder="..."
        )
        formatted_string += f"\n**[{i + 1}] Source: {title} ({source_type})**\n"
        formatted_string += f"    Chunk: {display_text}\n"
    return formatted_string

# --- Gradio Core Logic ---

initial_prompt = """
You are ThaiCodex, a polite, humble, and highly capable research assistant. Your role is to help find, summarize, and explain academic and technical information in both Thai and English. Always use an appropriate, respectful, and academically appropriate tone. When handling Thai content, preserve its linguistic nuances and formal structure. When responding in English, maintain a clear, formal, and scholarly tone.

When a user asks a question:
- Understand the language (Thai or English) and context of the query.
- Search for the most accurate, reliable, and up-to-date academic, technical, or research information.
- Summarize findings clearly and concisely.
- Use polite phrases and maintain a humble attitude in both languages.

End each response with a polite offer for further assistance.

ตัวอย่างคำลงท้าย:
- ภาษาไทย: "หากต้องการข้อมูลเพิ่มเติม โปรดแจ้งฉันได้เสมอนะคะ"
- English: "If you need any further assistance, please feel free to ask."

Be willing, kind, and precise in both languages.
"""

def run_research_chat_gradio(message: str, history: list, system_instructions: str) -> Tuple[list, str, str, str]:
    """
    Function to be called by the Gradio Blocks interface.
    Returns:
        - Updated history for the chatbot.
        - Formatted retrieved chunks string.
        - Current node stage string.
        - Current chunking info string.
    """
    topic = message
    source = "thaijo"  # This could be made a UI input later
    logger.info(
        f"Gradio Blocks request: topic='{topic}', source='{source}', history_len='{len(history)}', sys_instr_len='{len(system_instructions)}'"
    )

    
    try:
        current_turn_history: List[ChatMessage] = []

        # Reconstruct the chat history from the Gradio input
        if system_instructions.strip():
                    current_turn_history.append(
                        ChatMessage(role="system", content=system_instructions)
                    )

        for user_msg, bot_msg in history:
            if user_msg:
                current_turn_history.append(ChatMessage(role="user", content=user_msg))
            if bot_msg:
                current_turn_history.append(
                    ChatMessage(role="assistant", content=bot_msg)
                )
        if message: # Add current user message to history for this turn's context
            current_turn_history.append(ChatMessage(role="user", content=message))

        research_flow = create_research_agent_flow(source=source)
        shared_data_chat: SharedStore = cast(
            SharedStore,
            {
                # neccessary fields for the flow to run
                "topic": topic,
                "chat_history": current_turn_history,
                "system_instructions": system_instructions if system_instructions.strip() else None,

                # Initialize placeholders for new fields that nodes should populate
                "current_node": None,
                "query_intent": None, 
                "search_keywords": None,
                "fetched_papers": [],
                "retrieved_chunks_with_source": [],
            },
        )

        research_flow.run(shared_data_chat) # Backend nodes need to update the new SharedStore fields

        final_answer = shared_data_chat.get(
            "final_answer", "Sorry, I couldn't generate an answer."
        )
        retrieved_chunks_list: List[RetrievedChunk] = shared_data_chat.get(
            "retrieved_chunks_with_source", []
        )
        retrieved_chunks_display_str = format_chunks_for_display(retrieved_chunks_list)

        # --- New display information (retrieved from SharedStore) ---
        current_node_stage = shared_data_chat.get("current_node_name", "Stage: N/A")
        doc_being_chunked = shared_data_chat.get("current_doc_being_chunked", "")
        generated_chunks_preview: List[str] = shared_data_chat.get("generated_chunks_preview", [])

        chunking_info_display = "--- Chunking Status ---\n"
        if doc_being_chunked:
            chunking_info_display += f"**Document being processed:**\n{textwrap.shorten(doc_being_chunked, width=300, placeholder='...')}\n\n"
        if generated_chunks_preview:
            chunking_info_display += "**Generated Chunks (preview):**\n"
            for i, chunk_text in enumerate(generated_chunks_preview[:3]): # Show first 3
                chunking_info_display += f"- {textwrap.shorten(chunk_text, width=100, placeholder='...')}\n"
            if len(generated_chunks_preview) > 3:
                chunking_info_display += f"... and {len(generated_chunks_preview) - 3} more.\n"
        if not doc_being_chunked and not generated_chunks_preview and current_node_stage != "Initializing...": # Avoid showing "No info" if flow hasn't really started
             if "chunk" in str(current_node_stage).lower() or "embed" in str(current_node_stage).lower() : # Heuristic
                chunking_info_display = "Chunking Status: Processing..."
             else:
                chunking_info_display = "Chunking Status: No active chunking information or not applicable to current stage."

        updated_history = history + [[message, final_answer]]
        return updated_history, retrieved_chunks_display_str, str(current_node_stage), chunking_info_display

    except ValueError as ve:
        logger.error(f"Value error during Gradio Blocks request: {ve}")
        error_msg = f"Error: {str(ve)}"
        return history + [[message, error_msg]], "No chunks retrieved.", "Error", "Error in processing."
    except Exception as e:
        logger.exception(
            f"Unexpected error during Gradio Blocks request for topic '{topic}': {e}"
        )
        error_msg = f"An unexpected error occurred: {str(e)}"
        return history + [[message, error_msg]], "No chunks retrieved.", "Error", "Error in processing."

# --- Gradio UI with Blocks ---
with gr.Blocks(theme=gr.themes.Ocean(primary_hue=gr.themes.colors.blue)) as blocks_iface:
    gr.Markdown("# ThaiCodex Agent with Enhanced Controls")
    gr.Markdown("Interact with the research agent, monitor its progress, and provide system-level instructions.")

    with gr.Row():
        with gr.Column(scale=3): # Main interaction area
            chatbot = gr.Chatbot(label="Conversation", height=550, bubble_full_width=False)
            with gr.Row():
                msg_textbox = gr.Textbox(
                    label="Your Question:",
                    placeholder="Enter your research question here...",
                    lines=2,
                    scale=4,
                    elem_id="user_input_textbox"
                )
                submit_button = gr.Button("Send", variant="primary", scale=1, elem_id="send_button")

            system_instructions_textbox = gr.Textbox(
                label="System Instructions (Optional):",
                placeholder="e.g., 'Answer concisely.' or 'Focus on the impact of the research.'",
                value=initial_prompt,
                lines=3,
                info="These instructions guide the AI's response generation.",
            )

        with gr.Column(scale=2): # Status and context area
            gr.Markdown("### Agent Status & Context")
            with gr.Accordion("Current Stage & Chunking Details", open=True):
                node_stage_display = gr.Markdown(value="Stage: Idle", label="Current Agent Stage")
                chunking_info_display = gr.Markdown(value="Chunking Status: Idle", label="Live Chunking Details")
            with gr.Accordion("Retrieved Context (from knowledge base)", open=True):
                retrieved_chunks_display_area = gr.Markdown(value="No context retrieved yet.", label="Retrieved Chunks")

    # Define interactions
    submit_button.click(
        fn=run_research_chat_gradio,
        inputs=[msg_textbox, chatbot, system_instructions_textbox],
        outputs=[chatbot, retrieved_chunks_display_area, node_stage_display, chunking_info_display],
        api_name="research_chat_submit" # For API access if needed
    )
    msg_textbox.submit(
        fn=run_research_chat_gradio,
        inputs=[msg_textbox, chatbot, system_instructions_textbox],
        outputs=[chatbot, retrieved_chunks_display_area, node_stage_display, chunking_info_display],
        api_name="research_chat_enter" # For API access if needed
    )

# Mount the Gradio Blocks app to the FastAPI app
# The original file had `app = gr.mount_gradio_app(app, chat_iface, path="/")`
# We replace chat_iface with blocks_iface.
app = gr.mount_gradio_app(app, blocks_iface, path="/")

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for FastAPI/Gradio app with Blocks UI...")
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True)
