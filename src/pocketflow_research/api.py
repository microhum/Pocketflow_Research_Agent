import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import textwrap
from typing import List, cast # Added for typing

from pocketflow_research.flow import create_research_agent_flow
from src.pocketflow_research.models import SharedStore, RetrievedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str
    source: str = 'thaijo' # Default to thaijo as per Gradio

class ResearchResponse(BaseModel):
    answer: str
    # Optionally, include retrieved_chunks_with_source if API clients need it
    # retrieved_chunks: List[Dict[str, Any]] = [] 

@app.post("/api/research", response_model=ResearchResponse)
async def run_research_api(request: ResearchRequest):
    """
    API endpoint to run the research agent flow.
    """
    logger.info(f"API request received: topic='{request.topic}', source='{request.source}'")
    try:
        research_flow = create_research_agent_flow(source=request.source)
        # Initialize shared_data with the 'topic' key, conforming to SharedStore
        shared_data_api = cast(SharedStore, {"topic": request.topic})
        
        research_flow.run(shared_data_api) # Run the flow synchronously

        final_answer = shared_data_api.get('final_answer', "No answer generated.")
        logger.info(f"API request processed. Answer length: {len(final_answer)}")
        
        # If you want to return chunks via API:
        # retrieved_chunks_api = shared_data_api.get('retrieved_chunks_with_source', [])
        # # Convert Pydantic models to dicts if necessary for the response
        # chunks_for_response = [chunk.model_dump() for chunk in retrieved_chunks_api]
        # return ResearchResponse(answer=final_answer, retrieved_chunks=chunks_for_response)
        
        return ResearchResponse(answer=final_answer)
    except ValueError as ve:
        logger.error(f"Value error during API request: {ve}")
        return ResearchResponse(answer=f"Error: {str(ve)}") # Consider HTTP 400
    except Exception as e:
        logger.exception(f"Unexpected error during API request for topic '{request.topic}': {e}")
        return ResearchResponse(answer=f"An unexpected error occurred: {str(e)}") # Consider HTTP 500

# --- Gradio Setup ---

def format_chunks_for_display(chunks_with_source: List[RetrievedChunk]) -> str:
    """Formats the retrieved chunks (RetrievedChunk Pydantic models) into a Markdown string."""
    if not chunks_with_source:
        return "No relevant chunks were retrieved."

    formatted_string = "```markdown\n--- Retrieved Chunks ---\n"
    for i, item in enumerate(chunks_with_source):
        # item is now a RetrievedChunk Pydantic model
        text = item.text
        source_info = item.original_source # This is a ChunkSourceInfo Pydantic model
        title = source_info.title
        source_type = source_info.source

        display_text = textwrap.shorten(text.replace('\n', ' '), width=200, placeholder="...")

        formatted_string += f"\n[{i+1}] Source: {title} ({source_type})\n"
        formatted_string += f"    Chunk: {display_text}\n"
    formatted_string += "```"
    return formatted_string

def run_research_chat(message: str, history: list) -> str:
    """
    Function to be called by the Gradio ChatInterface.
    Uses the message as the topic and defaults to 'thaijo' source.
    """
    topic = message
    source = 'thaijo' # Defaulting source for chat interface
    logger.info(f"Gradio Chat request received: topic='{topic}', source='{source}'")
    try:
        research_flow = create_research_agent_flow(source=source)
        # Initialize shared_data with the 'topic' key, conforming to SharedStore
        shared_data_chat: SharedStore = cast(SharedStore, {"topic": topic})
        
        research_flow.run(shared_data_chat)

        final_answer = shared_data_chat.get('final_answer', "Sorry, I couldn't generate an answer.")
        # retrieved_chunks_with_source is List[RetrievedChunk]
        retrieved_chunks: List[RetrievedChunk] = shared_data_chat.get('retrieved_chunks_with_source', [])

        chunks_display = format_chunks_for_display(retrieved_chunks)
        full_response = f"{final_answer}\n\n{chunks_display}"

        logger.info(f"Gradio Chat request processed. Response length: {len(full_response)}")
        return full_response
    except ValueError as ve:
        logger.error(f"Value error during Gradio Chat request: {ve}")
        return f"Error: {str(ve)}"
    except Exception as e:
        logger.exception(f"Unexpected error during Gradio Chat request for topic '{topic}': {e}")
        return f"An unexpected error occurred: {str(e)}"

# Create the Gradio ChatInterface
chat_iface = gr.ChatInterface(
    fn=run_research_chat,
    chatbot=gr.Chatbot(height=600),
    title="ThaiCodex Agent",
    description="Ask a research question. The agent will use ThaiJo papers to answer and show retrieved context.",
    cache_examples=False,
)

# Mount the Gradio Chat app to the FastAPI app
app = gr.mount_gradio_app(app, chat_iface, path="/")

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for FastAPI/Gradio app...")
    # Note: For `uvicorn.run`, the app string should be "api:app" if this file is api.py
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True)
