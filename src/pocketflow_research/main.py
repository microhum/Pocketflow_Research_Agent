# main.py
import logging
import argparse
from pocketflow_research.flow import create_research_agent_flow
from pocketflow_research.models import SharedStore, ChatMessage
from typing import cast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_turn(research_flow, shared_data: SharedStore, current_input: str):
    """
    Runs a single turn of the research agent flow.
    Updates shared_data with the new user input and topic.
    """

    shared_data["chat_history"].append(ChatMessage(role="user", content=current_input))
    shared_data["topic"] = current_input

    logging.info(f"Effective topic for this turn: {shared_data.get('topic')}")
    logging.info("Current Chat History:")
    for i, msg in enumerate(shared_data.get("chat_history", [])):
        logging.info(f"  [{i}] - {msg.role}: {msg.content}")

    shared_data["query_intent"] = None
    shared_data["search_keywords"] = None
    shared_data["fetched_papers"] = []
    shared_data["all_chunks"] = []
    shared_data["chunk_source_map"] = {}
    shared_data["temp_embeddings"] = None
    shared_data["temp_text_map"] = {}
    shared_data["temp_index"] = None
    shared_data["query_embedding"] = None
    shared_data["retrieved_chunks_with_source"] = []
    shared_data["answer_text"] = None
    shared_data["answer_sources"] = []
    shared_data["final_answer"] = None


    try:
        logging.info("Running the Research Agent flow for the current turn...")
        research_flow.run(shared_data)
        logging.info("Research Agent flow finished for the current turn.")
    except Exception as e:
        logging.error(f"An error occurred during flow execution: {e}", exc_info=True)
        shared_data["final_answer"] = f"Error during flow execution: {e}"
        # Assistant still "says" something, even if it's an error message
        shared_data["chat_history"].append(ChatMessage(role="assistant", content=shared_data["final_answer"]))
        return

    final_answer_for_turn = shared_data.get("final_answer", "No answer generated for this turn.")
    
    shared_data["chat_history"].append(ChatMessage(role="assistant", content=final_answer_for_turn))

    print(f"\nAssistant: {final_answer_for_turn}")
    
    answer_sources_for_turn = shared_data.get("answer_sources")
    if answer_sources_for_turn:
        print("\nSources considered for this answer:")
        for i, src in enumerate(answer_sources_for_turn):
            print(f"  {i+1}. {src.title} (Source: {src.source})")
    print("--------------------")


def main():
    """
    Main function to run the Research Agent flow in a CLI loop.
    Parses command-line arguments for initial setup.
    """
    parser = argparse.ArgumentParser(description="Run the Research Agent flow interactively.")
    parser.add_argument(
        '--source',
        type=str,
        choices=['arxiv', 'thaijo'],
        default='thaijo',
        help="Specify the paper source: 'arxiv' or 'thaijo' (default: arxiv)"
    )
    parser.add_argument(
        '--initial_topic',
        type=str,
        required=False,
        help="Specify an initial research topic to start the conversation."
    )
    args = parser.parse_args()
    selected_source = args.source
    initial_topic_arg = args.initial_topic

    logging.info(f"Starting Interactive Research Agent with source: {selected_source.upper()}...")

    shared_data: SharedStore = cast(SharedStore, {"chat_history": []})

    # --- Flow Creation (once) ---
    try:
        research_flow = create_research_agent_flow(source=selected_source)
        logging.info(f"Research Agent flow created using {selected_source.upper()} source.")
    except ValueError as ve:
        logging.error(f"Flow creation error: {ve}")
        return
    except Exception as e:
        logging.error(f"Failed to create the flow: {e}", exc_info=True)
        return

    # --- Initial Turn if --initial_topic is provided ---
    if initial_topic_arg:
        print(f"Starting with initial topic: {initial_topic_arg}")
        run_single_turn(research_flow, shared_data, initial_topic_arg)

    # --- CLI Loop ---
    print("\nEnter your research query. Type 'exit' or 'quit' to end.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                logging.info("Exiting interactive session.")
                break
            if not user_input.strip():
                print("Please enter a query.")
                continue
            
            run_single_turn(research_flow, shared_data, user_input)

        except KeyboardInterrupt:
            logging.info("\nExiting interactive session due to KeyboardInterrupt.")
            break
        except EOFError:
            logging.info("\nExiting interactive session due to EOF.")
            break

    logging.info("Interactive session ended.")
    if shared_data.get("chat_history"):
        print("\n--- Full Conversation History ---")
        for msg in shared_data["chat_history"]:
            print(f"  {msg.role.capitalize()}: {msg.content}")
        print("-------------------------------")

if __name__ == "__main__":
    main()
