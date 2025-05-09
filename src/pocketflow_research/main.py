# main.py
import logging
import argparse
from pocketflow_research.flow import create_research_agent_flow
from src.pocketflow_research.models import SharedStore
from typing import cast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the Research Agent flow.
    Parses command-line arguments to select the paper source.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Research Agent flow.")
    parser.add_argument(
        '--source',
        type=str,
        choices=['arxiv', 'thaijo'],
        default='arxiv',
        help="Specify the paper source: 'arxiv' or 'thaijo' (default: arxiv)"
    )
    parser.add_argument(
        '--topic',
        type=str,
        required=False, # Make topic optional for now, can be prompted if None
        help="Specify the research topic directly."
    )
    args = parser.parse_args()
    selected_source = args.source
    initial_topic = args.topic

    logging.info(f"Starting Research Agent with source: {selected_source.upper()}...")

    # --- Shared Data Initialization ---
    # Initialize with only the 'topic' if provided, or None.
    # Other keys will be populated by the flow.
    # Using cast to satisfy type checker, though at runtime it's just a dict.
    if initial_topic:
        shared_data: SharedStore = cast(SharedStore, {"topic": initial_topic})
    else:
        # If no topic from args, prompt the user
        try:
            user_topic = input("Please enter the research topic: ")
            if not user_topic.strip():
                logging.error("No topic entered. Exiting.")
                return
            shared_data = cast(SharedStore, {"topic": user_topic})
        except EOFError: # Handle non-interactive environments
            logging.error("No topic provided and cannot prompt in non-interactive mode. Exiting.")
            return


    # --- Flow Creation ---
    try:
        # Pass the selected source to the flow
        research_flow = create_research_agent_flow(source=selected_source)
        logging.info(f"Research Agent flow created using {selected_source.upper()} source.")
    except ValueError as ve:
        logging.error(f"Flow creation error: {ve}")
        return
    except Exception as e:
        logging.error(f"Failed to create the flow: {e}", exc_info=True)
        return

    # Run the flow
    try:
        logging.info("Running the Research Agent flow...")
        research_flow.run(shared_data)
        logging.info("Research Agent flow finished.")
    except Exception as e:
        logging.error(f"An error occurred during flow execution: {e}", exc_info=True)
        pass

    # Print the final result
    final_answer = shared_data.get("final_answer", "No answer generated.")
    logging.info("--- Final Result ---")
    print(f"\nResearch Topic: {shared_data.get('topic', 'N/A')}") # Topic should always be there
    print(f"Final Answer/Summary:\n{final_answer}")
    
    # Optionally, print retrieved sources if available
    answer_sources = shared_data.get("answer_sources")
    if answer_sources:
        print("\nSources considered for the answer:")
        for i, src in enumerate(answer_sources):
            print(f"  {i+1}. {src.title} (Source: {src.source})")

    logging.info("--------------------")

if __name__ == "__main__":
    main()
