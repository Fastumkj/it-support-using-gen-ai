import os
import json
from typing import Dict, Any, List
from langchain_core.vectorstores import VectorStore
from ..models.schemas import QueryRequest, Resolution, AgentClassification
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import AzureChatOpenAI

# --- 1. MOCK SERVICE INTEGRATION ---
def mock_jira_api(classification: AgentClassification, user_id: str) -> str:
    """
    MOCK function simulating an API call to ServiceNow/Jira.
    This demonstrates the 'integration' job requirement without needing an actual API key.
    """
    if classification.category in ["Network", "Access"]:
        ticket_id = f"JIRA-{abs(hash(user_id)) % 10000}"
        print(f"MOCK API CALL: Created high-priority ticket {ticket_id} for {user_id} - Category: {classification.category}")
        return ticket_id
    else:
        return "None (Self-service resolution suggested)"

# --- 2. THE MAIN AGENT WORKFLOW ORCHESTRATOR ---

def run_support_workflow(
    user_query: str,
    user_id: str,
    vector_db: VectorStore,
    asset_data: List[Dict[str, Any]]
) -> Resolution:
    """
    The Orchestrator function that runs the multi-agent system (Classifier -> Retriever -> Resolver).
    """
    print(f"\n--- Starting Workflow for User {user_id} ---")
    
    # ----------------------------------------------------
    # STEP 1: Initialization (LLM Setup)
    # ----------------------------------------------------

    HF_API_KEY = os.getenv("HF_API_KEY")
    print(f"Using HF API Key: ", HF_API_KEY)

    #using free model (but will take longer time) 
    try:
        # Use a model that is good at instruction following, like a fine-tuned Mistral
        llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=512 # Use max_tokens instead of max_new_tokens for OpenAI
        )

    except Exception as e:
        # ⭐ FIX 1: Print the detailed error message for diagnosis
        print(f"Warning: HuggingFace LLM failed to initialize. Using a simple mock LLM. Details: {e}") 

        class MockLLM:
            def __init__(self):
                pass
            
            def invoke(self, prompt: str):
                if "CLASSIFY" in prompt and "Structured Asset Data:" in prompt:
                    # ⭐ FIX 2: Ensure the Mock LLM returns the necessary JSON for the Classifier Agent
                    return '{"category": "Network", "asset_id": "LAP-12345"}' 
                
                # The Resolver Agent mock output should also be a non-JSON string
                return "Mock resolution: Please try restarting your device. If the issue persists, a high-priority ticket has been automatically created. (This is a mock output.)"
            
        mock_instance = MockLLM()
        llm = RunnableLambda(mock_instance.invoke)

    # ----------------------------------------------------
    # STEP 2: Classifier Agent (Structured Output for Optimization)
    # ----------------------------------------------------
    print("Agent 1: Running Classification...")

    # We tell the LLM exactly what format to return using Pydantic (AgentClassification)
    parser = JsonOutputParser(pydantic_object=AgentClassification)
    
    # This prompt forces the LLM to categorize the issue and find the asset ID.
    classification_prompt = PromptTemplate(
        template="""CLASSIFY the user's issue and IDENTIFY the asset_id. 
        Focus on categories: Hardware, Network, Access, Software, General.
        User Query: "{user_query}"
        User ID: "{user_id}"
        Structured Asset Data: {asset_data}
        \n{format_instructions}\n
        """,
        input_variables=["user_query", "user_id", "asset_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    #create the chain: Prompt -> LLM -> Parser (forces structured output)
    classification_chain = classification_prompt | llm | parser
    
    #invoke the chain
    raw_classification_data = classification_chain.invoke({
        "user_query": user_query,
        "user_id": user_id,
        "asset_data": asset_data 
    })
    
    #validate the raw output against the Pydantic schema
    classification = AgentClassification(**raw_classification_data)
    print(f"   -> Classified as: {classification.category}, Asset: {classification.asset_id}")

    
    # ----------------------------------------------------
    # STEP 3: Retriever Agent (Integrating Structured + Unstructured Data)
    # ----------------------------------------------------
    print("Agent 2: Retrieving Context (Structured & Unstructured)...")
    
    #a. Get Structured Context (from the mock asset data list, using the classified ID)
    asset_context = next(
        (asset for asset in asset_data if asset.get("Asset_ID") == classification.asset_id),
        {"details": "Asset not found in inventory."}
    )
    
    #b. Get Unstructured Context (from RAG/ChromaDB)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    kb_documents = retriever.invoke(user_query)
    
    #format the documents into a simple string for the next agent
    kb_context = "\n---\n".join([doc.page_content for doc in kb_documents])
    print(f"   -> Retrieved {len(kb_documents)} KB documents.")


    # ----------------------------------------------------
    # STEP 4: Resolver Agent (Decision Making & Automation)
    # ----------------------------------------------------
    print("Agent 3: Generating Resolution and Automation Command...")
    
    #this prompt instructs the LLM to act as the final resolver, using all collected data.
    resolver_prompt = PromptTemplate(
        template="""You are the FINAL IT RESOLVER. Your goal is to provide a precise, action-oriented solution 
        based on ALL context provided.

        1. If a clear solution exists in the KB, provide it directly.
        2. If the issue is critical (Network/Access) or the KB is vague, suggest escalation and generate a mock JIRA ticket.
        3. Keep the response professional and concise.

        ---CONTEXT---
        User Query: {user_query}
        Category: {category}
        Asset Details: {asset_details}
        Knowledge Base Articles: {kb_context}
        ---RESOLUTION---
        """,
        input_variables=["user_query", "category", "asset_details", "kb_context"],
    )

    #simple chain for the final output
    resolution_chain = resolver_prompt | llm | StrOutputParser()
    
    final_response_text = resolution_chain.invoke({
        "user_query": user_query,
        "category": classification.category,
        "asset_details": json.dumps(asset_context), 
        "kb_context": kb_context
    })

    # ----------------------------------------------------
    # STEP 5: Automation Mock & Final Assembly
    # ----------------------------------------------------
    
    #trigger the mock JIRA API call based on the classified category
    jira_id = mock_jira_api(classification, user_id)
    
    #assemble the final Pydantic model for the API response
    final_resolution = Resolution(
        final_response=final_response_text.strip(),
        agent_status=f"Self-service resolution provided. Automation: {jira_id}",
        suggested_jira_ticket=jira_id,
        raw_classification=classification
    )
    
    print("Workflow Complete. Final Status:", final_resolution.agent_status)
    return final_resolution