from pydantic import BaseModel, Field
from typing import Literal, List

#input payload system received by the user when they submit a query to the IT Support co-pilot
class QueryRequest(BaseModel):
    """Defines the input payload for the IT Support Co-pilot."""
    user_query: str = Field(..., description="The user's original IT support question.")
    user_id: str = Field(..., description="The ID of the user submitting the query (e.g., 'pungkj').")
    
#schema defines structured output from the Classifier Agent
class AgentClassification(BaseModel):
    """Structured output for the Classifier Agent."""
    category: Literal["Hardware", "Network", "Access", "Software", "General"] = Field(
        ..., description="The primary category of the IT issue."
    )
    #agent extracts asset ID or look at user_id.
    asset_id: str = Field(..., description="The ID of the affected IT asset (e.g., LAP-4567, PRT-SG-FL3). Use 'N/A' if unknown.")

#schemas defines structured output from Resolver Agent
class Resolution(BaseModel):
    """Defines the final, structured response from the Resolver Agent."""
    final_response: str = Field(..., description="The generated resolution or next steps for the user.")
    agent_status: str = Field(..., description="Summary of the agent's action (e.g., 'Self-service provided', 'Escalated to Tier 2').")
    suggested_jira_ticket: str = Field(..., description="A mock Jira ticket created (e.g., 'JIRA-1234') or 'None'.")
    raw_classification: AgentClassification = Field(..., description="The raw classification data for transparency.")