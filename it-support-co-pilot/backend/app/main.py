from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .models.schemas import QueryRequest, Resolution
from .services.data_loader import setup_chroma_db, load_structured_assets
from .services.agent_workflow import run_support_workflow 

# Global cache for heavy resources (RAG Vector Store and Asset Data)
APP_CONTEXT = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and Shutdown event handler.
    Loads the ChromaDB and Asset Data before the application starts serving.
    """
    print("--- FastAPI Startup: Loading AI/Data Assets ---")
    APP_CONTEXT["vector_db"] = setup_chroma_db()
    APP_CONTEXT["asset_data"] = load_structured_assets()
    print("--- Startup Complete ---")
    yield
    APP_CONTEXT.clear()

app = FastAPI(
    title="Intelligent IT Support Co-pilot API",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/resolve", response_model=Resolution)
async def resolve_issue(query_data: QueryRequest):
    """
    API endpoint to submit an IT support query and receive a resolution 
    from the Multi-Agent workflow.
    """
    try:
        #pass the global resources and the user query to the core Agent logic
        resolution_result = run_support_workflow(
            user_query=query_data.user_query,
            user_id=query_data.user_id,
            vector_db=APP_CONTEXT["vector_db"],
            asset_data=APP_CONTEXT["asset_data"]
        )
        return resolution_result
        
    except Exception as e:
        print(f"Error processing query: {e}")
        # A clean error message is professional
        raise HTTPException(status_code=500, detail=f"Internal Agent Error: {e}")

#simple health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Co-pilot API is running"}