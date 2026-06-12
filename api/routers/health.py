from fastapi import APIRouter 
from api.schemas.models import HealthResponse 
from api.services.model_store import ModelStore 

router = APIRouter()

@router.get("/health", response_model = HealthResponse, summary = "API health check")
def health():
    """
    Returns API health 
    """
    db_ok = False 
    try:
        from data.loader import get_connection 
        conn = get_connection()
        conn.close()
        db_ok = True 
    except Exception:
        pass 
    
    return HealthResponse(
        status = "ok",
        models_loaded = ModelStore.loaded_keys(),
        db_connnected = db_ok,
    )