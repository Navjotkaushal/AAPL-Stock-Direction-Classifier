from fastapi import APIRouter, HTTPException 
from api.schemas.models import IngestRequest, IngestResponse 

router = APIRouter()

@router.post(
    "/run",
    response_model=IngestResponse,
    summary = "Trigger daily data ingestion from Yahoo Finance - MySQL",
)

def run_ingestion(body: IngestRequest):
    try:
        from data.loader import get_connection, fetch_from_yfinance 
        conn = get_connection()
        