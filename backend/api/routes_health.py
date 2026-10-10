from fastapi import APIRouter
from datetime import datetime
import psutil
from backend.config import AWS_REGION, AWS_S3_BUCKET, AWS_BEDROCK_MODEL_ID, ENABLE_AWS_SYNC

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "YOLO-ESI Debris Analysis Engine",
        "version": "1.0.0",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        },
        "aws": {
            "sync_enabled": ENABLE_AWS_SYNC,
            "region": AWS_REGION,
            "s3_bucket": AWS_S3_BUCKET,
            "bedrock_model": AWS_BEDROCK_MODEL_ID
        }
    }

