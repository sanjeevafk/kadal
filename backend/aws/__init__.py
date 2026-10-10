"""
AWS Service Integrations for KADAL Marine Intelligence.
Provides seamless Amazon S3 storage and Amazon Bedrock benthic impact assessment.
Includes graceful fallbacks so KADAL operates flawlessly in air-gapped/offline/local mode.
"""

from .s3_service import S3Service
from .bedrock_service import BedrockService

__all__ = ["S3Service", "BedrockService"]
