"""
Amazon S3 Service for KADAL Marine Intelligence.
Manages persistent hydrographic survey storage, evidence bundles, and geospatial reports.
Gracefully degrades to local disk storage if AWS credentials or network connectivity are absent.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from backend.config import AWS_REGION, AWS_S3_BUCKET, ENABLE_AWS_SYNC

logger = logging.getLogger("kadal.aws.s3")

class S3Service:
    def __init__(self, bucket_name: str = AWS_S3_BUCKET, region_name: str = AWS_REGION):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.enabled = ENABLE_AWS_SYNC
        self.client = None
        self._initialized = False

    def _get_client(self):
        if not self.enabled:
            return None
        if self._initialized:
            return self.client
        self._initialized = True
        try:
            import boto3
            self.client = boto3.client("s3", region_name=self.region_name)
            logger.info(f"[AWS S3] Initialized S3 client in region {self.region_name} for bucket '{self.bucket_name}'")
            return self.client
        except Exception as e:
            logger.info(f"[AWS S3] S3 client offline ({e}). Running in local fallback mode.")
            self.client = None
            return None

    def upload_file(self, file_path: str, s3_key: str, content_type: Optional[str] = None) -> Optional[str]:
        """
        Uploads a local file to the configured S3 bucket.
        Returns the s3:// URI or https S3 URL if successful, None otherwise.
        """
        client = self._get_client()
        if client is None:
            return None

        p = Path(file_path)
        if not p.exists():
            return None

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            client.upload_file(
                Filename=str(p),
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args if extra_args else None
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"[AWS S3] Successfully uploaded {p.name} -> {s3_uri}")
            return s3_uri
        except Exception as e:
            err_str = str(e)
            if "expired" in err_str.lower() or "credentials" in err_str.lower() or "nosuchbucket" in err_str.lower():
                logger.info(f"[AWS S3] Cloud upload bypassed for {p.name} (AWS credentials inactive or bucket pending). Retained in local storage.")
            else:
                logger.warning(f"[AWS S3] Failed to upload {file_path} to s3://{self.bucket_name}/{s3_key}: {e}")
            return None


    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Generates a presigned URL to view/download the S3 artifact."""
        client = self._get_client()
        if client is None:
            return None
        try:
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.warning(f"[AWS S3] Presigned URL generation failed for key '{s3_key}': {e}")
            return None

    def sync_analysis_artifacts(self, analysis_id: str, results_dir: Path) -> Dict[str, str]:
        """
        Uploads all generated survey assets for an analysis run to S3.
        Returns a mapping of asset name -> S3 URI or URL.
        """
        uploaded_map = {}
        if not results_dir.exists():
            return uploaded_map

        file_types = {
            "annotated.png": ("image/png", "annotated_image"),
            "original.png": ("image/png", "original_image"),
            "mask.png": ("image/png", "detection_mask"),
            "colormap.png": ("image/png", "colormap_image"),
            "evidence.png": ("image/png", "evidence_image"),
            "report.pdf": ("application/pdf", "pdf_report"),
            "detections.csv": ("text/csv", "csv_report"),
            "results.json": ("application/json", "json_report"),
            "dive_plan.kml": ("application/vnd.google-earth.kml+xml", "kml_export"),
            "waypoints.txt": ("text/plain", "nmea_export"),
            "evidence_bundle.zip": ("application/zip", "evidence_bundle")
        }

        for filename, (mime, key_label) in file_types.items():
            file_path = results_dir / filename
            if file_path.exists():
                s3_key = f"surveys/{analysis_id}/{filename}"
                uri = self.upload_file(str(file_path), s3_key, content_type=mime)
                if uri:
                    uploaded_map[key_label] = uri

        return uploaded_map
