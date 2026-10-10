"""
Amazon Bedrock Generative AI Service for KADAL Marine Intelligence.
Generates automated Benthic Ecological Impact & Cleanup Assessments for detected marine debris.
Uses Bedrock models (Claude 3.5 Sonnet / Amazon Titan) with robust, deterministic domain fallbacks
when credentials, access or network are unavailable.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from backend.config import AWS_BEDROCK_MODEL_ID, AWS_BEDROCK_REGION, ENABLE_AWS_SYNC

logger = logging.getLogger("kadal.aws.bedrock")

class BedrockService:
    def __init__(
        self,
        model_id: str = AWS_BEDROCK_MODEL_ID,
        region_name: str = AWS_BEDROCK_REGION
    ):
        self.model_id = model_id
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
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)
            logger.info(f"[AWS Bedrock] Initialized Bedrock runtime in {self.region_name} (Model: {self.model_id})")
            return self.client
        except Exception as e:
            logger.info(f"[AWS Bedrock] Bedrock client offline ({e}). Using deterministic rule-based benthic assessment.")
            self.client = None
            return None

    def generate_benthic_assessment(
        self,
        summary: Dict[str, Any],
        detections: List[Dict[str, Any]],
        geospatial_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a benthic ecological assessment and salvage recommendation.
        Returns a structured dictionary with:
        - ecological_risk_level: "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
        - risk_score: int [0-100]
        - summary: str
        - impact_points: List[str]
        - salvage_recommendation: str
        - provider: "amazon_bedrock" | "rule_based_engine"
        """
        total_targets = summary.get("total_detections", len(detections))

        # First, try calling Amazon Bedrock
        client = self._get_client()
        if client and total_targets > 0:
            try:
                bedrock_result = self._invoke_bedrock(summary, detections, geospatial_meta)
                if bedrock_result:
                    bedrock_result["provider"] = "amazon_bedrock"
                    return bedrock_result
            except Exception as e:
                err_str = str(e)
                if "expired" in err_str.lower() or "credentials" in err_str.lower():
                    logger.info("[AWS Bedrock] Live Bedrock call bypassed (credentials pending re-auth). Using local calibrated benthic engine.")
                else:
                    logger.warning(f"[AWS Bedrock] Bedrock API invocation failed ({e}). Reverting to rule-based benthic assessment engine.")

        # Deterministic Domain Rule-Based Ecological Assessment (Air-gapped / Local fallback)
        return self._generate_rule_based_assessment(summary, detections, geospatial_meta)


    def _invoke_bedrock(
        self,
        summary: Dict[str, Any],
        detections: List[Dict[str, Any]],
        geospatial_meta: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if not client:
            return None

        # Build prompt payload
        prompt = (
            f"You are a Senior Hydrographer and Marine Environmental Scientist analyzing side-scan sonar detection data.\n"
            f"Survey Summary:\n"
            f"- Total Detections: {summary.get('total_detections')}\n"
            f"- Class Distribution: {json.dumps(summary.get('class_counts'))}\n"
            f"- Acoustic Material Density: {json.dumps(summary.get('material_counts'))}\n"
            f"- Georeferenced: {geospatial_meta.get('georeferenced', False)}\n"
            f"Target Details: {json.dumps([{'type': d.get('object_type'), 'confidence': round(d.get('confidence', 0), 2), 'density': d.get('material_density'), 'threat': d.get('threat_score', 0)} for d in detections[:10]])}\n\n"
            f"Provide a concise JSON response strictly matching this schema with NO markdown fences:\n"
            f'{{"ecological_risk_level": "LOW|MODERATE|HIGH|CRITICAL", "risk_score": 0-100, "summary": "...", "impact_points": ["point 1", "point 2"], "salvage_recommendation": "..."}}'
        )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }

        response = client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        raw_text = response_body.get("content", [{}])[0].get("text", "").strip()

        # Clean JSON fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        return parsed

    def _generate_rule_based_assessment(
        self,
        summary: Dict[str, Any],
        detections: List[Dict[str, Any]],
        geospatial_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provides expert-calibrated benthic impact scoring even without live AWS connectivity."""
        total_targets = summary.get("total_detections", len(detections))
        if total_targets == 0:
            return {
                "ecological_risk_level": "LOW",
                "risk_score": 5,
                "summary": "Pristine seafloor baseline observed. Zero anthropogenic marine debris or acoustic anomalies detected in surveyed swath.",
                "impact_points": [
                    "Benthic habitat structure intact with standard acoustic backscatter uniformity.",
                    "No immediate obstruction hazard to marine biodiversity or commercial navigation."
                ],
                "salvage_recommendation": "No intervention required. Retain acoustic survey imagery for annual baseline monitoring.",
                "provider": "rule_based_engine"
            }

        material_counts = summary.get("material_counts", {})
        metal_count = material_counts.get("Hard (Metallic)", 0)
        synth_count = material_counts.get("Soft (Synthetic/Plastic)", 0)
        max_threat = max([d.get("threat_score", 0) for d in detections], default=0)

        # Compute calculated risk score
        risk_score = min(98, max(15, int(total_targets * 12 + metal_count * 8 + synth_count * 15 + max_threat * 0.3)))

        if risk_score >= 75:
            level = "CRITICAL"
        elif risk_score >= 50:
            level = "HIGH"
        elif risk_score >= 25:
            level = "MODERATE"
        else:
            level = "LOW"

        impacts = []
        if synth_count > 0:
            impacts.append(f"Synthetic ghost-gear/plastics ({synth_count} site{'s' if synth_count > 1 else ''}) pose persistent marine fauna entanglement and microplastic leaching hazards.")
        if metal_count > 0:
            impacts.append(f"Metallic submerged debris ({metal_count} site{'s' if metal_count > 1 else ''}) present galvanic corrosion, heavy-metal substrate contamination, and trawl-gear snag hazards.")
        if max_threat > 60:
            impacts.append("High acoustic relief profiles indicate significant bathymetric elevation capable of altering local hydrodynamic current scouring.")

        if not impacts:
            impacts.append("Anthropogenic debris footprint localized to identified acoustic coordinates.")

        salvage = (
            "Deploy autonomous ROV equipped with hydraulic grabber for non-invasive extraction."
            if synth_count > metal_count
            else "Coordinate tethered work-class ROV and crane barge for heavy maritime salvage and structural recovery."
        )

        return {
            "ecological_risk_level": level,
            "risk_score": risk_score,
            "summary": f"Survey identified {total_targets} anthropogenic marine debris target(s). Ecological impact assessed as {level} risk ({risk_score}/100) due to benthic habitat disruption.",
            "impact_points": impacts,
            "salvage_recommendation": salvage,
            "provider": "rule_based_engine"
        }
