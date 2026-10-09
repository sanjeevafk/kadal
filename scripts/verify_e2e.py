import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure kadal root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

def test_full_e2e():
    print("=" * 60)
    print("🚀 KADAL COMPREHENSIVE END-TO-END VERIFICATION")
    print("=" * 60)

    client = TestClient(app)

    # 1. Health check
    print("\n[Step 1] Checking /api/health endpoint...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code} {resp.text}"
    health = resp.json()
    print("✅ System Health:", health)
    assert health.get("status") == "healthy"
    model_info = health.get("model", {})
    print(f"   Model Provider: {model_info.get('active_provider')}")
    print(f"   Model SHA256:   {model_info.get('sha256')}")
    print(f"   Demo/Sim Mode:  {model_info.get('is_demo')}")

    # 2. Check preloaded samples
    print("\n[Step 2] Checking /api/samples endpoint...")
    resp = client.get("/api/samples")
    assert resp.status_code == 200, f"Samples endpoint failed: {resp.text}"
    samples = resp.json() if isinstance(resp.json(), list) else resp.json().get("samples", [])
    print(f"✅ Found {len(samples)} preloaded acoustic samples:")
    for s in samples:
        print(f"   - {s.get('id')}: {s.get('name')} ({s.get('type')})")

    # 3. Analyze NOAA Debris sample
    print("\n[Step 3] Analyzing sample_noaa_debris.png via /api/analyze...")
    debris_path = ROOT_DIR / "backend" / "static" / "samples" / "sample_noaa_debris.png"
    assert debris_path.exists(), "Sample debris image missing"
    with open(debris_path, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("sample_noaa_debris.png", f, "image/png")},
            data={"confidence_threshold": "0.20", "iou_threshold": "0.45"}
        )
    assert resp.status_code == 200, f"Analysis failed: {resp.text}"
    result = resp.json()
    analysis_id = result.get("analysis_id")
    detections = result.get("detections", [])
    print(f"✅ Analysis Successful!")
    print(f"   Analysis ID: {analysis_id}")
    print(f"   Inference Time: {result.get('timing', {}).get('inference_ms')} ms")
    print(f"   Detections Found: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"     [{i}] Class: {det.get('class_name')} | Conf: {det.get('confidence'):.2f} | Threat: {det.get('threat_score')}")

    # 4. Analyze GeoTIFF sample with real WGS84 coordinates
    print("\n[Step 4] Analyzing GeoTIFF with geospatial solver (sample_noaa_geotiff_debris.tif)...")
    geotiff_path = ROOT_DIR / "backend" / "static" / "samples" / "sample_noaa_geotiff_debris.tif"
    if geotiff_path.exists():
        with open(geotiff_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                files={"file": ("sample_noaa_geotiff_debris.tif", f, "image/tiff")},
                data={"confidence_threshold": "0.20", "iou_threshold": "0.45"}
            )
        assert resp.status_code == 200, f"GeoTIFF analysis failed: {resp.text}"
        geo_result = resp.json()
        geo_id = geo_result.get("analysis_id")
        geo_dets = geo_result.get("detections", [])
        print(f"✅ GeoTIFF Analysis Successful! (Analysis ID: {geo_id})")
        print(f"   Has Georeference: {geo_result.get('georeferenced', False)}")
        print(f"   Detections: {len(geo_dets)}")
        for d in geo_dets:
            coords = d.get("coordinates")
            if coords:
                print(f"     Target {d.get('class_name')} at WGS84: Lat {coords.get('latitude')}, Lon {coords.get('longitude')}")

    # 5. Export PDF Intelligence Report
    print(f"\n[Step 5] Generating Publication-Ready PDF Report for analysis {analysis_id}...")
    resp = client.get(f"/api/export/{analysis_id}/pdf")
    assert resp.status_code == 200, f"PDF export failed: {resp.text}"
    pdf_bytes = resp.content
    print(f"✅ PDF Generated successfully! Size: {len(pdf_bytes):,} bytes")
    assert pdf_bytes.startswith(b"%PDF"), "Output is not valid PDF binary"

    # 6. Export CSV
    print(f"\n[Step 6] Generating CSV Export for analysis {analysis_id}...")
    resp = client.get(f"/api/export/{analysis_id}/csv")
    assert resp.status_code == 200, f"CSV export failed: {resp.text}"
    print(f"✅ CSV Generated successfully! Lines: {len(resp.text.strip().splitlines())}")

    # 7. Export Evidence Bundle ZIP
    print(f"\n[Step 7] Generating One-Click Evidence Bundle ZIP for analysis {analysis_id}...")
    resp = client.get(f"/api/export/{analysis_id}/bundle")
    assert resp.status_code == 200, f"Bundle export failed: {resp.text}"
    print(f"✅ ZIP Evidence Bundle Generated! Size: {len(resp.content):,} bytes")

    print("\n" + "=" * 60)
    print("🎉 ALL END-TO-END TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_e2e()
