def test_run_comparison():
    from drt_opt.web.service import run_comparison

    data = run_comparison(seed=1, max_frames=50)
    assert "baseline" in data
    assert "optimized" in data
    assert len(data["stops"]) > 0
    assert len(data["baseline"]["frames"]) > 0
    assert len(data["optimized"]["frames"]) > 0
    assert data["baseline"]["final_metrics"]["total_requests"] == data["optimized"]["final_metrics"]["total_requests"]


def test_web_app_network():
    from fastapi.testclient import TestClient
    from drt_opt.web.app import app

    client = TestClient(app)
    r = client.get("/api/network")
    assert r.status_code == 200
    body = r.json()
    assert "stops" in body
    assert len(body["stops"]) > 0
