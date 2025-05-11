from fastapi.testclient import TestClient
from fastapi_app import app

client = TestClient(app)

def test_read_root():
    """Test that the root endpoint returns a 200 status code."""
    response = client.get("/")
    assert response.status_code == 200 