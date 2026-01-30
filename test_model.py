from src.model import build_lstm

def test_model_creation():
    model = build_lstm((60,1))
    assert model is not None
