import importlib
import pickle
import streamlit as st


def test_google_cache_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "secrets", {
        "OPENAI_API_KEY": "x",
        "GOOGLE_API_KEY": "x",
        "GOOGLE_CX": "x",
        "CHECKO_API_KEY": "x",
        "DYXLESS_TOKEN": "x",
    })
    un = importlib.import_module("un")
    cache_file = tmp_path / "cache.pkl"
    monkeypatch.setattr(un, "CACHE_FILE", cache_file)
    un.clear_google_cache()
    un.GOOGLE_CACHE[("q", 1)] = [("u", "t")]
    un.QUERY_HISTORY.append("q")
    un._save_cache()
    data = pickle.loads(cache_file.read_bytes())
    assert data["cache"] == {("q", 1): [("u", "t")]}
    assert data["history"] == ["q"]
    un.clear_google_cache()
    assert un.GOOGLE_CACHE == {}
    assert un.QUERY_HISTORY == []
