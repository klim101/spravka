import importlib
import streamlit as st


def test_extract_people_rounds_share(monkeypatch):
    monkeypatch.setattr(st, "secrets", {
        "OPENAI_API_KEY": "x",
        "GOOGLE_API_KEY": "x",
        "GOOGLE_CX": "x",
        "CHECKO_API_KEY": "x",
        "DYXLESS_TOKEN": "x",
    })
    un = importlib.import_module("un")
    cell = [{"ФИО": "Иван", "ИНН": "123", "Доля": {"Процент": 12.34}}]
    assert un.extract_people(cell) == ["Иван (ИНН 123, доля 12.3%)"]

