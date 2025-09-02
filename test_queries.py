import importlib
import asyncio
import streamlit as st
import re

def test_rag_queries_templates(monkeypatch):
    monkeypatch.setattr(st, "secrets", {
        "OPENAI_API_KEY": "x",
        "GOOGLE_API_KEY": "x",
        "GOOGLE_CX": "x",
        "CHECKO_API_KEY": "x",
        "DYXLESS_TOKEN": "x",
    })
    un = importlib.import_module("un")
    async def fake_gpt(*args, **kwargs):
        return ""
    monkeypatch.setattr(un, "_gpt", fake_gpt)
    rag = un.RAG("comp", website="", market="")
    ql = asyncio.run(rag._queries())
    joined = "\n".join(ql)
    assert re.search(r"\bсотрудники\b", joined)
    assert re.search(r"производственн[а-я]* мощности", joined)
    assert re.search(r"\bконкурент", joined)
