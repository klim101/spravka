import importlib, asyncio, aiohttp, streamlit as st

def test_leaders_bio_and_news(monkeypatch):
    monkeypatch.setattr(st, "secrets", {
        "OPENAI_API_KEY": "x",
        "GOOGLE_API_KEY": "x",
        "GOOGLE_CX": "x",
        "CHECKO_API_KEY": "x",
        "DYXLESS_TOKEN": "x",
    })
    un = importlib.import_module("un")

    async def fake_google(sess, q, n=3):
        return [("https://kommersant.ru/test", "snippet"), ("https://example.com", "other")]

    async def fake_gpt(messages, **kwargs):
        return "career; legal; assets; net worth"

    async def fake_image(sess, q):
        return "https://img.com/pic.jpg"

    monkeypatch.setattr(un, "_google", fake_google)
    monkeypatch.setattr(un, "_gpt", fake_gpt)
    monkeypatch.setattr(un, "_image", fake_image)

    cinfo = {
        "leaders_raw": un.extract_people([{ "ФИО": "Иванов Иван", "ИНН": "1", "Доля": {"Процент": 33.333}}]),
        "founders_raw": un.extract_people([{ "ФИО": "Петров Петр", "ИНН": "2", "Доля": {"Процент": 66.666}}]),
    }

    rag = un.FastLeadersInterviews("Comp", company_info=cinfo)

    async def run():
        async with aiohttp.ClientSession() as sess:
            people, _, _ = await rag._leaders(sess)
            return people
    people = asyncio.run(run())

    assert people[0]["name"].startswith("Иванов Иван")
    assert "(генеральный директор," in people[0]["name"]
    assert "доля 33.3%" in people[0]["name"]
    assert people[0]["bio"] == "career; legal; assets; net worth"
    assert people[0]["news"] == ["https://kommersant.ru/test"]
    assert people[0]["photo"] == "https://img.com/pic.jpg"
