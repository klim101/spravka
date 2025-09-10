#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:

from __future__ import annotations
import asyncio, re, logging, json
import aiohttp
from dataclasses import dataclass, asdict
from typing import Iterable
from urllib.parse import urlparse
# Устанавливаем API-ключ OpenAI
import os
import requests
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import openai
from typing import List, Dict, Any, Tuple
import json
import streamlit as st
from collections import defaultdict
import asyncio, aiohttp, re, textwrap, nest_asyncio, openai, tiktoken
from collections import defaultdict
from urllib.parse import urlparse
import tldextract, re, asyncio, aiohttp                      # если tldextract уже импортирован – эту строку можно короче
from functools import partial
import threading
import time
import functools
import pickle
import ast
from pathlib import Path
KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
    "DYXLESS_TOKEN": st.secrets["DYXLESS_TOKEN"]
}

DYXLESS_TOKEN = KEYS["DYXLESS_TOKEN"]

CACHE_FILE = Path("google_cache.pkl")
GOOGLE_CACHE: dict = {}
QUERY_HISTORY: list = []

def _save_cache():
    try:
        CACHE_FILE.write_bytes(pickle.dumps({"cache": GOOGLE_CACHE, "history": QUERY_HISTORY}))
    except Exception:
        pass

def clear_google_cache():
    GOOGLE_CACHE.clear()
    QUERY_HISTORY.clear()
    try:
        CACHE_FILE.unlink()
    except FileNotFoundError:
        pass


# In[ ]:


# ─────────────────── app.py ────────────────────
import os, re, asyncio, aiohttp, requests, nest_asyncio, logging
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

nest_asyncio.apply()
logging.basicConfig(level=logging.WARNING)
import os, re, html, textwrap, asyncio, logging, nest_asyncio
import aiohttp, requests, streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import tldextract, openai


import html, re

_URL_PAT = re.compile(r"https?://[^\s)]+", flags=re.I)

def _linkify(text) -> str:
    """Безопасно превращает URL в <a …>ссылка</a>."""
    if not isinstance(text, str):                      # << главное
        text = "" if text is None else str(text)

    def repl(m):
        u = html.escape(m.group(0))
        return f'<a href="{u}" target="_blank">ссылка</a>'
    return _URL_PAT.sub(repl, text)



def long_job(total: int, key: str):
    """Долгая задача: пишет прогресс в st.session_state[key]"""
    for i in range(total + 1):
        time.sleep(1)                         # здесь ваша тяжёлая логика
        st.session_state[key] = i / total     # от 0.0 до 1.0
    st.session_state[key] = 1.0               # финализируем



# ── helpers ────────────────────────────────────────────────
async def _site_snippet(domain: str) -> str:
    """Возвращает первый Google-сниппет для site:domain (или '')."""
    if not domain:
        return ""
    async with aiohttp.ClientSession() as sess:
        q = f"site:{domain}"
        snips = await _google(sess, q, n=1)
    return snips[0][1] if snips else ""



@st.cache_data(ttl=3_600, show_spinner=False)
def dyxless_query(query: str,
                  token: str,
                  max_rows: int = 20_000) -> Dict[str, Any]:
    """
    Обёртка Dyxless. Если записей > max_rows – возвращаем только max_rows,
    дополнительно пишем truncated=True и original_counts.
    """
    url = "https://api-dyxless.cfd/query"
    try:
        r = requests.post(url, json={"query": query, "token": token}, timeout=15)
        r.raise_for_status()
        res = r.json()

        if res.get("status") and isinstance(res.get("data"), list):
            full = len(res["data"])
            if full > max_rows:
                res["data"] = res["data"][:max_rows]
                res.update(truncated=True, original_counts=full, counts=max_rows)
        return res
    except requests.RequestException as e:
        return {"status": False, "error": str(e)}












# ╭─🔧  вспомогалки ───────────────────────────────╮
_BAD = ("vk.com", "facebook.", ".pdf", ".jpg", ".png")
HEADERS = {"User-Agent": "Mozilla/5.0 (Win64) AppleWebKit/537.36 Chrome/125 Safari/537.36"}
def _bad(u: str) -> bool: return any(b in u.lower() for b in _BAD)

async def _google(sess, q, n=3):
    q = re.sub(r'[\"\'“”]', " ", q)[:80]
    params = {"key": KEYS["GOOGLE_API_KEY"], "cx": KEYS["GOOGLE_CX"],
              "q": q, "num": n, "hl": "ru", "gl": "ru"}
    async with sess.get("https://www.googleapis.com/customsearch/v1",
                         params=params, headers=HEADERS, timeout=8) as r:
        if r.status != 200:
            logging.warning(f"Google error {r.status}")
            return []
        js = await r.json()
        return [(i["link"], i.get("snippet", "")) for i in js.get("items", [])
                if not _bad(i["link"])]






async def _gpt(messages, *, model="gpt-5-mini", T=0.1):
    """Асинхронный вызов OpenAI ChatCompletion → str."""
    chat = await openai.ChatCompletion.acreate(
        model=model, temperature=T, messages=messages)
    return chat.choices[0].message.content.strip()

# ────────── основная обёртка (в стиле вашего RAG) ────────────────────────────
class SiteRAG:
    """
    url        – адрес сайта (можно без http/https)
    max_chunk  – максимальная длина одного куска HTML, который уйдёт LLM
    summary    – итоговый подробный паспорт компании
    chunks_out – список промежуточных резюме c каждой части HTML
    html_size  – размер исходного HTML-файла, bytes
    """
    def __init__(self, url: str, *, model="gpt-4o-mini",
                 max_chunk: int = 6_000, T: float = 0.18):
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        self.url       = url
        self.model     = model
        self.max_chunk = max_chunk
        self.T         = T

    # ---------- 1. скачиваем HTML ------------------------------------------------
    async def _fetch(self) -> str:
        h = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=h) as sess:
            async with sess.get(self.url, timeout=20) as r:
                if r.status == 200 and "text/html" in r.headers.get("Content-Type", ""):
                    return await r.text("utf-8", errors="ignore")
                raise RuntimeError(f"Не удалось скачать {self.url} (status={r.status})")

    # ---------- 2. делим HTML на «безопасные» куски -----------------------------
    def _split(self, html_raw: str) -> list[str]:
        # пробуем резать по крупным тегам, чтобы куски были связны
        body = re.split(r"</?(?:body|div|section|article)[^>]*>", html_raw, flags=re.I)
        chunks, buf = [], ""
        for part in body:
            if len(buf) + len(part) > self.max_chunk:
                chunks.append(buf); buf = part
            else:
                buf += part
        if buf:
            chunks.append(buf)
        return chunks

    # ---------- 3. map-фаза: конспектируем каждый кусок -------------------------
    async def _summarise_chunk(self, n: int, total: int, chunk: str) -> str:
        sys = (
            "Ты – профессиональный аналитик. Прочитай данный HTML-фрагмент и "
            "выпиши ВСЕ значимые факты о компании (продукты, услуги, история, "
            "география, клиенты, цифры, команда, контакты и пр.). "
            "Удали навигацию/footer/скрипты. Сохрани структуру абзацами."
        )
        return await _gpt([
            {"role": "system", "content": sys},
            {"role": "user",
             "content": f"HTML_CHUNK_{n}/{total} (len={len(chunk):,}):\n{chunk}"}],
            model=self.model, T=self.T)

    # ---------- 4. reduce-фаза: делаем финальный паспорт ------------------------
    async def _summarise_overall(self, parts: list[str]) -> str:
        sys = (
            "Ниже уже подготовленные конспекты разных частей сайта. "
            "На их основе составь один ПОЛНЫЙ и связный паспорт компании: "
            "• кто они и чем занимаются; • продукты / услуги; • рынок и клиенты; "
            "• история и ключевые события; • география и масштабы; "
            "• руководители / команда; • любые цифры и факты; "
            "• вывод о позиции и перспективах. Ничего важного не упусти."
        )
        merged = "\n\n".join(parts)
        return await _gpt([
            {"role": "system", "content": sys},
            {"role": "user",   "content": merged}],
            model=self.model, T=self.T)

    # ---------- orchestrator ----------------------------------------------------
    async def _run_async(self):
        html_raw = await self._fetch()
        chunks   = self._split(html_raw)

        # map
        part_summaries = []
        for idx, ch in enumerate(chunks, 1):
            print(f"→ LLM chunk {idx}/{len(chunks)} …")
            part_summaries.append(await self._summarise_chunk(idx, len(chunks), ch))

        # reduce
        summary_final = await self._summarise_overall(part_summaries)

        return {"summary":    summary_final,
                "chunks_out": part_summaries,
                "html_size":  f"{len(html_raw):,} bytes",
                "url":        self.url}

    # ---------- публичный синхронный интерфейс ----------------------------------
    def run(self) -> dict:
        loop = asyncio.get_event_loop()
        if loop and loop.is_running():
            return loop.run_until_complete(self._run_async())
        return asyncio.run(self._run_async())




# ---------- helper: синхронно достаём паспорт сайта -----------------
def _site_passport_sync(url: str, *, max_chunk: int = 6_000) -> str:
    """Вызывает SiteRAG(url).run() и возвращает только summary."""
    try:
        return SiteRAG(url, max_chunk=max_chunk).run()["summary"]
    except Exception as exc:
        return f"[site passport error: {exc}]"



class RAG:
    """
    summary    – финальный отчёт (Google-сниппеты + паспорт сайта)
    queries    – запросы, которые LLM сгенерировала для Google
    snippets   – список (url, text) из Google
    news_snippets – сниппеты с крупных новостных сайтов
    site_ctx   – короткий сниппет «site:<домен> …»
    site_pass  – подробный паспорт сайта (готовый summary от SiteRAG)
    """
    def __init__(self, company: str, *, website: str = "", market: str = "",
                 years=(2022, 2023, 2024), country: str = "Россия",
                 steps: int = 3, snips: int = 4,
                 llm_model: str = "gpt-4o-mini",company_info: dict | None = None,):
        self.company   = company.strip()
        self.website   = website.strip()
        self.market    = market.strip()
        self.country   = country
        self.years     = years
        self.steps     = steps
        self.snips     = snips
        self.llm_model = llm_model
        self.company_info = company_info or {}

    # ---------- site-snippet из Google ---------------------------------
    async def _site_ctx(self) -> str:
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        snip = await _site_snippet(dom)
        if snip:
            return f"{snip}\nрынок компании – {self.market}" if self.market else snip
        return f"рынок компании – {self.market}" if self.market else ""

    # ---------- GPT → поисковые запросы --------------------------------
    async def _queries(self, hist="") -> list[str]:
        dom  = tldextract.extract(self.website).registered_domain if self.website else ""
        base = f'"{self.company}"' + (f' OR site:{dom}' if dom else "")
        sys  = (
            "ТЫ — ОПЫТНЫЙ ИССЛЕДОВАТЕЛЬ РЫНКОВ И ДАННЫХ. СФОРМУЛИРУЙ НЕ МЕНЕЕ 30 ПРОСТЫХ РАЗНООБРАЗНЫХ GOOGLE-ЗАПРОСОВ НА РУССКОМ ЯЗЫКЕ, "
            f"ПОЗВОЛЯЮЩИХ СОБРАТЬ ИНФОРМАЦИЮ О КОМПАНИИ «{self.company}» НА РЫНКЕ «{self.market}» "
            f"({self.country}, {', '.join(map(str, self.years))}).\n"
            "КАЖДЫЙ ЗАПРОС ОБЯЗАТЕЛЬНО ДОЛЖЕН СОДЕРЖАТЬ НАЗВАНИЕ КОМПАНИИ.\n"
            "### ОБЯЗАТЕЛЬНЫЕ БЛОКИ\n"
            "1. ОПИСАНИЕ КОМПАНИИ И БРЕНДЫ.\n"
            "2. ЧИСЛЕННОСТЬ СОТРУДНИКОВ.\n"
            "3. ПРОИЗВОДСТВЕННЫЕ МОЩНОСТИ.\n"
            "4. ИНВЕСТИЦИИ И РАСШИРЕНИЯ.\n"
            "5. АДРЕСА ШТАБ-КВАРТИРЫ И ПРОИЗВОДСТВ.\n"
            "6. СОЦИАЛЬНЫЕ СЕТИ.\n"
            "7. ИСТОРИЯ.\n"
            "8. ПРИБЫЛЬ И ОБЪЁМЫ ПРОДУКЦИИ.\n"
            "9. КОНКУРЕНТЫ (НАЗВАНИЕ И САЙТ).\n"
            "10. УПОМИНАНИЯ НА ФОРУМАХ И В РЕЙТИНГАХ.\n"
            "ПО КАЖДОМУ БЛОКУ СДЕЛАЙ НЕСКОЛЬКО РАЗНЫХ ЗАПРОСОВ.\n"
            "### СОВЕТЫ ПО КОНСТРУКЦИИ ЗАПРОСОВ\n"
            "- ИСПОЛЬЗУЙ ОПЕРАТОРЫ: `site:`, `intitle:`, `inurl:`, `filetype:pdf`, `OR`.\n"
            "- ДОБАВЛЯЙ ГОДЫ И НАЗВАНИЯ ПРОДУКТОВ И БРЕНДОВ, ЕСЛИ НУЖНО.\n"
            f"- ДЛЯ ОФИЦИАЛЬНЫХ ДАННЫХ ПРИМЕНЯЙ `site:{dom}` ИЛИ САЙТЫ РЕГУЛЯТОРОВ.\n"
            "### ПРАВИЛА\n"
            "- НЕ ДУБЛИРУЙ ЗАПРОСЫ.\n"
            "- НЕ ДОБАВЛЯЙ КОММЕНТАРИИ, НУМЕРАЦИЮ И ЭМОДЗИ.\n"
            "- ВЫВОДИ ТОЛЬКО СТРОКИ В ВИДЕ `QUERY: ...`.\n"
            "### CHAIN OF THOUGHTS (ВНУТРЕННЕ, НЕ ВЫВОДИТЬ)\n"
            "1. ПОНЯТЬ задачу.\n"
            "2. СФОРМИРОВАТЬ информационные пробелы.\n"
            "3. СГЕНЕРИРОВАТЬ ключевые термины.\n"
            "4. СКОМБИНИРОВАТЬ их с операторами.\n"
            "5. ВЫВЕСТИ строки `QUERY:`.\n"
        )
        raw = await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": f'base={base}{hist}'}],
            model=self.llm_model, T=0.1)
        ql = re.findall(r"QUERY:\s*(.+)", raw, flags=re.I)

        if not hist:
            templates = [
                f'"{self.company}" описание',
                f'"{self.company}" бренды',
                f'"{self.company}" сотрудники',
                f'"{self.company}" численность',
                f'"{self.company}" производственные мощности',
                f'"{self.company}" инвестиции',
                f'"{self.company}" расширение',
                f'"{self.company}" адрес',
                f'"{self.company}" история',
                f'"{self.company}" прибыль',
                f'"{self.company}" объём производства',
                f'"{self.company}" конкуренты',
                f'"{self.company}" рейтинг',
                f'форум "{self.company}"',
                f'site:news.* "{self.company}"',
            ]
            ql = templates + [q for q in ql if q not in templates]

        # ─── целевые соцсети и официальный сайт ──────────────────────
        social_sites = ["vk.com", "facebook.com", "linkedin.com",
                        "youtube.com", "ok.ru"]
        extras = [f'"{self.company}" site:{s}' for s in social_sites]
        if dom:
            extras.append(f'"{self.company}" site:{dom}')

        # dedup сохраняя порядок
        ql.extend(extras)
        ql = list(dict.fromkeys(ql))
        return ql

    # ---------- финальный отчёт ----------------------------------------
    async def _summary(self, ctx: str) -> str:
        sys = (
            "ТЫ — ВЫСОКОКВАЛИФИЦИРОВАННЫЙ АНАЛИТИК РЫНКОВ. СОСТАВЬ СТРУКТУРИРОВАННЫЙ "
            "АНАЛИТИЧЕСКИЙ ОТЧЁТ О КОМПАНИИ ИЗ ПОСЛЕДОВАТЕЛЬНЫХ АБЗАЦЕВ В СЛЕДУЮЩЕМ "
            "ФИКСИРОВАННОМ ПОРЯДКЕ: "
            "1) ОПИСАНИЕ; "
            "2) БРЕНДЫ (перечень и краткое описание); "
            "3) ЧИСЛЕННОСТЬ СОТРУДНИКОВ; "
            "4) ПРОИЗВОДСТВЕННЫЕ МОЩНОСТИ (площадь, объёмы по годам/дням); "
            "5) ИНВЕСТИЦИИ И ПРОЕКТЫ РАСШИРЕНИЯ (сумма, планы, рынки); "
            "6) АДРЕС HQ И ПРОИЗВОДСТВЕННЫХ ПЛОЩАДОК; "
            "7) СОЦИАЛЬНЫЕ СЕТИ (ВК, Facebook, LinkedIn, YouTube, Одноклассники, сайт); "
            "8) ИСТОРИЯ И КЛЮЧЕВЫЕ СОБЫТИЯ; "
            "9) ПРИБЫЛЬ/ОБЪЁМЫ ПРОДУКЦИИ; "
            "10) КОНКУРЕНТЫ (названия, сайты, краткое описание); "
            "11) УЧАСТИЕ В ФОРУМАХ/НОВОСТЯХ/РЕЙТИНГАХ. "
            "ПОСЛЕ КАЖДОГО ФАКТА ОБЯЗАТЕЛЬНО УКАЗЫВАЙ ССЫЛКУ-ИСТОЧНИК В КРУГЛЫХ СКОБКАХ (ПОЛНЫЙ URL). "
            "ЕСЛИ ДАННЫХ НЕТ — ПИШИ 'не найдено'. "
            "НЕ ДУБЛИРУЙ ИНФОРМАЦИЮ И НЕ ВЫДУМЫВАЙ ФАКТОВ. "
            "НЕ ИСПОЛЬЗУЙ MARKDOWN, НЕ УКАЗЫВАЙ ВЫРУЧКУ (REVENUE) НИ В КАКОМ ВИДЕ, НО МОЖНО УКАЗЫВАТЬ ПРИБЫЛЬ ПО ПРОДУКТАМ.\n"
        )
        
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": ctx[:20_000]}],
            model=self.llm_model, T=0.25)

    def _normalize_sections(self, summary: str) -> str:
        sections = summary.split("\n\n")
        norm = []
        for sec in sections:
            lines = [l.strip() for l in sec.splitlines() if l.strip()]
            seen, uniq = set(), []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    uniq.append(line)
            norm.append("\n".join(uniq) if uniq else "не найдено")
        return "\n\n".join(norm)

    # ---------- orchestrator -------------------------------------------
    async def _run_async(self):
        # paralell: сниппет + детальный паспорт сайта
        site_ctx_task = asyncio.create_task(self._site_ctx())
        site_pass_task = (
            asyncio.create_task(asyncio.to_thread(_site_passport_sync, self.website))
            if self.website else None
        )
        

        queries, snippets, hist = [], [], ""
        news_snippets: list[tuple[str, str]] = []
        async with aiohttp.ClientSession() as s:
            for _ in range(self.steps):
                ql = await self._queries(hist)
                ql = [f"{q} {self.market}" if self.market and
                      self.market.lower() not in q.lower() else q for q in ql]
                queries += ql
                res = await asyncio.gather(*[_google(s, q, self.snips) for q in ql])
                snippets += sum(res, [])
                hist = f"\nСниппетов: {len(snippets)}"

            news_domains = [
                "rbc.ru",
                "kommersant.ru",
                "vedomosti.ru",
                "tass.ru",
                "forbes.ru",
            ]
            news_queries = [f'site:{d} "{self.company}"' for d in news_domains]
            queries += news_queries
            res = await asyncio.gather(*[_google(s, q, self.snips) for q in news_queries])
            news_snippets = sum(res, [])
            snippets += news_snippets

        site_ctx  = await site_ctx_task
        site_pass = await site_pass_task if site_pass_task else ""

        # выделяем сниппеты с соцсетями
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        social_domains = ["vk.com", "facebook.com", "linkedin.com",
                          "youtube.com", "ok.ru"]
        if dom:
            social_domains.append(dom)
        social_snips = [
            (u, t) for u, t in snippets
            if any(sd in u.lower() or sd in t.lower() for sd in social_domains)
        ]

        # ---------- собираем единый контекст для GPT -----------------
        ctx_parts: list[str] = []

        # 1) короткий сниппет из Google («site:…»)
        if site_ctx:
            ctx_parts.append(f"SITE_SNIPPET:\n{site_ctx}")

        # 2) полный паспорт, сгенерированный SiteRAG
        if site_pass:
            ctx_parts.append(f"SITE_PASSPORT:\n{site_pass}")

        # 3) факты из checko / fin-API
        company_doc_txt = ""
        if self.company_info:                       # ← был передан в __init__
            def _pair(k, v):
                if v in (None, "", []): return ""
                if isinstance(v, list):
                    v = "; ".join(map(str, v[:10]))
                return f"* **{k}:** {v}"
            company_doc_txt = "\n".join(
                p for p in (_pair(k, v) for k, v in self.company_info.items()) if p
            )
            if company_doc_txt:
                ctx_parts.append(f"COMPANY_DOC:\n{company_doc_txt}")

        # 4) соцсети
        if social_snips:
            ctx_parts.append(
                "SOCIAL_SNIPPETS:\n" +
                "\n".join(f"URL:{u}\nTXT:{t}" for u, t in social_snips)
            )

        # 5) Google-сниппеты
        ctx_parts.append(
            "\n".join(f"URL:{u}\nTXT:{t}" for u, t in snippets)
        )

        # ---------- финальный отчёт ----------------------------------
        summary = await self._summary("\n\n".join(ctx_parts))
        summary = self._normalize_sections(summary)

        return {
            "summary":     summary,
            "queries":     queries,
            "snippets":    snippets,
            "news_snippets": news_snippets,
            "site_ctx":    site_ctx,
            "site_pass":   site_pass,
            "company_doc": company_doc_txt   # ← новый ключ (если нужен во фронте)
        }


    # ---------- публичный синхронный интерфейс -----------------
    def run(self) -> dict:
        """
        Возвращает dict со всеми полями.
        Корректно работает и когда event-loop уже запущен
        (например, в Jupyter, внутри Streamlit-callback и т.п.).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop and loop.is_running():
                # nest_asyncio.patch() уже вызван выше, поэтому можно:
                return loop.run_until_complete(self._run_async())
        except RuntimeError:
            # get_event_loop() может бросить, если цикла нет — тогда просто создаём новый
            pass

        return asyncio.run(self._run_async())












# ╭─🌐  Market RAG helpers ───────────────────────────╮
async def google_snippets(query: str, num: int = 4):
    q = re.sub(r'[\"\'“”]', '', query)[:80]
    params = {
        "key": KEYS["GOOGLE_API_KEY"],
        "cx":  KEYS["GOOGLE_CX"],
        "q":   q, "num": num, "hl": "ru", "gl": "ru"
    }
    async with aiohttp.ClientSession() as sess:
        async with sess.get("https://www.googleapis.com/customsearch/v1",
                            params=params, headers=HEADERS, timeout=8) as r:
            if r.status != 200:
                logging.warning(f"[Google] {r.status}")
                return []
            js = await r.json()
            return [(it["link"], it.get("snippet", ""))
                    for it in js.get("items", []) if not _bad(it["link"])]

async def gpt_async(messages, T=0.2, model="gpt-4o-mini"):
    chat = await openai.ChatCompletion.acreate(
        model=model, temperature=T, messages=messages)
    return chat.choices[0].message.content.strip()

class FastMarketRAG:
    """Возвращает dict(summary, queries, snippets) за ~10 с."""
    def __init__(self, market, country="Россия",
                 years=(2021, 2022, 2023, 2024),
                 steps=1, snips=6):
        self.market, self.country = market, country
        self.years, self.steps, self.snips = years, steps, snips



    
    async def _queries(self, hist=""):
        sys = (
            "ТЫ — ОПЫТНЫЙ ИССЛЕДОВАТЕЛЬ РЫНКОВ И ДАННЫХ. СФОРМУЛИРУЙ 10–12 ТОЧНЫХ И ЭФФЕКТИВНЫХ GOOGLE-ЗАПРОСОВ, "
            f"НАПРАВЛЕННЫХ НА СБОР СТРУКТУРИРОВАННОЙ ИНФОРМАЦИИ О РЫНКЕ «{self.market}» В СТРАНЕ {self.country.upper()} ЗА ПЕРИОД {', '.join(map(str, self.years))}. "
            "ПОИСКОВЫЕ ЗАПРОСЫ ДОЛЖНЫ ОХВАТЫВАТЬ СЛЕДУЮЩИЕ АСПЕКТЫ РЫНКА: "
            "1) ОБЪЁМ И ДИНАМИКА РЫНКА, "
            "2) СТРУКТУРА И СЕГМЕНТАЦИЯ, "
            "3) ОСНОВНЫЕ ИГРОКИ И ИХ ДОЛИ, "
            "4) ЦЕНЫ И ЦЕНОВЫЕ ТЕНДЕНЦИИ, "
            "5) КЛЮЧЕВЫЕ ТРЕНДЫ И ИНОВАЦИИ, "
            "6) РЕГИОНАЛЬНЫЙ РАЗРЕЗ, "
            "7) ФАКТОРЫ РОСТА И БАРЬЕРЫ ВХОДА, "
            "8) СДЕЛКИ, IPO, СЛИЯНИЯ, "
            "9) АНАЛИТИЧЕСКИЕ ОТЧЁТЫ И ДОКЛАДЫ "
            "ФОРМАТ ОТВЕТА: QUERY: <СТРОКА ДЛЯ ПОИСКА В GOOGLE>. "
            "НЕ ПОВТОРЯЙ ЗАПРОСЫ. НЕ ДОБАВЛЯЙ ЛИШНИХ ПРЕДИСЛОВИЙ — ТОЛЬКО СПИСКОМ."
)
        raw = await gpt_async([
            {"role": "system", "content": sys},
            {"role": "user",   "content": hist}
        ], T=0.12)
        return re.findall(r"QUERY:\s*(.+)", raw, flags=re.I)

    async def _run_async(self):
        snippets, hist = [], ""
        for _ in range(self.steps):
            ql = await self._queries(hist)
            tasks = [google_snippets(q, self.snips) for q in ql]
            for res in await asyncio.gather(*tasks):
                snippets.extend(res)
            hist = f"сниппетов={len(snippets)}"

        context = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in snippets)[:18000]
        sys = (
            f"ТЫ — ВЫСОКОКЛАССНЫЙ АНАЛИТИК РЫНКА «{self.market}» В СТРАНЕ {self.country.upper()}. "
            "СФОРМИРУЙ ПОГОДОВОЙ ОБЗОР РЫНКА, ГДЕ КАЖДЫЙ ГОД ПРЕДСТАВЛЕН ОТДЕЛЬНЫМ НАПОЛНЕННЫМ АБЗАЦЕМ, ВКЛЮЧАЮЩИМ СЛЕДУЮЩИЕ ЭЛЕМЕНТЫ: "
            "1) ОБЪЁМ РЫНКА (единицы измерения и оценка), "
            "2) ТЕМП РОСТА (в процентах или абсолютных значениях), "
            "3) СТРУКТУРА И СЕГМЕНТАЦИЯ (по типу продукта, клиенту, каналу и др.), "
            "4) РЕГИОНАЛЬНЫЕ РАЗРЕЗЫ (ключевые регионы внутри страны), "
            "5) ОСНОВНЫЕ ИГРОКИ И ИХ ДОЛИ (с долями в %, если доступны), "
            "6) КРУПНЫЕ СДЕЛКИ И СОБЫТИЯ (M&A, IPO, партнерства), "
            "7) ЦЕНОВОЙ АНАЛИЗ (уровни цен, динамика, факторы влияния), "
            "8) КЛЮЧЕВЫЕ ТРЕНДЫ (технологии, спрос, регулирование и др.), "
            "9) БАРЬЕРЫ И ОГРАНИЧЕНИЯ (вход, логистика, нормативка), "
            "10) ВЫВОДЫ ПО ГОДУ (ключевые итоги и сдвиги). "
            "11) в конце выведи таблицу формата 'Год | Объём рынка' по всем упомянутым годам; таблица будет использована для построения графика без осей и подписей"
            "ВСЕ ФАКТЫ ДОЛЖНЫ БЫТЬ УНИКАЛЬНЫМИ, НЕ ПОВТОРЯТЬСЯ И ПОДТВЕРЖДЁННЫ РЕАЛЬНЫМИ ССЫЛКАМИ НА ИСТОЧНИКИ В КРУГЛЫХ СКОБКАХ (ФОРМАТ: ПОЛНЫЙ URL). "
            "НЕ ИСПОЛЬЗУЙ MARKDOWN, НЕ ПРИДУМЫВАЙ ФАКТЫ — ТОЛЬКО ДОКУМЕНТИРОВАННЫЕ ДАННЫЕ. "
            "всегда оставляй ссылки"

)
        summary = await gpt_async([
            {"role": "system", "content": sys},
            {"role": "user",   "content": context}
        ], T=0.19)
        return {"summary": summary, "queries": ql, "snippets": snippets}

    def run(self):
        return asyncio.run(self._run_async())

# кешируем, чтобы при повторных запросах было мгновенно
@st.cache_data(ttl=86_400, show_spinner="🔎 Генерируем рыночный отчёт…")
def get_market_rag(market):
    return FastMarketRAG(market).run()








# ---------------------------------------------------------------
# Внешние зависимости (должны быть определены в проекте):
#   _google(sess, query: str, k: int) -> list[tuple[str, str]]
#   _gpt(messages, model: str, T: float) -> str
#   _image(sess, query: str) -> str | None
#   _linkify(text: str) -> str
#   ck_company(inn: str) -> dict  # (опционально)
#   _site_passport_sync(url: str) -> str  # (опционально)
# ---------------------------------------------------------------

RUS_NEWS_DOMAINS: set[str] = {
    # федеральные
    "kommersant.ru", "vedomosti.ru", "rbc.ru", "forbes.ru", "tass.ru",
    "interfax.ru", "iz.ru", "ria.ru", "thebell.io", "lenta.ru", "gazeta.ru",
    # профильные/региональные
    "vc.ru", "cnews.ru", "fontanka.ru", "dp.ru", "banki.ru",
}

INTERVIEW_PAT = re.compile(r"\bинтервью\b|подкаст|Q&A|q&a|в беседе|дал интервью|exclusive|эксклюзив", re.I)

LEGAL_KEYWORDS = {
    "prosecuted": [
        "уголовное дело", "преслед", "возбуждено дело", "расследуется",
        "под следствием", "подозревается", "обвин", "дело в отношении",
    ],
    "imprisoned": [
        "арестован", "задержан", "в сизо", "в колонии", "приговор",
        "осужд", "лишение свободы", "в тюрьме", "домашний арест",
    ],
    "sanctioned": [
        "санкц", "ofac", "sdn", "санкционных спис", "uk sanctions", "eu sanctions",
    ],
}

# ----------------- Helpers -----------------

def _norm_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_name(fio: str) -> str:
    fio = _norm_ws(fio).replace("\xa0", " ")
    fio = re.sub(r"\s+", " ", fio)
    return fio.strip(" \t\n\r\f\v-—\"'")


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        k = x.lower()
        if k not in seen:
            seen.add(k); out.append(x)
    return out


def _is_news(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(d in host for d in RUS_NEWS_DOMAINS)


def _scan_legal(text: str) -> dict[str, bool]:
    t = text.lower()
    return {k: any(kw in t for kw in kws) for k, kws in LEGAL_KEYWORDS.items()}


@dataclass
class Person:
    name: str
    role: str
    bio: str | None = None
    news: list[str] | None = None
    photo: str | None = None
    sources: list[str] | None = None
    legal: dict[str, bool] | None = None

    @property
    def tagged(self) -> str:
        if "(" in self.name and self.name.endswith(")"):
            return self.name
        return f"{self.name} ({self.role})"


class FastLeadersInterviews:
    """
    Возвращает dict(summary, names, queries, snippets).

    company_info ждёт структуру Checko/FNS (best-effort):
      • leaders_raw / founders_raw
      • general_director / managers / "Руковод"
      • founders / "Учред_ФЛ"
      • ИНН / inn
    """
    def get_leaders_rag(name: str, *, website: str = "", market: str = "", company_info: dict | None = None):
        return FastLeadersInterviews(name, website=website, market=market, company_info=company_info).run()

    def _safe_div(a, b):
        try:
            if a is None or b in (None, 0):
                return None
            return a / b
        except Exception:
            return None

    
    def __init__(self, company: str, *, website: str = "", market: str = "",
                 company_info: dict | None = None, model: str = "gpt-4o-mini"):
        self.c      = company.strip()
        self.site   = website.strip()
        self.market = market.strip()
        self.cinfo  = company_info or {}
        self.model  = model

    # ---------------- helpers ----------------
    def _domain(self) -> str:
        try:
            import tldextract
            return tldextract.extract(self.site).registered_domain if self.site else ""
        except Exception:
            return ""

    async def _llm_queries(self, prompt: str, n: int = 8) -> list[str]:
        raw = await _gpt([
            {"role": "system", "content": "Сгенерируй короткие Google-запросы, по одному на строку."},
            {"role": "user",   "content": prompt}
        ], model=self.model, T=0.18)
        qs = [q.strip().lstrip("Q:").strip() for q in raw.splitlines() if q.strip()]
        return _dedupe(qs)[:n]

    # ---------- 1) РУКОВОДИТЕЛИ / ВЛАДЕЛЬЦЫ ----------
    async def _leaders(self, sess: aiohttp.ClientSession):
        people: list[Person] = []
        queries: list[str] = []
        snips:   list[tuple[str, str]] = []

        # 0) Из company_info (несколько возможных ключей)
        def _push_name(x, role):
            if isinstance(x, str) and x.strip():
                people.append(Person(name=_normalize_name(x), role=role))
            elif isinstance(x, dict):
                fio = x.get("name") or x.get("fio") or x.get("ФИО") or ""
                if fio: people.append(Person(name=_normalize_name(fio), role=role))
            elif isinstance(x, list):
                for y in x: _push_name(y, role)

        _push_name(self.cinfo.get("leaders_raw"),  "генеральный директор")
        _push_name(self.cinfo.get("founders_raw"), "акционер")
        _push_name(self.cinfo.get("general_director"), "генеральный директор")
        _push_name(self.cinfo.get("Руковод"), "генеральный директор")
        _push_name(self.cinfo.get("Учред_ФЛ"), "акционер")
        _push_name(self.cinfo.get("founders"), "акционер")

        # 1) Checko по ИНН (мягко)
        inn = self.cinfo.get("ИНН") or self.cinfo.get("inn")
        if inn:
            try:
                cdata = ck_company(str(inn))
                fio_dir = _normalize_name(cdata.get("Руководитель", {}).get("ФИО") or cdata.get("CEO") or "")
                if fio_dir:
                    people.append(Person(name=fio_dir, role="генеральный директор"))
                for f in cdata.get("УчредителиФЛ") or []:
                    fio = _normalize_name(f.get("ФИО") or f.get("fio") or "")
                    if fio:
                        people.append(Person(name=fio, role="акционер"))
            except Exception as e:
                logging.warning(f"[ck_company] {inn}: {e}")

        # 2) Если пусто → сначала найдём ФИО гендира из Google
        if not people:
            base_q = [f'"{self.c}" генеральный директор', f'"{self.c}" CEO']
            for q in base_q:
                queries.append(q)
                snips.extend(await _google(sess, q, 3))
            if snips:
                sys = (
                    "Ты аналитик. Из сниппетов извлеки ФИО генерального директора. "
                    "Формат: NAME: <ФИО>"
                )
                txt = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in snips)[:10_000]
                llm = await _gpt([
                    {"role": "system", "content": sys},
                    {"role": "user",   "content": txt}
                ], model=self.model, T=0.1)
                m = re.search(r"NAME:\s*(.+)$", llm.strip())
                if m:
                    people.append(Person(name=_normalize_name(m.group(1)), role="генеральный директор"))

        # 3) Дедуп по (имя, роль)
        seen = set(); ded = []
        for p in people:
            k = (p.name.lower(), p.role.lower())
            if k in seen: continue
            seen.add(k); ded.append(p)
        people = ded

        if not people:
            return [], queries, snips

        # 4) Обогащение: биографии/юрстатус/фото/СМИ-ссылки
        sem_http = asyncio.Semaphore(6)
        sem_llm  = asyncio.Semaphore(2)

        async def enrich(p: Person) -> Person:
            fio = p.name.split("(")[0].strip()

            qset = {
                f'"{fio}" {self.c}',
                f'"{fio}" {self.c} биография',
                f'"{fio}" биография',
                f'"{fio}" {self.c} карьера',
                f'"{fio}" активы', f'"{fio}" состояние', f'"{fio}" капитал', f'"{fio}" Forbes',
                f'"{fio}" супруг', f'"{fio}" супруга', f'"{fio}" жена', f'"{fio}" муж',
                f'"{fio}" дети', f'"{fio}" сын', f'"{fio}" дочь',
                f'"{fio}" родственник', f'"{fio}" связаны', f'"{fio}" родственник чиновник',
                f'"{fio}" друг бизнесмен', f'"{fio}" чиновник',
                f'"{fio}" уголовное дело', f'"{fio}" арест', f'"{fio}" санкции',
            }
            try:
                prompt = (
                    f"Сгенерируй 4–8 гугл‑запросов для фактов о '{fio}' из '{self.c}': "
                    f"карьера/где работал; активы/состояние/Forbes; семья (супруг(а), дети); "
                    f"связи/родственники (госслужба, бизнесмены); юридический статус (дела, аресты, санкции)."
                )
                qset.update(await self._llm_queries(prompt))
            except Exception as e:
                logging.warning(f"[llm-queries] {fio}: {e}")

            search_all: list[tuple[str, str]] = []
            for q in _dedupe(list(qset)):
                async with sem_http:
                    try:
                        res = await _google(sess, q, 3)
                    except Exception as e:
                        logging.warning(f"[google] {q}: {e}")
                        res = []
                queries.append(q)
                snips.extend(res)
                search_all.extend(res)

            ctx = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in search_all)[:10_000]
            sys_bio = (
                "Сжато и только проверяемые факты. Выведи пунктами:\n"
                "1) Карьера/где работал (2–3 факта).\n"
                "2) Активы и оценка состояния (если есть).\n"
                "3) Семья: супруг(а), дети (если упоминаются).\n"
                "4) Связи/родственники: госслужащие/бизнесмены (если подтверждено).\n"
                "5) Юридический статус: преследуется/сидел/под санкциями (если есть).\n"
                "Если данных нет — пиши 'нет данных'. В конце строка: Источники: URL1; URL2; URL3 (до 3)."
            )
            async with sem_llm:
                bio = await _gpt([
                    {"role": "system", "content": sys_bio},
                    {"role": "user",   "content": ctx},
                ], model=self.model, T=0.15)
            p.bio = _norm_ws(bio)

            p.legal = _scan_legal("\n".join(t for _, t in search_all))
            p.news  = _dedupe([u for u, _ in search_all if _is_news(u)])

            async with sem_http:
                try:
                    p.photo = await _image(sess, f"{fio} {self.c}")
                except Exception as e:
                    logging.warning(f"[photo] {fio}: {e}")
                    p.photo = None

            p.sources = [u for u, _ in search_all]
            return p

        people = await asyncio.gather(*[enrich(p) for p in people])
        return [asdict(p) | {"name": p.tagged} for p in people], queries, snips

    # ---------- 2) Интервью и новости о компании (СМИ‑только) ----------
    async def _interviews(self, names: list[dict], sess: aiohttp.ClientSession):
        dom = self._domain()
        sc  = await self._site_ctx(sess)
        base_ctx = (f"SITE_CONTEXT:\n{sc}\nрынок компании – {self.market}\n\n" if sc else "")

        def qpack_person(fio: str) -> list[str]:
            qs = [
                f'"{fio}" интервью', f'"{fio}" комментарий', f'"{fio}" дал интервью',
                f'"{fio}" подкаст', f'"{fio}" выступил', f'"{fio}" {self.c}',
            ]
            if dom: qs.append(f'"{fio}" site:{dom}')
            return qs

        q_company = [
            f'"{self.c}" интервью', f'"{self.c}" комментарий', f'"{self.c}" новость',
        ]
        if dom: q_company.append(f'"{self.c}" site:{dom}')

        all_queries: list[str] = []
        all_snips:   list[tuple[str, str]] = []

        async def pull(q: str):
            all_queries.append(q)
            try:
                all_snips.extend(await _google(sess, q, 3))
            except Exception as e:
                logging.warning(f"[interviews-google] {q}: {e}")

        tasks = [pull(q) for q in q_company]
        for p in (names or []):
            fio = p["name"].split("(")[0].strip()
            tasks += [pull(q) for q in qpack_person(fio)]
        await asyncio.gather(*tasks)

        # СМИ-фильтр
        news_snips = [(u, t) for u, t in all_snips if _is_news(u)]
        if not news_snips:
            return all_queries, [], "Свежих интервью и новостей не найдено."

        # Классификация по ключевым словам
        interviews = _dedupe([u for u, t in news_snips if INTERVIEW_PAT.search(u) or INTERVIEW_PAT.search(t)])
        news_urls  = _dedupe([u for u, _ in news_snips])

        ctx = base_ctx + "\n".join(f"URL:{u}\nTXT:{t}" for u, t in news_snips)[:16_000]
        sys = (
            "Ты — контент-аналитик. Составь краткий дайджест ТОЛЬКО по ссылкам СМИ.\n"
            "Секции: 1) Интервью; 2) Новости о компании.\n"
            "Для каждого пункта: дата (если видна), 1–2 фразы сути, ссылка. Макс 8 пунктов на секцию."
        )
        digest = await _gpt([
            {"role": "system", "content": sys},
            {"role": "user",   "content": ctx},
        ], model=self.model, T=0.18)

        # Можно дополнительно вернуть сырые URL
        extra = {"interview_urls": interviews, "news_urls": news_urls}
        return all_queries, news_snips, _norm_ws(digest) + "\n\n" + json.dumps(extra, ensure_ascii=False)

    # ---------- 3) Orchestrator ----------
    async def _run_async(self):
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            people, q_lead, s_lead = await self._leaders(sess)
            q_int,  s_int, digest  = await self._interviews(people, sess)

        # ① Владельцы/топ-менеджеры c биографиями
        if people:
            blocks = []
            for p in people:
                block = p["name"]
                if p.get("bio"):   block += f"\nБиография:\n{p['bio']}"
                lf = (p.get("legal") or {})
                if any(lf.values()):
                    tags = []
                    if lf.get("prosecuted"): tags.append("преследуется")
                    if lf.get("imprisoned"): tags.append("был(а) в тюрьме/арест")
                    if lf.get("sanctioned"): tags.append("под санкциями")
                    if tags: block += "\n⚖️ Правовой статус: " + ", ".join(tags)
                if p.get("news"): block += "\nСМИ-ссылки:\n" + "\n".join(p["news"][:6])
                photo = p.get("photo")
                block += f"\nФото: {photo or 'изображение не найдено'}"
                blocks.append(block)
            owners_block = "\n\n".join(blocks)
        else:
            owners_block = "Топ-менеджеры и владельцы не найдены."

        # ② Контакты
        contacts_block = ""
        cdata = self.cinfo.get("Контакты") or {}
        if cdata:
            phones = ", ".join(cdata.get("Тел", []))
            emails = ", ".join(cdata.get("Емэйл", []))
            site   = cdata.get("ВебСайт") or ""
            lines  = []
            if phones: lines.append(f"Тел: {phones}")
            if emails: lines.append(f"E-mail: {emails}")
            if site:   lines.append(f"Сайт: {site}")
            if lines:  contacts_block = "Контакты:\n" + "\n".join(lines)

        # ③ HTML
        body = "\n\n".join([part for part in (owners_block, contacts_block, digest) if part])
        summary_html = _linkify(body)

        return {
            "summary":  summary_html,
            "names":    people,
            "queries":  q_lead + q_int,
            "snippets": s_lead + s_int,
        }

    # ---------- sync wrapper ----------
    def run(self) -> dict:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio; nest_asyncio.apply()
                return loop.run_until_complete(self._run_async())
        except RuntimeError:
            pass
        return asyncio.run(self._run_async())

    # ---------- optional site context ----------
    async def _site_ctx(self, sess: aiohttp.ClientSession) -> str:
        if not getattr(self, "site", None):
            return ""
        loop = asyncio.get_running_loop()
        try:
            from functools import partial
            return await loop.run_in_executor(None, partial(_site_passport_sync, self.site))
        except Exception:
            return ""

        

def run_ai_insight_tab() -> None:
    st.title("📊 AI Company Insight")
    st.markdown("Введите данные (каждая компания — в отдельной строке).")

    c1, c2, c3, c4 = st.columns(4)
    with c1: inns_raw  = st.text_area("ИНН")
    with c2: names_raw = st.text_area("Название")
    with c3: mkts_raw  = st.text_area("Рынок")
    with c4: sites_raw = st.text_area("Сайт")

    aggregate_mode = st.checkbox("🧮 Суммировать финансы по всем ИНН")
    blocks = st.multiselect(
        "Выберите блоки отчёта",
        ["Описание компании", "Рыночный отчёт", "Руководители и интервью"],
        default=["Описание компании", "Рыночный отчёт", "Руководители и интервью"],
    )
    gen_doc = "Описание компании" in blocks
    gen_market = "Рыночный отчёт" in blocks
    gen_leads = "Руководители и интервью" in blocks

    if st.button("🔍 Построить отчёт", key="ai_build"):
        with st.spinner("Считаем отчёт…"):

    
            # ---------- парсинг ----------
            split = lambda s: [i.strip() for i in s.splitlines() if i.strip()]
            inns   = split(inns_raw)
            names  = split(names_raw)
            mkts   = split(mkts_raw)
            sites  = split(sites_raw)
            
            # ---------- валидация ----------
            if not inns:
                st.error("Укажите хотя бы один ИНН."); st.stop()
            
            if aggregate_mode:            # Σ-режим
                # при необходимости растягиваем одиночные значения
                if len(names) == 1 and len(inns) > 1:  names *= len(inns)
                if len(mkts)  == 1 and len(inns) > 1:  mkts  *= len(inns)
                if len(sites) == 1 and len(inns) > 1:  sites *= len(inns)
            
                # теперь всё либо пустое, либо совпадает по длине
                for lst, lbl in [(names, "Название"), (mkts, "Рынок")]:
                    if lst and len(lst) != len(inns):
                        st.error(f"Число строк «{lbl}» должно быть 1 или совпадать с числом ИНН."); st.stop()
            
            else:                         # одиночный режим
                if not (names and mkts):
                    st.error("Заполните все поля — ИНН, Название и Рынок."); st.stop()
                if len({len(inns), len(names), len(mkts)}) != 1:
                    st.error("Число строк во всех трёх полях должно совпадать."); st.stop()
                if sites and len(sites) != len(inns):
                    st.error("Число строк «Сайт» должно совпадать с числом ИНН."); st.stop()
            
            # ---------- выравниваем длины списков ----------
            pad = lambda lst: lst if lst else [""] * len(inns)
            names_full = pad(names)
            mkts_full  = pad(mkts)
            sites_full = pad(sites)
            YEARS = ["2022", "2023", "2024"]
            df_companies = pd.DataFrame([ck_company(i) for i in inns])
            doc = None
            mkt_res = None
            lead_res = None

            def parse_people_cell(cell) -> list[str]:
                """
                Принимает содержимое ячейки «Руковод» или «Учред_ФЛ»
                и возвращает список строк «ФИО (ИНН xxxx, доля yy%)».
                Работает и если cell = NaN, '', список, dict, строка-JSON.
                """
                # пусто / NaN
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    return []
            
                # если пришла строка — пробуем превратить в объект
                if isinstance(cell, str):
                    cell = cell.strip()
                    if not cell:
                        return []
                    try:
                        cell = ast.literal_eval(cell)      # '[{…}]' -> python
                    except (ValueError, SyntaxError):
                        # просто строка с одним ФИО
                        return [cell]
            
                # одиночный dict
                if isinstance(cell, dict):
                    cell = [cell]
            
                # list
                if isinstance(cell, list):
                    out = []
                    for item in cell:
                        if isinstance(item, str):          # уже готовая строка
                            out.append(item.strip())
                        elif isinstance(item, dict):       # наш основной случай
                            fio   = item.get("ФИО") or ""
                            inn   = item.get("ИНН") or ""
                            share = item.get("Доля", {}).get("Процент")
                            line  = fio
                            if inn:
                                line += f" (ИНН {inn}"

                                if share is not None:
                                    line += f", доля {float(share):.1f}%)"
                                else:
                                    line += ")"
            
                                line += f", доля {float(share):.1f}%)" if share is not None else ")"
  
                            out.append(line)
                    return [s for s in out if s]
                # fallback
                return [str(cell)]
            
            def row_people_json(row: pd.Series) -> dict:
                """Возвращает {'leaders_raw': [...], 'founders_raw': [...]}."""
                # ── 1. руководители ──────────────────────────────────────────
                leaders = parse_people_cell(row.get("Руковод"))
            
                # ── 2. учредители: колонка 'Учред' → dict → ключ 'ФЛ' ────────
                founders_cell = None
                uc = row.get("Учред")
                if isinstance(uc, dict):
                    founders_cell = uc.get("ФЛ")          # список словарей
                else:
                    founders_cell = uc                    # fallback (если формат другой)
            
                founders = parse_people_cell(founders_cell)
            
                return {"leaders_raw": leaders, "founders_raw": founders}
            
            people_cols = df_companies.apply(row_people_json, axis=1, result_type="expand")
            df_companies = pd.concat([df_companies, people_cols], axis=1)

            

            
            PNL_CODES = [                       # всё, что хотим видеть в длинной таблице
                ("Выручка (₽ млн)",                "2110"),
                ("Себестоимость продаж (₽ млн)",   "2120"),
                ("Валовая прибыль (₽ млн)",        "2200"),
                ("Коммерческие расходы (₽ млн)",   "2210"),
                ("Управленческие расходы (₽ млн)", "2220"),
                ("Прибыль от продаж (₽ млн)",      "2300"),
                ("Доходы от участия (₽ млн)",      "2310"),
                ("Проценты к получению (₽ млн)",   "2320"),
                ("Проценты к уплате (₽ млн)",      "2330"),
                ("Прочие доходы (₽ млн)",          "2340"),
                ("Прочие расходы (₽ млн)",         "2350"),
                ("Чистая прибыль (₽ млн)",         "2400"),
                ("Совокупный долг (₽ млн)",        "_total_debt"),
                ("Денежные средства (₽ млн)",      "_cash"),
                ("Кредиторская задолженность (₽ млн)", "1520"),
                ("Чистый долг (₽ млн)",            "_net_debt"),
                ("EBIT margin (%)",                "_ebit_margin"),
                ("Net Debt / EBIT",                "_netdebt_ebit"),
            ]
            
            # ---------- ① сводная вкладка, если нужна ----------
            def build_agg_finances() -> dict[str, dict[str, float | None]]:
                """Суммирует все ИНН и возвращает словарь agg[year][code]."""
                NUMERIC = {c for _, c in PNL_CODES if c.isdigit()} | {"1250", "1400", "1500"}
                raw = {y: defaultdict(float) for y in YEARS}
            
                for inn in inns:
                    fin = ck_fin(inn)
                    for y in YEARS:
                        for code in NUMERIC:
                            v = fin.get(y, {}).get(code)
                            if isinstance(v, (int, float)):
                                raw[y][code] += v / 1e6        # → млн
            
                agg = {}
                for y in YEARS:
                    rev   = raw[y]["2110"]
                    ebit  = raw[y]["2200"]
                    cash  = raw[y]["1250"]
                    debt  = raw[y]["1400"] + raw[y]["1500"]
                    pay   = raw[y]["1520"]
            
                    net_debt    = debt - cash - pay
                    ebit_margin = _safe_div(ebit, rev)
                    ebit_margin = ebit_margin * 100 if ebit_margin is not None else None
                    nd_ebit     = _safe_div(net_debt, ebit)
            
            
                    agg[y] = {**raw[y],
                              "_total_debt":  debt,
                              "_cash":        cash,
                              "_net_debt":    net_debt,
                              "_ebit_margin": ebit_margin,
                              "_netdebt_ebit":nd_ebit}
                return agg
            
            # создаём вкладки заранее (чтобы переменная tabs СУЩЕСТВОВАЛА всегда)
            if aggregate_mode:
                tabs = st.tabs(["Σ Сводно"] + ([] if len(inns) == 1 else
                                               [f"{n} ({inn})" for inn, n in zip(inns, names_full)]))
                # блок Σ Сводно — всегда первый
                with tabs[0]:
                    agg = build_agg_finances()
            
                    st.header("Σ Сводная финансовая таблица")
                    tbl = pd.DataFrame({"Показатель": [n for n, _ in PNL_CODES]})
                    for y in YEARS:
                        tbl[y] = [agg[y].get(code) for _, code in PNL_CODES]
            
                    def _fmt(v, pct=False, d=1):
                        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
                        return f"{v:.{d}f}{'%' if pct else ''}".replace(".", ",")
                    for i, (nm, _) in enumerate(PNL_CODES):
                        pct  = nm.endswith("%")
                        digs = 2 if ("Net" in nm or pct) else 1
                        tbl.iloc[i, 1:] = [_fmt(v, pct, digs) for v in tbl.iloc[i, 1:]]
            
                    st.dataframe(tbl.set_index("Показатель"),
                                 use_container_width=True,
                                 height=min(880, 40 * len(PNL_CODES)))
            
            
                    # график
                    # --- график: выручка / EBIT / чистая прибыль + EBIT-margin -----------
                    fig, ax1 = plt.subplots(figsize=(7, 3.5))
                    x = np.arange(len(YEARS)); w = 0.25
                    
                    bars_rev  = ax1.bar(x - w, [agg[y]["2110"] or 0 for y in YEARS],
                                        w, label="Выручка")
                    bars_ebit = ax1.bar(x,     [agg[y]["2200"] or 0 for y in YEARS],
                                        w, label="EBIT")
                    bars_prof = ax1.bar(x + w, [agg[y]["2400"] or 0 for y in YEARS],
                                        w, label="Чистая прибыль")
                    
                    # подписи на столбцах
                    for b in (*bars_rev, *bars_ebit, *bars_prof):
                        h = b.get_height()
                        if h and not np.isnan(h):
                            ax1.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h),
                                         xytext=(0, 3), textcoords="offset points",
                                         ha="center", fontsize=8)
                    
                    # линия EBIT-margin (%)
                    ax2 = ax1.twinx()
                    margins = [agg[y]["_ebit_margin"] if agg[y]["_ebit_margin"] else np.nan for y in YEARS]
                    ax2.plot(x, margins, linestyle="--", marker="o", label="EBIT margin, %")
                    
                    # подписи «х %» над точками линии
                    for xx, yy in zip(x, margins):
                        if not np.isnan(yy):
                            ax2.annotate(f"{yy:.1f}%", xy=(xx, yy),
                                         xytext=(0, 5), textcoords="offset points",
                                         ha="center", fontsize=8)
                    
                    # объединяем легенду
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
                    # ⟵  прячем шкалы Y
                    ax1.set_yticks([]); ax2.set_yticks([])
                    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
                    
                    for ax in (ax1, ax2):
                        ax.spines[:].set_visible(False)
                    
                    # оформление
                    ax1.set_xticks(x); ax1.set_xticklabels(YEARS, fontsize=10)
                    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
                    for ax in (ax1, ax2):
                        ax.spines[:].set_visible(False)
                    
                    fig.tight_layout(pad=1.0)
                    st.pyplot(fig)
            
                    # ────── Профиль / рынок / руководители (+ паспорт сайта) ──────────────
                    first_name = names_full[0] or "Компания"
                    first_mkt  = mkts_full[0]
                    first_site = sites_full[0]
                    first_inn = inns[0] if inns else None
                    
                    if gen_doc:
                        st.subheader("📝 Описание компании")
                        with st.spinner("Генерируем описание компании…"):
                            doc = RAG(first_name, website=first_site, market=first_mkt).run()

                        html_main = _linkify(doc["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{html_main}</div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(doc["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(doc["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )

                        if doc.get("site_pass"):
                            with st.expander("🌐 Паспорт сайта"):
                                st.markdown(
                                    f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                                    f"border-radius:8px;padding:18px;line-height:1.55'>"
                                    f"{_linkify(doc['site_pass']).replace(chr(10), '<br>')}</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("Паспорт сайта не получен (нет URL, ошибка загрузки или истек тай-аут).")
                    
                    if gen_market and first_mkt:
                        st.subheader("📈 Рыночный отчёт")
                        with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                            mkt_res = get_market_rag(first_mkt)

                        mkt_html = _linkify(mkt_res["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{mkt_html}</div>",
                            unsafe_allow_html=True,
                        )

                        rows, chart_rows = [], []
                        for line in mkt_res["summary"].splitlines():
                            m = re.match(r"\s*(\d{4})\s*[|:-]\s*(.+)", line)
                            if m:
                                year, val = m.groups()
                                rows.append((year, val.strip()))
                                num = re.search(r"[\d,.]+", val.replace(" ", ""))
                                if num:
                                    chart_rows.append((year, float(num.group(0).replace(",", ".")), val.strip()))
                        if rows:
                            df_tbl = pd.DataFrame(rows, columns=["Год", "Объём рынка"])
                            st.dataframe(df_tbl, use_container_width=True)
                            if chart_rows:
                                years = [r[0] for r in chart_rows]
                                nums  = [r[1] for r in chart_rows]
                                labels = [r[2] for r in chart_rows]
                                fig, ax = plt.subplots(figsize=(6, 3))
                                bars = ax.bar(years, nums)
                                for b, lab in zip(bars, labels):
                                    ax.annotate(lab, xy=(b.get_x()+b.get_width()/2, b.get_height()),
                                                xytext=(0,3), textcoords="offset points",
                                                ha="center", fontsize=8)
                                    
                                ax.set_yticks([])
                                for spine in ax.spines.values():
                                    spine.set_visible(False)
                                st.pyplot(fig)

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(mkt_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )
                    
                    if gen_leads:
                        st.subheader("👥 Руководители и интервью")
                        with st.spinner("Собираем руководителей и интервью…"):
                            company_info = df_companies.iloc[0].to_dict()
                            lead_res = get_leaders_rag(
                                first_name,
                                website=first_site,
                                market=first_mkt,
                                company_info=company_info,
                            )

                        st.markdown(
                            f"<div style='background:#F9FAFB;border:1px solid #ddd;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>"
                            f"{lead_res['summary'].replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(lead_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            if lead_res["snippets"]:
                                df = (
                                    pd.DataFrame(lead_res["snippets"], columns=["URL", "Snippet"])
                                    .drop_duplicates(subset="URL")
                                    .head(15)
                                )
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("Сниппеты не найдены.")
                    
                    # ─────── конец блока, дальше ваш код (если был) ───────────────────────
            
            # ---------- ② вкладки по отдельным компаниям ----------
            if aggregate_mode and len(inns) > 1:
                tabs = st.tabs(["Σ Сводно"] + [f"{n} ({inn})"
                                               for inn, n in zip(inns, names_full)])
            else:                                   # одиночный режим
                tabs = st.tabs([f"{n} ({inn})" for inn, n
                                in zip(inns, names_full)])
            
            start_idx = 1 if (aggregate_mode and len(inns) > 1) else 0
            
            for idx, (tab, inn, name, mkt, site) in enumerate(
                    zip(
                        tabs[start_idx:],   # пропускаем Σ-вкладку при необходимости
                        inns,
                        names_full,
                        mkts_full,
                        sites_full,
                    )
            ):
                with tab:
                    st.header(f"{name} — {inn}")
                    st.caption(f"Рынок: **{mkt or '—'}**")
            
                    # ---------- Финансовый профиль ----------
                    fin = ck_fin(inn)
                    calc = {y: {} for y in YEARS}
            
                    for y in YEARS:
                        yr = fin.get(y, {})
                        # прямые строки отчёта
                        for _, code in PNL_CODES:
                            if code.isdigit():
                                v = yr.get(code)
                                calc[y][code] = (v / 1e6) if isinstance(v, (int, float)) else None
            
                        # расчётные показатели
                        cash  = yr.get("1250", 0) / 1e6
                        debt  = (yr.get("1400", 0) + yr.get("1500", 0)) / 1e6
                        pay   = yr.get("1520", 0) / 1e6
                        rev   = calc[y].get("2110")
                        ebit  = calc[y].get("2200")
                        net_debt = debt - cash - pay
                        ebit_m = _safe_div(ebit, rev)
                        ebit_m = ebit_m * 100 if ebit_m is not None else None
                        nd_eb     = _safe_div(net_debt, ebit)
            
                        calc[y].update({
                            "_total_debt": debt or None,
                            "_cash":       cash or None,
                            "_net_debt":   net_debt if rev is not None else None,
                            "_ebit_margin":ebit_m,
                            "_netdebt_ebit":nd_eb,
                        })
            
                    # --- длинная таблица ---
                    tbl = pd.DataFrame({"Показатель": [n for n, _ in PNL_CODES]})
                    for y in YEARS:
                        tbl[y] = [calc[y].get(code) for _, code in PNL_CODES]
            
                    def fmt(v, pct=False, d=1):
                        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
                        return f"{v:.{d}f}{'%' if pct else ''}".replace(".", ",")
            
                    for i, (nm, _) in enumerate(PNL_CODES):
                        pct  = nm.endswith("%")
                        digs = 2 if ("Net" in nm or pct) else 1
                        tbl.iloc[i, 1:] = [fmt(v, pct, digs) for v in tbl.iloc[i, 1:]]
            
                    st.dataframe(tbl.set_index("Показатель"),
                                 use_container_width=True,
                                 height=min(880, 40 * len(tbl)))
            
                    # --- график: выручка / EBIT / чистая прибыль + EBIT-margin ---
                    fig, ax1 = plt.subplots(figsize=(7, 3.5))
                    x = np.arange(len(YEARS)); w = 0.25
                    bars_r  = ax1.bar(x - w, [calc[y]["2110"] or 0 for y in YEARS], w, label="Выручка")
                    bars_e  = ax1.bar(x,     [calc[y]["2200"] or 0 for y in YEARS], w, label="EBIT")
                    bars_p  = ax1.bar(x + w, [calc[y]["2400"] or 0 for y in YEARS], w, label="Чистая прибыль")
            
                    for b in (*bars_r, *bars_e, *bars_p):
                        h = b.get_height()
                        if h and not np.isnan(h):
                            ax1.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h),
                                         xytext=(0,3), textcoords="offset points",
                                         ha="center", fontsize=8)
            
                    ax2 = ax1.twinx()
                    m_vals = [calc[y]["_ebit_margin"] if calc[y]["_ebit_margin"] else np.nan for y in YEARS]
                    ax2.plot(x, m_vals, linestyle="--", marker="o", label="EBIT margin, %")
                    # ----- единая легенда -----
                    h1, l1 = ax1.get_legend_handles_labels()   # bars
                    h2, l2 = ax2.get_legend_handles_labels()   # линия margin
                    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
            
            
                    
                    for xx, yy in zip(x, m_vals):
                        if yy and not np.isnan(yy):
                            ax2.annotate(f"{yy:.1f}%", xy=(xx, yy),
                                         xytext=(0,5), textcoords="offset points",
                                         ha="center", fontsize=8)
            
                    ax1.set_xticks(x); ax1.set_xticklabels(YEARS, fontsize=10)
                    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
                     # ⟵  прячем шкалы Y
                    ax1.set_yticks([]); ax2.set_yticks([])
                    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
                    
                    for ax in (ax1, ax2):
                        ax.spines[:].set_visible(False)
                    for ax in (ax1, ax2): ax.spines[:].set_visible(False)
                    ax1.legend(loc="upper left", fontsize=9)
                    fig.tight_layout(pad=1.0)
                    st.pyplot(fig)
            
                    
                    
                    if gen_doc:
                        st.subheader("📝 Описание компании")
                        with st.spinner("Генерируем описание компании…"):
                            doc = RAG(name, website=site, market=mkt).run()

                        main_html = _linkify(doc["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{main_html}</div>",
                            unsafe_allow_html=True
                        )

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(doc["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(doc["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )

                        if doc.get("site_pass"):
                            with st.expander("🌐 Паспорт сайта"):
                                st.markdown(
                                    f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                                    f"border-radius:8px;padding:18px;line-height:1.55'>"
                                    f"{_linkify(doc['site_pass']).replace(chr(10), '<br>')}</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("Паспорт сайта не получен (нет URL, ошибка загрузки или истек тай-аут).")
                    
                    if gen_market and mkt:
                        st.subheader("📈 Рыночный отчёт")
                        with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                            mkt_res = get_market_rag(mkt)

                        mkt_html = _linkify(mkt_res["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{mkt_html}</div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(mkt_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )
                    
                    if gen_leads:
                        st.subheader("👥 Руководители и интервью")
                        with st.spinner("Собираем руководителей и интервью…"):
                            company_info = {
                                "leaders_raw":  df_companies.loc[idx, "leaders_raw"]  or [],
                                "founders_raw": df_companies.loc[idx, "founders_raw"] or [],
                            }
                            lead_res = get_leaders_rag(
                                name,
                                website=site,
                                market=mkt,
                                company_info=company_info,
                            )

                        st.markdown(
                            f"<div style='background:#F9FAFB;border:1px solid #ddd;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>"
                            f"{lead_res['summary'].replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(lead_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")

                        with st.expander("🔍 Сниппеты (top-15)"):
                            if lead_res["snippets"]:
                                df = (
                                    pd.DataFrame(lead_res["snippets"], columns=["URL", "Snippet"])
                                    .drop_duplicates(subset="URL")
                                    .head(15)
                                )
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("Сниппеты не найдены.")

        report = {"tbl": tbl, "graphics": fig}
        if doc is not None:
            report["doc"] = doc
        if mkt_res is not None:
            report["mkt_res"] = mkt_res
        if lead_res is not None:
            report["lead_res"] = lead_res
        st.session_state["ai_report"] = report
        st.session_state["ai_result_ready"] = True

def long_job(total_sec: int = 180, key_prog: str = "ai_prog"):
    """Фоновая задача, каждые 1 с обновляет progress в session_state."""
    for i in range(total_sec + 1):
        time.sleep(1)
        st.session_state[key_prog] = i / total_sec     # 0 … 1
    st.session_state["ai_done"] = True                 # отчёт готов

# ─────────────────────────────────────────────────────────
# 2. UI-функции двух вкладок
# ─────────────────────────────────────────────────────────
def run_advance_eye_tab() -> None:
    st.header("👁️ Advance Eye")

    user_query = st.text_input("Введите ИНН или ФИО")
    if st.button("🔍 Найти контакты") and user_query:
        with st.spinner("Запрашиваем Dyxless…"):
            res = dyxless_query(user_query, token=DYXLESS_TOKEN, max_rows=20_000)

        if res.get("status"):
            st.success(f"Показано записей: **{res['counts']}**")
            st.json(res["data"] or {"note": "Ничего не найдено"})
        else:
            st.warning(f"Ошибка запроса: {res.get('error', 'пустой ответ')}")

# ─────────────────────────────────────────────────────────
# 3. Инициализируем состояние (один раз за сессию)
# ─────────────────────────────────────────────────────────
st.session_state.setdefault("ai_prog", None)   # float 0…1 или None
st.session_state.setdefault("ai_done", False)  # отчёт готов?

# ─────────────────────────────────────────────────────────
# 4. Две вкладки
# ─────────────────────────────────────────────────────────
tab_ai, tab_eye = st.tabs(["📊 AI-Insight", "👁️ Advance Eye"])

# === вкладка 1: AI-Insight =========================================

with tab_ai:
    run_ai_insight_tab()       # вся логика внутри

with tab_eye:
    run_advance_eye_tab()      # поиск Dyxless
