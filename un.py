import os
import re
import html
import json
import time
import pickle
import logging
import asyncio
import threading
import functools
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

import requests
import aiohttp
import nest_asyncio
import pandas as pd
import numpy as np
import streamlit as st
import tldextract
import openai
# tiktoken / BeautifulSoup опциональны; оставляем импорт на будущее
# import tiktoken
# from bs4 import BeautifulSoup

nest_asyncio.apply()
logging.basicConfig(level=logging.WARNING)

# Секреты из Streamlit
KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
    "DYXLESS_TOKEN":  st.secrets["DYXLESS_TOKEN"],
}
openai.api_key = KEYS["OPENAI_API_KEY"]
DYXLESS_TOKEN = KEYS["DYXLESS_TOKEN"]

# Кэш Google
CACHE_FILE = Path(".google_cache.pkl")
try:
    _cache = pickle.loads(CACHE_FILE.read_bytes())
    GOOGLE_CACHE: dict = _cache.get("cache", {})
    QUERY_HISTORY: list = _cache.get("history", [])
except Exception:
    GOOGLE_CACHE, QUERY_HISTORY = {}, []

def _save_cache():
    try:
        CACHE_FILE.write_bytes(pickle.dumps({"cache": GOOGLE_CACHE,
                                             "history": QUERY_HISTORY}))
    except Exception:
        pass

def clear_google_cache():
    GOOGLE_CACHE.clear()
    QUERY_HISTORY.clear()
    _save_cache()

# ──────────────────────────────────────────────────────────────────────────────
# Отрасли и подсказки
# ──────────────────────────────────────────────────────────────────────────────
GROUPS = ["Industrials", "Consumer", "O&G", "M&M", "Retail",
          "Logistics", "FIG", "Services", "Agro", "TMT"]

GROUP_QUERY_TEMPLATES: dict[str, list[Callable[[str], str]]] = {
    "Industrials": [
        lambda c: f'"{c}" объём производства',
        lambda c: f'"{c}" производственная мощность',
        lambda c: f'"{c}" фабрика',
        lambda c: f'"{c}" логистика',
        lambda c: f'"{c}" оборудование',
        lambda c: f'"{c}" r&d',
        lambda c: f'"{c}" патенты',
        lambda c: f'"{c}" сертификат ISO',
        lambda c: f'"{c}" список оборудования',
        lambda c: f'"{c}" виды продукции',
    ],
    "Consumer": [
        lambda c: f'"{c}" бренды',
        lambda c: f'"{c}" логотип',
        lambda c: f'"{c}" целевая аудитория',
    ],
    "O&G": [
        lambda c: f'"{c}" объём добычи',
        lambda c: f'"{c}" переработка',
        lambda c: f'"{c}" запасы',
    ],
    "M&M": [
        lambda c: f'"{c}" объём добычи',
        lambda c: f'"{c}" запасы',
    ],
    "Retail": [
        lambda c: f'"{c}" количество магазинов',
        lambda c: f'"{c}" площадь магазинов',
        lambda c: f'"{c}" квадратные метры',
    ],
    "Logistics": [
        lambda c: f'"{c}" парк',
        lambda c: f'"{c}" объём грузов',
        lambda c: f'"{c}" объём пассажиров',
    ],
    "FIG": [
        lambda c: f'"{c}" баланс',
        lambda c: f'"{c}" чистые активы',
        lambda c: f'"{c}" процентный доход',
        lambda c: f'"{c}" комиссионный доход',
        lambda c: f'"{c}" доход от страхования',
    ],
    "Services": [
        lambda c: f'"{c}" gmv услуг',
        lambda c: f'"{c}" проникновение',
    ],
    "Agro": [
        lambda c: f'"{c}" площадь земли',
        lambda c: f'"{c}" регион',
        lambda c: f'"{c}" культуры',
        lambda c: f'"{c}" животные',
        lambda c: f'"{c}" объём производства',
        lambda c: f'"{c}" склады',
        lambda c: f'"{c}" элеваторы',
    ],
    "TMT": [
        lambda c: f'"{c}" количество пользователей',
        lambda c: f'"{c}" ежедневные просмотры',
        lambda c: f'"{c}" проникновение',
        lambda c: f'"{c}" операционные показатели',
        lambda c: f'"{c}" gmv',
        lambda c: f'"{c}" gbv',
        lambda c: f'"{c}" количество просмотров',
    ],
}

GROUP_SUMMARY_HINTS: dict[str, str] = {
    "Industrials": (
        "учитывай объём производства и capacity, расположение фабрики, "
        "логистику, описание оборудования, наличие R&D и патентов, "
        "сертификаты ISO, список оборудования с характеристиками и виды продукции"
    ),
    "Consumer": "добавляй описание брендов с логотипами и целевую аудиторию",
    "O&G": "указывай объём добычи, переработки, запасы и ресурсы",
    "M&M": "указывай объём добычи, запасы и ресурсы",
    "Retail": "приводи количество и площадь магазинов, общую площадь в кв. метрах",
    "Logistics": "описывай парк и объёмы грузов или пассажиров",
    "FIG": (
        "добавляй баланс, чистые активы, процентный и комиссионный доход, "
        "доход от страхования и прочих финансовых продуктов"
    ),
    "Services": "указывай GMV услуг и уровень проникновения",
    "Agro": (
        "описывай объём земли и регион, выращиваемые культуры или животных, "
        "объёмы производства, наличие складов и элеваторов"
    ),
    "TMT": (
        "приводи количество пользователей, ежедневные просмотры, проникновение, "
        "релевантные операционные и крупные финансовые показатели"
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────
_URL_PAT = re.compile(r"https?://[^\s)]+", flags=re.I)
def _linkify(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    def repl(m):
        u = html.escape(m.group(0))
        return f'<a href="{u}" target="_blank">ссылка</a>'
    return _URL_PAT.sub(repl, text)

def long_job(total: int, key: str):
    for i in range(total + 1):
        time.sleep(1)
        st.session_state[key] = i / total
    st.session_state[key] = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# Dyxless (как было)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def dyxless_query(query: str,
                  token: str,
                  max_rows: int = 20_000) -> Dict[str, Any]:
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

# ──────────────────────────────────────────────────────────────────────────────
# Google Search helpers (правки: PDF разрешён, лимит q=128, кэш, rerank)
# ──────────────────────────────────────────────────────────────────────────────
# НЕ баним PDF (важные источники). Баним соцсети/картинки.
_BAD = ("vk.com", "facebook.", ".jpg", ".png")
HEADERS = {"User-Agent": "Mozilla/5.0 (Win64) AppleWebKit/537.36 Chrome/125 Safari/537.36"}

def _bad(u: str) -> bool:
    return any(b in u.lower() for b in _BAD)

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(x.strip())
    return out

# «сигналы мощности» для rerank
POWER_SIGNALS = [
    r"\bт/сут\b", r"\bтонн[аы] в сутки\b", r"\bт/год\b", r"\bтонн[аы] в год\b",
    r"\bмощн\w+\b", r"\bcapacity\b", r"\bм²\b", r"\bкв\.?\s*м\b",
    r"\bISO\s*9\d{2,}\b", r"\bMW\b|\bМВт\b|\bGWh\b",
]
_SIG = [re.compile(p, flags=re.I) for p in POWER_SIGNALS]

def score_snip(url: str, txt: str) -> int:
    s = 0
    for rgx in _SIG:
        if rgx.search(url) or rgx.search(txt):
            s += 1
    if url.lower().endswith(".pdf"): s += 1
    if re.search(r"\.(gov|edu)\b", url.lower()): s += 1
    return s

def rerank_snippets(snips: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, uniq_snips = set(), []
    for u, t in snips:
        if u not in seen:
            seen.add(u)
            uniq_snips.append((u, t))
    return sorted(uniq_snips, key=lambda x: score_snip(*x), reverse=True)

async def _google(sess: aiohttp.ClientSession, q: str, n: int = 3):
    q = re.sub(r'[\"\'“”]', " ", q)[:128]
    cache_key = (q, n)
    if cache_key in GOOGLE_CACHE:
        QUERY_HISTORY.append(q); return GOOGLE_CACHE[cache_key]

    params = {
        "key": KEYS["GOOGLE_API_KEY"], "cx": KEYS["GOOGLE_CX"],
        "q": q, "num": n, "hl": "ru", "gl": "ru"
    }
    try:
        async with sess.get("https://www.googleapis.com/customsearch/v1",
                            params=params, headers=HEADERS, timeout=8) as r:
            if r.status != 200:
                logging.warning(f"Google error {r.status}")
                return []
            js = await r.json()
            res = [(it["link"], it.get("snippet", "")) for it in js.get("items", []) if not _bad(it["link"])]
    except asyncio.TimeoutError:
        logging.warning("[Google] timeout")
        res = []

    GOOGLE_CACHE[cache_key] = res
    QUERY_HISTORY.append(q)
    _save_cache()
    return res

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI helpers
# ──────────────────────────────────────────────────────────────────────────────
async def _gpt(messages, *, model="gpt-4o-mini", T=0.2) -> str:
    chat = await openai.ChatCompletion.acreate(model=model, temperature=T, messages=messages)
    return chat.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# SiteRAG (как было, без лишних дублей)
# ──────────────────────────────────────────────────────────────────────────────
class SiteRAG:
    def __init__(self, url: str, *, model="gpt-4o-mini", max_chunk: int = 6_000, T: float = 0.18):
        if url and not url.startswith(("http://", "https://")):
            url = "http://" + url
        self.url       = url
        self.model     = model
        self.max_chunk = max_chunk
        self.T         = T

    async def _fetch(self) -> str:
        if not self.url:
            raise RuntimeError("URL is empty")
        h = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=h) as sess:
            async with sess.get(self.url, timeout=20) as r:
                if r.status == 200 and "text/html" in (r.headers.get("Content-Type", "")):
                    return await r.text("utf-8", errors="ignore")
                raise RuntimeError(f"Не удалось скачать {self.url} (status={r.status})")

    def _split(self, html_raw: str) -> list[str]:
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

    async def _summarise_chunk(self, n: int, total: int, chunk: str) -> str:
        sys = (
            "Ты – профессиональный аналитик. Прочитай HTML-фрагмент и выпиши ВСЕ значимые факты "
            "(продукты, услуги, история, география, клиенты, цифры, команда, контакты и пр.). "
            "Игнорируй навигацию/footer/скрипты. Пиши кратко, абзацами."
        )
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": f"HTML_CHUNK_{n}/{total} (len={len(chunk):,}):\n{chunk}"}],
            model=self.model, T=self.T
        )

    async def _summarise_overall(self, parts: list[str]) -> str:
        sys = (
            "Ниже конспекты частей сайта. На их основе составь один ПОЛНЫЙ и связный паспорт компании: "
            "кто они; продукты/услуги; рынок и клиенты; история и ключевые события; география и масштабы; "
            "руководители/команда; любые цифры и факты; вывод о позиции и перспективах."
        )
        merged = "\n\n".join(parts)
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": merged}],
            model=self.model, T=self.T
        )

    async def _run_async(self):
        html_raw = await self._fetch()
        chunks   = self._split(html_raw)
        part_summaries = []
        for idx, ch in enumerate(chunks, 1):
            part_summaries.append(await self._summarise_chunk(idx, len(chunks), ch))
        summary_final = await self._summarise_overall(part_summaries)
        return {"summary": summary_final, "chunks_out": part_summaries,
                "html_size": f"{len(html_raw):,} bytes", "url": self.url}

    def run(self) -> dict:
        try:
            loop = asyncio.get_event_loop()
            if loop and loop.is_running():
                return loop.run_until_complete(self._run_async())
        except RuntimeError:
            pass
        return asyncio.run(self._run_async())

def _site_passport_sync(url: str, *, max_chunk: int = 6_000) -> str:
    if not url:
        return ""
    try:
        return SiteRAG(url, max_chunk=max_chunk).run()["summary"]
    except Exception as exc:
        return f"[site passport error: {exc}]"

# ──────────────────────────────────────────────────────────────────────────────
# Шаблоны запросов (новые детерминированные)
# ──────────────────────────────────────────────────────────────────────────────
def BASE_TEMPLATES(C: str) -> list[str]:
    return unique([
        f'"{C}" описание',
        f'"{C}" бренды',
        f'"{C}" численность',
        f'"{C}" количество сотрудников',
        f'"{C}" производственные мощности',
        f'"{C}" мощность завода',
        f'"{C}" производственная площадка',
        f'"{C}" объём выпуска',
        f'"{C}" производительность',
        f'"{C}" инвестиции',
        f'"{C}" расширение',
        f'"{C}" строительство завода',
        f'"{C}" адрес',
        f'"{C}" офис',
        f'"{C}" история',
        f'"{C}" прибыль',
        f'"{C}" объём производства',
        f'"{C}" конкуренты',
        f'"{C}" рейтинг',
        f'форум "{C}"',
        f'site:news.* "{C}"',
        f'"{C}" filetype:pdf',
        f'"{C}" презентация filetype:pdf',
        f'"{C}" тендер filetype:pdf',
    ])

def SOCIAL_TEMPLATES(C: str, dom: str) -> list[str]:
    base = [f'"{C}" site:{s}' for s in ["vk.com","facebook.com","linkedin.com","youtube.com","ok.ru"]]
    if dom:
        base.append(f'"{C}" site:{dom}')
    base += [f'"{C}" сайт', f'"{C}" linkedin', f'"{C}" youtube', f'"{C}" вк']
    return unique(base)

# ──────────────────────────────────────────────────────────────────────────────
# RAG (переписан: генерация запросов, rerank, 2-шаговый summary)
# ──────────────────────────────────────────────────────────────────────────────
class RAG:
    """
    summary     – финальный отчёт (Google-сниппеты + паспорт сайта)
    queries     – список всех запросов
    snippets    – [(url, snippet)] после rerank и dedup
    news_snips  – сниппеты с крупных новостных сайтов
    site_ctx    – короткий сниппет «site:<домен> …»
    site_pass   – паспорт сайта
    """
    def __init__(
        self,
        company: str,
        *,
        website: str = "",
        market: str = "",
        years=(2022, 2023, 2024),
        country: str = "Россия",
        steps: int = 2,
        snips: int = 4,
        llm_model: str = "gpt-4o-mini",
        facts_model: str = "gpt-4o",         # можно указать более сильную модель
        render_model: str = "gpt-4o-mini",
        company_info: dict | None = None,
        group: str = "",
    ):
        self.company   = company.strip()
        self.website   = website.strip()
        self.market    = market.strip()
        self.country   = country
        self.years     = years
        self.steps     = steps
        self.snips     = snips
        self.llm_model = llm_model
        self.facts_model = facts_model or llm_model
        self.render_model = render_model or llm_model
        self.company_info = company_info or {}
        self.group = group.strip()

    # ── контекст «site:домен»
    async def _site_ctx(self) -> str:
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        if not dom:
            return f"рынок компании – {self.market}" if self.market else ""
        async with aiohttp.ClientSession() as sess:
            snips = await _google(sess, f"site:{dom}", n=1)
            base = snips[0][1] if snips else ""
        if base and self.market:
            return f"{base}\nрынок компании – {self.market}"
        return base or (f"рынок компании – {self.market}" if self.market else "")

    # ── генерация запросов (детерминированное ядро + 10 LLM-запросов)
    async def _queries(self) -> list[str]:
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        C = self.company

        seeds = BASE_TEMPLATES(C)
        if self.group:
            seeds += [tpl(C) for tpl in GROUP_QUERY_TEMPLATES.get(self.group, [])]
        seeds = unique(seeds)

        extras = SOCIAL_TEMPLATES(C, dom)
        ql = unique(seeds + extras)

        sys = (
            "Дай 10 доп. Google-запросов на русском для компании "
            f"«{C}» (каждый должен содержать название компании), "
            "упор на мощности/объёмы/адреса/инвестпроекты. "
            "Используй site:, intitle:, inurl:, filetype:pdf, OR. "
            "Формат: QUERY: <строка>. Ничего больше."
        )
        raw = await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": ""}],
            model=self.llm_model, T=0.05
        )
        llm_q = re.findall(r"QUERY:\s*(.+)", raw, flags=re.I)

        def add_market(q: str) -> str:
            if self.market and self.market.lower() not in q.lower():
                return f'{q} "{self.market}"'
            return q

        ql = unique([add_market(q) for q in (ql + llm_q)])
        return ql[:60]   # жёсткий лимит

    # ── 2-шаговый вывод: факты(JSON) → рендер текста
    async def _facts_json(self, ctx: str) -> str:
        hint = GROUP_SUMMARY_HINTS.get(self.group, "")
        sys = (
            "Извлеки СТРУКТУРИРОВАННЫЕ ФАКТЫ о компании в JSON без лишнего текста. "
            "Схема: {"
            '"description": str|null,'
            '"brands": [str],'
            '"headcount": {"value": number|null, "year": number|null, "sources":[string]},'
            '"capacity": [{"metric": str, "value": str, "year": number|null, "site": str|null, "sources":[string]}],'
            '"investments": [{"desc": str, "amount": str|null, "year": number|null, "sources":[string]}],'
            '"addresses": [{"type":"hq|plant", "value": str, "sources":[string]}],'
            '"socials": [{"type":"site|vk|fb|linkedin|youtube|ok", "url": str}],'
            '"history": [{"year": number|null, "event": str, "sources":[string]}],'
            '"production": [{"product": str, "volume": str, "period": str|null, "sources":[string]}],'
            '"competitors": [{"name": str, "site": str|null}],'
            '"mentions": [{"type":"forum|rating|news", "url": str}]'
            "}. "
            "Пиши только валидный JSON. Если данных нет — null/[] соответственно. "
            + (f"Учитывай отраслевой контекст: {hint} " if hint else "")
        )
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": ctx[:18000]}],
            model=self.facts_model, T=0.05
        )

    async def _render_text_from_facts(self, facts_json: str) -> str:
        sys = (
            "Возьми этот JSON с фактами и сверстани непротиворечивый текст без Markdown "
            "строгим порядком разделов: 1) Описание; 2) Бренды; 3) Численность; "
            "4) Производственные мощности; 5) Инвестиции и расширения; 6) Адреса HQ и площадок; "
            "7) Соцсети; 8) История; 9) Прибыль/объёмы продукции; 10) Конкуренты; 11) Упоминания. "
            "После каждого факта оставляй ссылку в круглых скобках (полный URL). "
            "Не указывай выручку. Коротко и по делу."
        )
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user", "content": facts_json}],
            model=self.render_model, T=0.15
        )

    def _normalize_sections(self, summary: str) -> str:
        sections = summary.split("\n\n")
        norm = []
        for sec in sections:
            lines = [l.strip() for l in sec.splitlines() if l.strip()]
            seen, uniq = set(), []
            for line in lines:
                if line not in seen:
                    seen.add(line); uniq.append(line)
            norm.append("\n".join(uniq) if uniq else "не найдено")
        return "\n\n".join(norm)

    async def _run_async(self):
        # параллельно: site snippet и паспорт сайта
        site_ctx_task = asyncio.create_task(self._site_ctx())
        site_pass_task = (
            asyncio.create_task(asyncio.to_thread(_site_passport_sync, self.website))
            if self.website else None
        )

        queries, snippets = [], []
        news_snippets: list[tuple[str, str]] = []

        async with aiohttp.ClientSession() as s:
            # генерация запросов 1 раз (стабильнее)
            ql = await self._queries()
            queries += ql

            # основной сбор сниппетов
            res = await asyncio.gather(*[_google(s, q, self.snips) for q in ql])
            snippets += sum(res, [])

            # крупные новости
            news_domains = ["rbc.ru","kommersant.ru","vedomosti.ru","tass.ru","forbes.ru"]
            news_queries = [f'site:{d} "{self.company}"' for d in news_domains]
            queries += news_queries
            res = await asyncio.gather(*[_google(s, q, self.snips) for q in news_queries])
            news_snippets = sum(res, [])
            snippets += news_snippets

        # rerank + dedup
        snippets = rerank_snippets(snippets)

        # соцсети отдельно
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        social_domains = ["vk.com", "facebook.com", "linkedin.com", "youtube.com", "ok.ru"]
        if dom: social_domains.append(dom)
        social_snips = [(u, t) for u, t in snippets if any(sd in u.lower() or sd in t.lower() for sd in social_domains)]

        site_ctx  = await site_ctx_task
        site_pass = await site_pass_task if site_pass_task else ""

        # собираем контекст для LLM: берём топ после rerank (лимит по размеру)
        ctx_parts: list[str] = []
        if site_ctx:  ctx_parts.append(f"SITE_SNIPPET:\n{site_ctx}")
        if site_pass: ctx_parts.append(f"SITE_PASSPORT:\n{site_pass}")

        # факты из company_info (если переданы)
        company_doc_txt = ""
        if self.company_info:
            def _pair(k, v):
                if v in (None, "", []): return ""
                if isinstance(v, list): v = "; ".join(map(str, v[:10]))
                return f"* **{k}:** {v}"
            company_doc_txt = "\n".join(
                p for p in (_pair(k, v) for k, v in self.company_info.items()) if p
            )
            if company_doc_txt:
                ctx_parts.append(f"COMPANY_DOC:\n{company_doc_txt}")

        if social_snips:
            ctx_parts.append(
                "SOCIAL_SNIPPETS:\n" + "\n".join(f"URL:{u}\nTXT:{t}" for u, t in social_snips[:30])
            )

        # ограничиваем общий объём сниппетов (после rerank)
        main_snips_txt = []
        total_len = 0
        for u, t in snippets:
            line = f"URL:{u}\nTXT:{t}"
            if total_len + len(line) > 16000:
                break
            main_snips_txt.append(line)
            total_len += len(line)
        ctx_parts.append("\n".join(main_snips_txt))

        # двухшаговый вывод
        facts = await self._facts_json("\n\n".join(ctx_parts))
        summary = await self._render_text_from_facts(facts)
        summary = self._normalize_sections(summary)

        return {
            "summary":       summary,
            "queries":       queries,
            "snippets":      snippets,
            "news_snippets": news_snippets,
            "site_ctx":      site_ctx,
            "site_pass":     site_pass,
            "company_doc":   company_doc_txt,
        }

    def run(self) -> dict:
        try:
            loop = asyncio.get_event_loop()
            if loop and loop.is_running():
                return loop.run_until_complete(self._run_async())
        except RuntimeError:
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
            "11) итоговым абзацем выведи объемы рынка по годам которые фигурировали в прошлых абзацах"
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


def _parse_market_volumes(summary: str) -> dict[str, float]:
    """Извлекает пары год–объём из последнего абзаца рыночного отчёта."""
    vols: dict[str, float] = {}
    lines = summary.strip().splitlines()
    if not lines:
        return vols
    last = lines[-1]
    for year, num in re.findall(r"(20\d{2})[^\d]{0,20}([\d\s,\.]+)", last):
        try:
            vols[year] = float(num.replace(" ", "").replace(",", "."))
        except ValueError:
            continue
    return vols






# ╭─🌐  Leaders & Interviews (context-aware)  ───────────────────────╮
import aiohttp, asyncio, re, html, logging, openai, streamlit as st, tldextract

HEADERS = {"User-Agent": "Mozilla/5.0"}
_URL_PAT = re.compile(r"https?://[^\s)]+")
def _linkify(txt:str)->str:
    return _URL_PAT.sub(lambda m:f'<a href="{html.escape(m.group(0))}" target="_blank">ссылка</a>', txt)

# --- быстрый сниппет Google ---------------------------------------
async def _snip(sess: aiohttp.ClientSession, query:str, n:int=4):
    q = re.sub(r'[\"\'“”]', '', query)[:90]
    params = {"key": KEYS["GOOGLE_API_KEY"], "cx": KEYS["GOOGLE_CX"],
              "q": q, "num": n, "hl": "ru", "gl": "ru"}
    try:
        async with sess.get("https://www.googleapis.com/customsearch/v1",
                             params=params, headers=HEADERS, timeout=8) as r:
            if r.status!=200:
                logging.warning(f"[Google] {r.status}"); return []
            js = await r.json()
            return [(it["link"], it.get("snippet",""))
                    for it in js.get("items",[]) if not _bad(it["link"])]
    except asyncio.TimeoutError:
        logging.warning("[Google] timeout"); return []

# --- контекст-сниппет по домену -----------------------------------
async def _site_snip(sess, domain:str)->str:
    if not domain: return ""
    res = await _snip(sess, f"site:{domain}", n=1)
    return res[0][1] if res else ""

class FastLeadersInterviews:
    """
    Возвращает dict(summary, names, queries, snippets).

    company_info ждёт структуру Checko/FNS:
       • general_director / managers / «Руковод»
       • founders        / «Учред_ФЛ»
    """
    def __init__(self, company: str, *,
                 website: str = "",
                 market:  str = "",
                 company_info: dict | None = None,
                 model: str = "gpt-4o-mini"):

        self.c        = company.strip()
        self.site     = website.strip()
        self.market   = market.strip()
        self.cinfo    = company_info or {}
        self.model    = model

    # ---------- helpers ------------------------------------------------
    def _domain(self) -> str:
        import tldextract
        return tldextract.extract(self.site).registered_domain if self.site else ""

    @staticmethod
    def _fmt_person(p: dict | list | None, default_role: str) -> str | None:
        # ДОБАВИЛИ ключи 'ФИО' и 'ИНН'
        if not p:
            return None
        if isinstance(p, list):
            p = next((d for d in p if isinstance(d, dict) and
                      (d.get("name") or d.get("fio") or d.get("ФИО"))), None)
            if not p:
                return None
        fio  = p.get("name") or p.get("fio") or p.get("ФИО")
        inn  = p.get("inn")  or p.get("ИНН")
        role = p.get("type") or p.get("post") or default_role
        if not fio:
            return None
        inn_txt = f", ИНН {inn}" if inn else ""
        return f"{fio} ({role}{inn_txt})"

    async def _llm_queries(self, prompt: str) -> list[str]:
        """
        Отправляет prompt в GPT-4o (или любую self.model) и
        вытаскивает строки вида  Q: <query>  из ответа.
        """
        raw = await _gpt(
            [{"role": "system", "content": prompt},
             {"role": "user",   "content": ""}],
            model=self.model,
            T=0.14,
        )
        import re
        return re.findall(r"(?:Q|QUERY)[:\-]\s*(.+)", raw, flags=re.I)

    
    # ---------- 1. РУКОВОДИТЕЛИ / ВЛАДЕЛЬЦЫ ---------------------------
    async def _leaders(self, sess):
        # 1) берём уже очищенные списки из self.cinfo
        names = []
        leaders_raw  = self.cinfo.get("leaders_raw")  or []
        founders_raw = self.cinfo.get("founders_raw") or []
        names.extend(leaders_raw)
        names.extend(founders_raw)
    
        # если списки нашлись, ничего больше не делаем
        if names:
            return list(dict.fromkeys(names)), [], [] 

        # 1-B. Если после очистки имена так и не появились → fallback на Google
        if not names:
            # ----------------------------------------------------------- #
            # 1) расширяем список ролей
            roles_kw = [
                # founders / owners
                "основатель", "сооснователь", "owner", "founder",
                # top-management
                "генеральный директор", "гендиректор", "CEO",
                "коммерческий директор", "CCO", "chief commercial officer",
                "директор по маркетингу", "маркетинговый директор", "CMO",
                "финансовый директор", "CFO",
            ]

            # 2) строим запросы двух типов:
            #    а) «кто {роль} "{компания}" "{рынок}"»
            #    б) «"{компания}" {роль}» (+ site:домен, если есть)
            dom   = self._domain()
            mkt   = f' "{self.market}"' if self.market else ""
            g_queries, g_snips = [], []

            for kw in roles_kw:
                g_queries.append(f'кто {kw} "{self.c}"{mkt}')
                plain_q = f'"{self.c}" {kw}' + (f' OR site:{dom}' if dom else "")
                g_queries.append(plain_q)

            # 3) выполняем поиск (≤3 выдачи на запрос, чтобы не шуметь)
            for q in g_queries:
                g_snips += await _google(sess, q, 3)

            # 4) если сниппеты есть — пускаем их через LLM-фильтр
            if g_snips:
                sys = ("Ты проф-аналитик. По сниппетам составь список "
                       "действующих руководителей и владельцев "
                       "(ФИО, должность).")
                llm_txt = await _gpt(
                    [{"role": "system", "content": sys},
                     {"role": "user",
                      "content": "\n".join(f'URL:{u}\nTXT:{t}'
                                           for u, t in g_snips)[:10_000]}],
                    model=self.model, T=0.12,
                )
                names += [ln.strip() for ln in llm_txt.splitlines() if ln.strip()]

        # dedup ---------------------------------------------------------
        uniq, seen = [], set()
        for n in names:
            k = n.lower()
            if k not in seen:
                seen.add(k); uniq.append(n)

        return uniq, g_queries, g_snips

    # ---------- 2. Интервью (оставьте вашу реализацию) -----------------
    async def _interviews(self, names: list[str], sess: aiohttp.ClientSession):
        if not names:
            return [], [], "Свежих интервью не найдено."
    
        dom   = self._domain()
        sc    = await self._site_ctx(sess)
        base_ctx = (f"SITE_CONTEXT:\n{sc}\nрынок компании – {self.market}\n\n"
                    if sc else "")
    
        all_queries, all_snips = [], []
        for fio_role in names:
            fio = fio_role.split("(")[0].strip()
            prompt = (f"Ты — медиа-аналитик. Сформулируй 4-6 Google-запросов, "
                      f"чтобы найти интервью / комментарии «{fio}» "
                      f"из компании «{self.c}». Формат: Q: <query>")
            qlist = await self._llm_queries(prompt)
            for q in qlist:
                full_q = q + (f' OR site:{dom}' if dom and "site:" not in q.lower() else "")
                all_queries.append(full_q)
                all_snips += await _google(sess, full_q, 3)
    
        if not all_snips:
            return all_queries, [], "Свежих интервью не найдено."
    
        ctx = base_ctx + "\n".join(f"URL:{u}\nTXT:{t}" for u, t in all_snips)[:16_000]
    
        sys = ("Ты — контент-аналитик. Составь дайджест релевантных интервью. "
               "Для каждого: ФИО, роль, дата, 1-2 фразы сути, ссылка.")
        digest = await _gpt([{"role": "system", "content": sys},
                             {"role": "user",   "content": ctx}],
                            model=self.model, T=0.18)
        return all_queries, all_snips, digest

    # ------------------------------------------------------------------
    # ---------- orchestrator ------------------------------------------------
    async def _run_async(self):
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)) as sess:
    
            names, q_lead, s_lead = await self._leaders(sess)
            q_int,  s_int, digest = await self._interviews(names, sess)
    
        # --- ① владельцы / топ-менеджеры ------------------------------------
        owners_block = ("Топ-менеджеры и владельцы:\n" + "\n".join(names)
                        if names else "Топ-менеджеры и владельцы не найдены.")
    
        # --- ② контакты ------------------------------------------------------
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
            if lines:
                contacts_block = "Контакты:\n" + "\n".join(lines)
    
        # --- ③ финальное HTML -----------------------------------------------
        body = "\n\n".join([part for part in (owners_block, contacts_block, digest) if part])
        summary_html = _linkify(body)
    
        return {
            "summary":  summary_html,
            "names":    names,
            "queries":  q_lead + q_int,
            "snippets": s_lead + s_int,
        }

    # ---------- публичный sync-интерфейс ------------------------------
    def run(self) -> dict:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():                 # Jupyter / Streamlit-callback
                import nest_asyncio; nest_asyncio.apply()
                return loop.run_until_complete(self._run_async())
        except RuntimeError:
            pass
        return asyncio.run(self._run_async())

    async def _site_ctx(self, sess: aiohttp.ClientSession) -> str | None:
        """
        Возвращает краткий паспорт сайта компании (или пустую строку,
        если self.site не указан). Запускается в отдельном потоке,
        чтобы не блокировать event-loop.
        """
        if not self.site:
            return ""

        loop = asyncio.get_running_loop()
        # _site_passport_sync блокирующий ⇒ отправляем в ThreadPool
        return await loop.run_in_executor(
            None,                              # default ThreadPoolExecutor
            partial(_site_passport_sync, self.site)
        )


# ───────────────────  обёртка для кэша  ─────────────────────────────
@st.cache_data(ttl=86_400,
               show_spinner="🔎 Ищем руководителей и интервью…")
def get_leaders_rag(company: str, *,
                    website: str = "",
                    market:  str = "",
                    company_info: dict | None = None) -> dict:
    """Streamlit-кэш вокруг FastLeadersInterviews."""
    return FastLeadersInterviews(
        company      = company,
        website      = website,
        market       = market,
        company_info = company_info,
    ).run()





# ---------- 1. Универсальный клиент Checko ----------
@st.cache_data(ttl=3_600)
def ck_call(endpoint: str, inn: str):
    """
    Универсальный вызов к Checko API.

    endpoint : 'company', 'finances', 'analytics', …
    inn      : строка ИНН
    """
    url = f"https://api.checko.ru/v2/{endpoint}"
    r = requests.get(
        url,
        params={"key": KEYS["CHECKO_API_KEY"], "inn": inn},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["data"]

# ---------- 2. Тонкие обёртки (по желанию) ----------
ck_company = functools.partial(ck_call, "company")
ck_fin     = functools.partial(ck_call, "finances")
# при желании можно добавить ck_analytics = functools.partial(ck_call, "analytics")



# ---------- 4. Помощник для лидеров / учредителей ----------
def extract_people(cell) -> list[str]:
    """
    Нормализует ячейку «Руковод» / «Учред_ФЛ» и
    возвращает список строк «ФИО (ИНН…, доля …%)».
    """
    # 0) сразу отсекаем None / NaN
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    # 1) если это строка → пробуем распарсить как Python-литерал
    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        try:
            cell = ast.literal_eval(cell)  # '[{…}]' → list | dict | str
        except (ValueError, SyntaxError):
            # просто строка с одним ФИО
            return [cell]

    # 2) одиночный dict → оборачиваем в list
    if isinstance(cell, dict):
        cell = [cell]

    # 3) если это уже list — обрабатываем каждый элемент
    if isinstance(cell, list):
        people = []
        for item in cell:
            if isinstance(item, str):
                people.append(item.strip())
            elif isinstance(item, dict):
                fio  = item.get("ФИО") or item.get("fio") or ""
                inn  = item.get("ИНН") or item.get("inn")
                share = item.get("Доля", {}).get("Процент")
                line = fio
                if inn:
                    line += f" (ИНН {inn}"
                    if share is not None:
                        line += f", доля {share}%)"
                    else:
                        line += ")"
                people.append(line)
        return [p for p in people if p]      # без пустых строк
    # 4) неизвестный тип → оборачиваем в строку
    return [str(cell)]



def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None






import openai, asyncio, nest_asyncio, logging
nest_asyncio.apply()

# кешируем, чтобы при повторных кликах не дергать LLM и сайт заново
@st.cache_data(ttl=86_400, show_spinner=False)
def get_site_passport(url: str) -> dict:
    """Синхронный обёртка SiteRAG.run() с кешированием."""
    if not url:
        return {"summary": "", "chunks_out": [], "html_size": "0", "url": url}
    try:
        return SiteRAG(url).run()
    except Exception as e:
        logging.warning(f"[SiteRAG] {url} → {e}")
        return {"summary": f"(не удалось распарсить сайт: {e})",
                "chunks_out": [], "html_size": "0", "url": url}









def run_ai_insight_tab() -> None:
        # ── 1. «очистка» (если пользователь хочет перезапустить отчёт)
    if st.session_state.get("ai_result_ready"):
        rep = st.session_state["ai_report"]
    
        # --- выводим всё из session_state вместо повторного расчёта ---
        st.markdown(rep["doc"]["summary_rendered_html"], unsafe_allow_html=True)
        st.dataframe(rep["tbl"], use_container_width=True)
        st.pyplot(rep["graphics"])
        # и т.д.
    
        # кнопка «Сбросить и построить заново»
        if st.button("🔄 Построить новый отчёт", type="primary"):
            st.session_state.pop("ai_result_ready", None)
            st.session_state.pop("ai_report", None)
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        return   
        

    # ╭─🎛  UI ──────────────────────────────────────────╮
    st.title("📊 AI Company Insight")
    if st.button("🗑️ Очистить кэш Google"):
        clear_google_cache()
        st.success("Кэш очищен")
    if QUERY_HISTORY:
        with st.expander("🕓 История запросов"):
            for i, q in enumerate(QUERY_HISTORY[-50:], 1):
                st.write(f"{i}. {q}")
    st.markdown("Введите данные (каждая компания — в отдельной строке).")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: inns_raw  = st.text_area("ИНН")          # ✅ без key=* — нам не нужны две копии
    with c2: names_raw = st.text_area("Название")
    with c3: mkts_raw  = st.text_area("Рынок")
    with c4: sites_raw = st.text_area("Сайт")
    with c5: group_sel = st.selectbox("Группа", GROUPS)
    
    aggregate_mode = st.checkbox("🧮 Суммировать финансы по всем ИНН")
    
    if st.button("🔍 Построить отчёт", key="ai_build"):
        with st.spinner("Считаем отчёт…"):

    
            # ---------- парсинг ----------
            split = lambda s: [i.strip() for i in s.splitlines() if i.strip()]
            inns   = split(inns_raw)
            names  = split(names_raw)
            mkts   = split(mkts_raw)
            sites  = split(sites_raw)
            groups = [group_sel] * len(inns)
            
            # ---------- валидация ----------
            if not inns:
                st.error("Укажите хотя бы один ИНН."); st.stop()
            
            if aggregate_mode:            # Σ-режим
                # при необходимости растягиваем одиночные значения
                if len(names) == 1 and len(inns) > 1:  names *= len(inns)
                if len(mkts)  == 1 and len(inns) > 1:  mkts  *= len(inns)
                if len(sites) == 1 and len(inns) > 1:  sites *= len(inns)
                if len(groups) == 1 and len(inns) > 1: groups *= len(inns)
            
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
                if groups and len(groups) != len(inns):
                    st.error("Число строк «Группа» должно совпадать с числом ИНН."); st.stop()
            
            # ---------- выравниваем длины списков ----------
            pad = lambda lst: lst if lst else [""] * len(inns)
            names_full = pad(names)
            mkts_full  = pad(mkts)
            sites_full = pad(sites)
            groups_full = pad(groups)
            YEARS = ["2022", "2023", "2024"]
            df_companies = pd.DataFrame([ck_company(i) for i in inns])

            
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
                                line += f", доля {int(share)}%)" if share is not None else ")"
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
                    
                    # --- единый RAG-пайплайн (Google-сниппеты + сайт) ---------------------
                    st.subheader("📝 Описание компании")
                    with st.spinner("Генерируем описание компании…"):
                        doc = RAG(first_name, website=first_site, market=first_mkt, group=groups_full[0]).run()
                    
                    # ----------- вывод основного отчёта -----------------------------------
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
                    
                    # ----------- отдельная плашка «Паспорт сайта» (если есть) --------------
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
                    
                    # ----------- Рыночный отчёт -------------------------------------------
                    if first_mkt:
                        st.subheader("📈 Рыночный отчёт")
                        with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                            mkt_res = get_market_rag(first_mkt)
                    
                        mkt_html = _linkify(mkt_res["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{mkt_html}</div>",
                            unsafe_allow_html=True,
                        )

                        vols = _parse_market_volumes(mkt_res["summary"])
                        if vols:
                            fig, ax = plt.subplots(figsize=(4, 2))
                            years = list(vols.keys())
                            vals = list(vols.values())
                            bars = ax.bar(range(len(years)), vals, color="#4C72B0")
                            ax.set_xticks(range(len(years)))
                            ax.set_xticklabels(years)
                            ax.set_yticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            for i, b in enumerate(bars):
                                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                                        f"{vals[i]:.1f}", ha="center", va="bottom", fontsize=8)
                            st.pyplot(fig)

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(mkt_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")
                    
                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )
                    
                    # ----------- Руководители и интервью -----------------------------------
                    st.subheader("👥 Руководители и интервью")
                    with st.spinner("Собираем руководителей и интервью…"):
                        # берём Checko-карточку первой компании из готового DataFrame
                        company_info = df_companies.iloc[0].to_dict()
                    
                        lead_res = get_leaders_rag(
                            first_name,
                            website=first_site,
                            market=first_mkt,
                            company_info=company_info,      # ← передаём dict с leaders_raw / founders_raw
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
            
                    
                    
                    # ────── Описание компании (Google + сайт) ───────────────────────────
                    st.subheader("📝 Описание компании")
                    with st.spinner("Генерируем описание компании…"):
                        doc = RAG(name, website=site, market=mkt, group=groups_full[idx]).run()     # ← новая переменная
                    
                    # основной отчёт
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
                    
                    # 🌐 Паспорт сайта (если получился)
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
                    
                    # ────── Рыночный отчёт ───────────────────────────────────────────────
                    if mkt:
                        st.subheader("📈 Рыночный отчёт")
                        with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                            mkt_res = get_market_rag(mkt)

                        mkt_html = _linkify(mkt_res["summary"]).replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{mkt_html}</div>",
                            unsafe_allow_html=True,
                        )

                        vols = _parse_market_volumes(mkt_res["summary"])
                        if vols:
                            fig, ax = plt.subplots(figsize=(4, 2))
                            years = list(vols.keys())
                            vals = list(vols.values())
                            bars = ax.bar(range(len(years)), vals, color="#4C72B0")
                            ax.set_xticks(range(len(years)))
                            ax.set_xticklabels(years)
                            ax.set_yticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            for i, b in enumerate(bars):
                                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                                        f"{vals[i]:.1f}", ha="center", va="bottom", fontsize=8)
                            st.pyplot(fig)

                        with st.expander("⚙️ Запросы к Google"):
                            for i, q in enumerate(mkt_res["queries"], 1):
                                st.markdown(f"**{i}.** {q}")
                    
                        with st.expander("🔍 Сниппеты (top-15)"):
                            st.dataframe(
                                pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                                use_container_width=True,
                            )
                    
                    # ────── Руководители и интервью ─────────────────────────────────────
                    st.subheader("👥 Руководители и интервью")
                    with st.spinner("Собираем руководителей и интервью…"):
                    
                        # ① берём сырые списки руководителей / учредителей из готового df_companies
                        company_info = {
                            "leaders_raw":  df_companies.loc[idx, "leaders_raw"]  or [],
                            "founders_raw": df_companies.loc[idx, "founders_raw"] or [],
                        }
                    
                        # ② запускаем пайплайн
                        lead_res = get_leaders_rag(
                            name,
                            website=site,
                            market=mkt,
                            company_info=company_info,   # ← только нужные ключи
                        )
                    
                    # вывод
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

        st.session_state["ai_report"] = {
            "doc":          doc,          # описание компании
            "mkt_res":      mkt_res,      # рыночный отчёт
            "lead_res":     lead_res,     # руководители/интервью
            "tbl":          tbl,          # фин. таблица DataFrame
            "graphics":     fig,          # объект matplotlib (если нужен повторный рендер)
            # … что-угодно ещё
        }
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


# In[6]:





# In[14]:





# In[13]:





# In[ ]:




