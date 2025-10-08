#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:
#from timesheet_tab import render_timesheet_tab, ensure_db
#ensure_db()  # создаст таблицы в выбранной БД при первом запуске

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
import ast
from timesheet_tab import render_timesheet_tab, ensure_db
ensure_db()  # безопасно дергать при старте (создаст недостающее)

KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
    "DYXLESS_TOKEN": st.secrets["DYXLESS_TOKEN"],
    "SONAR_API_KEY": st.secrets["SONAR_API_KEY"],
}

DYXLESS_TOKEN = KEYS["DYXLESS_TOKEN"]

try:
    st.cache_data.clear()
except Exception:
    pass

def _no_cache(*args, **kwargs):
    def _decorator(func):
        return func
    return _decorator

st.cache_data = _no_cache

# --- единая константа годов (в модуле, не внутри функций!) ---
YEARS = ["2022", "2023", "2024"]

# ── Общие константы (единые для всего файла)
HEADERS = {"User-Agent": "Mozilla/5.0 (Win64) AppleWebKit/537.36 Chrome/125 Safari/537.36"}
_URL_PAT = re.compile(r"https?://[^\s)<>\"']+", flags=re.I)

# Две чёткие версии linkify, чтобы не конфликтовать:
def linkify_as_word(text: str, label: str = "ссылка") -> str:
    """Заменяет URL на <a>label</a> (краткая версия)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _URL_PAT.sub(lambda m: f'<a href="{html.escape(m.group(0))}" target="_blank" rel="noopener">{label}</a>', text)

def linkify_keep_url(text: str) -> str:
    """Заменяет URL на <a>сам URL</a> (полная версия)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _URL_PAT.sub(lambda m: f'<a href="{html.escape(m.group(0))}" target="_blank" rel="noopener">{html.escape(m.group(0))}</a>', text)


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






async def _gpt(messages, *, model="gpt-4o-mini", T=0.2):
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

# ╭─🧾  INVEST SNAPSHOT (cheap, 1 call) — адреса мощностей, соцсети, новости, интервью, конкуренты, headcount ─╮
import json, requests, re
from typing import Optional

API_URL_INVEST = "https://api.perplexity.ai/chat/completions"

class PPLXError(Exception):
    ...

def _pplx_call_invest(
    prompt: str,
    model: str = "sonar",            # дешёвая модель
    recency: Optional[str] = None,   # None = широкий охват (5 лет)
    temperature: float = 0.0,
    max_tokens: int = 1500,
    timeout: int = 60,
) -> str:
    key = (os.getenv("SONAR_API_KEY") or os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY") or "").strip()
    assert key.startswith("pplx-"), "Установи SONAR_API_KEY = pplx-..."
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":(
                "Ты — аналитик M&A. Охват 5 лет. Строго факты из открытых источников. "
                "Запрещено упоминать финансовые показатели (выручка, прибыль, EBITDA и т.п.), "
                "ИНН/ОГРН и любые выводы на их основе. Можно упоминать имена владельцев/руководителей "
                "ТОЛЬКО в контексте ссылок на их интервью. "
                "Если данных по пункту нет — пиши 'нет данных'. Не повторяй одну и ту же информацию в разных разделах. "
                "В конце укажи прямые URL источников."
            )},
            {"role":"user","content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if recency in {"hour","day","week","month","year"}:
        payload["search_recency_filter"] = recency

    r = requests.post(API_URL_INVEST, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        try: err = r.json()
        except Exception: err = {"error": r.text}
        raise PPLXError(f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:900]}")
    return r.json()["choices"][0]["message"]["content"]

# фильтр запретных тем: финпоказатели и рег.номера (разрешаем владельцев только в интервью)
_FORBIDDEN_INVEST = re.compile(
    r"(\bвыручк|\bприбыл|\bebit(?:da)?\b|маржинал|рентабельн|финанс|\bинн\b|\bогрн\b|уставн\w*\s+капитал)",
    re.IGNORECASE
)

def _dedup_lines_invest(text: str) -> str:
    """Удалить точные повторы строк и повторяющиеся URL в строках, сохранить порядок."""
    seen_lines, seen_urls, out = set(), set(), []
    url_re = re.compile(r'https?://\S+')
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        urls = url_re.findall(ln)
        for u in urls:
            if u in seen_urls:
                ln = ln.replace(u, "")
            else:
                seen_urls.add(u)
        ln = re.sub(r'\s{2,}', ' ', ln).strip()
        if ln and ln not in seen_lines:
            out.append(ln); seen_lines.add(ln)
    cleaned = []
    for i, ln in enumerate(out):
        if ln == "" and (i == 0 or (i+1 < len(out) and out[i+1] == "")):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def sanitize_invest(text: str) -> str:
    """Убираем строки с финпоказателями/ИНН/ОГРН/уставным капиталом и чистим повторы."""
    keep = []
    for ln in text.splitlines():
        if _FORBIDDEN_INVEST.search(ln):
            continue
        keep.append(ln)
    return _dedup_lines_invest("\n".join(keep))

def build_invest_prompt(company: str, site_hint: Optional[str] = None) -> str:
    site = f"\nВозможный официальный сайт: {site_hint}." if site_hint else ""
    return f"""
Сделай структурированный отчёт по компании «{company}» на русском (охват: 5 лет).{site}
Формат строго Markdown с заголовками уровня ### и короткими абзацами (без списков). Не дублируй факты между разделами.

### Профиль
1–2 предложения: что за компания, чем занимается (категории товаров/услуг), география. Если есть одноимённые фирмы — кратко укажи дизамбигуацию по профилю/сайту.

### Бизнес-модель
Как компания зарабатывает: каналы (собственные продажи/дистрибуция/опт/розница/маркетплейсы/дилеры), сервисные модели, подписки/сервисные контракты, интеграции. Без цифр и оценок — только явные формулировки из источников (если есть).

### Активы и площадки
Адреса всех выявленных объектов (офисы, склады, РЦ, магазины/ПВЗ). Площади (м²) и характеристики (кол-во РЦ/складов, собственные/арендованные) — только если прямо указано в источниках.

### Производственные мощности и адреса
Укажи наличие/отсутствие собственного производства. Приведи АДРЕСА производственных площадок/цехов/комбинатов и по каждой — что производится. Мощности (ед./м²/тонн/месяц) и степень загрузки — если раскрыты. Локации логистики и складской сети — кратко.

### Инвестиционные планы и проекты
Заявленные/ожидаемые инвестиции в новые линии/склады/РЦ/площадки; сроки и статус. Ссылки на первоисточники в скобках.

### Клиенты и каналы сбыта
B2B/B2C; сегменты/вертикали; продажи через сайт/магазины/маркетплейсы/дилеров; ориентиры по числу клиентов — только если публично и с источником.

### Численность персонала
Численность сотрудников/штата (если публично), дата/период и источник в скобках. Если только оценки — передай формулировку источника.

### Конкуренты (Россия)
Перечисли 5–12 релевантных российских конкурентов по профилю бизнеса; формат: Название (официальный сайт) в одном абзаце через запятую. Только компании, реально работающие в РФ за последние 5 лет.

### Новости (последние 5 лет)
Дай 5–12 значимых новостей о компании: «Заголовок» — URL (дата). В одном абзаце; не повторяй ссылки.

### Интервью (владельцы/руководство)
Дай 3–8 релевантных интервью/публичных разговоров: «Спикер — Заголовок/тема» — URL (дата). В одном абзаце. Не раскрывай доли/структуру владения, только факт интервью.

### Цифровые каналы и контакты
Сайт (если есть), e-mail/телефоны (если публично), ВСЕ найденные соцсети (VK, Telegram, YouTube, RuTube, OK, Instagram*, Facebook*, LinkedIn), каталоги/карты (2ГИС, Яндекс.Карты, Google Maps) — приводи ПРЯМЫЕ URL.

### Источники
Перечисли все использованные прямые URL через запятую, без нумерации.

Требования:
— Не указывай финансовые метрики (выручка/прибыль/EBITDA и т.п.), ИНН/ОГРН/уставный капитал и выводы на их основе.
— Строго избегай повторов между разделами и повторов ссылок.
— Только подтверждаемые факты из открытых источников; если данных нет — 'нет данных'.
""".strip()

def invest_snapshot(company: str, site_hint: Optional[str] = None,
                    model: str = "sonar", recency: Optional[str] = None,
                    max_tokens: int = 1500) -> str:
    prompt = build_invest_prompt(company, site_hint=site_hint)
    raw = _pplx_call_invest(prompt, model=model, recency=recency, max_tokens=max_tokens)
    return sanitize_invest(raw)

@st.cache_data(ttl=86_400, show_spinner="📝 Собираем описание компании…")
def get_invest_snapshot(company: str,
                        site_hint: Optional[str] = None,
                        model: str = "sonar",
                        recency: Optional[str] = None,
                        max_tokens: int = 1500) -> dict:
    """
    Возвращает dict: {'md': markdown_text, 'raw': raw_text_for_debug}
    """
    try:
        md = invest_snapshot(company, site_hint=site_hint, model=model, recency=recency, max_tokens=max_tokens)
        return {"md": md, "raw": md}
    except PPLXError as e:
        return {"md": f"_Не удалось получить INVEST SNAPSHOT: {e}_", "raw": ""}
# ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯





# ╭─👥 Интервью: обогащаем INVEST SNAPSHOT лицами из Checko (Sonar-only) ─╮
#   • Этап 1 — интервью по ФИО из Checko
#   • Этап 2 — если нет/мало ФИО → discovery ФИО через Sonar
#   • Подмена секции "### Интервью (владельцы/руководство)" в готовом Markdown
# Требует: _pplx_call_invest, sanitize_invest из блока INVEST SNAPSHOT
# ╰───────────────────────────────────────────────────────────────────────╯
import re, html
from typing import Optional

def _clean_person(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)            # убираем хвост в скобках: (ИНН…, доля…)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _names_from_checko(company_info: dict | None) -> list[str]:
    if not isinstance(company_info, dict):
        return []
    raw = []
    for key in ("leaders_raw", "founders_raw"):
        v = company_info.get(key) or []
        if isinstance(v, list):
            raw.extend([str(x) for x in v if x])
        elif isinstance(v, str):
            raw.append(v)
    out, seen = [], set()
    for p in raw:
        fio = _clean_person(p)
        k = fio.lower()
        if fio and k not in seen:
            seen.add(k); out.append(fio)
    return out

def _domain_from_site(site_hint: str | None) -> str:
    if not site_hint:
        return ""
    m = re.search(r"^(?:https?://)?([^/]+)", site_hint.strip(), re.I)
    return (m.group(1) if m else "").lower()

def _extract_urls(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'https?://[^\s<>)"\'\]]+', text or "")))

def _build_people_discovery_prompt(company: str,
                                   site_hint: str | None,
                                   market: str | None) -> str:
    dom = _domain_from_site(site_hint)
    mkt = f"(рынок: {market}). " if market else ""
    site_line = f"Официальный сайт (если верно): {site_hint}. " if site_hint else ""
    return f"""
Найди действующих руководителей и/или основателей компании «{company}». {mkt}{site_line}
Охват 5 лет. Только подтверждённые факты с ПРЯМЫМИ URL.

Формат вывода — только строки:
PERSON: <ФИО> — <должность/роль> — <прямой URL на источник>

Требования:
— Не указывать ИНН/ОГРН, доли, структуру владения и финансовые показатели.
— Предпочтительно источники: официальный сайт{(' ('+dom+')' if dom else '')}, СМИ, профильные медиа, видео/подкасты, соцсети компании.
— Если данных нет — выведи «PERSON: нет данных».
""".strip()

def _parse_people_lines(text: str) -> list[str]:
    if not text:
        return []
    ppl = []
    for ln in text.splitlines():
        m = re.match(r"\s*PERSON:\s*(.+?)\s+—\s+.+?\s+—\s+https?://", ln.strip(), re.I)
        if m:
            fio = _clean_person(m.group(1))
            if fio:
                ppl.append(fio)
    return list(dict.fromkeys(ppl))

def _build_interviews_prompt(company: str,
                             names: list[str],
                             site_hint: str | None,
                             market: str | None) -> str:
    names_block = "; ".join(names[:10]) or "—"
    site_line = f"Официальный сайт: {site_hint}. " if site_hint else ""
    mkt = f"(рынок: {market})" if market else ""
    return f"""
Ты — медиа-аналитик. Найди интервью/публичные разговоры по людям [{names_block}] из компании «{company}» {mkt}.
{site_line}Охват 5 лет. Только подтверждаемые факты и ПРЯМЫЕ URL. Никаких ИНН/ОГРН/финансов.

Формат вывода — ОДИН абзац (без списков):
«ФИО — площадка/издание — краткая суть (1 фраза) — URL (YYYY-MM-DD)»;
записи разделяй точкой с запятой «;», не повторяй ссылки. Всего 3–8 записей.
В конце абзаца через пробел добавь: «Источники: <URL1>, <URL2>, ...» (уникальные).
""".strip()

_SEC_INTERV_RE = re.compile(r"(^|\n)###\s*Интервью[^\n]*\n.*?(?=\n###\s|\Z)",
                            flags=re.S | re.I)

def _replace_interviews_section(md: str, new_paragraph: str) -> str:
    block = f"\n### Интервью (владельцы/руководство)\n{new_paragraph.strip()}\n"
    if _SEC_INTERV_RE.search(md or ""):
        return _SEC_INTERV_RE.sub(block, md, count=1)
    # если раздела нет — вставим перед «Цифровые каналы» или в конец
    m = re.search(r"(^|\n)###\s*Цифровые\s+каналы[^\n]*", md or "", flags=re.I)
    if m:
        return md[:m.start()] + block + md[m.start():]
    return (md or "").rstrip() + block

def interviews_from_checko_sonar(company: str,
                                 company_info: dict | None = None,
                                 site_hint: str | None = None,
                                 market: str | None = None,
                                 max_people_discovery: int = 6) -> tuple[list[str], str]:
    """
    Возвращает (names, paragraph_markdown).
    names — финальный список ФИО, paragraph_markdown — один абзац с интервью.
    """
    # 1) имена из Checko
    names = _names_from_checko(company_info)

    # 2) если имён нет/мало — discovery через Sonar
    if len(names) < 2:
        try:
            p_disc = _build_people_discovery_prompt(company, site_hint, market)
            raw = _pplx_call_invest(p_disc, model="sonar", recency=None, max_tokens=900)
            discovered = _parse_people_lines(raw)
        except Exception:
            discovered = []
        for fio in discovered:
            if fio.lower() not in {n.lower() for n in names}:
                names.append(fio)
        names = names[:max_people_discovery] or ["нет данных"]

    # 3) интервью по итоговому списку
    try:
        p_int = _build_interviews_prompt(company, names, site_hint, market)
        digest = _pplx_call_invest(p_int, model="sonar", recency=None, max_tokens=1400)
    except Exception as e:
        digest = f"нет данных (ошибка поиска интервью: {e})"

    # дедуп URL внутри абзаца
    def _urls_in(t: str) -> list[str]: return _extract_urls(t)
    seen = set(); parts = []
    for part in re.split(r"\s*;\s*", (digest or "").strip()):
        u = next(iter(_urls_in(part)), None)
        if not u or u not in seen:
            parts.append(part.strip())
            if u: seen.add(u)
    paragraph = "; ".join(parts)
    paragraph = sanitize_invest(paragraph)   # фильтр финансов/ИНН/ОГРН

    return names, paragraph

def invest_snapshot_enriched(
    company: str,
    site_hint: Optional[str] = None,
    company_info: dict | None = None,
    market: str | None = None,
    model: str = "sonar",
    recency: Optional[str] = None,
    max_tokens: int = 1500,
) -> str:
    """
    1) Делает обычный INVEST SNAPSHOT (sonar).
    2) Собирает интервью в два шага:
       — сначала по ФИО из Checko,
       — затем ищет ФИО в интернете и дополняет.
    3) Подменяет секцию «### Интервью (владельцы/руководство)»
       блоком с ФИО и двумя дайджестами.
    """
    # 1) базовый отчёт
    base_md = invest_snapshot(
        company, site_hint=site_hint, model=model, recency=recency, max_tokens=max_tokens
    )

    # 2) «двойные интервью»: Checko → интернет
    dual = build_dual_interviews_from_v2(
        company,
        company_info=company_info,
        site_hint=site_hint,
        market=market,
        max_people_inet=8,
    )
    names_checko = ", ".join(dual.get("names_checko") or []) or "нет данных"
    names_inet   = ", ".join(dual.get("names_inet")   or []) or "нет данных"

    # 3) формируем текст новой секции (Markdown, без HTML)
    parts: list[str] = []
    parts.append(f"**ФИО (Checko):** {names_checko}")
    parts.append(f"**ФИО (интернет):** {names_inet}")

    digest_checko = (dual.get("digest_checko") or "").strip()
    digest_inet   = (dual.get("digest_inet")   or "").strip()

    if digest_checko and digest_checko.lower() != "нет данных":
        parts.append("")
        parts.append("**Дайджест интервью — Checko**")
        parts.append(digest_checko)

    if digest_inet and digest_inet.lower() != "нет данных":
        parts.append("")
        parts.append("**Дайджест интервью — интернет**")
        parts.append(digest_inet)

    # фильтруем финпоказатели/ИНН/ОГРН на всякий
    new_block = "\n".join(parts).strip()
    
    # 4) подменяем/вставляем секцию «Интервью»
    return _replace_interviews_section(base_md, new_block)


@st.cache_data(ttl=86_400, show_spinner="📝 Собираем описание (enriched)…")
def get_invest_snapshot_enriched(
    company: str,
    site_hint: Optional[str] = None,
    company_info: dict | None = None,
    market: str | None = None,
    model: str = "sonar",
    recency: Optional[str] = None,
    max_tokens: int = 1500,
) -> dict:
    try:
        md = invest_snapshot_enriched(
            company,
            site_hint=site_hint,
            company_info=company_info,
            market=market,
            model=model,
            recency=recency,
            max_tokens=max_tokens,
        )
        return {"md": md, "raw": md}
    except PPLXError as e:
        return {"md": f"_Не удалось получить INVEST SNAPSHOT (enriched): {e}_", "raw": ""}







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


# ╭─🧾  MARKET EVIDENCE (Perplexity) ───────────────────────────────╮
#   • Абзацы-источники + финальная сводка + 2 блока "СТРУКТУРА"
#   • Синхронный requests-вызов Perplexity, аккуратный вывод в Streamlit
# ╰────────────────────────────────────────────────────────────────╯
import os
import re
import json
import requests
import streamlit as st
from typing import Optional, Tuple, Dict

API_URL_PPLX = "https://api.perplexity.ai/chat/completions"


class PPLXError(Exception):
    pass


# Слова, которые не выводим (юр-сущности)
_FORBIDDEN = re.compile(r"(акционер|владельц|бенефициар|инн|огрн)", re.IGNORECASE)


def _sanitize_evidence(text: str) -> str:
    """Фильтрует запрещённые строки и чистит лишние пустые."""
    lines, out, blank = [], [], False
    for ln in (text or "").splitlines():
        if _FORBIDDEN.search(ln):
            continue
        lines.append(ln)
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln.rstrip())
            blank = False
    return "\n".join(out).strip()


def _get_pplx_key() -> str:
    key = (os.getenv("SONAR_API_KEY") or os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY") or "").strip()
    if (not key.startswith("pplx-")) or (len(key) < 40) or key.endswith("..."):
        raise PPLXError("Перепроверь Perplexity API key: должен начинаться с 'pplx-' и быть длинным (обычно >40 символов).")
    return key


def _call_pplx(
    prompt: str,
    *,
    model: str = "sonar",
    recency: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    timeout: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {_get_pplx_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "adv-market-evidence/1.1",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — аналитик рынков. Отвечай строго фактологично, с ПРЯМЫМИ URL. "
                    "НЕ упоминай владельцев/акционеров/бенефициаров, ИНН/ОГРН."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if recency in {"hour", "day", "week", "month", "year"}:
        payload["search_recency_filter"] = recency

    r = requests.post(API_URL_PPLX, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text[:800]}
        raise PPLXError(f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:800]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def build_market_evidence_prompt(
    market: str,
    country: str = "Россия",
    years_force: tuple = (2021, 2022, 2023, 2024),
    min_sources: int = 6,
) -> str:
    years_txt = ", ".join(str(y) for y in years_force)
    # ДВА ЯВНЫХ ЗАВЕРШАЮЩИХ FENCED-БЛОКА: ```text ... ```
    return f"""
Собери "evidence" по рынку «{market}» (страна: {country}) из разных ОТКРЫТЫХ источников.

ФОРМАТ ОТВЕТА — СТРОГО ТЕКСТ С АБЗАЦАМИ (БЕЗ СПИСКОВ/ТАБЛИЦ/CSV):
— Каждый НОВЫЙ АБЗАЦ посвящён ОДНОМУ ресурсу (источнику): начинай так — «Источник: <издатель/название>, <год/дата>, URL: <прямой_линк>.»
— Внутри абзаца НЕ используй маркеры. Пиши компактно, но включай ВСЕ найденные ЧИСЛА по рынку с единицами:
   • годовые объёмы рынка в деньгах (предпочтительно ₽; если только $, всё равно включай и помечай как $);
   • годовые NATURAL-объёмы (шт., м², т, посещения и т.п.), если есть;
   • если есть сегменты/регионы — кратко добавь ключевые цифры.
— По ГОДАМ {years_txt} старайся дать значения, если у источника они есть; если нет — явно напиши «за {years_txt} у источника: нет данных».
— У КАЖДОГО факта — ПРЯМОЙ URL в этом же абзаце.
— Минимум источников: {min_sources}. Разные домены/издатели (новости/аналитика/отчёты/госстат/профильные медиа).

Заключительный абзац (последний):
— Сформируй общую картину по найденным РЯДАМ И ПРОГНОЗАМ: перечисли, какие серии годовых значений мы получили (кто издатель/валюта/периметр), выстрой ХРОНОЛОГИЮ 2021→2024 и прогнозы (с единицами и ссылками), отметь расхождения (baseline vs альтернативы) и ограничения.

СТРУКТУРА (сводная матрица; ДОБАВЬ ПОСЛЕ заключительного абзаца) — ДВА fenced-блока:
1) Деньги:
```text
Источник | Период 1 | Период 2 | Период 3 | ...
<краткое_название_источника> | <YYYY>: <число> <валюта/масштаб> | <YYYY>: <число> <валюта/масштаб> | ...
2) Натуральные:
Источник | Период 1 | Период 2 | Период 3 | ...
<краткое_название_источника> | <YYYY>: <число> <ед.изм.> | <YYYY>: <число> <ед.изм.> | ...
[если у источника нет натуральных оценок — укажи «нет данных» одной ячейкой]
Требования к стилю:
— Только абзацы и два завершающих fenced-блока ```text, без списков/нумерации/таблиц/CSV.
— Единицы и валюта всегда рядом с числом (например, «млрд ₽», «$ млн», «тыс. посещений», «м²»).
— Не выдумывай числа — только подтверждаемые факты с ПРЯМЫМИ URL.
— Без упоминания владельцев/акционеров/ИНН/ОГРН.
""".strip()

def market_evidence_report(
    market: str,
    country: str = "Россия",
    min_sources: int = 6,
    model: str = "sonar",
    recency: Optional[str] = None,
    max_tokens: int = 1800,
) -> str:
    assert isinstance(min_sources, int) and 3 <= min_sources <= 15, "min_sources ∈ [3, 15]"
    prompt = build_market_evidence_prompt(market, country=country, min_sources=min_sources)
    raw = _call_pplx(prompt, model=model, recency=recency, max_tokens=max_tokens)
    return _sanitize_evidence(raw)


def _split_evidence_blocks(raw_text: str) -> Tuple[str, str, str]:
    if not raw_text:
        return "", "", ""
    blocks = [m.group(1) for m in re.finditer(r"```text\s*(.*?)\s*```", raw_text, flags=re.S|re.I)]
    money_block   = blocks[0].strip() if len(blocks) > 0 else ""
    natural_block = blocks[1].strip() if len(blocks) > 1 else ""
    plain = re.sub(r"```text\s*.*?\s*```", "", raw_text, flags=re.S|re.I).strip()
    return plain, money_block, natural_block


def _linkify(text: str) -> str:
    """
    Заменяет http/https ссылки на кликабельные <a>.
    Не трогаем остальной HTML.
    """
    url_re = re.compile(r"(https?://[^\s<>)\"']+)")
    return url_re.sub(r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>', text)


@st.cache_data(ttl=86_400, show_spinner="🔎 Собираем рыночные EVIDENCE…")
def get_market_evidence(
    market: str,
    country: str = "Россия",
    min_sources: int = 6,
    model: str = "sonar",
    recency: Optional[str] = None,
    max_tokens: int = 1800,
) -> Dict[str, str]:
    """
    Streamlit-кэш: возвращает dict с:
    • text_html — весь текст (абзацы-источники + финальный абзац), ссылки кликабельны
    • money_block / natural_block — содержимое двух матриц для st.code(..., language="text")
    • raw_text — исходный текст (для отладки)
    """
    try:
        raw = market_evidence_report(
            market=market,
            country=country,
            min_sources=min_sources,
            model=model,
            recency=recency,
            max_tokens=max_tokens,
        )
    except PPLXError as e:
        return {
            "text_html": f"<i>Не удалось получить MARKET EVIDENCE: {str(e)}</i>",
            "money_block": "",
            "natural_block": "",
            "raw_text": "",
        }

    plain, money, natural = _split_evidence_blocks(raw)
    text_html = linkify_keep_url(plain).replace("\n", "<br>")

    return {"text_html": text_html, "money_block": money, "natural_block": natural, "raw_text": raw}






# === Leaders & Interviews (2-pass + union, Sonar-only, no cache) ==============
# Использует: re, html, typing, _pplx_call_invest
import re, html
from typing import Optional, List, Dict, Tuple

_URL_RE = re.compile(r'https?://[^\s<>)"\'\]]+')

# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: Optional[str]) -> str:
    import re
    return re.sub(r"\s{2,}", " ", (s or "").strip())

def _extract_urls(text: str) -> List[str]:
    """Извлекает уникальные URL в порядке появления."""
    return list(dict.fromkeys(_URL_RE.findall(text or "")))

def _dedup_urls_in_paragraph(paragraph: str) -> str:
    """
    Режем по «;», оставляем первую запись с каждым уникальным URL.
    Возвращаем склеенный абзац тем же разделителем.
    """
    seen, out = set(), []
    for part in re.split(r"\s*;\s*", (paragraph or "").strip()):
        if not part:
            continue
        u = next(iter(_extract_urls(part)), None)
        if (not u) or (u not in seen):
            out.append(part.strip())
            if u:
                seen.add(u)
    return "; ".join(out)

def _clean_person(s: str) -> str:
    """Убирает хвосты в скобках и лишние пробелы: 'Иванов (ИНН..., доля...)' → 'Иванов'."""
    s = (s or "").strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _domain_from_site(site_hint: Optional[str]) -> str:
    if not site_hint:
        return ""
    m = re.search(r"^(?:https?://)?([^/]+)", site_hint.strip(), re.I)
    return (m.group(1) if m else "").lower()

# ─────────────────────────────────────────────────────────────────────────────
# Доли / проценты (строгая трактовка для таблицы)
# ─────────────────────────────────────────────────────────────────────────────
def _to_float_safe(x) -> Optional[float]:
    """
    Оставляем для совместимости (НЕ использовать для долей).
    Приводит строку/число к float без логики *100.
    """
    try:
        if x is None:
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _to_percent_strict(x) -> Optional[float]:
    """
    '25', '25%', '12,5' -> 25.0
    0.25 -> 0.25 (это 0.25%, НИЧЕГО не домножаем).
    """
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith("%"):
            s = s[:-1]
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None

def _share_from_checko_dict(item: dict) -> Optional[float]:
    """
    Строго берём item['Доля']['Процент'] или плоскую 'Доля' как проценты.
    Никаких эвристик.
    """
    d = item.get("Доля")
    if isinstance(d, dict) and ("Процент" in d):
        return _to_percent_strict(d.get("Процент"))
    if "Доля" in item and not isinstance(d, dict):
        return _to_percent_strict(item.get("Доля"))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Разбор Checko-ячейки в нормализованные записи персон
# ─────────────────────────────────────────────────────────────────────────────
def _parse_checko_cell(cell, role_hint: Optional[str] = None) -> List[Dict]:
    """
    Превращает leaders_raw / founders_raw в список словарей:
      {'fio','inn','share_pct','role'}
    ЛОГИКА ДОЛЕЙ — строго как в Checko: 'Доля.Процент' (или плоская 'Доля').
    """
    import ast
    out: List[Dict] = []

    def _emit(fio=None, inn=None, share=None, role=None):
        item = {
            "fio": _norm(fio),
            "inn": _norm(str(inn)) if inn else None,
            "share_pct": _to_percent_strict(share),
            "role": _norm(role or role_hint),
        }
        if item["fio"] or item["inn"]:
            out.append(item)

    if cell is None:
        return out

    # строка → пытаемся распарсить как литерал ([{…}] / {...}), иначе вытаскиваем хвосты
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return out
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = ast.literal_eval(s)
                return _parse_checko_cell(parsed, role_hint=role_hint)
            except Exception:
                pass
        m_inn = re.search(r"(?:ИНН|inn)\s*[:№]?\s*([0-9]{8,12})", s, re.I)
        inn = m_inn.group(1) if m_inn else None
        m_share = re.search(r"(?:доля|share)[^0-9]*([0-9]+[.,]?[0-9]*)\s*%?", s, re.I)
        share = m_share.group(1) if m_share else None
        fio = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
        _emit(fio=fio, inn=inn, share=share)
        return out

    # dict → читаем расширенный набор ключей
    if isinstance(cell, dict):
        fio   = cell.get("ФИО") or cell.get("fio") or cell.get("name")
        inn   = cell.get("ИНН") or cell.get("inn") or cell.get("tax_id")
        share = _share_from_checko_dict(cell)
        role  = cell.get("Должность") or cell.get("role") or role_hint
        _emit(fio=fio, inn=inn, share=share, role=role)
        return out

    # list → рекурсивно
    if isinstance(cell, list):
        for it in cell:
            out.extend(_parse_checko_cell(it, role_hint=role_hint))
        return out

    # fallback
    _emit(fio=str(cell))
    return out

def _pick_ceo(leaders: List[Dict], names_fallback: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Пытаемся найти гендиректора:
      1) по 'генераль';
      2) по 'директор'/'руковод'/'ceo';
      3) иначе первый из leaders;
      4) иначе фолбэк по первому имени.
    """
    for p in leaders:
        r = (p.get("role") or "").lower()
        if "генераль" in r or "ген. дир" in r or "гендир" in r or "general director" in r:
            return p
    for p in leaders:
        r = (p.get("role") or "").lower()
        if any(k in r for k in ("директор", "руковод", "ceo")):
            return p
    if leaders:
        return leaders[0]
    if names_fallback:
        return {"fio": names_fallback[0], "inn": None, "share_pct": None, "role": "руководитель"}
    return None

def _shareholders_from_founders(founders: List[Dict]) -> List[Dict]:
    """
    Готовим таблицу акционеров:
      — приводим доли строго (без *100),
      — auto-scale: если все ≤1 и сумма ≤1.5 — считаем доли от 1 → *100,
      — сортируем по убыванию.
    """
    rows_raw = []
    for p in founders:
        fio = (p.get("fio") or "").strip()
        inn = p.get("inn")
        share_raw = _to_percent_strict(p.get("share_pct"))
        rows_raw.append({"fio": fio, "inn": inn, "share_pct": share_raw})

    vals = [r["share_pct"] for r in rows_raw if r["share_pct"] is not None]
    scale = 1.0
    if vals:
        max_v = max(vals)
        ssum = sum(vals)
        if 0 < max_v <= 1.0 and 0 < ssum <= 1.5:
            scale = 100.0

    rows = []
    for r in rows_raw:
        v = r["share_pct"]
        rows.append({
            "fio": r["fio"],
            "inn": r["inn"],
            "share_pct": (v * scale) if v is not None else None
        })

    with_share = [r for r in rows if r["share_pct"] is not None]
    no_share   = [r for r in rows if r["share_pct"] is None]
    with_share.sort(key=lambda x: x["share_pct"], reverse=True)
    return with_share + no_share

def _markdown_shareholders_table(rows: List[Dict]) -> str:
    if not rows:
        return "_нет данных_"
    lines = ["| ФИО | ИНН | Доля, % |", "|---|---|---|"]
    for r in rows:
        fio  = r.get("fio") or ""
        inn  = r.get("inn") or ""
        val  = r.get("share_pct")
        share = "" if val is None else f"{float(val):.2f}"
        lines.append(f"| {fio} | {inn} | {share} |")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Имена из Checko (богатая версия → leaders/founders + dedup ФИО)
# ─────────────────────────────────────────────────────────────────────────────
def _names_from_checko_rich(company_info: Optional[Dict]) -> Tuple[List[Dict], List[Dict], List[str]]:
    leaders, founders = [], []
    if isinstance(company_info, dict):
        leaders = _parse_checko_cell(company_info.get("leaders_raw"), role_hint="руководитель")
        founders = _parse_checko_cell(company_info.get("founders_raw"), role_hint="акционер/учредитель")
    names, seen = [], set()
    for p in leaders + founders:
        fio = (p.get("fio") or "").strip()
        if fio:
            k = fio.lower()
            if k not in seen:
                seen.add(k); names.append(fio)
    return leaders, founders, names

def _names_from_checko(company_info: Optional[Dict]) -> List[str]:
    """
    Лёгкий режим: достаём строки из leaders_raw/founders_raw и чистим «(ИНН…, доля …)».
    Используется как бэкап.
    """
    if not isinstance(company_info, dict):
        return []
    raw: List[str] = []
    for k in ("leaders_raw", "founders_raw"):
        v = company_info.get(k) or []
        if isinstance(v, list):
            raw.extend([_clean_person(str(x)) for x in v if x])
        elif isinstance(v, str):
            raw.append(_clean_person(v))
    out, seen = [], set()
    for fio in raw:
        key = fio.lower()
        if fio and key not in seen:
            seen.add(key); out.append(fio)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Промпты/парсинг для дискавери персон и интервью
# ─────────────────────────────────────────────────────────────────────────────
def _build_people_discovery_prompt(company: str,
                                   site_hint: Optional[str],
                                   market: Optional[str]) -> str:
    dom = _domain_from_site(site_hint)
    mkt = f"(рынок: {market}). " if market else ""
    site_line = f"Официальный сайт (если верно): {site_hint}. " if site_hint else ""
    pref = f"— Предпочтительно источники: официальный сайт{(' ('+dom+')' if dom else '')}, СМИ, профильные медиа, видео/подкасты, соцсети компании."
    return f"""
Найди действующих руководителей и/или основателей компании «{company}». {mkt}{site_line}
Охват 5 лет. Только подтверждённые факты с ПРЯМЫМИ URL.

Формат вывода — только строки:
PERSON: <ФИО> — <должность/роль> — <прямой URL на источник>

Требования:
— Не указывать ИНН/ОГРН, доли, структуру владения и финпоказатели.
{pref}
— Если данных нет — выведи «PERSON: нет данных».
""".strip()

# Терпимый к разным тире/дефисам парсер
_PERSON_LINE_RE = re.compile(r"\s*PERSON:\s*(.+?)\s+[—–-]\s+.+?\s+[—–-]\s+https?://", re.I)

def _parse_people_lines(text: str) -> List[str]:
    if not text:
        return []
    ppl: List[str] = []
    for ln in text.splitlines():
        if "нет данных" in ln.lower():
            continue
        m = _PERSON_LINE_RE.match(ln.strip())
        if m:
            fio = _clean_person(m.group(1))
            if fio:
                ppl.append(fio)
    return list(dict.fromkeys(ppl))

def _build_interviews_prompt(company: str,
                             names: List[str],
                             site_hint: Optional[str],
                             market: Optional[str]) -> str:
    names_block = "; ".join(names[:10]) or "—"
    site_line = f"Официальный сайт: {site_hint}. " if site_hint else ""
    mkt = f"(рынок: {market})" if market else ""
    return f"""
Ты — медиа-аналитик. Найди интервью/публичные разговоры по людям [{names_block}] из компании «{company}» {mkt}.
{site_line}Охват 5 лет. Только подтверждаемые факты и ПРЯМЫЕ URL. Никаких ИНН/ОГРН/финансов.

Формат вывода — ОДИН абзац (без списков):
«ФИО — площадка/издание — краткая суть (1 фраза) — URL (YYYY-MM-DD)»;
записи разделяй точкой с запятой «;», не повторяй ссылки. Всего 3–8 записей.
В конце абзаца через пробел добавь: «Источники: <URL1>, <URL2>, ...» (уникальные).
""".strip()

# Мягкая санитизация интервью: вырезаем только ИНН/ОГРН (таблица при этом уже построена выше)
_FORBID_ID_RE = re.compile(r"\b(ИНН|ОГРН)\b", re.I)
def sanitize_interviews(text: str) -> str:
    parts = []
    for part in re.split(r"\s*;\s*", (text or "").strip()):
        if part and not _FORBID_ID_RE.search(part):
            parts.append(part.strip())
    return "; ".join(parts)

def _interviews_by_names(company: str,
                         names: List[str],
                         site_hint: Optional[str],
                         market: Optional[str]) -> str:
    """
    Ищем интервью по заданным ФИО и возвращаем один абзац:
      LLM → de-dup по URL → мягкая санитизация (без фин.фильтров).
    """
    if not names:
        return "нет данных"
    prompt = _build_interviews_prompt(company, names, site_hint, market)
    try:
        raw = _pplx_call_invest(prompt, model="sonar", recency=None, max_tokens=1400)
    except Exception as e:
        return f"нет данных (ошибка: {e})"
    para = _dedup_urls_in_paragraph(raw)   # 1) дубликаты
    para = sanitize_interviews(para)       # 2) убираем ИНН/ОГРН
    return para or "нет данных"

def _discover_people(company: str,
                     site_hint: Optional[str],
                     market: Optional[str],
                     top_n: int = 10) -> List[str]:
    """Дискавери ФИО через Sonar (устойчивый промпт, терпимый парсер)."""
    prompt = _build_people_discovery_prompt(company, site_hint, market)
    try:
        raw = _pplx_call_invest(prompt, model="sonar", recency=None, max_tokens=900)
    except Exception:
        return []
    names = _parse_people_lines(raw)
    return names[:top_n]

# ─────────────────────────────────────────────────────────────────────────────
# Главная функция блока
# ─────────────────────────────────────────────────────────────────────────────
def build_dual_interviews(
    company: str,
    company_info: Optional[Dict] = None,
    site_hint: Optional[str] = None,
    market: Optional[str] = None,
    max_people_inet: int = 10
) -> Dict[str, object]:
    """
    Возвращает структуру:
      {
        "names_checko": List[str],
        "digest_checko": str,
        "names_inet": List[str],
        "digest_inet": str,
        "names_union": List[str],
        "digest_union": str,
        "ceo": Dict|None,
        "shareholders": List[Dict],
        "md_block": str,  # ГОТОВЫЙ markdown (гендир, акционеры, интервью x3)
      }

    Важно:
      — ИНН/доли используются в таблице и ЛИШЬ ПОТОМ чистим интервью.
      — sanitize_invest на md_block НЕ применяем.
    """
    # 1) Checko → leaders/founders/names
    leaders, founders, names_checko = _names_from_checko_rich(company_info)

    # CEO + акционеры
    ceo = _pick_ceo(leaders, names_fallback=names_checko)
    shareholders = _shareholders_from_founders(founders)

    # 2) Интервью по Checko-именам
    digest_checko = _interviews_by_names(company, names_checko, site_hint, market) if names_checko else "нет данных"

    # 3) Интернет-дискавери → имена → интервью
    names_inet = _discover_people(company, site_hint, market, top_n=max_people_inet)
    digest_inet = _interviews_by_names(company, names_inet, site_hint, market) if names_inet else "нет данных"

    # 3.5) Объединённый проход (даёт больше хитов)
    names_union = list(dict.fromkeys((names_checko or []) + (names_inet or [])))[:12]
    digest_union = _interviews_by_names(company, names_union, site_hint, market) if names_union else "нет данных"

    # 4) Markdown-блок (без общей санитизации, чтобы не убить таблицу)
    ceo_line = "_нет данных_"
    if ceo:
        inn_txt = f"(ИНН {ceo['inn']})" if ceo.get("inn") else ""
        ceo_line = f"**Генеральный директор:** {ceo.get('fio','').strip()} {inn_txt}".strip()

    sh_tbl = _markdown_shareholders_table(shareholders)

    md_parts = [
        ceo_line,
        "",
        "**Акционеры**",
        sh_tbl,
        "",
        "**Интервью (по данным Checko):**",
        (digest_checko or "_нет данных_").strip(),
        "",
        "**Интервью (интернет-дискавери):**",
        (digest_inet or "_нет данных_").strip(),
        "",
        "**Интервью (объединённый поиск):**",
        (digest_union or "_нет данных_").strip(),
    ]
    md_block = "\n".join(md_parts).strip()

    return {
        "names_checko": names_checko,
        "digest_checko": digest_checko,
        "names_inet": names_inet,
        "digest_inet": digest_inet,
        "names_union": names_union,
        "digest_union": digest_union,
        "ceo": ceo,
        "shareholders": shareholders,
        "md_block": md_block,
    }

# Backward-compat alias
build_dual_interviews_from_v2 = build_dual_interviews




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




def _to_float_safe(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith("%"):
            s = s[:-1]
        s = s.replace(",", ".")
        val = float(s)
        # если дали долю в долях (0.25) — переведём в проценты
        if 0 < val <= 1.0:
            val = val * 100.0
        return val
    except Exception:
        return None

def _normalize_share_any(share_obj) -> Optional[float]:
    """
    Принимает:
    - число/строку: '25', '25%', '12,5', 0.25
    - dict вида {'Процент': 25} или {'Дробь': {'Числитель':1,'Знаменатель':2}}
    Возвращает процент float или None.
    """
    if share_obj is None:
        return None

    if isinstance(share_obj, dict):
        # 1) явный процент
        if "Процент" in share_obj:
            return _to_float_safe(share_obj.get("Процент"))
        if "percent" in share_obj:
            return _to_float_safe(share_obj.get("percent"))
        # 2) дробь
        frac = share_obj.get("Дробь") or share_obj.get("fraction")
        if isinstance(frac, dict):
            try:
                num = float(str(frac.get("Числитель") or frac.get("num") or 0).replace(",", "."))
                den = float(str(frac.get("Знаменатель") or frac.get("den") or 0).replace(",", "."))
                if den:
                    return (num / den) * 100.0
            except Exception:
                pass
        # 3) обойдём вложенные ключи
        for v in share_obj.values():
            got = _normalize_share_any(v)
            if got is not None:
                return got
        return None

    # число/строка
    return _to_float_safe(share_obj)

def _parse_checko_cell(cell, role_hint: Optional[str] = None) -> List[Dict]:
    """
    Делает из leaders_raw / founders_raw список:
      {'fio','inn','share_pct','role'}
    Поддерживает str | dict | list, разные схемы долей.
    """
    import re, ast
    out: List[Dict] = []

    def _emit(fio=None, inn=None, share=None, role=None):
        fio = _norm(fio)
        inn = _norm(str(inn)) if inn else None
        share_pct = _normalize_share_any(share)
        item = {"fio": fio, "inn": inn, "share_pct": share_pct, "role": _norm(role or role_hint)}
        if item["fio"] or item["inn"]:
            out.append(item)

    if cell is None:
        return out

    # если строка — попробуем распарсить JSON-подобное
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = ast.literal_eval(s)
                return _parse_checko_cell(parsed, role_hint=role_hint)
            except Exception:
                pass
        # выдернем ИНН/долю из строки
        m_inn = re.search(r"(?:ИНН|inn)\s*[:№]?\s*([0-9]{8,12})", s, re.I)
        inn = m_inn.group(1) if m_inn else None
        m_share = re.search(r"(?:доля|share)[^0-9]*([0-9]+[.,]?[0-9]*)\s*%?", s, re.I)
        share = m_share.group(1) if m_share else None
        fio = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
        _emit(fio=fio, inn=inn, share=share)
        return out

    if isinstance(cell, dict):
        fio   = cell.get("ФИО") or cell.get("fio") or cell.get("name")
        inn   = cell.get("ИНН") or cell.get("inn") or cell.get("tax_id")
        # 1) классика Checko: доля может быть dict или «плоская»
        share = None
        if isinstance(cell.get("Доля"), dict):
            share = cell["Доля"]
        elif "share" in cell:
            share = cell["share"]
        elif "Доля" in cell and not isinstance(cell.get("Доля"), dict):
            share = cell.get("Доля")
        role  = cell.get("Должность") or cell.get("role") or role_hint
        _emit(fio=fio, inn=inn, share=share, role=role)
        return out

    if isinstance(cell, list):
        for it in cell:
            out.extend(_parse_checko_cell(it, role_hint=role_hint))
        return out

    # fallback
    _emit(fio=str(cell))
    return out

def _pick_ceo(leaders: List[Dict], names_fallback: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Пытаемся найти гендиректора:
    1) по роли с приоритетом 'генераль';
    2) по 'директор'/'руководитель'/'CEO';
    3) иначе первый из leaders;
    4) иначе фолбэк по первому имени.
    """
    if not leaders and names_fallback:
        return {"fio": names_fallback[0], "inn": None, "share_pct": None, "role": "руководитель"}

    for p in leaders:
        r = (p.get("role") or "").lower()
        if "генераль" in r:
            return p
    for p in leaders:
        r = (p.get("role") or "").lower()
        if any(k in r for k in ("директор", "руковод", "ceo")):
            return p
    if leaders:
        return leaders[0]
    if names_fallback:
        return {"fio": names_fallback[0], "inn": None, "share_pct": None, "role": "руководитель"}
    return None

def _shareholders_from_founders(founders: List[Dict]) -> List[Dict]:
    """
    Чистим/сортируем акционеров. Если доля дана в долях (<=1),
    конвертируем в проценты. Сортируем по убыванию.
    """
    rows = []
    for p in founders:
        fio = (p.get("fio") or "").strip()
        inn = p.get("inn")
        share = _to_float_safe(p.get("share_pct"))  # на случай старых значений
        rows.append({"fio": fio, "inn": inn, "share_pct": share})

    with_share = [r for r in rows if r["share_pct"] is not None]
    no_share   = [r for r in rows if r["share_pct"] is None]
    with_share.sort(key=lambda x: x["share_pct"], reverse=True)
    return with_share + no_share



# кешируем, чтобы при повторных кликах не дергать LLM и сайт заново
@st.cache_data(ttl=86_400, show_spinner=False)










def run_ai_insight_tab() -> None:

        

    # ╭─🎛  UI ──────────────────────────────────────────╮
    st.title("📊 AI Company Insight")
    st.markdown("Введите данные (каждая компания — в отдельной строке).")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: inns_raw  = st.text_area("ИНН")          # ✅ без key=* — нам не нужны две копии
    with c2: names_raw = st.text_area("Название")
    with c3: mkts_raw  = st.text_area("Рынок")
    with c4: sites_raw = st.text_area("Сайт")
    
    aggregate_mode = st.checkbox("🧮 Суммировать финансы по всем ИНН")
    
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
                    desc_legacy = st.toggle("Legacy (Google/SiteRAG) description", value=False, key="desc_first")
                    
                    if desc_legacy:
                        with st.spinner("Генерируем описание (Legacy)…"):
                            doc = RAG(first_name, website=first_site, market=first_mkt).run()
                    
                        html_main = linkify_as_word(doc["summary"]).replace("\n", "<br>")
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
                    
                    else:
                        with st.spinner("Генерируем INVEST SNAPSHOT…"):
                            inv = get_invest_snapshot(
                                first_name,
                                site_hint=first_site,
                                model="sonar",
                                recency=None,
                                max_tokens=1500
                            )
                            # вырезаем раздел «Интервью» из описания
                            inv_clean = strip_interviews_section(inv["md"])
                            # делаем URL кликабельными (внутри div можно HTML)
                            inv_html = linkify_keep_url(inv_clean)
                            st.markdown(
                                f"<div style='background:#F7F9FA;border:1px solid #ccc;border-radius:8px;padding:18px;line-height:1.55'>{inv_html}</div>",
                                unsafe_allow_html=True,
                            )
                            doc = {"summary": inv_clean, "mode": "invest_snapshot"}
                        with st.expander("🔧 Отладка (сырой ответ)"):
                            st.text(inv.get("raw") or "—")
                    
                    # ----------- Рыночный отчёт (MARKET EVIDENCE) ------------------------
                    if first_mkt:
                        st.subheader("📈 Рыночный отчёт")
                        legacy = st.toggle("Legacy (Google/GPT) mode", value=False, key="legacy_first")
                    
                        if legacy:
                            with st.spinner("Собираем данные по рынку (Legacy)…"):
                                mkt_res = get_market_rag(first_mkt)
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
                        else:
                            with st.spinner("Собираем MARKET EVIDENCE…"):
                                ev = get_market_evidence(first_mkt, country="Россия", min_sources=8, recency=None)
                            st.markdown(
                                f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                                f"border-radius:8px;padding:18px;line-height:1.55'>{ev['text_html']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption("СТРУКТУРА по источникам — деньги:")
                            st.code(ev["money_block"] or "—", language="text")
                            st.caption("СТРУКТУРА по источникам — натуральные объёмы:")
                            st.code(ev["natural_block"] or "—", language="text")
                    
                            # (опционально) отладка сырого ответа
                            with st.expander("🔧 Отладка (сырой ответ)"):
                                st.text(ev["raw_text"] or "—")

                        if "mkt_res" not in locals():
                            mkt_res = {}
                    
                    # ----------- Руководители и интервью -----------------------------------
                    # ────── Руководители и интервью ─────────────────────────────────────
                    st.subheader("👥 Руководители и интервью")
                    
                    company_info_row = {
                        "leaders_raw":  (df_companies.loc[idx, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                        "founders_raw": (df_companies.loc[idx, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                    }
                    
                    with st.spinner("Ищем интервью (Checko → интернет)…"):
                        dual = build_dual_interviews_from_v2(
                            cmp_name, company_info=company_info_row, site_hint=site, market=mkt
                        )
                    
                    fio_checko = ", ".join(dual.get("names_checko") or []) or "нет данных"
                    fio_inet   = ", ".join(dual.get("names_inet") or [])   or "нет данных"
                    
                    digest_checko = sanitize_invest(dual.get("digest_checko") or "нет данных")
                    digest_inet   = sanitize_invest(dual.get("digest_inet") or "нет данных")
                    
                    # делаем ссылки кликабельными:
                    dig_checko_html = linkify_keep_url(digest_checko).replace("\n", "<br>")
                    dig_inet_html   = linkify_keep_url(digest_inet).replace("\n", "<br>")
                    
                    block_checko = (f"<h4 style='margin:6px 0'>Дайджест интервью — Checko</h4><div>{dig_checko_html}</div>"
                                    if digest_checko.strip().lower() != "нет данных" else "")
                    block_inet   = (f"<h4 style='margin:14px 0 6px'>Дайджест интервью — интернет</h4><div>{dig_inet_html}</div>"
                                    if digest_inet.strip().lower() != "нет данных" else "")
                    
                    st.markdown(
                        f"<div style='background:#F9FAFB;border:1px solid #ddd;border-radius:8px;padding:18px;line-height:1.6'>"
                        f"<p><b>ФИО (Checko):</b> {html.escape(fio_checko)}</p>"
                        f"<p><b>ФИО (интернет):</b> {html.escape(fio_inet)}</p>"
                        f"<hr style='border:none;border-top:1px solid #eee;margin:10px 0'>"
                        f"{block_checko}"
                        f"{block_inet}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    # 2) ⬇️ НОВОЕ: расширенный отчёт (2-шаговый PPLX: кандидаты → раскрытие юрлиц → факты + отдельный раздел «Интервью»)
                    adv_mode = st.toggle("Расширенный отчёт (бенефициары + факты + интервью)", value=True, key=f"owners_{idx}")
                    if adv_mode:
                        with st.spinner("Собираем расширенный отчёт по владельцам/бенефициарам…"):
                            try:
                                deep_text = two_step_perplexity_owners_v2(
                                    cmp_name,
                                    site_hint=site,
                                    market=mkt,
                                    model="sonar",
                                    recency=None,
                                    owners_limit=8,
                                    expand_org_limit=6,
                                    per_person_tokens=1500,
                                )
                                # делаем ссылки кликабельными и сохраняем разметку строк
                                deep_html = linkify_keep_url(deep_text).replace("\n", "<br>")
                            except Exception as e:
                                deep_html = f"<i>Не удалось собрать расширенный отчёт: {html.escape(str(e))}</i>"
                    
                        st.markdown(
                            f"<div style='background:#FFF;border:1px dashed #cbd5e1;border-radius:8px;padding:16px;line-height:1.6'>"
                            f"{deep_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    
                    # ─────── конец блока, дальше ваш код (если был) ───────────────────────
            
            # ---------- ② вкладки по отдельным компаниям ----------
            if aggregate_mode and len(inns) > 1:
                tabs = st.tabs(["Σ Сводно"] + [f"{n} ({inn})"
                                               for inn, n in zip(inns, names_full)])
            else:                                   # одиночный режим
                tabs = st.tabs([f"{n} ({inn})" for inn, n
                                in zip(inns, names_full)])
            
            start_idx = 1 if (aggregate_mode and len(inns) > 1) else 0
            
            for idx, (tab, inn, cmp_name, mkt, site) in enumerate(
                zip(tabs[start_idx:], inns, names_full, mkts_full, sites_full)
            ):
                with tab:
                    st.header(f"{cmp_name} — {inn}")
                    # дальше везде используй cmp_name вместо name
            
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
            
                    
                                      
                    # ────── Описание компании (INVEST SNAPSHOT, без интервью в выводе) ───────────────────────────
                    
                    # страховка: если паттерн/функция ещё не объявлены — объявим здесь
                    try:
                        _SEC_INTERV_RE
                    except NameError:
                        import re
                        _SEC_INTERV_RE = re.compile(
                            r"(^|\n)###\s*Интервью[^\n]*\n.*?(?=\n###\s|\Z)", flags=re.S | re.I
                        )
                    
                    def strip_interviews_section(md: str) -> str:
                        """Убирает из Markdown блок '### Интервью …' целиком (до следующего ### или конца)."""
                        if not md:
                            return ""
                        cleaned = _SEC_INTERV_RE.sub("\n", md).strip()
                        # сжимаем лишние пустые строки
                        return re.sub(r"\n{3,}", "\n\n", cleaned)
                    
                    # 1) готовим people из Checko для этой компании (используем дальше в двух местах)
                    company_info_row = {
                        "leaders_raw":  (df_companies.loc[idx, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                        "founders_raw": (df_companies.loc[idx, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                    }
                    
                    with st.spinner("Генерируем INVEST SNAPSHOT…"):
                        inv = get_invest_snapshot(
                            cmp_name,
                            site_hint=site,
                            model="sonar", recency=None, max_tokens=1500
                        )
                        # вырезаем раздел «Интервью …» из описания, чтобы он не попадал в блок «Описание компании»
                        inv_clean = strip_interviews_section(inv["md"])
                        inv_html  = linkify_keep_url(inv_clean)
                    
                        st.markdown(
                            f"<div style='background:#F7F9FA;border:1px solid #ccc;border-radius:8px;padding:18px;line-height:1.55'>{inv_html}</div>",
                            unsafe_allow_html=True,
                        )
                        doc = {"summary": inv_clean, "mode": "invest_snapshot"}
                    
                    with st.expander("🔧 Отладка (сырой ответ)"):
                        st.text(inv.get("raw") or "—")
                    
                    
                    # ────── Рыночный отчёт (MARKET EVIDENCE) ─────────────────────────────
                    if mkt:
                        st.subheader("📈 Рыночный отчёт")
                        legacy = st.toggle("Legacy (Google/GPT) mode", value=False, key=f"legacy_{idx}")
                    
                        if legacy:
                            with st.spinner("Собираем данные по рынку (Legacy)…"):
                                mkt_res = get_market_rag(mkt)
                            mkt_html = _linkify(mkt_res["summary"]).replace("\n", "<br>")
                            st.markdown(
                                f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;border-radius:8px;padding:18px;line-height:1.55'>{mkt_html}</div>",
                                unsafe_allow_html=True,
                            )
                            with st.expander("⚙️ Запросы к Google"):
                                for i, q in enumerate(mkt_res["queries"], 1):
                                    st.markdown(f"**{i}.** {q}")
                            with st.expander("🔍 Сниппеты (top-15)"):
                                df_leg = pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15)
                                st.dataframe(df_leg, use_container_width=True)
                        else:
                            with st.spinner("Собираем MARKET EVIDENCE…"):
                                ev = get_market_evidence(mkt, country="Россия", min_sources=8, recency=None)
                            st.markdown(
                                f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;border-radius:8px;padding:18px;line-height:1.55'>{ev['text_html']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption("СТРУКТУРА по источникам — деньги:")
                            st.code(ev["money_block"] or "—", language="text")
                            st.caption("СТРУКТУРА по источникам — натуральные объёмы:")
                            st.code(ev["natural_block"] or "—", language="text")
                            with st.expander("🔧 Отладка (сырой ответ)"):
                                st.text(ev["raw_text"] or "—")
                    
                    
                    # ────── Руководители и интервью ─────────────────────────────────────
                    st.subheader("👥 Руководители и интервью")
                    
                    company_info_row = {
                        "leaders_raw":  (df_companies.loc[idx, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                        "founders_raw": (df_companies.loc[idx, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                    }
                    
                    with st.spinner("Ищем интервью (Checko → интернет)…"):
                        dual = build_dual_interviews_from_v2(
                            cmp_name, company_info=company_info_row, site_hint=site, market=mkt
                        )
                    
                    # ===== 1) Гендиректор + Акционеры (чистый Markdown) =====
                    ceo = dual.get("ceo") or {}
                    fio_ceo = (ceo.get("fio") or "").strip()
                    if fio_ceo:
                        inn_txt = f"(ИНН {ceo.get('inn')})" if ceo.get("inn") else ""
                        st.markdown(f"**Генеральный директор:** {fio_ceo} {inn_txt}".strip())
                    else:
                        st.markdown("_Генеральный директор: нет данных_")
                    
                    st.markdown("")
                    st.markdown("**Акционеры**")
                    
                    shareholders = dual.get("shareholders") or []
                    st.markdown(_markdown_shareholders_table(shareholders))
                    
                    # ===== 2) Интервью (HTML с linkify_keep_url) =====
                    digest_checko = sanitize_invest(dual.get("digest_checko") or "нет данных")
                    digest_inet   = sanitize_invest(dual.get("digest_inet")   or "нет данных")
                    
                    blocks = []
                    if digest_checko.strip().lower() != "нет данных":
                        dig_checko_html = linkify_keep_url(digest_checko).replace("\n", "<br>")
                        blocks.append(
                            "<h4 style='margin:12px 0 6px'>Дайджест интервью — Checko</h4>"
                            f"<div>{dig_checko_html}</div>"
                        )
                    
                    if digest_inet.strip().lower() != "нет данных":
                        dig_inet_html = linkify_keep_url(digest_inet).replace("\n", "<br>")
                        blocks.append(
                            "<h4 style='margin:14px 0 6px'>Дайджест интервью — интернет</h4>"
                            f"<div>{dig_inet_html}</div>"
                        )
                    
                    if blocks:
                        st.markdown(
                            "<div style='background:#F9FAFB;border:1px solid #ddd;border-radius:8px;padding:18px;line-height:1.6'>"
                            + "".join(blocks) +
                            "</div>",
                            unsafe_allow_html=True,
                        )

                # === Q&A helpers (вставить ОДИН РАЗ выше UI) ===
                def _kb_collect_sections_for_company(cmp_name: str,
                                                     leaders_md: str = "",
                                                     digest_checko: str = "",
                                                     digest_inet: str = "") -> list[tuple[str,str]]:
                    kb = []
                    if leaders_md.strip():
                        kb.append(("Leaders & Shareholders", leaders_md))
                    if digest_checko.strip():
                        kb.append(("Interviews (Checko)", digest_checko))
                    if digest_inet.strip():
                        kb.append(("Interviews (Internet)", digest_inet))
                    return kb
                
                def _kb_simple_rank(query: str, sections: list[tuple[str,str]], top_k: int = 3) -> list[tuple[str,str,float]]:
                    import re
                    q = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
                    scored = []
                    for title, txt in sections:
                        tokens = re.findall(r"\w+", (txt or "").lower())
                        score = sum(tokens.count(t) for t in q)
                        if score > 0:
                            scored.append((title, txt, float(score)))
                    scored.sort(key=lambda x: x[2], reverse=True)
                    return scored[:top_k]
                
                _EMP_PATTERNS = [
                    r"численность\s+сотрудников[:\s\-~]*([0-9\s]+)\s*(?:чел|employees|сотр|сотруд)\b",
                    r"employees[:\s\-~]*([0-9\s]+)\b",
                ]
                
                def _extract_employee_count(text: str) -> tuple[int | None, str | None]:
                    import re
                    t = (text or "")
                    for pat in _EMP_PATTERNS:
                        m = re.search(pat, t, flags=re.I)
                        if m:
                            raw = re.sub(r"\D", "", m.group(1) or "")
                            if raw.isdigit():
                                return int(raw), "regex"
                    return None, None
                
                def _qa_prompt_for_web(company: str, user_q: str, site_hint: str | None = None) -> str:
                    site = f"Официальный сайт: {site_hint}. " if site_hint else ""
                    return f"""Ты — ассистент-исследователь. Найди точный ответ на вопрос про компанию «{company}».
                {site}Дай конкретный факт и дату/период, если важно. ВСЕГДА приводи ПРЯМЫЕ URL (2–4).
                ANSWER: <короткий ответ>
                DETAILS: <1–3 уточнения>
                SOURCES: <URL1>; <URL2>; <URL3>
                Q: {user_q}""".strip()
                
                def ask_guide(company: str, user_q: str, kb_sections: list[tuple[str,str]],
                              site_hint: str | None = None, allow_web: bool = True) -> dict:
                    # 1) локально
                    top_local = _kb_simple_rank(user_q, kb_sections, top_k=3)
                    merged = "\n\n".join(sec for _, sec, _ in top_local) if top_local else ""
                    emp_local, how = _extract_employee_count(merged)
                    if emp_local:
                        return {
                            "answer_md": f"**Ответ:** {emp_local:,} чел.".replace(",", " "),
                            "used": "local",
                            "sources": [],
                            "raw": f"LOCAL({how})",
                            "suggest_patch": {"section":"INVEST SNAPSHOT","md_line": f"**Численность:** {emp_local:,} чел.".replace(",", " ")},
                        }
                    # 2) веб
                    if allow_web:
                        prompt = _qa_prompt_for_web(company, user_q, site_hint)
                        raw = _pplx_call_invest(prompt, model="sonar", recency=None, max_tokens=800)
                        cleaned = sanitize_invest(raw)
                        urls = _extract_urls(cleaned)
                        emp_web, _ = _extract_employee_count(cleaned)
                        md_ans = cleaned
                        suggest = None
                        if emp_web:
                            md_ans = f"**Ответ:** {emp_web:,} чел.\n\n{cleaned}".replace(",", " ")
                            suggest = {"section":"INVEST SNAPSHOT","md_line": f"**Численность:** {emp_web:,} чел.  \nИсточники: " + "; ".join(urls[:3])}
                        return {"answer_md": md_ans, "used": "web", "sources": urls[:4], "raw": raw, "suggest_patch": suggest}
                    # 3) нет данных
                    return {"answer_md":"_Не удалось найти ответ_.","used":"none","sources":[],"raw":"","suggest_patch":None}
                
                def insert_or_append_line(md_text: str, section_title: str, new_line_md: str) -> str:
                    import re
                    if not new_line_md: return md_text or ""
                    text = md_text or ""
                    pat_h = re.compile(rf"(?m)^(#{1,6}\s*{re.escape(section_title)}\s*$)")
                    pat_b = re.compile(rf"(?m)^\*\*{re.escape(section_title)}\*\*\s*$")
                    m = pat_h.search(text) or pat_b.search(text)
                    if m:
                        pos = m.end()
                        return text[:pos] + "\n" + new_line_md.strip() + "\n\n" + text[pos:]
                    return (text + f"\n\n## {section_title}\n{new_line_md.strip()}\n").strip()

                    
                    # ────── Q&A: Спросить справку ─────────────────────────────────────
                    st.markdown("---")
                    st.subheader("🔎 Спросить справку")
                    
                    user_q = st.text_input(
                        "Ваш вопрос по компании (например: «найди численность сотрудников»)",
                        key=f"qa_{idx}"
                    )
                    
                    # Собираем локальный корпус только из того, что реально есть
                    kb_sections = []
                    # Блок лидеров/акционеров: соберём простой md
                    leaders_md = ""
                    if fio_ceo:
                        inn_txt = f"(ИНН {ceo.get('inn')})" if ceo.get("inn") else ""
                        leaders_md += f"Генеральный директор: {fio_ceo} {inn_txt}\n"
                    if shareholders:
                        for r in shareholders:
                            fio = (r.get("fio") or "").strip()
                            inn = r.get("inn") or ""
                            share = r.get("share_pct")
                            try:
                                share_f = float(str(share).replace(",", ".")) if share is not None else None
                                if share_f is not None and 0 < share_f <= 1.0:
                                    share_f *= 100.0
                            except Exception:
                                share_f = None
                            share_txt = (f"{share_f:.4g}".rstrip('0').rstrip('.') + "%") if share_f is not None else ""
                            leaders_md += f"- {fio}{f' (ИНН {inn})' if inn else ''}{f' — {share_txt}' if share_txt else ''}\n"
                    if leaders_md:
                        kb_sections.append(("Leaders & Shareholders", leaders_md))
                    
                    # интервью
                    if isinstance(digest_checko, str) and digest_checko.strip().lower() != "нет данных":
                        kb_sections.append(("Interviews (Checko)", digest_checko))
                    if isinstance(digest_inet, str) and digest_inet.strip().lower() != "нет данных":
                        kb_sections.append(("Interviews (Internet)", digest_inet))
                    
                    # кнопки
                    col_qa1, col_qa2 = st.columns([1,1])
                    
                    if col_qa1.button("Искать ответ", key=f"qa_go_{idx}") and (user_q or "").strip():
                        try:
                            qa = ask_guide(
                                company=cmp_name,
                                user_q=user_q.strip(),
                                kb_sections=kb_sections,
                                site_hint=site,
                                allow_web=True
                            )
                            ans_html = linkify_keep_url(qa.get("answer_md") or "").replace("\n", "<br>")
                            st.markdown(
                                f"<div style='background:#F6F8FA;border:1px solid #e2e8f0;border-radius:8px;padding:14px;line-height:1.6'>{ans_html}</div>",
                                unsafe_allow_html=True
                            )
                            if qa.get("sources"):
                                st.caption("Источники: " + " • ".join(qa["sources"]))
                    
                            # безопасное встраивание в справку (session_state), без привязки к inv
                            suggest = qa.get("suggest_patch")
                            if suggest and col_qa2.button(f"Вставить в раздел: {suggest['section']}", key=f"qa_apply_{idx}"):
                                ss_key = f"guide_section_{suggest['section']}_{idx}"
                                cur_md = st.session_state.get(ss_key, "")
                                new_md = insert_or_append_line(cur_md, suggest["section"], suggest["md_line"])
                                st.session_state[ss_key] = new_md  # можно потом отрендерить где нужно
                                st.success("Добавлено в локальную копию справки (session_state).")
                        except Exception as e:
                            st.error(f"Не удалось выполнить поиск: {e}")




# === BACKGROUND / PROGRESS =====================================================
import os, time, re, json, io
import requests
from datetime import date, timedelta
from typing import Optional, Tuple, List, Iterable
import streamlit as st

def long_job(total_sec: int = 180, key_prog: str = "ai_prog"):
    """Фоновая задача, каждые 1 с обновляет progress в session_state."""
    for i in range(total_sec + 1):
        time.sleep(1)
        st.session_state[key_prog] = i / total_sec     # 0 … 1
    st.session_state["ai_done"] = True                 # отчёт готов

# ─────────────────────────────────────────────────────────
# 2. UI-функция существующей вкладки "Advance Eye" (без изменений)
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

# === Q&A over Guide: локальный поиск → веб-фолбэк через Sonar ==================
# (оставлено как у вас; при необходимости можете править отдельно)
from typing import Tuple

def _kb_collect_sections_for_company(
    cmp_name: str,
    inv_md: str | None = None,
    owners_md: str | None = None,
    leaders_md: str | None = None,
    interviews_md: str | None = None,
    extra_sections: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    kb: list[tuple[str, str]] = []
    if inv_md:         kb.append(("INVEST SNAPSHOT", inv_md))
    if owners_md:      kb.append(("Owners/Beneficiaries", owners_md))
    if leaders_md:     kb.append(("Leaders & Shareholders", leaders_md))
    if interviews_md:  kb.append(("Interviews", interviews_md))
    if extra_sections:
        for k, v in extra_sections.items():
            if v: kb.append((k, v))
    return kb

def _kb_simple_rank(query: str, sections: list[tuple[str,str]], top_k: int = 3) -> list[tuple[str,str,float]]:
    import re
    q = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
    scored = []
    for title, txt in sections:
        tokens = re.findall(r"\w+", (txt or "").lower())
        score = sum(tokens.count(t) for t in q)
        if score > 0:
            scored.append((title, txt, float(score)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

_EMP_PATTERNS = [
    r"численность\s+сотрудников[:\s\-~]*([0-9\s]+)\s*(?:чел|сotr|сотруд|employees)\b",
    r"employees[:\s\-~]*([0-9\s]+)\b",
    r"персонал[:\s\-~]*([0-9\s]+)\s*(?:чел|сотруд)\b",
]

def _extract_employee_count(text: str) -> tuple[int | None, str | None]:
    import re
    t = (text or "")
    for pat in _EMP_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            raw = re.sub(r"\D", "", m.group(1) or "")
            if raw.isdigit():
                return int(raw), "regex"
    return None, None

def _qa_prompt_for_web(company: str, user_q: str, site_hint: str | None = None) -> str:
    site = f"Официальный сайт: {site_hint}. " if site_hint else ""
    return f"""
Ты — ассистент-исследователь. Найди точный ответ на вопрос про компанию «{company}». 
{site}Дай конкретный факт и дату/период, если важно. ВСЕГДА приводи ПРЯМЫЕ URL на первоисточники (не агрегаторы), 2–4 ссылки.
Формат вывода строго такой:
ANSWER: <краткий ответ в одну-две строки>
DETAILS: <1–3 коротких пояснения/уточнения, если нужно>
SOURCES: <URL1>; <URL2>; <URL3>
Q: {user_q}
""".strip()

def ask_guide(
    company: str,
    user_q: str,
    kb_sections: list[tuple[str,str]],
    site_hint: str | None = None,
    allow_web: bool = True,
) -> dict:
    top_local = _kb_simple_rank(user_q, kb_sections, top_k=3)
    merged_local = "\n\n".join(sec for _, sec, _ in top_local) if top_local else ""
    emp_local, how = _extract_employee_count(merged_local)
    if emp_local:
        ans = f"**Ответ:** {emp_local:,} чел.".replace(",", " ")
        return {
            "answer_md": ans,
            "used": "local",
            "sources": [],
            "raw": f"LOCAL({how})",
            "suggest_patch": {
                "section": "INVEST SNAPSHOT",
                "md_line": f"**Численность:** {emp_local:,} чел.",
            },
        }

    if allow_web:
        prompt = _qa_prompt_for_web(company, user_q, site_hint)
        raw = _pplx_call_invest(prompt, model="sonar", recency=None, max_tokens=800)
        cleaned = sanitize_invest(raw)
        urls = _extract_urls(cleaned)
        emp_web, _ = _extract_employee_count(cleaned)
        md_ans = cleaned
        suggest = None
        if emp_web:
            md_ans = f"**Ответ:** {emp_web:,} чел.\n\n{cleaned}".replace(",", " ")
            suggest = {"section":"INVEST SNAPSHOT", "md_line": f"**Численность:** {emp_web:,} чел.  \nИсточники: " + "; ".join(urls[:3])}
        return {
            "answer_md": md_ans,
            "used": "web",
            "sources": urls[:4],
            "raw": raw,
            "suggest_patch": suggest,
        }

    return {
        "answer_md": "_Не удалось найти ответ ни локально, ни в вебе._",
        "used": "none",
        "sources": [],
        "raw": "",
        "suggest_patch": None,
    }

def insert_or_append_line(md_text: str, section_title: str, new_line_md: str) -> str:
    import re
    if not new_line_md:
        return md_text or ""
    text = md_text or ""
    pat_h1 = re.compile(rf"(?m)^(#{1,6}\s*{re.escape(section_title)}\s*$)")
    pat_b  = re.compile(rf"(?m)^\*\*{re.escape(section_title)}\*\*\s*$")

    def _add_after(pos: int) -> str:
        before = text[:pos]
        after  = text[pos:]
        return before + "\n" + new_line_md.strip() + "\n\n" + after

    m = pat_h1.search(text) or pat_b.search(text)
    if m:
        insert_pos = m.end()
        return _add_after(insert_pos)
    block = f"\n\n## {section_title}\n{new_line_md.strip()}\n"
    return (text + block).strip()

# ==============================================================================
# === NEWS RUN (last N days, multi-company, NO auto-matrix, NO ```text) ========
# ==============================================================================
import re, io
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple, List, Iterable    


API_URL_PPLX_NEWS = "https://api.perplexity.ai/chat/completions"

# ==============================================================================
# === NEWS RUN (таблица, N=15 по умолчанию, multi-company) =====================
# ==============================================================================

import re, io
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple, List, Iterable

API_URL_PPLX_NEWS = "https://api.perplexity.ai/chat/completions"

class PPLXNewsError(Exception):
    pass

def _get_pplx_key_news() -> str:
    key = (os.getenv("SONAR_API_KEY") or os.getenv("PPLX_API_KEY") or
           os.getenv("PERPLEXITY_API_KEY") or st.secrets.get("SONAR_API_KEY", "")).strip()
    if (not key.startswith("pplx-")) or (len(key) < 40) or key.endswith("..."):
        raise PPLXNewsError("Perplexity API key отсутствует или выглядит как заглушка. Задай pplx-<…> (длинный).")
    return key

def call_pplx_news(
    prompt: str,
    model: str = "sonar",
    recency: Optional[str] = None,    # 'week' / 'month' / 'year'
    temperature: float = 0.0,
    max_tokens: int = 1600,
    timeout: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {_get_pplx_key_news()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "news-run-multi/1.1",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Ты — редактор новостей и факт-чекер. Возвращай только проверяемые факты с ПРЯМЫМИ URL. "
                "Никогда не придумывай ссылки и новости. Формат строго по инструкции пользователя. Без владельцев/ИНН/ОГРН."
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if recency in {"hour", "day", "week", "month", "year"}:
        payload["search_recency_filter"] = recency

    r = requests.post(API_URL_PPLX_NEWS, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text[:800]}
        raise PPLXNewsError(f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:800]}")
    return r.json()["choices"][0]["message"]["content"]

# ---- sanitize / split / dedup -------------------------------------------------
_FORBIDDEN_NEWS = re.compile(r"(акционер|владельц|бенефициар|инн|огрн)", re.IGNORECASE)

def sanitize_news(text: str) -> str:
    lines = []
    for ln in (text or "").splitlines():
        if _FORBIDDEN_NEWS.search(ln):
            continue
        lines.append(ln.rstrip())
    out, blank = [], False
    for ln in lines:
        if not ln.strip():
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()

_PAR_SPLIT_NEWS = re.compile(r"(?=^Источник:\s*)", flags=re.IGNORECASE | re.MULTILINE)

def _normalize_companies_news(company: Iterable[str] | str | None) -> List[str]:
    if company is None:
        return []
    if isinstance(company, str):
        parts = re.split(r"[;,|\n]", company)
        comps = [p.strip() for p in parts if p.strip()]
    else:
        comps = [str(x).strip() for x in company if str(x).strip()]
    seen, out = set(), []
    for c in comps:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

# -------- новый формат промпта: одна строка с ключами через " | " --------------
def build_news_prompt_window(
    company: str,
    country: str,
    since: date,
    until: date,
    min_items: int,
    keywords: list[str] | None = None,
) -> str:
    user_terms = [k.strip() for k in (keywords or []) if k and k.strip()]
    # компания добавляется как «якорь»
    term_hint = ", ".join(dict.fromkeys([company] + user_terms))

    return f"""
Сделай новостной дайджест про «{company}» в стране {country}.
Включай ТОЛЬКО публикации в диапазоне [{since.isoformat()} … {until.isoformat()}] включительно.
Ищи по терминам: {term_hint}

СТРОГО ОДНА СТРОКА на материал, начинай с 'Источник:' и используй ключи в таком порядке (через ' | '):
Источник: <издатель> | Дата: <YYYY-MM-DD> | Заголовок: <title> | Ключевое: <1–2 факта/цифры> | URL: <прямой_линк>

— Дай минимум {min_items} материалов (если релевантных меньше — верни сколько есть, но не выходи за пределы даты и не придумывай).
— Ссылки должны быть прямыми (https://...).
— Не упоминай владельцев/ИНН/ОГРН.
""".strip()

# -------- фильтрация и дедуп по окну/URL --------------------------------------
_DATE_RE_NEWS = re.compile(r"\bДата:\s*(\d{4}-\d{2}-\d{2})\b")
_URL_RE_NEWS  = re.compile(r"\bURL:\s*([^\s|]+)")
_PUB_RE       = re.compile(r"^Источник:\s*(.*?)\s*\|", flags=re.I|re.S)
_TITLE_RE     = re.compile(r"\bЗаголовок:\s*(.*?)\s*\|\s*Ключевое:", flags=re.I|re.S)
_KEY_RE       = re.compile(r"\bКлючевое:\s*(.*?)\s*\|\s*URL:", flags=re.I|re.S)

def _parse_date_safe_news(s: str) -> Optional[date]:
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        return None

def filter_output_by_window_news(text: str, since: date, until: date) -> str:
    kept = []
    for p in [p.strip() for p in _PAR_SPLIT_NEWS.split(text or "") if p.strip()]:
        m = _DATE_RE_NEWS.search(p)
        if not m:
            continue
        dt = _parse_date_safe_news(m.group(1))
        if dt and since <= dt <= until:
            kept.append(p)
    return "\n".join(kept)

def _dedup_by_url_news(paragraphs: List[str]) -> List[str]:
    seen, out = set(), []
    for p in paragraphs:
        m = _URL_RE_NEWS.search(p)
        url = m.group(1).strip() if m else ""
        if url and url not in seen:
            seen.add(url)
            out.append(p)
    return out

def _split_paragraphs_news(text: str) -> List[str]:
    return [p.strip() for p in _PAR_SPLIT_NEWS.split(text or "") if p.strip()]

# -------- парсим в строки таблицы ---------------------------------------------
def parse_news_rows(text: str) -> list[dict]:
    rows = []
    for p in _split_paragraphs_news(text):
        try:
            dt = _DATE_RE_NEWS.search(p).group(1)
            title = _TITLE_RE.search(p).group(1).strip()
            key   = _KEY_RE.search(p).group(1).strip()
            url   = _URL_RE_NEWS.search(p).group(1).strip()
            pub_m = _PUB_RE.search(p)
            pub   = pub_m.group(1).strip() if pub_m else ""
            rows.append({"Дата": dt, "Заголовок": title, "Ключевое": key, "Ссылка": url, "Издатель": pub})
        except Exception:
            # пропускаем криво сформированный абзац
            continue
    return rows

# -------- основная функция: всегда таргет = 15 --------------------------------
def news_run_last_days(
    company: Iterable[str] | str | None = None,   # пользователь вводит
    country: str = "Россия",
    last_days: int = 31,
    keywords: str | list[str] | None = None,
    model: str = "sonar",
) -> str:
    today = date.today()
    since = today - timedelta(days=last_days)
    companies = _normalize_companies_news(company)
    if not companies:
        raise ValueError("Список компаний пуст. Укажите минимум одну компанию/сущность.")

    # распарсим keywords
    if isinstance(keywords, str):
        kw = [k.strip() for k in re.split(r"[;,|\n]", keywords) if k.strip()]
    else:
        kw = [str(k).strip() for k in (keywords or []) if str(k).strip()]

    TARGET = 15  # ← фиксированное количество на весь вывод
    n = len(companies)
    base = max(1, TARGET // n)
    extra = max(0, TARGET - base * n)

    recency = "week" if last_days <= 7 else ("month" if last_days <= 31 else "year")

    all_pars: List[str] = []
    for idx, comp in enumerate(companies):
        need = base + (1 if idx < extra else 0)
        prompt = build_news_prompt_window(comp, country, since, today, need, keywords=kw)
        raw = call_pplx_news(prompt, model=model, recency=recency, max_tokens=1600)
        clean = sanitize_news(raw)
        filtered = filter_output_by_window_news(clean, since, today)
        if not filtered:
            continue
        all_pars.extend(_split_paragraphs_news(filtered))

    if not all_pars:
        return f"[Окно дат: {since.isoformat()} — {today.isoformat()}]\n\nнет свежих новостей в заданном окне."

    # дедуп по URL и ограничение до TARGET
    all_pars = _dedup_by_url_news(all_pars)[:TARGET]

    header = f"[Окно дат: {since.isoformat()} — {today.isoformat()}] Компании: {', '.join(companies)}"
    body = "\n".join(all_pars)
    return header + "\n\n" + body

# -------- UI вкладка: рисуем таблицу с кликабельной ссылкой -------------------
def run_news_run_tab() -> None:
    st.header("🗞 News Run")

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        companies_raw = st.text_area(
            "Компании/сущности (по одной в строке или через запятую)",
            placeholder="Примеры:\nЛукойл\nВосток Ойл\nТМК",
            height=140
        )
    with c2:
        last_days = st.number_input("За сколько дней искать", min_value=1, max_value=365, value=31, step=1)
    with c3:
        country = st.text_input("Страна", value="Россия")

    keywords_raw = st.text_area(
        "Ключевые слова (по одному в строке или через запятую)",
        placeholder="контракты\nтендер\nпроизводство\nэкспорт",
        height=120
    )

    col_run, col_dl_txt, col_dl_csv = st.columns([1,1,1])
    if col_run.button("Запустить дайджест"):
        companies = [x.strip() for x in re.split(r"[;,|\n]", companies_raw or "") if x.strip()]
        if not companies:
            st.error("Укажите минимум одну компанию/сущность.")
            st.stop()
        keywords = [k.strip() for k in re.split(r"[;,|\n]", keywords_raw or "") if k.strip()]

        try:
            with st.spinner("Ищем и агрегируем новости…"):
                text = news_run_last_days(
                    company=companies,
                    country=country,
                    last_days=int(last_days),
                    keywords=keywords,
                    model="sonar",
                )

            # Парсим в строки таблицы
            rows = parse_news_rows(text)
            if not rows:
                st.warning("Не удалось распарсить новости в таблицу. Показываю сырой текст.")
                st.code(text, language="markdown")
                return

            # Сортировка по дате (по убыванию)
            df = pd.DataFrame(rows)
            # убеждаемся, что формат YYYY-MM-DD — отсортируем и оставим строкой
            df["_sort"] = pd.to_datetime(df["Дата"], errors="coerce")
            df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])

            st.dataframe(
                df[["Дата", "Заголовок", "Ключевое", "Ссылка"]],
                use_container_width=True,
                column_config={
                    "Ссылка": st.column_config.LinkColumn("Ссылка", help="Открыть источник"),
                }
            )

            # Кнопки выгрузки
            col_dl_txt.download_button(
                "Скачать TXT",
                data=io.BytesIO(text.encode("utf-8")),
                file_name=f"news_run_{date.today().isoformat()}.txt",
                mime="text/plain"
            )
            col_dl_csv.download_button(
                "Скачать CSV",
                data=io.BytesIO(df.to_csv(index=False).encode("utf-8")),
                file_name=f"news_run_{date.today().isoformat()}.csv",
                mime="text/csv"
            )
        except (PPLXNewsError, ValueError) as e:
            st.error(str(e))


# ─────────────────────────────────────────────────────────
# 5. Вкладки приложения: добавляем News Run между AI-Insight и Advance Eye
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# 5. Вкладки приложения: добавляем News Run между AI-Insight и Advance Eye
# ─────────────────────────────────────────────────────────
tab_ts, tab_ai, tab_news, tab_eye = st.tabs(
    ["⏱️ Timesheet", "📊 AI-Insight", "🗞 News Run", "👁️ Advance Eye"]
)

with tab_ts:
    # nikabot-style форма учёта времени (autoselect пользователя, недельная сетка)
    render_timesheet_tab()

with tab_ai:
    run_ai_insight_tab()

with tab_news:
    run_news_run_tab()

with tab_eye:
    run_advance_eye_tab()
