#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
    "DYXLESS_TOKEN": st.secrets["DYXLESS_TOKEN"],
    "SONAR_API_KEY": st.secrets["SONAR_API_KEY"],
}

DYXLESS_TOKEN = KEYS["DYXLESS_TOKEN"]


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

def invest_snapshot_enriched(company: str,
                             site_hint: Optional[str] = None,
                             company_info: dict | None = None,
                             market: str | None = None,
                             model: str = "sonar",
                             recency: Optional[str] = None,
                             max_tokens: int = 1500) -> str:
    """
    Строит обычный INVEST SNAPSHOT, потом заменяет/вставляет секцию «Интервью»
    материалом, собранным на базе Checko (+ Sonar discovery при необходимости).
    """
    base_md = invest_snapshot(company, site_hint=site_hint, model=model,
                              recency=recency, max_tokens=max_tokens)
    _, paragraph = interviews_from_checko_sonar(company,
                                                company_info=company_info,
                                                site_hint=site_hint,
                                                market=market)
    return _replace_interviews_section(base_md, paragraph)

@st.cache_data(ttl=86_400, show_spinner="📝 Собираем описание (enriched)…")
def get_invest_snapshot_enriched(company: str,
                                 site_hint: Optional[str] = None,
                                 company_info: dict | None = None,
                                 market: str | None = None,
                                 model: str = "sonar",
                                 recency: Optional[str] = None,
                                 max_tokens: int = 1500) -> dict:
    try:
        md = invest_snapshot_enriched(company, site_hint=site_hint,
                                      company_info=company_info, market=market,
                                      model=model, recency=recency, max_tokens=max_tokens)
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
    """
    Возвращает (plain_text_without_code, money_block, natural_block).
    Ищем два fenced-блока: text ...
    """
    if not raw_text:
        return "", "", ""
    code_blocks = re.findall(r"text\s*(.*?)\s*", raw_text, flags=re.S | re.I)
    money_block = code_blocks[0].strip() if len(code_blocks) >= 1 else ""
    natural_block = code_blocks[1].strip() if len(code_blocks) >= 2 else ""
    # Вырезаем найденные блоки из основного текста
    plain = raw_text
    for blk in code_blocks[:2]:
        plain = plain.replace(f"text\n{blk}\n", "")
        plain = plain.replace(f"text\r\n{blk}\r\n", "")
    return plain.strip(), money_block, natural_block


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
        summary_html = linkify_as_word(body)
    
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
                            company_info_first = {
                                "leaders_raw":  (df_companies.loc[0, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                                "founders_raw": (df_companies.loc[0, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                            }
                            
                            inv = get_invest_snapshot_enriched(
                                first_name,
                                site_hint=first_site,
                                company_info=company_info_first,
                                market=first_mkt,
                                model="sonar",
                                recency=None,
                                max_tokens=1500
                            )
                    
                        # inv['md'] уже Markdown → без _linkify
                        st.markdown(
                            f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{inv['md']}</div>",
                            unsafe_allow_html=True,
                        )
                        with st.expander("🔧 Отладка (сырой ответ)"):
                            st.text(inv["raw"] or "—")
                    
                    # (опционально) запоминаем «doc» для session_state
                    doc = doc if desc_legacy else {"summary": inv["md"], "mode": "invest_snapshot"}
                    
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
                    
                    use_legacy_leaders_first = st.toggle(
                        "Показать расширенный поиск интервью (legacy)",
                        value=False,
                        key="leaders_global"  # <- было "leaders_first"
                    )
                    
                    if use_legacy_leaders_first:
                        with st.spinner("Собираем руководителей и интервью (legacy)…"):
                            # берём ТОЛЬКО нужные ключи из df_companies (а не всю карточку)
                            company_info = {
                                "leaders_raw":  (df_companies.loc[0, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                                "founders_raw": (df_companies.loc[0, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                            }
                    
                            lead_res = get_leaders_rag(
                                first_name,
                                website=first_site,
                                market=first_mkt,
                                company_info=company_info,  # ← имена из Checko → дальше поиск интервью в интернете
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
                    else:
                        st.info("Базовые интервью уже есть в блоке «Интервью (владельцы/руководство)» внутри INVEST SNAPSHOT выше. "
                                "Включите переключатель, чтобы выполнить расширенный поиск по именам из Checko и в интернете.")

                    # --- страховки, если тумблер был выключен (переменные не создались) ---
                    if "lead_res" not in locals():
                        lead_res = {"summary": "", "queries": [], "snippets": []}
                    
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
                    desc_legacy_tab = st.toggle("Legacy (Google/SiteRAG) description", value=False, key=f"desc_{idx}")
                    
                    if desc_legacy_tab:
                        with st.spinner("Генерируем описание (Legacy)…"):
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
                    
                    else:
                        with st.spinner("Генерируем INVEST SNAPSHOT…"):
                            inv = get_invest_snapshot(name, site_hint=site, model="sonar", recency=None, max_tokens=1500)
                    
                        st.markdown(
                            f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                            f"border-radius:8px;padding:18px;line-height:1.55'>{inv['md']}</div>",
                            unsafe_allow_html=True
                        )
                        with st.expander("🔧 Отладка (сырой ответ)"):
                            st.text(inv["raw"] or "—")
                    
                    # для совместимости с session_state ниже
                    doc = doc if desc_legacy_tab else {"summary": inv["md"], "mode": "invest_snapshot"}
                    
                    # ────── Рыночный отчёт (MARKET EVIDENCE) ─────────────────────────────
                    if mkt:
                        st.subheader("📈 Рыночный отчёт")
                        legacy = st.toggle("Legacy (Google/GPT) mode", value=False, key=f"legacy_{idx}")
                    
                        if legacy:
                            with st.spinner("Собираем данные по рынку (Legacy)…"):
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
                                df_leg = pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15)
                                st.dataframe(df_leg, use_container_width=True)
                        else:
                            with st.spinner("Собираем MARKET EVIDENCE…"):
                                ev = get_market_evidence(mkt, country="Россия", min_sources=8, recency=None)
                            st.markdown(
                                f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                                f"border-radius:8px;padding:18px;line-height:1.55'>{ev['text_html']}</div>",
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
                    use_legacy_leaders = st.toggle(
                        "Показать расширенный поиск интервью (legacy)",
                        value=False,
                        key=f"leaders_{idx}"  # <- вместо "leaders_first"
                    )
                    
                    if use_legacy_leaders:
                        with st.spinner("Собираем руководителей и интервью (legacy)…"):
                            row_idx = inns.index(inn)  # индекс текущего ИНН в изначальном списке
                            company_info = {
                                "leaders_raw":  (df_companies.loc[row_idx, "leaders_raw"]  if "leaders_raw"  in df_companies.columns else []) or [],
                                "founders_raw": (df_companies.loc[row_idx, "founders_raw"] if "founders_raw" in df_companies.columns else []) or [],
                            }
                        
                            lead_res = get_leaders_rag(
                                name,                # <- было first_name
                                website=site,        # <- было first_site
                                market=mkt,          # <- было first_mkt
                                company_info=company_info,
                            )
                        st.markdown(
                            f"<div style='background:#F9FAFB;border:1px solid #ddd;border-radius:8px;padding:18px;line-height:1.55'>"
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
                    else:
                        st.info("Интервью смотрите в разделе «Интервью (владельцы/руководство)» внутри INVEST SNAPSHOT выше.")

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



