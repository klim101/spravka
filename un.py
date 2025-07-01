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


KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
}


# In[2]:





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







# ── helpers ────────────────────────────────────────────────
async def _site_snippet(domain: str) -> str:
    """Возвращает первый Google-сниппет для site:domain (или '')."""
    if not domain:
        return ""
    async with aiohttp.ClientSession() as sess:
        q = f"site:{domain}"
        snips = await _google(sess, q, n=1)
    return snips[0][1] if snips else ""
















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

async def _gpt(msgs, T=0.2, model="gpt-4o-mini"):
    chat = await openai.ChatCompletion.acreate(
        model=model, temperature=T, messages=msgs)
    return chat.choices[0].message.content.strip()





class RAG:
    """
    summary   – итоговый отчёт (plain-text)
    queries   – список Гугл-запросов (уже с добавленным рынком)
    snippets  – Google-сниппеты, которые отданы GPT
    site_ctx  – контекст сайта («site:…» + “рынок компании – …”)
    """
    def __init__(self, company: str, website: str = "", market: str = "",
                 years=(2022, 2023, 2024), country: str = "Россия",
                 steps: int = 2, snips: int = 3):
        self.company  = company.strip()
        self.website  = website.strip()
        self.market   = market.strip()       # ← индустрия
        self.country  = country
        self.years    = years
        self.steps    = steps
        self.snips    = snips

    # ----------  контекст сайта -----------------------------------
    async def _site_ctx(self) -> str:
        dom = tldextract.extract(self.website).registered_domain if self.website else ""
        snip = await _site_snippet(dom)
        if snip:
            return f"{snip}\nрынок компании – {self.market}" if self.market else snip
        return f"рынок компании – {self.market}" if self.market else ""

    # ----------  GPT → 10-12 поисковых запросов -------------------
    async def _queries(self, hist="") -> list[str]:
        dom  = tldextract.extract(self.website).registered_domain if self.website else ""
        base = f'"{self.company}"'
        if dom:
            base += f' OR site:{dom}'

        sys = (
            "ТЫ — ОПЫТНЫЙ ИССЛЕДОВАТЕЛЬ РЫНКОВ И ДАННЫХ. СФОРМУЛИРУЙ 10–12 ТОЧНЫХ GOOGLE-ЗАПРОСОВ, "
            f"ПОЗВОЛЯЮЩИХ СОБРАТЬ ИНФОРМАЦИЮ О КОМПАНИИ «{self.company}» НА РЫНКЕ «{self.market}» "
            f"({self.country}, {', '.join(map(str, self.years))}). "
            "ФОРМАТ: QUERY: <строка запроса>. БЕЗ ДОПОЛНИТЕЛЬНЫХ КОММЕНТАРИЕВ."
        )
        usr = f"base={base}" + hist
        raw = await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            T=0.10
        )
        return re.findall(r"QUERY:\s*(.+)", raw, flags=re.I)

    # ----------  формируем GPT-конспект ---------------------------
    async def _summary(self, ctx: str, site_ctx: str = "") -> str:
        sys = (
            "ТЫ — ВЫСОКОКВАЛИФИЦИРОВАННЫЙ АНАЛИТИК РЫНКОВ. СОСТАВЬ СТРУКТУРИРОВАННЫЙ АНАЛИТИЧЕСКИЙ ОТЧЁТ О КОМПАНИИ В ФОРМЕ ПОСЛЕДОВАТЕЛЬНЫХ АБЗАЦЕВ ПО СЛЕДУЮЩИМ ТЕМАТИЧЕСКИМ БЛОКАМ: "
            "1) ОПИСАНИЕ (миссия, род деятельности, сфера), "
            "2) ОБЩАЯ ИНФОРМАЦИЯ (юридический статус, дата и место основания, штаб-квартира), "
            "3) ПАРТНЁРСТВА (ключевые альянсы и сотрудничества), "
            "4) НАПРАВЛЕНИЯ (ключевые линии бизнеса и инициативы), "
            "5) ИСТОРИЯ (вехи развития, ключевые события), "
            "6) ЦИФРЫ (объёмы производства, доля рынка, активы и др. — КРОМЕ ВЫРУЧКИ), "
            "7) ПРОДУКТЫ (основные категории товаров и услуг), "
            "8) ГЕОГРАФИЯ (рынки присутствия, регионы продаж, производственные мощности), "
            "9) СОТРУДНИКИ (численность персонала, ключевые фигуры, корпоративная культура), "
            "10) УНИКАЛЬНОСТЬ (конкурентные преимущества, отличительные черты), "
            "11) ВЫВОДЫ (оценка позиции на рынке, перспективы, вызовы). "
            "ПОСЛЕ КАЖДОГО ФАКТА ОБЯЗАТЕЛЬНО УКАЗЫВАЙ ПОДТВЕРЖДЁННУЮ ССЫЛКУ-ИСТОЧНИК В КРУГЛЫХ СКОБКАХ (ФОРМАТ: ПОЛНЫЙ URL). "
            "НЕ ИСПОЛЬЗУЙ Markdown, НЕ УКАЗЫВАЙ ВЫРУЧКУ НИ В КАКОМ ВИДЕ."
        )  
        if site_ctx:
            ctx = f"SITE_CONTEXT:\n{site_ctx}\n\n{ctx}"
        return await _gpt(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": ctx}],
            T=0.25
        )

    # ----------  основной пайплайн -------------------------------
    async def _run_async(self):
        site_ctx = await self._site_ctx()
        queries, snippets, hist = [], [], ""

        async with aiohttp.ClientSession() as s:
            for _ in range(self.steps):
                q_raw = await self._queries(hist)        # GPT-запросы
                # ▸ добавляем к КАЖДОМУ запросу рынок (если он указан)
                ql = [f"{q} {self.market}" if self.market and self.market.lower() not in q.lower()
                      else q for q in q_raw]

                queries  += ql
                res = await asyncio.gather(*[_google(s, q, self.snips) for q in ql])
                snippets += sum(res, [])
                hist = f"\nСниппетов: {len(snippets)}"

        ctx  = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in snippets)[:20_000]
        summ = await self._summary(ctx, site_ctx)
        return {"summary":  summ,
                "queries":  queries,
                "snippets": snippets,
                "site_ctx": site_ctx}

    def run(self):
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
                 steps=1, snips=4):
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
            "ВСЕ ФАКТЫ ДОЛЖНЫ БЫТЬ УНИКАЛЬНЫМИ, НЕ ПОВТОРЯТЬСЯ И ПОДТВЕРЖДЁННЫ РЕАЛЬНЫМИ ССЫЛКАМИ НА ИСТОЧНИКИ В КРУГЛЫХ СКОБКАХ (ФОРМАТ: ПОЛНЫЙ URL). "
            "НЕ ИСПОЛЬЗУЙ MARKDOWN, НЕ ПРИДУМЫВАЙ ФАКТЫ — ТОЛЬКО ДОКУМЕНТИРОВАННЫЕ ДАННЫЕ."
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

# ── основной класс -------------------------------------------------
class FastLeadersInterviews:
    """
    summary  – HTML-блок (руководители + дайджест интервью)
    names    – список «ФИО (роль)»
    queries  – все поисковые запросы
    snippets – сниппеты, отданные GPT
    """
    def __init__(self, company:str, website:str="", market:str="", country:str="Россия"):
        self.c, self.site, self.market, self.cntry = company, website.strip(), market.strip(), country

    # ---- домен сайта ----------------------------------------------
    def _domain(self):
        return tldextract.extract(self.site).registered_domain if self.site else ""

    # ---- 1. руководители ------------------------------------------
    async def _get_names(self, sess):
        roles = ["генеральный директор","CEO","учредитель",
                 "собственник","владельцы","основатель","директор"]
        dom  = self._domain()
        site_ctx = await _site_snip(sess, dom)

        tasks=[]
        for r in roles:
            q = f'"{self.c}" {r}'
            if dom: q += f' OR site:{dom}'
            tasks.append(_snip(sess, q, 4))
        snips = sum(await asyncio.gather(*tasks), [])

        ctx = ""
        if site_ctx:
            ctx += f"SITE_CONTEXT:\n{site_ctx}\nрынок компании – {self.market}\n\n"
        ctx += "\n".join(f"URL:{u}\nTXT:{t}" for u,t in snips)[:10_000]

        if not snips:
            return []

        sys = ("ТЫ — ПРОФЕССИОНАЛЬНЫЙ АНАЛИТИК КОМПАНИЙ. НАЙДИ И ВЕРНИ ТОЧНЫЙ СПИСОК "
               "РУКОВОДИТЕЛЕЙ/ВЛАДЕЛЬЦЕВ ИМЕННО ЭТОЙ КОМПАНИИ, "
               "ИГНОРИРУЯ ОДНОФАМИЛЬЦЕВ И НЕСВЯЗАННЫЕ УПОМИНАНИЯ. "
               "ФОРМАТ: «ФИО (роль)» на каждую строку, без лишнего.")
        txt = await gpt_async([{"role":"system","content":sys},
                               {"role":"user",  "content":ctx}], T=0.12)
        return [ln.strip() for ln in txt.splitlines() if ln.strip()]

    # ---- 2. интервью ----------------------------------------------
    async def _get_interviews(self, sess, names):
        dom = self._domain()
        if not names:
            return [], [], "Свежих интервью не найдено."
    
        all_snips, queries = [], []
        for fio in names:
            fio_cut = fio.split("(")[0].strip()
            q_tpl   = ["интервью","интервью 2024","интервью 2023",
                       "комментарий","exclusive interview","цитата"]
    
            for q in q_tpl:
                full_q = f"{fio_cut} {self.c} {q}"
                if dom:
                    full_q += f' OR site:{dom}'
                queries.append(full_q)
                # ↓ теперь используем именно full_q
                all_snips += await _snip(sess, full_q, 3)
    
        if not all_snips:
            return queries, [], "Свежих интервью не найдено."

        # ---- строим контекст для GPT --------------------------------
        site_ctx = await _site_snip(sess, dom)
        ctx = ""
        if site_ctx:
            ctx += f"SITE_CONTEXT:\n{site_ctx}\nрынок компании – {self.market}\n\n"
        ctx += "\n".join(f"URL:{u}\nTXT:{t}" for u,t in all_snips)[:16_000]

        sys = ("ТЫ — АНАЛИТИК-КОНТЕНТОЛОГ. СОЗДАЙ ДАЙДЖЕСТ ТОЛЬКО ТЕХ ИНТЕРВЬЮ, "
               "КОТОРЫЕ ОТНОСЯТСЯ К НУЖНОЙ КОМПАНИИ И ЕЁ СФЕРЕ. "
               "ЕСЛИ СНИППЕТ НЕ СООТВЕТСТВУЕТ КОНТЕКСТУ САЙТА/РЫНКА — ПРОПУСТИ ЕГО. "
               "ДЛЯ КАЖДОГО ОСТАВШЕГОСЯ ИНТЕРВЬЮ ВЫВЕДИ: ФИО, роль, дата, краткая суть (≤2 предложения), ссылка.")
        summary = await gpt_async([{"role":"system","content":sys},
                                   {"role":"user",  "content":ctx}], T=0.15)
        return queries, all_snips, summary

    # ---- orchestrator ---------------------------------------------
    async def _run(self):
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as sess:
            names = await self._get_names(sess)
            int_q, snips, int_sum = await self._get_interviews(sess, names)

        owners_block = ("Топ-менеджеры и владельцы:\n" + "\n".join(names)
                        if names else "Топ-менеджеры и владельцы не найдены.")
        summary_html = _linkify(owners_block + "\n\n" + int_sum)

        return {"summary": summary_html,
                "names": names,
                "queries": int_q,
                "snippets": snips}

    def run(self):
        return asyncio.run(self._run())

@st.cache_data(
        ttl=86_400,
        show_spinner="🔎 Ищем руководителей и интервью…")
def get_leaders_rag(company:str, website:str="", market:str=""):
    """
    Thin wrapper: возвращает dict(summary, names, queries, snippets)
    с кешированием на сутки.
    """
    return FastLeadersInterviews(company, website, market).run()







@st.cache_data(ttl=3_600)
def ck_company(inn):
    r = requests.get("https://api.checko.ru/v2/company",
                     params={"key": KEYS["CHECKO_API_KEY"], "inn": inn}, timeout=10)
    r.raise_for_status(); return r.json()["data"]

@st.cache_data(ttl=3_600)
def ck_fin(inn):
    r = requests.get("https://api.checko.ru/v2/finances",
                     params={"key": KEYS["CHECKO_API_KEY"], "inn": inn}, timeout=10)
    r.raise_for_status(); return r.json()["data"]







# ╭─🎛  UI ────────────────────────────────────────────╮
st.title("📊 AI Company Insight")
st.markdown("Введите данные (каждая компания — в отдельной строке).")

c1,c2,c3,c4=st.columns(4)
with c1: inns = st.text_area("ИНН", key="inns")
with c2: names= st.text_area("Название", key="names")
with c3: mkts = st.text_area("Рынок", key="mkts")
with c4: sites= st.text_area("Сайт", key="sites")

split = lambda s:[i.strip() for i in s.splitlines() if i.strip()]
inns,names,mkts,sites = map(split, (inns,names,mkts,sites))
if sites and len(sites)!=len(names):
    st.warning("Число строк «Сайт» должно совпадать."); st.stop()

if st.button("🔍 Получить данные"):
    # --- валидация ---
    if not (inns and names and mkts):
        st.error("Заполните все поля."); st.stop()
    if len({len(inns), len(names), len(mkts)}) != 1:
        st.error("Число строк должно совпадать."); st.stop()

    # --- вкладки по компаниям ---
    # --- вкладки по компаниям ---
    tabs = st.tabs([f"{n} ({inn})" for inn, n in zip(inns, names)])
    YEARS = ["2022", "2023", "2024"]
    
    # --- Подготовим список сайтов такой же длины как и имена компаний
    sites_full = sites if sites and len(sites) == len(names) else [""] * len(names)
    
    for tab, inn, name, mkt, site in zip(tabs, inns, names, mkts, sites_full):
        with tab:
            st.header(f"{name} — {inn}")
            st.caption(f"Рынок: **{mkt}**")
    
            # --------- Финансовый профиль ---------
            st.subheader("💰 Финансовый профиль компании")
            fin_raw = ck_fin(inn)
            C_REVENUE, C_EBIT, C_CASH = "2110", "2200", "1250"
            C_DEBT_LT, C_DEBT_ST = "1400", "1500"
    
            revenue, ebit, cash, total_debt, net_debt, margin, netdebt_ebit = {}, {}, {}, {}, {}, {}, {}
            for y in YEARS:
                yr = fin_raw.get(y, {})
                revenue[y] = (yr.get(C_REVENUE) / 1e6) if isinstance(yr.get(C_REVENUE), (int, float)) else None
                ebit[y]    = (yr.get(C_EBIT)    / 1e6) if isinstance(yr.get(C_EBIT), (int, float))    else None
                cash[y]    = (yr.get(C_CASH)    / 1e6) if isinstance(yr.get(C_CASH), (int, float))    else None
                debt_lt    = (yr.get(C_DEBT_LT) / 1e6) if isinstance(yr.get(C_DEBT_LT), (int, float)) else 0
                debt_st    = (yr.get(C_DEBT_ST) / 1e6) if isinstance(yr.get(C_DEBT_ST), (int, float)) else 0
                total_debt[y] = debt_lt + debt_st if (debt_lt or debt_st) else None
                net_debt[y]   = (total_debt[y] - cash[y]) if (total_debt[y] is not None and cash[y] is not None) else None
                margin[y]     = (ebit[y] / revenue[y] * 100) if (revenue[y] not in (None, 0) and ebit[y] is not None) else None
                netdebt_ebit[y] = (net_debt[y] / ebit[y]) if (ebit[y] not in (None, 0) and net_debt[y] is not None) else None
    
            table_df = pd.DataFrame({
                "Метрика": [
                    "Выручка (₽ млн)", "EBIT (₽ млн)", "EBIT margin (%)",
                    "Совокупный долг (₽ млн)", "Денежные средства (₽ млн)",
                    "Чистый долг (₽ млн)", "Net Debt / EBIT"
                ]
            })
            for y in YEARS:
                table_df[y] = [
                    revenue[y], ebit[y], margin[y],
                    total_debt[y], cash[y], net_debt[y], netdebt_ebit[y]
                ]
    
            def fmt(v, is_percent=False, n=1):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return f"{v:.{n}f}%" if is_percent else f"{v:.{n}f}"
    
            table_df.iloc[2, 1:] = [fmt(v, is_percent=True) for v in table_df.iloc[2, 1:]]  # EBIT margin
            table_df.iloc[6, 1:] = [fmt(v, is_percent=False, n=2) for v in table_df.iloc[6, 1:]]  # Net Debt/EBIT
            for i in [0, 1, 3, 4, 5]:
                table_df.iloc[i, 1:] = [fmt(v) for v in table_df.iloc[i, 1:]]
    
            st.dataframe(table_df.set_index("Метрика"), use_container_width=True)
    
            # --- красивый график ---
            fig, ax1 = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(YEARS)); w = 0.3
            bars1 = ax1.bar(x - w/2, [revenue[y] if revenue[y] is not None else 0 for y in YEARS], w, label="Выручка")
            bars2 = ax1.bar(x + w/2, [ebit[y] if ebit[y] is not None else 0 for y in YEARS], w, label="EBIT")
    
            # Подписи столбцов
            for b in (*bars1, *bars2):
                h = b.get_height()
                if h and not np.isnan(h):
                    ax1.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h), (0, 3),
                                 textcoords="offset points", ha="center", fontsize=9)
    
            ax2 = ax1.twinx()
            m_vals = [margin[y] if margin[y] is not None else np.nan for y in YEARS]
            ax2.plot(x, m_vals, color="#2196f3", linestyle="--", marker='o', label="EBIT margin, %")
            for xx, yy in zip(x, m_vals):
                if yy is not None and not np.isnan(yy):
                    ax2.annotate(f"{yy:.1f}%", (xx, yy), (0, 5), textcoords="offset points",
                                 ha='center', fontsize=9, color="#2196f3")
    
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(y) for y in YEARS], fontsize=10)
            ax1.set_xlim(-0.7, len(YEARS) - 0.3); ax1.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            for ax in (ax1, ax2):
                ax.spines[:].set_visible(False)
            ax1.legend(loc="upper left", fontsize=9)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)




            
            
            st.subheader("📝 Описание компании")
            
            with st.spinner("Генерируем описание компании…"):
                res_p = RAG(name, website=site, market=mkt).run()   # ↓  только вызов
            
            #  ▸ всё, что использует результат, пишем уже снаружи
            passport_html = _linkify(res_p['summary']).replace('\n', '<br>')
            st.markdown(
                f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                f"border-radius:8px;padding:18px;line-height:1.55'>{passport_html}</div>",
                unsafe_allow_html=True
            )
            
            with st.expander("⚙️ Запросы"):
                for i, q in enumerate(res_p["queries"], 1):
                    st.markdown(f"**{i}.** {q}")
            
            with st.expander("🔍 Сниппеты (top-15)"):
                st.dataframe(
                    pd.DataFrame(res_p["snippets"], columns=["URL", "Snippet"]).head(15),
                    use_container_width=True
            )




            # ───────────────────  📈 Рыночный отчёт  ────────────────────
            if mkt:                                   # если поле «Рынок» заполнено
                st.subheader("📈 Рыночный отчёт")
                with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                    mkt_res = get_market_rag(mkt)
            
                market_html = _linkify(mkt_res['summary']).replace(chr(10), '<br>')
                st.markdown(
                    f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                    f"border-radius:8px;padding:18px;line-height:1.55'>{market_html}</div>",
                    unsafe_allow_html=True)
            
                with st.expander("⚙️ Запросы к Google"):
                    for i, q in enumerate(mkt_res["queries"], 1):
                        st.markdown(f"**{i}.** {q}")
            
                with st.expander("🔍 Сниппеты (top-15)"):
                    st.dataframe(
                        pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                        use_container_width=True)


                                
            # ─────────── 👥 Руководители и интервью ────────────
            st.subheader("👥 Руководители и интервью")
            with st.spinner("Собираем руководителей и интервью…"):
                lead_res = get_leaders_rag(name, website=site, market=mkt)
            st.markdown(
                f"<div style='background:#F9FAFB;border:1px solid #ddd;"
                f"border-radius:8px;padding:18px;line-height:1.55'>"
                f"{lead_res['summary'].replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True)
                
            with st.expander("⚙️ Запросы к Google"):
                for i, q in enumerate(lead_res["queries"][:20], 1):
                    st.markdown(f"**{i}.** {q}")
            
            with st.expander("🔍 Сниппеты (top-15)"):
                if lead_res["snippets"]:
                    df = (pd.DataFrame(lead_res["snippets"], columns=["URL", "Snippet"])
                            .drop_duplicates(subset="URL").head(15))
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Сниппеты не найдены.")
                                
                    
                    
                    
                    







# In[9]:




