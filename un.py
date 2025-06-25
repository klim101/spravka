#!/usr/bin/env python
# coding: utf-8

# In[4]:


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



KEYS = {
    "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "GOOGLE_CX":      st.secrets["GOOGLE_CX"],
    "CHECKO_API_KEY": st.secrets["CHECKO_API_KEY"],
}


# In[2]:


def make_full_word_report(company, inn, market, passport, finances_df, fig, market_report):
    doc = Document()
    doc.add_heading(f"Отчёт по компании: {company}", 0)
    doc.add_paragraph(f"ИНН: {inn}")
    if market:
        doc.add_paragraph(f"Рынок: {market}")

    doc.add_heading("Паспорт компании", level=1)
    for para in passport.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_heading("Финансовый профиль", level=1)
    # Таблица финансов (finances_df — DataFrame)
    table = doc.add_table(rows=1, cols=len(finances_df.columns)+1)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Метрика"
    for i, col in enumerate(finances_df.columns):
        hdr_cells[i+1].text = str(col)
    for idx, row in finances_df.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(idx)
        for j, val in enumerate(row):
            row_cells[j+1].text = str(val)
    doc.add_paragraph("")

    # График (fig — matplotlib Figure)
    img_stream = BytesIO()
    fig.savefig(img_stream, format='png', bbox_inches='tight')
    img_stream.seek(0)
    doc.add_picture(img_stream, width=Inches(5.5))
    doc.add_paragraph("")

    doc.add_heading("GPT-анализ рынка", level=1)
    for para in market_report.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output


# In[ ]:


# ─────────────────── app.py ────────────────────
import os, re, asyncio, aiohttp, requests, nest_asyncio, logging
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

nest_asyncio.apply()
logging.basicConfig(level=logging.WARNING)
import openai


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
    def __init__(self, company, country="Россия", steps=2, snips=3):
        self.company, self.country, self.steps, self.snips = company, country, steps, snips
    async def _queries(self, hist=""):
        sys  = "Ты — исследователь. Придумай 5–10 Google-запросов…"
        ask  = f"Нужно описание {self.company} ({self.country}). Формат: QUERY: <...>"
        raw  = await _gpt([{"role": "system", "content": sys},
                           {"role": "user",   "content": ask + hist}], T=0.1)
        return re.findall(r"QUERY:\s*(.+)", raw, flags=re.I)
    async def _summary(self, ctx):
        sys = ("Ты — аналитик рынков. Составь абзацы: описание, общая инф., партнёрства, "
               "направления, история, цифры, продукты, география, сотрудники, уникальность, выводы. "
               "Без Markdown и без упоминаний выручки.")
        return await _gpt([{"role": "system", "content": sys},
                           {"role": "user",   "content": ctx}], T=0.25)
    async def _run_async(self):
        queries, snippets, hist = [], [], ""
        async with aiohttp.ClientSession() as s:
            for _ in range(self.steps):
                ql = await self._queries(hist);  queries += ql
                for res in await asyncio.gather(*[_google(s, q, self.snips) for q in ql]):
                    snippets += res
                hist = f"\nСниппетов: {len(snippets)}"
        ctx = "\n".join(f"URL: {u}\nTXT: {t}" for u, t in snippets)[:20_000]
        return {"summary": await self._summary(ctx), "queries": queries, "snippets": snippets}
    def run(self):
        return asyncio.get_event_loop().run_until_complete(self._run_async())

@st.cache_data(ttl=86_400)
def get_rag(name): return RAG(name).run()

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
            "Ты — исследователь рынков. Сформулируй 7-12 Google-запросов, "
            f"чтобы найти объём, структуру, сегменты, динамику, игроков и тренды "
            f"рынка «{self.market}» ({self.country}) за {', '.join(map(str, self.years))}. "
            "Формат: QUERY: <строка запроса>")
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
            f"Ты — аналитик по рынку «{self.market}» ({self.country}). "
            "Для каждого года напиши абзац: объём, рост, сегменты, регионы, "
            "игроки/доли, сделки, ценовые срезы, тренды, барьеры, вывод. "
            "Уникальные факты, без повторов, источники в скобках.")
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







# ╭─🌐  Leaders & Interviews (super-fast) ─────────────────────╮
import aiohttp, asyncio, re, html, logging, openai, streamlit as st

HEADERS = {"User-Agent": "Mozilla/5.0"}           # короткий UA

# ── helper: превращаем URL в кликабельную ссылку ------------
_URL_PAT = re.compile(r"https?://[^\s)]+")
def _linkify(text: str) -> str:
    def repl(m):
        u = html.escape(m.group(0))
        return f'<a href="{u}" target="_blank">ссылка</a>'
    return _URL_PAT.sub(repl, text)

# ── один запрос к Google CSE (только сниппеты) --------------
async def _snip(sess: aiohttp.ClientSession, query: str, n: int = 4):
    q = re.sub(r'[\"\'“”]', '', query)[:90]
    params = {
        "key": KEYS["GOOGLE_API_KEY"],
        "cx":  KEYS["GOOGLE_CX"],
        "q":   q, "num": n, "hl": "ru", "gl": "ru"
    }
    try:
        async with sess.get("https://www.googleapis.com/customsearch/v1",
                            params=params, headers=HEADERS, timeout=8) as r:
            if r.status != 200:
                logging.warning(f"[Google] {r.status}")
                return []
            js = await r.json()
            return [(it["link"], it.get("snippet", ""))
                    for it in js.get("items", []) if not _bad(it["link"])]
    except asyncio.TimeoutError:
        logging.warning("[Google] timeout")
        return []

# ── основной класс ------------------------------------------
class FastLeadersInterviews:
    """⚡️ Выдаёт dict(summary, names, queries, snippets) за ~10-15 с."""
    def __init__(self, company: str, country: str = "Россия"):
        self.c, self.cntry = company, country

    # ---------- 1. ФИО --------------------------------------
    async def _get_names(self, sess):
        roles = ["генеральный директор", "CEO", "учредитель",
                 "собственник", "владельцы", "основатель", "директор"]
        tasks = [_snip(sess, f"{self.c} {self.cntry} {r}", 4) for r in roles]
        snippets = sum(await asyncio.gather(*tasks), [])
        if not snippets:
            return []
        ctx = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in snippets)[:10_000]
        sys = ("Выдели всех руководителей/владельцев компании: "
               "верни список «ФИО (роль)» без лишнего.")
        txt = await gpt_async([{"role": "system", "content": sys},
                               {"role": "user",   "content": ctx}], T=0.12)
        return [ln.strip() for ln in txt.splitlines() if ln.strip()]

    # ---------- 2. Интервью ---------------------------------
    async def _get_interviews(self, sess, names):
        if not names:
            return [], [], "Свежих интервью не найдено."
        all_snips, queries = [], []
        for fio in names:
            fio_cut = fio.split("(")[0].strip()
            q_tpl = ["интервью", "интервью 2024", "интервью 2023",
                     "комментарий", "exclusive interview", "цитата"]
            queries += [f"{fio_cut} {self.c} {q}" for q in q_tpl]
            tasks = [_snip(sess, f"{fio_cut} {self.c} {q}", 3) for q in q_tpl]
            all_snips += sum(await asyncio.gather(*tasks), [])
        if not all_snips:
            return queries, [], "Свежих интервью не найдено."
        ctx = "\n".join(f"URL:{u}\nTXT:{t}" for u, t in all_snips)[:16_000]
        sys = ("Собери краткий дайджест интервью: ФИО, роль, дата (если есть), "
               "краткая суть, ссылка. Без повторов.")
        summary = await gpt_async([{"role": "system", "content": sys},
                                   {"role": "user",   "content": ctx}], T=0.15)
        return queries, all_snips, summary

    # ---------- orchestrator --------------------------------
    async def _run(self):
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as sess:

            names                = await self._get_names(sess)
            int_q, snips, intsum = await self._get_interviews(sess, names)

        owners_block = ("Топ-менеджеры и владельцы:\n" + "\n".join(names)
                        if names else "Топ-менеджеры и владельцы не найдены.")
        summary_html = _linkify(owners_block + "\n\n" + intsum)

        return {"summary": summary_html, "names": names,
                "queries": int_q, "snippets": snips}

    def run(self):
        return asyncio.run(self._run())

# ── Streamlit-кэш (24 ч) -----------------------------------
@st.cache_data(ttl=86_400,
               show_spinner="🔎 Ищем руководителей и интервью…")
def get_leaders_rag(company: str):
    return FastLeadersInterviews(company).run()











# ╭─🎛  UI ────────────────────────────────────────────╮
st.title("📊 Company Insight — Checko + GPT-4o")
st.markdown("Введите данные (каждая компания — в одной строке):")

c1, c2, c3 = st.columns(3)
with c1: inns  = st.text_area("ИНН",        key="inns_in")
with c2: names = st.text_area("Название",   key="names_in")
with c3: mkts  = st.text_area("Рынок",      key="mkts_in")

inns  = [i.strip() for i in inns.splitlines()  if i.strip()]
names = [i.strip() for i in names.splitlines() if i.strip()]
mkts  = [i.strip() for i in mkts.splitlines()  if i.strip()]

if st.button("🔍 Получить данные"):
    # --- валидация ---
    if not (inns and names and mkts):
        st.error("Заполните все поля."); st.stop()
    if len({len(inns), len(names), len(mkts)}) != 1:
        st.error("Число строк должно совпадать."); st.stop()

    # --- вкладки по компаниям ---
    tabs = st.tabs([f"{n} ({inn})" for inn, n in zip(inns, names)])
    YEARS = ["2022", "2023", "2024"]
    for tab, inn, name, mkt in zip(tabs, inns, names, mkts):
        with tab:
            st.header(f"{name} — {inn}")
            st.caption(f"Рынок: **{mkt}**")
    
            # --- Checko profile ---
            st.subheader("🪪 Профиль (Checko)")
            st.json(ck_company(inn), expanded=False)
    
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
    
            # --- GPT паспорт ---
            st.subheader("📝 GPT-паспорт")
            res = get_rag(name)
            st.markdown(
                f"<div style='background:#F7F9FA;border:1px solid #ccc;"
                f"border-radius:8px;padding:18px;line-height:1.55'>"
                f"{res['summary'].replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True)
    
            with st.expander("⚙️ Запросы"):
                for i, q in enumerate(res["queries"], 1):
                    st.markdown(f"**{i}.** {q}")
    
            with st.expander("🔍 Сниппеты (top-15)"):
                st.dataframe(pd.DataFrame(res["snippets"],
                            columns=["URL", "Snippet"]).head(15),
                            use_container_width=True)
    




            # ───────────────────  📈 Рыночный отчёт  ────────────────────
            if mkt:                                   # если поле «Рынок» заполнено
                st.subheader("📈 Рыночный отчёт")
                with st.spinner("Собираем данные по рынку и генерируем анализ…"):
                    mkt_res = get_market_rag(mkt)
            
                st.markdown(
                    f"<div style='background:#F1F5F8;border:1px solid #cfd9e2;"
                    f"border-radius:8px;padding:18px;line-height:1.55'>"
                    f"{mkt_res['summary'].replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True)
            
                with st.expander("⚙️ Запросы к Google"):
                    for i, q in enumerate(mkt_res["queries"], 1):
                        st.markdown(f"**{i}.** {q}")
            
                with st.expander("🔍 Сниппеты (top-15)"):
                    st.dataframe(
                        pd.DataFrame(mkt_res["snippets"], columns=["URL", "Snippet"]).head(15),
                        use_container_width=True)

                if st.button(f"💾 Скачать полный Word-отчёт — {name}", key=f"word_{inn}"):
                    # --- паспорт компании ---
                    passport_text = res["summary"]
                    # --- финансы ---
                    # table_df (у тебя он уже есть, но индексы — строки)
                    finances_df = table_df.set_index("Метрика") if "Метрика" in table_df.columns else table_df
                    # --- график (fig) --- он у тебя создан выше
                    # --- рыночный отчет ---
                    market_text = mkt_res["summary"] if mkt else ""
   



                    passport_text = res["summary"]
                    finances_df = table_df.set_index("Метрика") if "Метрика" in table_df.columns else table_df
                    market_text = mkt_res["summary"] if mkt else ""
                                
            # ─────────── 👥 Руководители и интервью ────────────
            st.subheader("👥 Руководители и интервью")
            with st.spinner("Собираем руководителей и интервью…"):
                lead_res = get_leaders_rag(name)
            
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
                                
                    
                    
                    
                    






