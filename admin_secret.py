# admin_secret.py
from __future__ import annotations
import datetime as dt
from typing import Optional, Sequence
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# 1) Встраиваем слушатель безопасной клавиатурной последовательности
#    По умолчанию — Konami (без Ctrl/Shift), чтобы браузер не перехватывал хоткей.
#    При совпадении -> добавляем ?admin=1 к URL и перезагружаем страницу.
# ─────────────────────────────────────────────────────────────────────────────
_KONAMI = ["ArrowUp","ArrowUp","ArrowDown","ArrowDown","ArrowLeft","ArrowRight","ArrowLeft","ArrowRight","b","a","Enter"]

def inject_secret_listener(sequence: Optional[Sequence[str]] = None, key: str = "admin_secret_js") -> None:
    seq = list(sequence or _KONAMI)
    # Важно: height=0, чтобы не занимать место; use_container_width не нужен
    components.html(f"""
<div></div>
<script>
(function() {{
  const seq = {seq!r};
  let buf = [];
  // ловим на самом верхнем уровне, без модификаторов
  window.addEventListener('keydown', function(e) {{
    const k = e.key;
    // Игнорируем, если нажаты модификаторы — чтобы не конфликтовать с браузером
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    buf.push(k);
    if (buf.length > seq.length) buf.shift();
    let ok = true;
    for (let i=0; i<seq.length; i++) {{
      if (buf[i] !== seq[i]) {{ ok = false; break; }}
    }}
    if (ok) {{
      try {{
        const url = new URL(window.parent.location.href);
        url.searchParams.set('admin','1');
        window.parent.history.replaceState({{}},'', url);
        window.parent.location.reload();
      }} catch(err) {{
        // fall back — просто редирект с параметром
        window.parent.location.href = window.parent.location.href.split('#')[0].split('?')[0] + '?admin=1';
      }}
    }}
  }}, true);
}})();
</script>
""", height=0, key=key)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Инициализация режима админа по query-параметру + кнопка выхода
# ─────────────────────────────────────────────────────────────────────────────
def init_admin_mode(auto_inject: bool = True, sequence: Optional[Sequence[str]] = None) -> bool:
    if auto_inject:
        inject_secret_listener(sequence=sequence)

    params = st.experimental_get_query_params()
    is_admin = (params.get("admin", ["0"])[0] == "1")
    return is_admin

def exit_admin():
    st.experimental_set_query_params(admin="0")
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Рендер админ-панели: график часов по сотрудникам за выбранный период
#    Автоматически подбираем колоноки по распространённым названиям.
# ─────────────────────────────────────────────────────────────────────────────
_EMP_COL_CANDIDATES = ["employee","user","name","first_name","ФИО","Сотрудник","сотрудник"]
_DATE_COL_CANDIDATES = ["work_date","date","Дата","дата"]
_HOURS_COL_CANDIDATES = ["hours","Часы","часы"]

def _guess_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # второй шанс: по типам
    if candidates is _HOURS_COL_CANDIDATES:
        # самая "часовая" колонка — числовая с макс. средним значением в разумном диапазоне
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return num_cols[0] if num_cols else None
    if candidates is _DATE_COL_CANDIDATES:
        date_like = [c for c in df.columns if "date" in c.lower() or "дата" in c.lower()]
        return date_like[0] if date_like else None
    return None

def render_admin_panel(
    df_times: pd.DataFrame,
    employee_col: Optional[str] = None,
    date_col: Optional[str] = None,
    hours_col: Optional[str] = None,
    title: str = "🔧 Админ-панель — загрузка по часам",
    default_days: int = 30,
):
    st.title(title)
    st.caption("Секретная панель. Нажмите «Выйти», чтобы вернуться в обычный режим.")
    st.button("⏏️ Выйти из админ-режима", on_click=exit_admin)

    if df_times is None or df_times.empty:
        st.warning("Нет данных для построения графика. Передайте DataFrame в render_admin_panel().")
        return

    # авто-детект колонок
    employee_col = employee_col or _guess_col(df_times, _EMP_COL_CANDIDATES)
    date_col     = date_col     or _guess_col(df_times, _DATE_COL_CANDIDATES)
    hours_col    = hours_col    or _guess_col(df_times, _HOURS_COL_CANDIDATES)

    missing = [("сотрудник", employee_col), ("дата", date_col), ("часы", hours_col)]
    missing = [label for label, col in missing if not col]
    if missing:
        st.error("Не удалось определить колонки: " + ", ".join(missing))
        st.info("Подсказка: укажите аргументы employee_col=…, date_col=…, hours_col=… вручную.")
        st.dataframe(df_times.head())
        return

    # приводим типы
    df = df_times.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=[date_col])
    if not pd.api.types.is_numeric_dtype(df[hours_col]):
        df[hours_col] = pd.to_numeric(df[hours_col], errors="coerce")
    df = df.dropna(subset=[hours_col])

    # диапазон дат
    today = dt.date.today()
    min_d = df[date_col].min() or (today - dt.timedelta(days=default_days))
    max_d = df[date_col].max() or today
    default_start = max(max_d - dt.timedelta(days=default_days), min_d)

    start, end = st.date_input(
        "Период",
        (default_start, max_d),
        min_value=min_d,
        max_value=max_d
    )
    if isinstance(start, tuple):
        # Streamlit < 1.30 иногда так отдаёт значение
        start, end = start

    # фильтр
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    dff = df.loc[mask].copy()

    # агрегация по сотрудникам
    agg = (
        dff.groupby(employee_col, dropna=False)[hours_col]
        .sum()
        .reset_index()
        .rename(columns={employee_col: "Сотрудник", hours_col: "Часы"})
    )
    agg = agg.sort_values("Часы", ascending=False)

    c = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            y=alt.Y("Сотрудник:N", sort="-x", title="Сотрудник"),
            x=alt.X("Часы:Q", title="Сумма часов"),
            tooltip=["Сотрудник", "Часы"]
        )
        .properties(height=max(300, 20 * len(agg)), width="container")
    )
    st.altair_chart(c, use_container_width=True)

    # Дополнительно: динамика по дням (stacked area)
    with st.expander("Показать динамику по дням"):
        dline = (
            dff.groupby([date_col, employee_col])[hours_col]
            .sum()
            .reset_index()
            .rename(columns={date_col: "Дата", employee_col: "Сотрудник", hours_col: "Часы"})
        )
        lc = (
            alt.Chart(dline)
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X("Дата:T", title="Дата"),
                y=alt.Y("Часы:Q", stack="zero", title="Часы"),
                color=alt.Color("Сотрудник:N", legend=alt.Legend(title="Сотрудник")),
                tooltip=["Дата","Сотрудник","Часы"]
            )
            .properties(height=350, width="container")
        )
        st.altair_chart(lc, use_container_width=True)
