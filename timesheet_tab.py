# timesheet_tab.py
# -*- coding: utf-8 -*-
"""
Streamlit-вкладка «Timesheet» для внесения часов в таблицу log.
Похоже на nikatime.com: недельная сетка (Пн–Вс), строки — проекты, ячейки — часы.
Хранение — в PostgreSQL Supabase (вам не нужно отдельно управлять PostgreSQL — это тот же движок под капотом Supabase).

Зависимости (в requirements.txt):
    streamlit
    sqlalchemy
    psycopg2-binary
    pandas

Требуемые секреты (Streamlit → Secrets):
    # возьмите в Supabase: Project Settings → Database → Connection string → SQLAlchemy
    POSTGRES_DSN = "postgresql+psycopg2://postgres:*****@db.<id>.supabase.co:5432/postgres"

Опционально:
    DEFAULT_TG_ID = 123456789  # если хотите предвыбирать пользователя по tg_id при первом заходе
"""
from __future__ import annotations



import streamlit as st
from sqlalchemy.engine import URL


import socket
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

@st.cache_resource(show_spinner=False)
def get_engine():
    # 1) Если есть готовый DSN в секретах/переменных — используем его БЕЗ вариантов
    dsn = (
        st.secrets.get("POSTGRES_DSN")
        or st.secrets.get("SUPABASE_DB_URL")
        or os.getenv("DATABASE_URL")
    )
    if dsn:
        eng = create_engine(dsn, pool_pre_ping=True, pool_recycle=1800, future=True)
        # Небольшая проверка и подсказка, каким драйвером реально подключились
        try:
            with eng.connect() as c:
                c.exec_driver_sql("SELECT 1")
            st.caption(f"DB OK · {eng.dialect.name}+{eng.dialect.driver}")
        except Exception as e:
            st.error(f"Не удалось подключиться по POSTGRES_DSN: {e}")
            st.stop()
        return eng

    # 2) Запасной путь (если вдруг DSN не дали): собираем URL под pg8000 с SSL
    host = st.secrets["SUPA_HOST"]
    user = st.secrets.get("SUPA_USER", "postgres")
    pwd  = st.secrets["SUPA_PASSWORD"]
    db   = st.secrets.get("SUPA_DB", "postgres")

    url = URL.create(
        "postgresql+pg8000",
        username=user,
        password=pwd,
        host=host,
        port=5432,
        database=db,
        query={"ssl": "true"},
    )
    eng = create_engine(url, pool_pre_ping=True, pool_recycle=1800, future=True)
    with eng.connect() as c:
        c.exec_driver_sql("SELECT 1")
    st.caption(f"DB OK · {eng.dialect.name}+{eng.dialect.driver}")
    return eng




import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect

# ──────────────────────────────────────────────────────────────────────────────
# DB utilities
# ──────────────────────────────────────────────────────────────────────────────

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

def _dsn_from_secrets() -> str:
    import os, streamlit as st

    dsn = (
        st.secrets.get("POSTGRES_DSN")
        or st.secrets.get("SUPABASE_DB_URL")
        or os.getenv("DATABASE_URL")
    )
    if not dsn:
        raise RuntimeError("Не найден DSN к PostgreSQL Supabase. Добавьте POSTGRES_DSN (или SUPABASE_DB_URL) в st.secrets.")

    # Нормализуем префикс:
    dsn = dsn.replace("postgres://", "postgresql://", 1)

    # Всегда используем pg8000 + SSL
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+pg8000://", 1)
    if ("sslmode=" not in dsn) and ("ssl=" not in dsn):
        dsn += ("&" if "?" in dsn else "?") + "ssl=true"

    return dsn




@st.cache_resource(show_spinner=False)
def get_engine():
    dsn = _dsn_from_secrets()
    eng = create_engine(dsn, pool_pre_ping=True, pool_recycle=1800)
    return eng


def _detect_user_table(engine) -> str:
    """
    В некоторых БД таблица пользователей называется 'user' (зарезерв. слово в SQL),
    у вас — вероятно 'klim101'. Определим корректное имя.
    """
    insp = inspect(engine)
    names = {t.lower() for t in insp.get_table_names()}
    if "klim101" in names:
        return "klim101"
    if "user" in names:
        return '"user"'  # экранируем зарезервированное слово
    # на крайний случай создадим 'klim101'
    return "klim101"


def ensure_db():
    """
    Создаёт недостающие таблицы/индексы. Безопасно выполнять при каждом старте.
    Ничего не трёт.
    """
    eng = get_engine()
    user_tbl = _detect_user_table(eng)

    ddl_user = f"""
    CREATE TABLE IF NOT EXISTS {user_tbl} (
        id         BIGINT PRIMARY KEY,
        tg_id      BIGINT NOT NULL,
        first_name VARCHAR NOT NULL
    );
    """

    ddl_project = """
    CREATE TABLE IF NOT EXISTS project (
        id   BIGINT PRIMARY KEY,
        name VARCHAR NOT NULL UNIQUE
    );
    """

    ddl_log = f"""
    CREATE TABLE IF NOT EXISTS log (
        id         BIGINT PRIMARY KEY,
        user_id    BIGINT NOT NULL REFERENCES {user_tbl} (id),
        project_id BIGINT NOT NULL REFERENCES project (id),
        work_date  DATE    NOT NULL,
        hours      FLOAT   NOT NULL
    );
    """

    # Индекс для быстрых апдейтов/вставок по ключу (user, project, date)
    ddl_uniq = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes WHERE indexname = 'ux_log_user_project_date'
        ) THEN
            CREATE UNIQUE INDEX ux_log_user_project_date
            ON log (user_id, project_id, work_date);
        END IF;
    END $$;
    """

    with eng.begin() as con:
        con.execute(text(ddl_user))
        con.execute(text(ddl_project))
        con.execute(text(ddl_log))
        # попытка создать uniq индекс; если уже есть — игнорируем
        try:
            con.execute(text(ddl_uniq))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TimesheetWeek:
    monday: date
    dates: List[date]  # 7 дат Пн–Вс

    @staticmethod
    def from_any(d: date | datetime | None = None) -> "TimesheetWeek":
        if d is None:
            d = date.today()
        d = d.date() if isinstance(d, datetime) else d
        monday = d - timedelta(days=(d.weekday() % 7))  # Monday=0
        return TimesheetWeek(monday=monday, dates=[monday + timedelta(days=i) for i in range(7)])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_projects() -> pd.DataFrame:
    eng = get_engine()
    q = text("SELECT id, name FROM project ORDER BY name")
    return pd.read_sql(q, eng)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_users() -> pd.DataFrame:
    eng = get_engine()
    user_tbl = _detect_user_table(eng)
    q = text(f"SELECT id, first_name, tg_id FROM {user_tbl} ORDER BY first_name")
    return pd.read_sql(q, eng)


def fetch_week_hours(user_id: int, week: TimesheetWeek) -> pd.DataFrame:
    eng = get_engine()
    q = text("""
        SELECT project_id, work_date, SUM(hours) AS hours
        FROM log
        WHERE user_id = :uid AND work_date BETWEEN :d1 AND :d7
        GROUP BY project_id, work_date
    """)
    df = pd.read_sql(q, eng, params={"uid": user_id, "d1": week.dates[0], "d7": week.dates[-1]})
    return df


def _next_log_ids(con, n: int) -> List[int]:
    cur = con.execute(text("SELECT COALESCE(MAX(id), 0) FROM log"))
    start = int(cur.scalar() or 0) + 1
    return list(range(start, start + n))


def upsert_week(payload_rows: List[Dict], user_id: int, week: TimesheetWeek) -> None:
    """
    payload_rows: список строк вида:
        {'project_id': int, 'hours': {date: float, ...}}
    Правило: для каждой пары (user_id, project_id, work_date) полностью ПЕРЕЗАПИСЫВАЕМ часы:
    — если новая ячейка > 0: UPSERT
    — если 0 или NaN: DELETE
    """
    eng = get_engine()
    with eng.begin() as con:
        to_insert: List[Tuple[int,int,int,date,float]] = []
        to_delete: List[Tuple[int,int,date]] = []
        for row in payload_rows:
            pid = int(row["project_id"])
            hours_map: Dict[date, float] = row.get("hours", {})
            for d in week.dates:
                v = hours_map.get(d, 0.0)
                try:
                    h = float(v) if v is not None and str(v).strip() != "" else 0.0
                except Exception:
                    h = 0.0
                if h > 0:
                    to_insert.append((0, user_id, pid, d, h))
                else:
                    to_delete.append((user_id, pid, d))

        if to_delete:
            con.execute(
                text("""
                    DELETE FROM log
                    WHERE (user_id, project_id, work_date) IN (
                        SELECT * FROM UNNEST(:uids::bigint[], :pids::bigint[], :dates::date[])
                    )
                """),
                params={
                    "uids":  [u for (u, _, _) in to_delete],
                    "pids":  [p for (_, p, _) in to_delete],
                    "dates": [d for (_, _, d) in to_delete],
                },
            )

        if to_insert:
            ids = _next_log_ids(con, len(to_insert))
            uids  = [u for (_, u, _, _, _) in to_insert]
            pids  = [p for (_, _, p, _, _) in to_insert]
            dates = [d for (_, _, _, d, _) in to_insert]
            hrs   = [h for (_, _, _, _, h) in to_insert]

            try:
                con.execute(
                    text("""
                        INSERT INTO log (id, user_id, project_id, work_date, hours)
                        SELECT * FROM UNNEST(:ids::bigint[], :uids::bigint[], :pids::bigint[], :dates::date[], :hrs::float[])
                        ON CONFLICT (user_id, project_id, work_date) DO UPDATE
                        SET hours = EXCLUDED.hours;
                    """),
                    params={"ids": ids, "uids": uids, "pids": pids, "dates": dates, "hrs": hrs},
                )
            except Exception:
                con.execute(
                    text("""
                        DELETE FROM log
                        WHERE (user_id, project_id, work_date) IN (
                            SELECT * FROM UNNEST(:uids::bigint[], :pids::bigint[], :dates::date[])
                        )
                    """),
                    params={"uids": uids, "pids": pids, "dates": dates},
                )
                con.execute(
                    text("""
                        INSERT INTO log (id, user_id, project_id, work_date, hours)
                        SELECT * FROM UNNEST(:ids::bigint[], :uids::bigint[], :pids::bigint[], :dates::date[], :hrs::float[])
                    """),
                    params={"ids": ids, "uids": uids, "pids": pids, "dates": dates, "hrs": hrs},
                )


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
h1, h2, h3 {font-weight: 700;}
.ts-grid {border-collapse: separate; border-spacing: 0; width: 100%;}
.ts-grid th {position: sticky; top: 0; background: #fff; z-index: 2;}
.ts-grid th, .ts-grid td {border-bottom: 1px solid #e8e8e8; padding: 8px 10px;}
.ts-grid thead tr {border-bottom: 2px solid #dcdcdc;}
.ts-grid .muted {color: #888;}
.ts-total {text-align: right; font-weight: 700;}
.ts-controls {display:flex; flex-wrap: wrap; gap: 10px; align-items: center;}
.ts-controls > * {flex: 0 0 auto;}
.stDataFrame {border: 1px solid #eee; border-radius: 12px;}
.small {font-size: 12px; color:#666;}
</style>
"""


def _day_cols(week: TimesheetWeek) -> List[str]:
    ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return [f"{ru[i]} {d.strftime('%d.%m')}" for i, d in enumerate(week.dates)]


def _empty_df(projects: pd.DataFrame, week: TimesheetWeek) -> pd.DataFrame:
    cols = ["Проект"] + _day_cols(week) + ["Итого"]
    df = pd.DataFrame(columns=cols)
    return df


def _prefill_df(user_id: int, week: TimesheetWeek, projects: pd.DataFrame) -> pd.DataFrame:
    day_cols = _day_cols(week)
    df = _empty_df(projects, week)
    raw = fetch_week_hours(user_id, week)
    if raw.empty:
        return df
    pid2name = dict(projects[["id", "name"]].values)
    rows: Dict[int, Dict[str, float]] = {}
    for _, r in raw.iterrows():
        pid = int(r["project_id"])
        wd  = pd.to_datetime(r["work_date"]).date()
        h   = float(r["hours"] or 0.0)
        label = pid2name.get(pid, f"#{pid}")
        rows.setdefault(pid, {"Проект": label, **{c: 0.0 for c in day_cols}})
        for i, d in enumerate(week.dates):
            if d == wd:
                rows[pid][day_cols[i]] = h
                break
    if not rows:
        return df
    df = pd.DataFrame(list(rows.values()))
    df["Итого"] = df[day_cols].sum(axis=1)
    df = df.sort_values("Проект").reset_index(drop=True)
    return df


def _as_payload(edited: pd.DataFrame, projects: pd.DataFrame, week: TimesheetWeek) -> List[Dict]:
    name2pid = {n: int(i) for i, n in projects[["id", "name"]].values}
    day_cols = _day_cols(week)
    out: List[Dict] = []
    for _, row in edited.iterrows():
        pname = str(row.get("Проект") or "").strip()
        if not pname:
            continue
        pid = name2pid.get(pname)
        if not pid:
            continue
        hours_map: Dict[date, float] = {}
        for i, col in enumerate(day_cols):
            v = row.get(col, 0.0)
            try:
                h = float(v) if v is not None and str(v).strip() != "" else 0.0
            except Exception:
                h = 0.0
            hours_map[week.dates[i]] = h
        out.append({"project_id": pid, "hours": hours_map})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Persisting chosen user (select once)
# ──────────────────────────────────────────────────────────────────────────────

def _get_saved_uid() -> Optional[int]:
    # 1) query params
    try:
        qp = st.experimental_get_query_params()  # совместимость с разными версиями Streamlit
        uid = qp.get("uid", [None])[0]
    except Exception:
        uid = None
    if uid:
        try:
            return int(uid)
        except Exception:
            return None
    # 2) session_state
    if "uid" in st.session_state:
        try:
            return int(st.session_state["uid"])
        except Exception:
            pass
    return None


def _save_uid(uid: int) -> None:
    st.session_state["uid"] = int(uid)
    try:
        st.experimental_set_query_params(uid=str(uid))
    except Exception:
        pass


def _clear_saved_uid() -> None:
    st.session_state.pop("uid", None)
    try:
        st.experimental_set_query_params()  # очистит параметры
    except Exception:
        pass


def _header_controls(users: pd.DataFrame, projects: pd.DataFrame) -> Tuple[int, TimesheetWeek, List[str]]:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.subheader("⏱️ Timesheet")

    col1, col2, col3 = st.columns([1.1, 1.4, 2])
    with col1:
        picked = st.date_input("Неделя", value=date.today(), format="DD.MM.YYYY")
        week = TimesheetWeek.from_any(picked)

    # Выбор/фиксация пользователя: "выбери один раз и больше не спрашиваем"
    saved_uid = _get_saved_uid()
    user_id: Optional[int] = None

    valid_ids = set(users["id"].astype(int).tolist())

    with col2:
        if saved_uid and int(saved_uid) in valid_ids:
            # Найдём имя пользователя
            row = users[users["id"] == int(saved_uid)].iloc[0]
            st.markdown(f"**Пользователь:** {row['first_name']}  ·  id={int(row['id'])}")
            if st.button("Сменить пользователя"):
                _clear_saved_uid()
                st.experimental_rerun()
            user_id = int(saved_uid)
        else:
            # Первый заход: показываем выбор один раз
            # Предвыбор по DEFAULT_TG_ID (если есть)
            default_idx = 0
            default_tg = st.secrets.get("DEFAULT_TG_ID")
            if default_tg:
                try:
                    default_idx = users.index[users["tg_id"] == int(default_tg)][0]
                except Exception:
                    default_idx = 0

            ulabel = st.selectbox(
                "Выберите пользователя (один раз)",
                options=list(users["first_name"] + "  ·  id=" + users["id"].astype(str)),
                index=default_idx,
            )
            user_id = int(ulabel.split("id=")[-1])
            # фиксируем выбор, чтобы больше не спрашивать
            _save_uid(user_id)

    with col3:
        st.markdown(
            f"""
            <div class="small">Неделя: <b>{week.dates[0].strftime('%d.%m.%Y')}</b> — <b>{week.dates[-1].strftime('%d.%m.%Y')}</b><br>
            Введите часы в ячейки. 0 или пусто — удаление записи. Нажмите «Сохранить».</div>
            """,
            unsafe_allow_html=True,
        )

    days = _day_cols(week)
    return user_id, week, days


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

def render_timesheet_tab():
    """Основная функция вкладки Timesheet. Вызывайте из вашего главного файла."""
    ensure_db()  # безопасно
    projects = fetch_projects()
    users = fetch_users()

    if projects.empty or users.empty:
        st.info("Добавьте хотя бы одного пользователя и проект в БД, чтобы начать работу.")
        st.stop()

    user_id, week, day_cols = _header_controls(users, projects)

    # Кнопки быстрого заполнения
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 4])
    with cc1:
        if st.button("Заполнить 8ч по будням", use_container_width=True):
            st.session_state.setdefault("ts_quick_fill", {})[user_id] = {d: (8.0 if i < 5 else 0.0) for i, d in enumerate(week.dates)}
    with cc2:
        if st.button("Очистить неделю", use_container_width=True):
            st.session_state.setdefault("ts_quick_fill", {})[user_id] = {d: 0.0 for d in week.dates}
    with cc3:
        if st.button("Скопировать с прошлой недели", use_container_width=True):
            prev = TimesheetWeek(week.monday - timedelta(days=7), [week.monday - timedelta(days=7) + timedelta(days=i) for i in range(7)])
            prev_df = fetch_week_hours(user_id, prev)
            st.session_state.setdefault("ts_paste_prev", {})[user_id] = prev_df

    base_df = _prefill_df(user_id, week, projects)

    quick = st.session_state.get("ts_quick_fill", {}).get(user_id)
    paste_prev = st.session_state.get("ts_paste_prev", {}).get(user_id)

    if paste_prev is not None and not paste_prev.empty:
        pid2name = dict(projects[["id", "name"]].values)
        proj_ids = sorted(paste_prev["project_id"].unique().tolist())
        df = pd.DataFrame({"Проект": [pid2name.get(pid, f"#{pid}") for pid in proj_ids]})
        for i, d in enumerate(week.dates):
            col = day_cols[i]
            prev_day = d - timedelta(days=7)
            merged = paste_prev[paste_prev["work_date"] == pd.to_datetime(prev_day)]
            hours_by_pid = merged.set_index("project_id")["hours"].to_dict()
            df[col] = [float(hours_by_pid.get(pid, 0.0)) for pid in proj_ids]
        df["Итого"] = df[day_cols].sum(axis=1)
        base_df = df

    if quick is not None:
        for i, d in enumerate(week.dates):
            base_df[day_cols[i]] = float(quick.get(d, 0.0))
        base_df["Итого"] = base_df[day_cols].sum(axis=1)

    st.markdown("<hr>", unsafe_allow_html=True)

    edited = st.data_editor(
        base_df,
        use_container_width=True,
        hide_index=True,
        column_config={},
        num_rows="dynamic",
        key=f"ts_editor_{user_id}_{week.monday.isoformat()}",
    )
    try:
        edited["Итого"] = edited[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    except Exception:
        edited["Итого"] = 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("💾 Сохранить", type="primary", use_container_width=True):
            known_names = set(projects["name"].tolist())
            bad = [str(x) for x in edited["Проект"].tolist() if str(x) not in known_names]
            if bad:
                st.error("Эти проекты отсутствуют в справочнике: " + ", ".join(bad))
            else:
                payload = _as_payload(edited, projects, week)
                try:
                    upsert_week(payload, user_id, week)
                    st.success("Сохранено ✔")
                    fetch_week_hours.clear()
                    fetch_projects.clear()
                except Exception as e:
                    st.error(f"Ошибка сохранения: {e}")
    with c2:
        st.markdown(
            "<span class='small'>Подсказка: добавлять/переименовывать проекты и пользователей лучше через SQL или вашу админку. "
            "Эта вкладка редактирует только часы в log. Чтобы сменить пользователя — нажмите «Сменить пользователя».</span>",
            unsafe_allow_html=True,
        )

    total_week = float(edited["Итого"].sum()) if not edited.empty else 0.0
    st.markdown(f"**Итого за неделю:** {total_week:.2f} ч")










