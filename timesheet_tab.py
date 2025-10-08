# timesheet_tab.py
# -*- coding: utf-8 -*-
"""
Streamlit-вкладка «Timesheet» (nikatime-style).
Для каждого дня недели: строки из выпадающих списков [Проект] + [Часы].
Сохранение в таблицу log (Supabase Postgres).

Требуемые секреты (Streamlit → Secrets):
  POSTGRES_DSN = "<SQLAlchemy DSN к Supabase>"
  # пример для pooler (pg8000 + SSL):
  # postgresql+pg8000://postgres.hvntnpffdnywlxhlrxcm:<URL-ENC-PASS>@aws-1-eu-north-1.pooler.supabase.com:6543/postgres?ssl=true

Опционально:
  DEFAULT_TG_ID = 123456789  # предвыбор пользователя по tg_id при первом заходе
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text

# ──────────────────────────────────────────────────────────────────────────────
# Подключение к БД
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine():
    dsn = st.secrets.get("POSTGRES_DSN", "").strip()
    if not dsn:
        st.error(
            "В secrets нет POSTGRES_DSN. Укажите SQLAlchemy строку подключения "
            "(лучше к Transaction/Session pooler)."
        )
        st.stop()
    # нормализуем postgres:// → postgresql:// (если вдруг)
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return create_engine(dsn, pool_pre_ping=True, pool_recycle=1800, future=True)

# ──────────────────────────────────────────────────────────────────────────────
# Схема/DDL
# ──────────────────────────────────────────────────────────────────────────────

def _detect_user_table(engine) -> str:
    insp = inspect(engine)
    names = {n.lower() for n in insp.get_table_names()}
    if "klim101" in names:
        return "klim101"
    if "user" in names:
        return '"user"'  # зарезервированное слово
    return "klim101"

def ensure_db():
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
        try:
            con.execute(text(ddl_uniq))
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Модель недели + запросы
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
        monday = d - timedelta(days=(d.weekday() % 7))
        return TimesheetWeek(monday=monday, dates=[monday + timedelta(days=i) for i in range(7)])

@st.cache_data(ttl=60, show_spinner=False)
def fetch_projects() -> pd.DataFrame:
    eng = get_engine()
    return pd.read_sql(text("SELECT id, name FROM project ORDER BY name"), eng)

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
    start = int(con.execute(text("SELECT COALESCE(MAX(id), 0) FROM log")).scalar() or 0) + 1
    return list(range(start, start + n))

def upsert_week(payload_rows: List[Dict], user_id: int, week: TimesheetWeek) -> None:
    """
    Принимает список вида:
      [{"project_id": 11, "hours": {<date>: 2.0, ...}}, ...]
    ИЛИ построчно:
      [{"project_id": 11, "work_date": <date>, "hours": 2.0}, ...]
    Полностью перезаписывает часы недели для переданных (user, project, day).
    """
    # Нормализуем к форме с картой дат
    normalized: Dict[int, Dict[date, float]] = {}
    for row in payload_rows:
        pid = int(row["project_id"])
        if "work_date" in row:
            d = row["work_date"]
            if isinstance(d, str):
                d = datetime.fromisoformat(d).date()
            h = float(row.get("hours") or 0.0)
            if h > 0:
                normalized.setdefault(pid, {}).setdefault(d, 0.0)
                normalized[pid][d] += h  # суммируем, если несколько строк на день
        else:
            for d, h in (row.get("hours") or {}).items():
                if isinstance(d, str):
                    d = datetime.fromisoformat(d).date()
                h = float(h or 0.0)
                if h > 0:
                    normalized.setdefault(pid, {})[d] = h

    inserts: List[Tuple[int,int,int,date,float]] = []
    deletes: List[Tuple[int,int,date]] = []

    for pid, daymap in normalized.items():
        for d in week.dates:
            val = float(daymap.get(d, 0.0))
            if val > 0:
                inserts.append((0, user_id, pid, d, val))
            else:
                deletes.append((user_id, pid, d))

    eng = get_engine()
    with eng.begin() as con:
        if deletes:
            con.execute(
                text("""
                    DELETE FROM log
                     WHERE (user_id, project_id, work_date) IN (
                        SELECT * FROM UNNEST(:uids::bigint[], :pids::bigint[], :dates::date[])
                     )
                """),
                params={
                    "uids": [u for (u, _, _) in deletes],
                    "pids": [p for (_, p, _) in deletes],
                    "dates": [d for (_, _, d) in deletes],
                },
            )
        if inserts:
            ids  = _next_log_ids(con, len(inserts))
            uids = [u for (_, u, _, _, _) in inserts]
            pids = [p for (_, _, p, _, _) in inserts]
            dts  = [d for (_, _, _, d, _) in inserts]
            hrs  = [h for (_, _, _, _, h) in inserts]
            con.execute(
                text("""
                  INSERT INTO log (id, user_id, project_id, work_date, hours)
                  SELECT * FROM UNNEST(:ids::bigint[], :uids::bigint[], :pids::bigint[],
                                       :dts::date[], :hrs::float[])
                  ON CONFLICT (user_id, project_id, work_date) DO UPDATE
                    SET hours = EXCLUDED.hours
                """),
                params={"ids": ids, "uids": uids, "pids": pids, "dts": dts, "hrs": hrs},
            )

# ──────────────────────────────────────────────────────────────────────────────
# Query params helpers (новый API + fallback на experimental)
# ──────────────────────────────────────────────────────────────────────────────

def qp_get(name: str, default=None):
    try:
        v = st.query_params.get(name, default)  # новый API
    except Exception:
        try:
            v = st.experimental_get_query_params().get(name, [default])[0]
        except Exception:
            v = default
    if isinstance(v, list):
        return v[0] if v else default
    return v

def qp_update(**kwargs):
    payload = {k: ("" if v is None else str(v)) for k, v in kwargs.items()}
    try:
        st.query_params.update(payload)
    except Exception:
        try:
            current = st.experimental_get_query_params()
        except Exception:
            current = {}
        current.update({k: [v] for k, v in payload.items()})
        st.experimental_set_query_params(**{k: vv[0] if isinstance(vv, list) else vv for k, vv in current.items()})

def qp_delete(*names):
    try:
        for n in names:
            if n in st.query_params:
                del st.query_params[n]
    except Exception:
        try:
            current = st.experimental_get_query_params()
        except Exception:
            current = {}
        for n in names:
            current.pop(n, None)
        st.experimental_set_query_params(**{k: vv[0] if isinstance(vv, list) else vv for k, vv in current.items()})

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers (CSS, выбор пользователя)
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.small {font-size: 12px; color:#666;}
.day-card {padding: 10px 14px; border: 1px solid #eee; border-radius: 12px; margin-bottom: 10px;}
.day-title {font-weight: 700; margin-bottom: 6px;}
.hr {height:1px; background:#eee; border:0; margin:12px 0;}
</style>
"""

HOUR_CHOICES = [0.5 * i for i in range(1, 25)]  # 0.5..12.0
PROJECT_PLACEHOLDER = "— выберите проект —"
HOURS_PLACEHOLDER = "— ч —"

def _fmt_hours(v):
    if isinstance(v, (int, float)): return f"{float(v):g} ч"
    return HOURS_PLACEHOLDER

def _get_saved_uid() -> Optional[int]:
    uid = qp_get("uid")
    if uid not in (None, "", "None"):
        try:
            return int(uid)
        except Exception:
            pass
    if "uid" in st.session_state:
        try:
            return int(st.session_state["uid"])
        except Exception:
            pass
    return None

def _save_uid(uid: int) -> None:
    st.session_state["uid"] = int(uid)
    qp_update(uid=uid)

def _clear_saved_uid() -> None:
    st.session_state.pop("uid", None)
    qp_delete("uid")

def _header_controls(users: pd.DataFrame) -> Tuple[int, TimesheetWeek]:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.subheader("⏱️ Timesheet")

    col1, col2, col3 = st.columns([1.1, 1.6, 2])
    with col1:
        picked = st.date_input("Неделя", value=date.today(), format="DD.MM.YYYY")
        week = TimesheetWeek.from_any(picked)

    saved_uid = _get_saved_uid()
    valid_ids = set(users["id"].astype(int).tolist())
    user_id: Optional[int] = None

    with col2:
        choose_mode = st.session_state.get("ts_choose_user", saved_uid is None)
        if (not choose_mode) and saved_uid and int(saved_uid) in valid_ids:
            row = users[users["id"] == int(saved_uid)].iloc[0]
            st.markdown(f"**Пользователь:** {row['first_name']} · id={int(row['id'])}")
            if st.button("Сменить пользователя"):
                _clear_saved_uid()
                st.session_state["ts_choose_user"] = True
                st.rerun()
            user_id = int(saved_uid)
        else:
            ids = users["id"].astype(int).tolist()
            labels = {int(r.id): f"{r.first_name} · id={int(r.id)}" for r in users.itertuples(index=False)}
            default_idx = 0
            default_tg = st.secrets.get("DEFAULT_TG_ID")
            if default_tg:
                try:
                    default_idx = users.index[users["tg_id"] == int(default_tg)][0]
                except Exception:
                    default_idx = 0
            if saved_uid and int(saved_uid) in ids:
                default_idx = ids.index(int(saved_uid))

            selected_id = st.selectbox(
                "Выберите пользователя (один раз)",
                options=ids,
                index=default_idx if 0 <= default_idx < len(ids) else 0,
                format_func=lambda i: labels.get(int(i), f"id={i}"),
                key="ts_select_user",
            )
            if st.button("✅ Выбрать этого пользователя"):
                _save_uid(int(selected_id))
                st.session_state["ts_choose_user"] = False
                st.rerun()

    with col3:
        st.caption(
            f"Неделя: {week.dates[0].strftime('%d.%m.%Y')} — {week.dates[-1].strftime('%d.%m.%Y')}. "
            "В каждой строке выберите проект и часы. Пустая строка добавляется автоматически."
        )

    if user_id is None:
        st.stop()

    return user_id, week

# ──────────────────────────────────────────────────────────────────────────────
# Nika-style day blocks
# ──────────────────────────────────────────────────────────────────────────────

def _ctx_key(user_id, week):
    return f"{int(user_id)}::{week.monday.isoformat()}"

def _init_day_from_db(ctx: str, day: date, df: pd.DataFrame, pid2name: Dict[int, str]):
    key = f"ts_rows_{ctx}_{day.isoformat()}"
    if key in st.session_state:
        return
    rows = []
    day_df = df[df["work_date"].dt.date == day] if not df.empty else df
    for _, r in day_df.iterrows():
        rows.append({"project": pid2name.get(int(r["project_id"])), "hours": float(r["hours"])})
    rows.append({"project": None, "hours": None})  # пустая строка в конце
    st.session_state[key] = rows

def _render_day(ctx: str, day: date, project_names: List[str]) -> float:
    key = f"ts_rows_{ctx}_{day.isoformat()}"
    rows = st.session_state.setdefault(key, [{"project": None, "hours": None}])

    proj_opts = [PROJECT_PLACEHOLDER] + project_names
    hrs_opts  = [HOURS_PLACEHOLDER] + HOUR_CHOICES

    def _idx(options, val):
        if val is None: return 0
        try: return options.index(val)
        except Exception: return 0

    st.markdown('<div class="day-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="day-title">{day.strftime("%A, %d.%m.%Y")}</div>', unsafe_allow_html=True)

    to_delete = []
    for i, row in enumerate(rows):
        pref = f"{key}_{i}"
        c1, c2, c3 = st.columns([3, 1, 0.6])
        with c1:
            proj_val = st.selectbox(
                "Проект", proj_opts, index=_idx(proj_opts, row["project"]),
                key=f"{pref}_p", label_visibility="collapsed",
            )
        with c2:
            hrs_val = st.selectbox(
                "Часы", hrs_opts, index=_idx(hrs_opts, row["hours"]),
                key=f"{pref}_h", label_visibility="collapsed", format_func=_fmt_hours,
            )
        with c3:
            can_rm = not (len(rows) == 1 and proj_val == PROJECT_PLACEHOLDER and hrs_val == HOURS_PLACEHOLDER)
            if can_rm and st.button("✖", key=f"{pref}_rm"):
                to_delete.append(i)

        rows[i] = {
            "project": None if proj_val == PROJECT_PLACEHOLDER else proj_val,
            "hours":   None if hrs_val == HOURS_PLACEHOLDER else float(hrs_val),
        }

    if to_delete:
        for i in sorted(to_delete, reverse=True):
            rows.pop(i)
        if not rows:
            rows.append({"project": None, "hours": None})
        st.session_state[key] = rows
        st.rerun()

    if rows and rows[-1]["project"] is not None and rows[-1]["hours"] is not None:
        rows.append({"project": None, "hours": None})
        st.session_state[key] = rows

    day_total = sum(float(r["hours"]) for r in rows if r["project"] and r["hours"] is not None)
    st.caption(f"Итого за день: {day_total:g} ч")
    st.markdown('</div>', unsafe_allow_html=True)
    return day_total

def _collect_payload(ctx: str, week: TimesheetWeek, name2pid: Dict[str, int]) -> List[Dict]:
    """Превращаем state в структуру для upsert_week (соберём в карту дат по проекту)."""
    per_project: Dict[int, Dict[date, float]] = {}
    for d in week.dates:
        rows = st.session_state.get(f"ts_rows_{ctx}_{d.isoformat()}", [])
        for r in rows:
            proj, hrs = r.get("project"), r.get("hours")
            if proj and hrs and hrs > 0:
                pid = name2pid.get(str(proj))
                if pid is None:
                    continue
                per_project.setdefault(int(pid), {})
                per_project[int(pid)][d] = per_project[int(pid)].get(d, 0.0) + float(hrs)

    # в формат upsert_week
    out = [{"project_id": pid, "hours": hmap} for pid, hmap in per_project.items()]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Public API (не меняем имя — чтобы un.py ничего не трогать)
# ──────────────────────────────────────────────────────────────────────────────

def render_timesheet_tab():
    """Вызвать внутри вкладки '⏱️ Timesheet'."""
    ensure_db()
    projects = fetch_projects()
    users = fetch_users()

    if projects.empty or users.empty:
        st.info("Добавьте хотя бы одного пользователя и проект в БД, чтобы начать работу.")
        st.stop()

    user_id, week = _header_controls(users)
    ctx = _ctx_key(user_id, week)

    # Смена пользователя/недели → сброс локального стейта и предзаполнение из БД
    if st.session_state.get("ts_ctx") != ctx:
        # очистить старые дневные ключи
        for k in list(st.session_state.keys()):
            if k.startswith("ts_rows_"):
                del st.session_state[k]
        st.session_state["ts_ctx"] = ctx

        df = fetch_week_hours(user_id, week).copy()
        if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["work_date"]):
            df["work_date"] = pd.to_datetime(df["work_date"])
        pid2name = dict(projects[["id", "name"]].values)
        for d in week.dates:
            _init_day_from_db(ctx, d, df, pid2name)

    # Справочники для селектов
    name2pid = {str(n): int(i) for i, n in projects[["id", "name"]].itertuples(index=False)}
    proj_names = [str(x) for x in sorted(projects["name"].tolist(), key=str.lower)]

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Рисуем 7 блоков дней
    week_total = 0.0
    for d in week.dates:
        week_total += _render_day(ctx, d, proj_names)

    # Кнопки и подсказки
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("💾 Сохранить", type="primary", use_container_width=True):
            payload = _collect_payload(ctx, week, name2pid)
            try:
                upsert_week(payload, user_id, week)
                st.success("Сохранено ✔")
                fetch_week_hours.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка сохранения: {e}")
    with c2:
        st.caption("Строки добавляются автоматически. ✖ удаляет строку. "
                   "Чтобы сменить пользователя — нажмите «Сменить пользователя» выше.")

    st.markdown(f"**Итого за неделю:** {week_total:g} ч")
