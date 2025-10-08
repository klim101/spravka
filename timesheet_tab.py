# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text

# ──────────────────────────────────────────────────────────────────────────────
# Подключение к Supabase (PostgreSQL)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine():
    dsn = st.secrets.get("POSTGRES_DSN", "")
    if not dsn:
        st.error(
            "В secrets нет POSTGRES_DSN. Укажи строку для Supabase Pooler "
            "(например, postgresql+psycopg2://postgres.<proj>:***@aws-1-...pooler.supabase.com:5432/postgres?sslmode=require)"
        )
        st.stop()
    # нормализуем префикс на всякий случай
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return create_engine(dsn, pool_pre_ping=True, pool_recycle=1800, future=True)

# ──────────────────────────────────────────────────────────────────────────────
# DDL / структура
# ──────────────────────────────────────────────────────────────────────────────

def _detect_user_table(engine) -> str:
    insp = inspect(engine)
    names = {t.lower() for t in insp.get_table_names()}
    if "klim101" in names:
        return "klim101"
    if "user" in names:
        return '"user"'
    return "klim101"

def _ensure_db_impl():
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

@st.cache_resource(show_spinner=False)
def ensure_db_once():
    _ensure_db_impl()
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Модель недели + выборки
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TimesheetWeek:
    monday: date
    dates: List[date]  # 7 дней: Пн..Вс

    @staticmethod
    def from_any(d: date | datetime | None = None) -> "TimesheetWeek":
        if d is None:
            d = date.today()
        d = d.date() if isinstance(d, datetime) else d
        monday = d - timedelta(days=(d.weekday() % 7))
        return TimesheetWeek(monday=monday, dates=[monday + timedelta(days=i) for i in range(7)])

@st.cache_data(ttl=60, show_spinner=False)
def fetch_projects() -> pd.DataFrame:
    q = text("SELECT id, name FROM project ORDER BY name")
    return pd.read_sql(q, get_engine())

@st.cache_data(ttl=60, show_spinner=False)
def fetch_users() -> pd.DataFrame:
    eng = get_engine()
    user_tbl = _detect_user_table(eng)
    q = text(f"SELECT id, first_name, tg_id FROM {user_tbl} ORDER BY first_name")
    return pd.read_sql(q, eng)

@st.cache_data(ttl=30, show_spinner=False)
def fetch_week_rows(user_id: int, week: TimesheetWeek) -> pd.DataFrame:
    """
    Отдаёт строки лога этой недели: project_id, work_date(date), hours(float)
    """
    q = text("""
        SELECT project_id, work_date, hours
        FROM log
        WHERE user_id=:uid AND work_date BETWEEN :d1 AND :d7
        ORDER BY work_date, project_id
    """)
    df = pd.read_sql(q, get_engine(), params={"uid": user_id, "d1": week.dates[0], "d7": week.dates[-1]})
    if not df.empty:
        df["work_date"] = pd.to_datetime(df["work_date"]).dt.date
        df["project_id"] = df["project_id"].astype(int)
        df["hours"] = df["hours"].astype(float)
    return df

def _next_ids(con, n: int) -> List[int]:
    cur = con.execute(text("SELECT COALESCE(MAX(id), 0) FROM log"))
    start = int(cur.scalar() or 0) + 1
    return list(range(start, start + n))

def save_week_replace(user_id: int, week: TimesheetWeek, tuples: List[Tuple[int, date, float]]) -> int:
    """
    tuples: [(project_id, work_date, hours)]
    Алгоритм: DELETE (вся неделя) -> пакетная вставка executemany.
    Возвращает количество вставленных строк.
    """
    eng = get_engine()
    with eng.begin() as con:
        # 1) Сносим всё за неделю для данного пользователя
        con.execute(
            text("DELETE FROM log WHERE user_id=:uid AND work_date BETWEEN :d1 AND :d7"),
            {"uid": user_id, "d1": week.dates[0], "d7": week.dates[-1]},
        )

        if not tuples:
            return 0

        # 2) Генерим новые id и готовим payload
        ids = _next_ids(con, len(tuples))
        payload = [
            {"id": ids[i], "uid": user_id, "pid": int(pid), "dt": dt, "hr": float(hr)}
            for i, (pid, dt, hr) in enumerate(tuples)
        ]

        # 3) Пакетная вставка (executemany) — надёжно везде
        stmt = text(
            "INSERT INTO log (id, user_id, project_id, work_date, hours) "
            "VALUES (:id, :uid, :pid, :dt, :hr)"
        )
        con.execute(stmt, payload)
        return len(payload)


# ──────────────────────────────────────────────────────────────────────────────
# Query params helpers (универсально для новых/старых Streamlit)
# ──────────────────────────────────────────────────────────────────────────────

def qp_get(name: str, default=None):
    try:
        v = st.query_params.get(name, default)
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
        st.experimental_set_query_params(**{k: (vv[0] if isinstance(vv, list) else vv) for k, vv in current.items()})

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
        st.experimental_set_query_params(**{k: (vv[0] if isinstance(vv, list) else vv) for k, vv in current.items()})

# ──────────────────────────────────────────────────────────────────────────────
# UI: выпадающие списки по дням, автодобавление строк, гидратация из БД
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_PLACEHOLDER = "Выберите проект…"
HOURS_PLACEHOLDER   = "часы…"
# кратные 0.25 до 12
HOUR_CHOICES = [x / 4 for x in range(1, 49)]  # 0.25..12.0

_CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1200px}
.day-card{border:1px solid #eee;border-radius:12px;padding:10px 12px;margin-bottom:10px}
.day-title{font-weight:700;margin-bottom:6px}
.small{font-size:12px;color:#777}
</style>
"""

def _mark_dirty(ctx: str):
    st.session_state[f"ts_dirty_{ctx}"] = True



def _fmt_hours(v):
    try:
        f = float(v)
        return f"{f:g}"
    except Exception:
        return str(v)

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

def _hydrate_week_state(ctx: str, user_id: int, week: TimesheetWeek, projects: pd.DataFrame):
    """
    Заполняем st.session_state[*] строками из БД один раз для пары (user, week).
    """
    sig_key = f"{ctx}_hydrated_sig"
    signature = f"{user_id}:{week.monday.isoformat()}"
    if st.session_state.get(sig_key) == signature:
        return  # уже гидрировано

    df = fetch_week_rows(user_id, week)
    pid2name = {int(i): str(n) for i, n in projects[["id", "name"]].values}

    for d in week.dates:
        key = f"ts_rows_{ctx}_{d.isoformat()}"
        # соберём строки по этому дню из БД
        day_rows = []
        if not df.empty:
            day_df = df[df["work_date"] == d]
            for _, r in day_df.iterrows():
                pname = pid2name.get(int(r["project_id"]))
                if pname:
                    day_rows.append({"project": pname, "hours": float(r["hours"])})
        # обязательно последняя пустая строка для автодобавления
        day_rows.append({"project": None, "hours": None})
        st.session_state[key] = day_rows

    st.session_state[sig_key] = signature

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
                "Проект",
                proj_opts,
                index=_idx(proj_opts, row["project"]),
                key=f"{pref}_p",
                label_visibility="collapsed",
                on_change=_mark_dirty, args=(ctx,),
            )
        with c2:
            hrs_val = st.selectbox(
                "Часы",
                hrs_opts,
                index=_idx(hrs_opts, row["hours"]),
                key=f"{pref}_h",
                label_visibility="collapsed",
                format_func=_fmt_hours,
                on_change=_mark_dirty, args=(ctx,),
            )
        with c3:
            can_rm = not (len(rows) == 1 and proj_val == PROJECT_PLACEHOLDER and hrs_val == HOURS_PLACEHOLDER)
            if can_rm and st.button("✖", key=f"{pref}_rm"):
                to_delete.append(i)

        rows[i] = {
            "project": None if proj_val == PROJECT_PLACEHOLDER else proj_val,
            "hours":   None if hrs_val == HOURS_PLACEHOLDER else float(hrs_val),
        }

    # удаление → мгновенно перерисуемся
    if to_delete:
        for i in sorted(to_delete, reverse=True):
            rows.pop(i)
        if not rows:
            rows.append({"project": None, "hours": None})
        st.session_state[key] = rows
        st.rerun()

    # последняя строка стала заполненной → добавим новую и перерисуемся
    if rows and rows[-1]["project"] is not None and rows[-1]["hours"] is not None:
        rows.append({"project": None, "hours": None})
        st.session_state[key] = rows
        st.rerun()

    day_total = sum(float(r["hours"]) for r in rows if r["project"] and r["hours"] is not None)
    st.caption(f"Итого за день: {day_total:g} ч")
    st.markdown('</div>', unsafe_allow_html=True)
    return day_total

def _collect_rows_by_day(ctx: str, week: TimesheetWeek, name2pid: Dict[str, int]) -> List[Tuple[int, date, float]]:
    out: List[Tuple[int, date, float]] = []
    for d in week.dates:
        rows = st.session_state.get(f"ts_rows_{ctx}_{d.isoformat()}", [])
        for r in rows:
            proj, hrs = r.get("project"), r.get("hours")
            if proj and hrs and hrs > 0:
                pid = name2pid.get(str(proj))
                if pid:
                    out.append((int(pid), d, float(hrs)))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Заголовок/контролы и рендер вкладки
# ──────────────────────────────────────────────────────────────────────────────

def _header_controls(users: pd.DataFrame) -> Tuple[Optional[int], TimesheetWeek]:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.subheader("⏱️ Timesheet")

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        picked = st.date_input("Неделя", value=date.today(), format="DD.MM.YYYY")
        week = TimesheetWeek.from_any(picked)

    saved_uid = _get_saved_uid()
    valid_ids = set(users["id"].astype(int).tolist())

    with col2:
        choose_mode = st.session_state.get("ts_choose_user", saved_uid is None)

        if (not choose_mode) and saved_uid and int(saved_uid) in valid_ids:
            row = users[users["id"] == int(saved_uid)].iloc[0]
            st.markdown(f"**Пользователь:** {row['first_name']} · id={int(row['id'])}")
            c11, c12 = st.columns([1, 1])
            with c11:
                if st.button("Сменить пользователя"):
                    _clear_saved_uid()
                    st.session_state["ts_choose_user"] = True
                    st.rerun()
            with c12:
                if st.button("🔄 Обновить из БД"):
                    # сбросим «сигнатуру» гидратации, чтобы перечитать
                    for k in list(st.session_state.keys()):
                        if "_hydrated_sig" in k:
                            del st.session_state[k]
                    st.rerun()
            uid = int(saved_uid)
        else:
            ids = users["id"].astype(int).tolist()
            labels = {int(r.id): f"{r.first_name}  ·  id={int(r.id)}" for r in users.itertuples(index=False)}

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
            uid = None

    return uid, week

def render_timesheet_tab():
    """Вкладка Timesheet: теперь с автосохранением недели при любом изменении."""
    ensure_db_once()

    projects = fetch_projects()
    users = fetch_users()
    if projects.empty or users.empty:
        st.info("Добавьте хотя бы одного пользователя и проект.")
        st.stop()

    user_id, week = _header_controls(users)
    if not user_id:
        st.stop()

    # контекст на пару (user, week)
    ctx = f"u{user_id}_{week.monday.isoformat()}"

    # один раз при заходе/смене недели/пользователя — заполним UI из БД
    _hydrate_week_state(ctx, user_id, week, projects)

    # сетка дней
    st.markdown(
        f"<div class='small'>Неделя: <b>{week.dates[0].strftime('%d.%m.%Y')}</b> — "
        f"<b>{week.dates[-1].strftime('%d.%m.%Y')}</b>. Заполняйте проект и часы, новые строки добавляются автоматически.</div>",
        unsafe_allow_html=True,
    )

    proj_names = projects["name"].astype(str).tolist()
    totals = []
    for d in week.dates:
        totals.append(_render_day(ctx, d, proj_names))

    # ---------- АВТОСОХРАНЕНИЕ (replace всей недели) ----------
    name2pid = {str(n): int(i) for i, n in projects[["id", "name"]].values}
    tuples = _collect_rows_by_day(ctx, week, name2pid)
    
    import hashlib, json
    def _digest(rows):
        norm = sorted([(int(pid), d.isoformat(), float(hr)) for (pid, d, hr) in rows])
        return hashlib.md5(json.dumps(norm, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    
    cur_hash = _digest(tuples)
    hash_key  = f"ts_last_hash_{ctx}"
    dirty_key = f"ts_dirty_{ctx}"
    
    need_save = st.session_state.get(dirty_key, False) or (st.session_state.get(hash_key) != cur_hash)
    
    if need_save:
        try:
            n = save_week_replace(user_id, week, tuples)  # DELETE неделя -> INSERT актуальных строк
            fetch_week_rows.clear()                       # сброс кеша читаемой недели
            st.session_state[hash_key]  = cur_hash
            st.session_state[dirty_key] = False
            st.toast(f"Автосохранено ({n} строк)")
        except Exception as e:
            st.warning(f"Не удалось автосохранить: {e}")
    # ----------------------------------------------------------


    total_week = sum(totals)
    st.markdown(f"**Итого за неделю:** {total_week:g} ч")


