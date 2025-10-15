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
    """Читаем выбранного пользователя только из session_state (без query params)."""
    uid = st.session_state.get("uid")
    try:
        return int(uid) if uid is not None else None
    except Exception:
        return None


def _save_uid(uid: int) -> None:
    st.session_state["uid"] = int(uid)


def _clear_saved_uid() -> None:
    st.session_state.pop("uid", None)


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
    day_key = f"ts_rows_{ctx}_{day.isoformat()}"
    rows = st.session_state.setdefault(day_key, [{"project": None, "hours": None}])

    proj_opts = [PROJECT_PLACEHOLDER] + project_names
    hrs_opts  = [HOURS_PLACEHOLDER] + HOUR_CHOICES

    def _idx(options, val):
        if val is None:
            return 0
        try:
            return options.index(val)
        except Exception:
            return 0

    st.markdown('<div class="day-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="day-title">{day.strftime("%A, %d.%m.%Y")}</div>', unsafe_allow_html=True)

    for i, row in enumerate(rows):
        pref  = f"{day_key}_{i}"
        key_p = f"{pref}_p"
        key_h = f"{pref}_h"

        c1, c2, c3 = st.columns([3, 1, 0.6])

        with c1:
            st.selectbox(
                "Проект",
                proj_opts,
                index=_idx(proj_opts, row.get("project")),
                key=key_p,
                label_visibility="collapsed",
                on_change=_on_row_change,
                args=(day_key, i, key_p, key_h, ctx),
            )

        with c2:
            st.selectbox(
                "Часы",
                hrs_opts,
                index=_idx(hrs_opts, row.get("hours")),
                key=key_h,
                label_visibility="collapsed",
                format_func=_fmt_hours,
                on_change=_on_row_change,
                args=(day_key, i, key_p, key_h, ctx),
            )

        with c3:
            # Кнопка удаления как колбэк (без ручного rerun)
            can_rm = not (len(rows) == 1 and row.get("project") is None and row.get("hours") is None)
            st.button("✖", key=f"{pref}_rm", disabled=not can_rm,
                      on_click=_on_remove_row, args=(day_key, i, ctx))

    # считаем итог уже по актуальному состоянию из session_state
    rows_now = st.session_state[day_key]
    day_total = sum(float(r["hours"]) for r in rows_now
                    if r.get("project") and r.get("hours") is not None)
    st.caption(f"Итого за день: {day_total:g} ч")
    st.markdown('</div>', unsafe_allow_html=True)
    return day_total


def _on_row_change(day_key: str, idx: int, key_proj: str, key_hrs: str, ctx: str):
    """Обновляет строку idx и при необходимости добавляет пустой хвост."""
    rows = st.session_state.get(day_key, [{"project": None, "hours": None}])

    proj_val = st.session_state.get(key_proj, PROJECT_PLACEHOLDER)
    hrs_val  = st.session_state.get(key_hrs, HOURS_PLACEHOLDER)

    proj = None if proj_val == PROJECT_PLACEHOLDER else proj_val
    hrs  = None if hrs_val  == HOURS_PLACEHOLDER   else float(hrs_val)

    # гарантируем наличие нужного индекса
    while idx >= len(rows):
        rows.append({"project": None, "hours": None})

    rows[idx] = {"project": proj, "hours": hrs}

    # если последняя строка теперь заполнена — добавим ещё одну пустую
    if rows and rows[-1]["project"] is not None and rows[-1]["hours"] is not None:
        rows.append({"project": None, "hours": None})

    st.session_state[day_key] = rows
    st.session_state[f"ts_dirty_{ctx}"] = True  # помечаем «изменено» для автосохранения


def _on_remove_row(day_key: str, idx: int, ctx: str):
    """Удаляет строку idx без st.rerun()."""
    rows = st.session_state.get(day_key, [{"project": None, "hours": None}])
    if 0 <= idx < len(rows):
        rows.pop(idx)
    if not rows:
        rows.append({"project": None, "hours": None})
    st.session_state[day_key] = rows
    st.session_state[f"ts_dirty_{ctx}"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Заголовок/контролы и рендер вкладки
# ──────────────────────────────────────────────────────────────────────────────

def _header_controls(users: pd.DataFrame) -> Tuple[Optional[int], TimesheetWeek]:
    st.subheader("⏱️ Timesheet")

    col1, col2, col3 = st.columns([1.1, 1.4, 1.2])
    with col1:
        picked = st.date_input("Неделя", value=date.today(), format="DD.MM.YYYY")
        week = TimesheetWeek.from_any(picked)

    saved_uid = _get_saved_uid()
    valid_ids = set(users["id"].astype(int).tolist())

    # колбэки, чтобы не вызывать st.rerun()
    def _cb_set_uid():
        sel = st.session_state.get("ts_select_user_value")
        if sel is not None:
            _save_uid(int(sel))
            st.session_state["ts_choose_user"] = False

    def _cb_change_user():
        _clear_saved_uid()
        st.session_state["ts_choose_user"] = True

    with col2:
        choose_mode = st.session_state.get("ts_choose_user", saved_uid is None)

        if (not choose_mode) and saved_uid and int(saved_uid) in valid_ids:
            row = users[users["id"] == int(saved_uid)].iloc[0]
            st.markdown(f"**Пользователь:** {row['first_name']} · id={int(row['id'])}")
            st.button("Сменить пользователя", on_click=_cb_change_user)
            user_id = int(saved_uid)
        else:
            ids = users["id"].astype(int).tolist()
            labels = {int(r.id): f"{r.first_name} · id={int(r.id)}" for r in users.itertuples(index=False)}

            # default по tg_id — опционально
            default_idx = 0
            default_tg = st.secrets.get("DEFAULT_TG_ID")
            if default_tg:
                try:
                    default_idx = users.index[users["tg_id"] == int(default_tg)][0]
                except Exception:
                    default_idx = 0

            st.selectbox(
                "Выберите пользователя (один раз)",
                options=ids,
                index=default_idx if 0 <= default_idx < len(ids) else 0,
                format_func=lambda i: labels.get(int(i), f"id={i}"),
                key="ts_select_user_value",
            )
            st.button("✅ Выбрать", on_click=_cb_set_uid)
            user_id = _get_saved_uid()  # мог установиться колбэком в этом же проходе

    with col3:
        st.button("🔄 Обновить из БД", help="Перечитать данные недели из базы",
                  on_click=lambda: [fetch_projects.clear(), fetch_users.clear(), fetch_week_rows.clear()])

    return user_id, week



def _collect_rows_by_day(ctx: str, week: TimesheetWeek, name2pid: dict) -> list[tuple[int, date, float]]:
    """
    Собирает все заполненные строки за неделю из session_state и
    превращает в [(project_id, date, hours), ...].
    Плейсхолдеры и пустые строки игнорируются.
    """
    tuples: list[tuple[int, date, float]] = []

    for d in week.dates:
        day_key = f"ts_rows_{ctx}_{d.isoformat()}"
        rows = st.session_state.get(day_key, [])
        if not isinstance(rows, list):
            continue

        for r in rows:
            proj_name = r.get("project")
            hrs_val   = r.get("hours")

            if proj_name in (None, "", PROJECT_PLACEHOLDER):
                continue
            if hrs_val in (None, "", HOURS_PLACEHOLDER):
                continue

            pid = name2pid.get(str(proj_name))
            if not pid:
                # незнакомый проект — пропускаем
                continue

            try:
                hours = float(hrs_val)
            except Exception:
                continue

            if hours <= 0:
                continue

            tuples.append((int(pid), d, hours))

    return tuples




def render_timesheet_tab():
    """
    Вкладка Timesheet.
    — Без st.rerun()
    — Автосохранение всей недели с дебаунсом (0.4s)
    — Не трогаем query params без необходимости
    """
    _inject_admin_hotkey()           # ← ловим Ctrl+Shift+A / Esc
    if _is_admin():
        render_admin_panel()         # ← показываем админ-панель наверху

    ensure_db_once()


    projects = fetch_projects()
    users = fetch_users()
    if projects.empty or users.empty:
        st.info("Добавьте хотя бы одного пользователя и проект.")
        return

    user_id, week = _header_controls(users)
    if not user_id:
        return

    # Контекст «пользователь + неделя»
    ctx = f"u{user_id}_{week.monday.isoformat()}"

    # Гидратация UI из БД — только один раз на (user, week)
    _hydrate_week_state(ctx, user_id, week, projects)

    # Инструкция
    st.markdown(
        f"<div class='small'>Неделя: <b>{week.dates[0].strftime('%d.%m.%Y')}</b> — "
        f"<b>{week.dates[-1].strftime('%d.%m.%Y')}</b>. Заполняйте проект и часы — "
        f"новые строки добавляются автоматически, изменения сохраняются сами.</div>",
        unsafe_allow_html=True,
    )

    # Рендерим 7 дней
    proj_names = projects["name"].astype(str).tolist()
    totals = [ _render_day(ctx, d, proj_names) for d in week.dates ]

    # -------- АВТОСОХРАНЕНИЕ (replace всей недели) --------
    name2pid = {str(n): int(i) for i, n in projects[["id", "name"]].values}
    tuples = _collect_rows_by_day(ctx, week, name2pid)  # [(pid, date, hours), ...]

    # Хэш содержимого недели (чтобы не писать одинаковое по кругу)
    import time, hashlib, json
    norm = sorted([(int(pid), d.toordinal(), float(hr)) for (pid, d, hr) in tuples])
    cur_hash = hashlib.md5(json.dumps(norm, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

    hash_key  = f"ts_hash_{ctx}"          # последний сохранённый снимок
    dirty_key = f"ts_dirty_{ctx}"         # флаг «что-то меняли»
    tick_key  = f"ts_last_change_{ctx}"   # время последнего изменения (для дебаунса)

    now = time.time()
    # если колбэк что-то менял — он уже поставил dirty=True; запомним момент
    if st.session_state.get(dirty_key):
        st.session_state[tick_key] = now

    last_change = st.session_state.get(tick_key, 0.0)

    # сохраняем только если:
    # 1) было изменение и прошло >0.4с тишины; или
    # 2) хэш отличается от сохранённого (например, после «Обновить из БД»)
    should_save = (
        (st.session_state.get(dirty_key) and now - last_change > 0.4) or
        (st.session_state.get(hash_key) != cur_hash)
    )

    if should_save:
        try:
            n = save_week_replace(user_id, week, tuples)  # DELETE неделя -> INSERT актуальных строк
            fetch_week_rows.clear()                       # сброс кеша читаемой недели
            st.session_state[hash_key]  = cur_hash
            st.session_state[dirty_key] = False
            # никаких toast/alert — чтобы не трогать DOM и не мешать вкладкам
        except Exception as e:
            # Покажем предупреждение, но без rerun
            st.warning(f"Автосохранение не удалось: {e}")
    # ------------------------------------------------------

    st.markdown(f"**Итого за неделю:** {sum(totals):g} ч")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_hours_interval(d1: date, d2: date) -> pd.DataFrame:
    """
    Возвращает DataFrame: work_date(date), user_id(int), first_name(str), hours(float)
    """
    eng = get_engine()
    user_tbl = _detect_user_table(eng)
    q = text(f"""
        SELECT l.work_date::date   AS work_date,
               u.id::bigint        AS user_id,
               u.first_name::text  AS first_name,
               SUM(l.hours)::float AS hours
        FROM log l
        JOIN {user_tbl} u ON u.id = l.user_id
        WHERE l.work_date BETWEEN :d1 AND :d2
        GROUP BY l.work_date, u.id, u.first_name
        ORDER BY l.work_date, u.first_name
    """)
    df = pd.read_sql(q, eng, params={"d1": d1, "d2": d2})
    if not df.empty:
        df["work_date"] = pd.to_datetime(df["work_date"]).dt.date
        df["hours"] = df["hours"].astype(float)
    return df


def _is_admin() -> bool:
    # признак из query params → session_state
    v = str(qp_get("admin", "0")).lower()
    if v in ("1", "true", "yes"):
        st.session_state["__admin__"] = True
    return bool(st.session_state.get("__admin__", False))


def _admin_off():
    st.session_state["__admin__"] = False
    qp_delete("admin")


def _inject_admin_hotkey():
    # Ловим Ctrl+Shift+A + Esc. Меняем query param и триггерим rerun.
    from streamlit.components.v1 import html as st_html
    st_html(
        """
        <script>
        (function () {
          const setAdmin = (on) => {
            const url = new URL(window.location.href);
            if (on) url.searchParams.set('admin','1'); else url.searchParams.delete('admin');
            window.history.replaceState({}, '', url);
            // попросим Streamlit сделать rerun
            window.parent.postMessage({isStreamlitMessage:true, type:'streamlit:rerun'}, '*');
          };
          window.addEventListener('keydown', (e) => {
            // Ctrl+Shift+A  → включить
            if (e.ctrlKey && e.shiftKey && !e.altKey && e.code === 'KeyA') setAdmin(true);
            // Esc → выключить
            if (!e.ctrlKey && !e.shiftKey && !e.altKey && e.code === 'Escape') setAdmin(false);
          }, {passive:true});
        })();
        </script>
        """,
        height=0,
    )


def render_admin_panel():
    import altair as alt

    st.markdown(
        """
        <div style="
            border:1px solid #e3e3e7;border-radius:12px;padding:14px 16px;margin:8px 0 16px 0;
            background:linear-gradient(180deg, #fff, #f9fafc);">
          <div style="font-weight:700;">🔐 Админ-панель · загрузка сотрудников</div>
          <div style="font-size:12px;color:#666;">Нажми Esc, чтобы выйти из режима.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Период
    c1, c2, c3 = st.columns([1.1, 1.1, 1])
    with c1:
        d2 = st.date_input("До", value=date.today(), format="DD.MM.YYYY")
    with c2:
        d1_default = date.today() - timedelta(days=28)
        d1 = st.date_input("С", value=d1_default, format="DD.MM.YYYY", max_value=d2)
    with c3:
        agg = st.selectbox("Детализация", ["По дням", "По неделям"])

    df = fetch_hours_interval(d1, d2)

    if df.empty:
        st.info("За выбранный период нет данных.")
    else:
        plot_df = df.copy()
        if agg == "По неделям":
            ts = pd.to_datetime(plot_df["work_date"])
            week_start = (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date
            plot_df = (
                plot_df.assign(week_start=week_start)
                       .groupby(["week_start", "first_name"], as_index=False)["hours"].sum()
            )
            x_field, x_title = "week_start:T", "Неделя (пн)"
        else:
            plot_df = plot_df.groupby(["work_date", "first_name"], as_index=False)["hours"].sum()
            x_field, x_title = "work_date:T", "Дата"

        chart = (
            alt.Chart(plot_df)
               .mark_bar()
               .encode(
                   x=alt.X(x_field, title=x_title),
                   y=alt.Y("sum(hours):Q", title="Часы"),
                   color=alt.Color("first_name:N", title="Сотрудник"),
                   tooltip=[x_field, "first_name:N", alt.Tooltip("sum(hours):Q", title="Часы")]
               )
               .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        # сводка по людям
        summary = plot_df.groupby("first_name", as_index=False)["hours"].sum().sort_values("hours", ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # выгрузка CSV
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Экспорт итогов (CSV)", data=csv, file_name=f"workload_{d1}_{d2}.csv", mime="text/csv")

    st.button("Выйти из админ-режима", type="secondary", on_click=_admin_off)







