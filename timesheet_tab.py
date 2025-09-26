# timesheet_tab.py
from __future__ import annotations
import os
from datetime import date, datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sqlmodel import SQLModel, Field, Session, select, create_engine

# ───────── DB ─────────
DB_URL = os.getenv("TIMESHEET_DB_URL") or "sqlite:///worklog.db"
_is_sqlite = DB_URL.startswith("sqlite")
_engine = create_engine(DB_URL, echo=False,
                        connect_args={"check_same_thread": False} if _is_sqlite else {})

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tg_id: Optional[int] = Field(default=None, index=True)
    first_name: str = Field(index=True)

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

class Log(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    hours: float
    work_date: date = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

def ensure_db():
    SQLModel.metadata.create_all(_engine)

def get_or_create_user(name: str) -> User:
    with Session(_engine) as s:
        u = s.exec(select(User).where(User.first_name == name.strip())).first()
        if u: return u
        u = User(first_name=name.strip())
        s.add(u); s.commit(); s.refresh(u); return u

def list_projects() -> List[Project]:
    with Session(_engine) as s:
        return s.exec(select(Project).order_by(Project.name)).all()

def add_project(name: str) -> Project:
    with Session(_engine) as s:
        ex = s.exec(select(Project).where(Project.name == name.strip())).first()
        if ex: return ex
        p = Project(name=name.strip()); s.add(p); s.commit(); s.refresh(p); return p

def add_logs_bulk(user_id: int, rows: list[tuple[int, float]], d: date) -> int:
    if not rows: return 0
    logs = [Log(user_id=user_id, project_id=pid, hours=float(h), work_date=d) for pid, h in rows if h > 0]
    with Session(_engine) as s:
        for l in logs: s.add(l)
        s.commit()
    return len(logs)

def logs_for_user_date(user_id: int, d: date):
    with Session(_engine) as s:
        q = (select(Log, Project.name)
             .where(Log.user_id == user_id, Log.work_date == d)
             .join(Project, Project.id == Log.project_id))
        return [(l.id, pname, l.hours) for l, pname in s.exec(q).all()]

def delete_log(log_id: int) -> None:
    with Session(_engine) as s:
        row = s.get(Log, log_id)
        if row: s.delete(row); s.commit()

def week_dataframe(user_id: int) -> pd.DataFrame:
    with Session(_engine) as s:
        since = (date.today() - timedelta(days=6))
        q = (select(Log, Project.name)
             .where(Log.user_id == user_id, Log.work_date >= since)
             .join(Project, Project.id == Log.project_id))
        recs = s.exec(q).all()
    if not recs: return pd.DataFrame(columns=["date","project","hours"])
    data = [{"date": l.work_date, "project": pname, "hours": l.hours} for l, pname in recs]
    return pd.DataFrame(data)

# ───────── UI ─────────
_HOURS_CHOICES = [x/2 for x in range(1, 21)]  # 0.5 … 10.0

def _init_state():
    if "ts_rows" not in st.session_state:
        st.session_state.ts_rows = [{"project_id": None, "hours": 2.0}]
    if "ts_user" not in st.session_state:
        st.session_state.ts_user = ""

def _css():
    st.markdown("""
    <style>
      .ts-card {background:#fff;border:1px solid #E6EEF5;border-radius:12px;padding:16px 16px 10px;margin-top:6px;}
      .ts-title {font-weight:600;margin-bottom:2px;}
      .ts-muted {color:#6b7280;font-size:12px;margin-bottom:10px;}
      .ts-add {font-size:13px;color:#2563eb;cursor:pointer;}
      .ts-chip {background:#F1F5F9;border:1px solid #E2E8F0;border-radius:10px;padding:4px 8px;font-size:12px;margin-right:6px;}
      div[data-testid="column"] > div > label {margin-bottom:4px}
    </style>
    """, unsafe_allow_html=True)

def render_timesheet_tab():
    ensure_db(); _init_state(); _css()

    st.subheader("⏱️ Timesheet — стиль Nikabot")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="ts-card">', unsafe_allow_html=True)
        st.markdown('<div class="ts-title">let’s catch up!</div>', unsafe_allow_html=True)
        st.markdown('<div class="ts-muted">Which project did you work on today?</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            work_date = st.date_input("Дата", value=date.today(), format="YYYY-MM-DD")
        with c2:
            st.session_state.ts_user = st.text_input("Ваше имя", st.session_state.ts_user, placeholder="Например, Klim")
        user_name = (st.session_state.ts_user or "").strip()

        # проекты
        projects = list_projects()
        proj_by_name = {p.name: p.id for p in projects}
        proj_names = list(proj_by_name.keys())

        with st.expander("➕ Добавить проект"):
            new_p = st.text_input("Название проекта")
            if st.button("Добавить проект", key="ts_add_proj") and new_p.strip():
                p = add_project(new_p.strip())
                st.success(f"Проект «{p.name}» добавлен.")
                st.rerun()

        with st.form("ts_form"):
            # строки «проект + время» (как в Nikabot)
            for i, row in enumerate(st.session_state.ts_rows):
                r1, r2, r3 = st.columns([2.5, 1, 0.7])
                with r1:
                    sel = st.selectbox("Проект", proj_names if proj_names else [""],
                                       index=0 if proj_names else 0, key=f"proj_{i}",
                                       placeholder="Select a project…")
                    st.session_state.ts_rows[i]["project_id"] = proj_by_name.get(sel)
                with r2:
                    sel_h = st.selectbox("Время", [f"{h:g}h" for h in _HOURS_CHOICES],
                                         index=3, key=f"hrs_{i}")
                    st.session_state.ts_rows[i]["hours"] = float(sel_h.rstrip("h"))
                with r3:
                    if st.form_submit_button("✖", type="secondary", use_container_width=True, key=f"del_{i}"):
                        st.session_state.ts_rows.pop(i); st.rerun()

            if st.form_submit_button("＋ Add project", type="secondary"):
                st.session_state.ts_rows.append({"project_id": None, "hours": 2.0})

            saved = st.form_submit_button("💾 SAVE", type="primary", use_container_width=True)
            if saved:
                if not user_name:
                    st.error("Введите имя.")
                else:
                    u = get_or_create_user(user_name)
                    rows = [(r["project_id"], float(r["hours"])) for r in st.session_state.ts_rows
                            if r.get("project_id") and float(r.get("hours", 0)) > 0]
                    n = add_logs_bulk(u.id, rows, work_date)
                    if n:
                        st.success(f"Сохранено записей: {n}")
                        st.session_state.ts_rows = [{"project_id": None, "hours": 2.0}]
                        st.rerun()
                    else:
                        st.warning("Нет валидных строк для сохранения.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if user_name:
            u = get_or_create_user(user_name)
            st.markdown("**Записи за выбранную дату**")
            day_rows = logs_for_user_date(u.id, work_date)
            if day_rows:
                for log_id, pname, hrs in day_rows:
                    a, b = st.columns([0.75, 0.25])
                    with a: st.write(f"• {pname} — {hrs:g} ч.")
                    with b:
                        if st.button("Удалить", key=f"delbtn_{log_id}"):
                            delete_log(log_id); st.rerun()
            else:
                st.info("Пока нет записей на эту дату.")

            # график «моя неделя»
            df = week_dataframe(u.id)
            if not df.empty:
                st.markdown("**Моя неделя**")
                days = pd.date_range(date.today() - timedelta(days=6), periods=7).date
                pivot = (df.pivot_table(index="date", columns="project", values="hours", aggfunc="sum")
                           .reindex(days, fill_value=0))
                x = np.arange(len(days))
                fig, ax = plt.subplots(figsize=(6.8, 3.2))
                bottom = np.zeros(len(days))
                for proj in pivot.columns:
                    vals = pivot[proj].values
                    ax.bar(x, vals, bottom=bottom, label=proj)
                    bottom += vals
                ax.set_xticks(x)
                ax.set_xticklabels([d.strftime("%d.%m") for d in days])
                ax.set_ylabel("Часы"); ax.set_xlabel("Дни")
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Нет данных за последние 7 дней.")
        else:
            st.info("Введите имя, чтобы увидеть записи и график.")
