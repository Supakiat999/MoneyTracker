from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Any, List, Tuple
import uuid

import pandas as pd
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta


# ========= STORAGE: PROJECT FOLDER =========
# Save DB in the SAME folder as app.py
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR
APP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB = APP_DIR / "spending.db"
BACKUP_DIR = APP_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Default path for OLD CSV from previous simple app (adjust if needed)
LEGACY_CSV_DEFAULT = Path.home() / "spending_tracker" / "spendings.csv"

# ========= CONSTANTS =========
DEFAULT_CATEGORIES = [
    "Food & Drinks", "Groceries", "Transport", "Shopping", "Bills",
    "Entertainment", "Health", "Education", "Rent", "Other",
]
DEFAULT_ACCOUNTS = ["Cash", "Debit Card", "Credit Card", "Bank Transfer"]
TX_TYPES = ["expense", "income", "transfer"]


# ========= UTILS =========
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_yyyy_mm_dd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def open_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT,
            is_hidden INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL UNIQUE,
            monthly_limit REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS recurring_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            tx_type TEXT NOT NULL,
            amount REAL NOT NULL,
            category_id INTEGER,
            account_id INTEGER,
            start_date TEXT NOT NULL,
            frequency TEXT NOT NULL,          -- daily / weekly / monthly
            interval_n INTEGER NOT NULL DEFAULT 1,
            next_run_date TEXT NOT NULL,
            end_date TEXT,
            note TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            tx_date TEXT NOT NULL,            -- YYYY-MM-DD
            tx_type TEXT NOT NULL,            -- expense/income/transfer
            amount REAL NOT NULL,
            category_id INTEGER,
            account_id INTEGER,
            merchant TEXT,
            note TEXT,
            group_id TEXT,                    -- for splits
            recurring_id INTEGER,
            deleted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL,
            FOREIGN KEY (recurring_id) REFERENCES recurring_rules(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(tx_date);
        CREATE INDEX IF NOT EXISTS idx_tx_deleted ON transactions(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(tx_type);
        CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category_id);
        CREATE INDEX IF NOT EXISTS idx_tx_account ON transactions(account_id);

        CREATE UNIQUE INDEX IF NOT EXISTS uniq_recurring_occurrence
        ON transactions(recurring_id, tx_date)
        WHERE recurring_id IS NOT NULL;
        """
    )
    conn.commit()


def q(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def exec1(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> None:
    conn.execute(sql, params)
    conn.commit()


def seed_defaults(conn: sqlite3.Connection) -> None:
    for name in DEFAULT_CATEGORIES:
        exec1(conn, "INSERT OR IGNORE INTO categories(name, created_at) VALUES (?, ?)", (name, now_iso()))
    for name in DEFAULT_ACCOUNTS:
        exec1(conn, "INSERT OR IGNORE INTO accounts(name, created_at) VALUES (?, ?)", (name, now_iso()))


def get_categories(conn: sqlite3.Connection, include_hidden: bool = False) -> pd.DataFrame:
    if include_hidden:
        rows = q(conn, "SELECT id, name, color, is_hidden FROM categories ORDER BY LOWER(name)")
    else:
        rows = q(conn, "SELECT id, name, color, is_hidden FROM categories WHERE is_hidden=0 ORDER BY LOWER(name)")
    return pd.DataFrame([dict(r) for r in rows])


def get_accounts(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = q(conn, "SELECT id, name FROM accounts ORDER BY LOWER(name)")
    return pd.DataFrame([dict(r) for r in rows])


def advance_date(d: date, frequency: str, interval_n: int) -> date:
    if frequency == "daily":
        return d + timedelta(days=interval_n)
    if frequency == "weekly":
        return d + timedelta(weeks=interval_n)
    if frequency == "monthly":
        return d + relativedelta(months=+interval_n)
    raise ValueError("Unsupported frequency")


def run_recurring_catchup(conn: sqlite3.Connection, today: date) -> int:
    created = 0
    rules = q(conn, "SELECT * FROM recurring_rules WHERE is_active=1")
    for r in rules:
        next_run = parse_yyyy_mm_dd(r["next_run_date"])
        end_date = parse_yyyy_mm_dd(r["end_date"]) if r["end_date"] else None

        while next_run <= today and (end_date is None or next_run <= end_date):
            try:
                exec1(
                    conn,
                    """
                    INSERT INTO transactions(
                        id, tx_date, tx_type, amount, category_id, account_id,
                        merchant, note, group_id, recurring_id,
                        deleted_at, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        fmt_yyyy_mm_dd(next_run),
                        r["tx_type"],
                        float(r["amount"]),
                        r["category_id"],
                        r["account_id"],
                        None,
                        r["note"],
                        None,
                        r["id"],
                        now_iso(),
                        now_iso(),
                    ),
                )
                created += 1
            except sqlite3.IntegrityError:
                # duplicate occurrence for that day & rule -> skip
                pass

            next_run = advance_date(next_run, r["frequency"], int(r["interval_n"]))

        exec1(
            conn,
            "UPDATE recurring_rules SET next_run_date=?, updated_at=? WHERE id=?",
            (fmt_yyyy_mm_dd(next_run), now_iso(), r["id"]),
        )
    return created


def fetch_transactions(
    conn: sqlite3.Connection,
    start_d: date,
    end_d: date,
    types: Optional[List[str]] = None,
    category_ids: Optional[List[int]] = None,
    account_ids: Optional[List[int]] = None,
    search: str = "",
    include_deleted: bool = False,
) -> pd.DataFrame:
    where = ["t.tx_date BETWEEN ? AND ?"]
    params: List[Any] = [fmt_yyyy_mm_dd(start_d), fmt_yyyy_mm_dd(end_d)]

    if not include_deleted:
        where.append("t.deleted_at IS NULL")

    if types:
        where.append(f"t.tx_type IN ({','.join('?' for _ in types)})")
        params += types

    if category_ids:
        where.append(f"t.category_id IN ({','.join('?' for _ in category_ids)})")
        params += category_ids

    if account_ids:
        where.append(f"t.account_id IN ({','.join('?' for _ in account_ids)})")
        params += account_ids

    if search.strip():
        where.append("(LOWER(COALESCE(t.note,'')) LIKE ? OR LOWER(COALESCE(t.merchant,'')) LIKE ?)")
        s = f"%{search.strip().lower()}%"
        params += [s, s]

    sql = f"""
    SELECT
        t.id, t.tx_date, t.tx_type, t.amount,
        c.name AS category,
        a.name AS account,
        t.merchant, t.note, t.group_id, t.recurring_id,
        t.deleted_at, t.created_at, t.updated_at
    FROM transactions t
    LEFT JOIN categories c ON c.id=t.category_id
    LEFT JOIN accounts a ON a.id=t.account_id
    WHERE {' AND '.join(where)}
    ORDER BY t.tx_date DESC, t.created_at DESC
    """
    rows = q(conn, sql, tuple(params))
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def insert_transaction(
    conn: sqlite3.Connection,
    tx_date: date,
    tx_type: str,
    amount: float,
    category_id: Optional[int],
    account_id: Optional[int],
    merchant: str,
    note: str,
    group_id: Optional[str] = None,
) -> None:
    exec1(
        conn,
        """
        INSERT INTO transactions(
            id, tx_date, tx_type, amount, category_id, account_id,
            merchant, note, group_id, recurring_id, deleted_at, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            fmt_yyyy_mm_dd(tx_date),
            tx_type,
            float(amount),
            category_id,
            account_id,
            merchant.strip() or None,
            note.strip() or None,
            group_id,
            now_iso(),
            now_iso(),
        ),
    )


def soft_delete_transactions(conn: sqlite3.Connection, ids: List[str]) -> int:
    if not ids:
        return 0
    exec1(
        conn,
        f"UPDATE transactions SET deleted_at=?, updated_at=? WHERE id IN ({','.join('?' for _ in ids)})",
        tuple([now_iso(), now_iso()] + ids),
    )
    return len(ids)


def restore_transactions(conn: sqlite3.Connection, ids: List[str]) -> int:
    if not ids:
        return 0
    exec1(
        conn,
        f"UPDATE transactions SET deleted_at=NULL, updated_at=? WHERE id IN ({','.join('?' for _ in ids)})",
        tuple([now_iso()] + ids),
    )
    return len(ids)


def purge_transactions(conn: sqlite3.Connection, ids: List[str]) -> int:
    if not ids:
        return 0
    exec1(conn, f"DELETE FROM transactions WHERE id IN ({','.join('?' for _ in ids)})", tuple(ids))
    return len(ids)


def get_budget_table(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = q(
        conn,
        """
        SELECT b.id, c.id AS category_id, c.name AS category, b.monthly_limit, b.updated_at
        FROM budgets b
        JOIN categories c ON c.id=b.category_id
        ORDER BY LOWER(c.name)
        """,
    )
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["monthly_limit"] = pd.to_numeric(df["monthly_limit"], errors="coerce")
    return df


def upsert_budget(conn: sqlite3.Connection, category_id: int, monthly_limit: float) -> None:
    exec1(
        conn,
        """
        INSERT INTO budgets(category_id, monthly_limit, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(category_id) DO UPDATE SET
            monthly_limit=excluded.monthly_limit,
            updated_at=excluded.updated_at
        """,
        (category_id, float(monthly_limit), now_iso(), now_iso()),
    )


def create_backup(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = BACKUP_DIR / f"spending_{ts}.db"
    out.write_bytes(db_path.read_bytes())
    return out


def money_fmt(sym: str, x: float) -> str:
    return f"{sym}{x:,.2f}"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_to_date(val) -> Optional[date]:
    dt = pd.to_datetime(val, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def import_csv_transactions(conn: sqlite3.Connection, df: pd.DataFrame, mapping: dict[str, str]) -> tuple[int, int]:
    """Generic CSV importer (maps arbitrary columns)."""
    inserted = 0
    skipped = 0

    cats = get_categories(conn, include_hidden=True)
    accs = get_accounts(conn)
    cat_map = {str(n).strip().lower(): int(i) for i, n in zip(cats["id"], cats["name"])} if not cats.empty else {}
    acc_map = {str(n).strip().lower(): int(i) for i, n in zip(accs["id"], accs["name"])} if not accs.empty else {}

    for _, row in df.iterrows():
        d = safe_to_date(row.get(mapping["date"], ""))
        try:
            amt = float(row.get(mapping["amount"], ""))
        except Exception:
            amt = None

        if d is None or amt is None:
            skipped += 1
            continue

        tx_type = str(row.get(mapping["type"], "expense")).strip().lower()
        if tx_type not in TX_TYPES:
            tx_type = "expense"

        cat_name = str(row.get(mapping["category"], "")).strip()
        acc_name = str(row.get(mapping["account"], "")).strip()
        note = str(row.get(mapping["note"], "")).strip()
        merchant = str(row.get(mapping["merchant"], "")).strip()

        cat_id = None
        if cat_name:
            key = cat_name.lower()
            if key not in cat_map:
                exec1(conn, "INSERT OR IGNORE INTO categories(name, created_at) VALUES (?,?)", (cat_name, now_iso()))
                cats = get_categories(conn, include_hidden=True)
                cat_map = {str(n).strip().lower(): int(i) for i, n in zip(cats["id"], cats["name"])}
            cat_id = cat_map.get(key)

        acc_id = None
        if acc_name:
            key = acc_name.lower()
            if key not in acc_map:
                exec1(conn, "INSERT OR IGNORE INTO accounts(name, created_at) VALUES (?,?)", (acc_name, now_iso()))
                accs = get_accounts(conn)
                acc_map = {str(n).strip().lower(): int(i) for i, n in zip(accs["id"], accs["name"])}
            acc_id = acc_map.get(key)

        insert_transaction(conn, d, tx_type, amt, cat_id, acc_id, merchant, note)
        inserted += 1

    return inserted, skipped


def import_legacy_csv(conn: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    """
    Import from OLD simple-app CSV (expected columns: date, amount, category, note).
    Everything is treated as 'expense', account=None, merchant=None.
    Avoids duplicates by (date, amount, category_id, note).
    """
    if not csv_path.exists():
        return 0, 0

    df = pd.read_csv(csv_path, encoding_errors="ignore")
    needed = ["date", "amount", "category", "note"]
    for col in needed:
        if col not in df.columns:
            return 0, 0

    cats = get_categories(conn, include_hidden=True)
    cat_map = {str(n).strip().lower(): int(i) for i, n in zip(cats["id"], cats["name"])} if not cats.empty else {}

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        d = safe_to_date(row.get("date", ""))
        try:
            amt = float(row.get("amount", ""))
        except Exception:
            amt = None

        if d is None or amt is None:
            skipped += 1
            continue

        cat_name = str(row.get("category", "")).strip()
        note = str(row.get("note", "")).strip()

        cat_id = None
        if cat_name:
            key = cat_name.lower()
            if key not in cat_map:
                exec1(conn, "INSERT OR IGNORE INTO categories(name, created_at) VALUES (?,?)", (cat_name, now_iso()))
                cats = get_categories(conn, include_hidden=True)
                cat_map = {str(n).strip().lower(): int(i) for i, n in zip(cats["id"], cats["name"])}
            cat_id = cat_map.get(key)

        # Check for duplicate
        existing = q(
            conn,
            """
            SELECT 1
            FROM transactions
            WHERE tx_date=? AND amount=? AND IFNULL(category_id,0)=? AND IFNULL(note,'')=?
              AND deleted_at IS NULL
            """,
            (fmt_yyyy_mm_dd(d), float(amt), cat_id or 0, note),
        )
        if existing:
            skipped += 1
            continue

        insert_transaction(conn, d, "expense", float(amt), cat_id, None, "", note)
        inserted += 1

    return inserted, skipped


# ========= MAIN APP =========
def main() -> None:
    st.set_page_config(page_title="Spending Tracker Pro", page_icon="💸", layout="wide")

    st.title("💸 Spending Tracker Pro (local-first)")

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Settings")
        currency = st.text_input("Currency symbol", value="฿", key="currency")
        db_path_str = st.text_input("Database path", value=str(DEFAULT_DB), key="db_path")
        db_path = Path(db_path_str).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        st.caption("Local storage (SQLite). CSV + backup options inside the app.")

    conn = open_conn(db_path)
    init_db(conn)
    seed_defaults(conn)

    created = run_recurring_catchup(conn, date.today())
    if created:
        st.toast(f"Added {created} recurring transaction(s).", icon="🔁")

    cats_df = get_categories(conn, include_hidden=False)
    accs_df = get_accounts(conn)
    cat_label_to_id = {r["name"]: int(r["id"]) for _, r in cats_df.iterrows()} if not cats_df.empty else {}
    acc_label_to_id = {r["name"]: int(r["id"]) for _, r in accs_df.iterrows()} if not accs_df.empty else {}

    tabs = st.tabs(["Dashboard", "Add", "Transactions", "Budgets", "Recurring", "Import/Export", "Categories", "Backup"])

    # ----- Dashboard -----
    with tabs[0]:
        st.subheader("Overview")
        today = date.today()
        start_default = today - timedelta(days=90)

        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 2])
        with c1:
            start_d = st.date_input("Start", value=start_default, key="dash_start")
        with c2:
            end_d = st.date_input("End", value=today, key="dash_end")
        with c3:
            tx_types = st.multiselect("Types", TX_TYPES, default=["expense", "income"], key="dash_types")
        with c4:
            search = st.text_input("Search (note/merchant)", value="", key="dash_search")

        df_dash = fetch_transactions(conn, start_d, end_d, types=tx_types, search=search)
        if df_dash.empty:
            st.info("No data in this range yet.")
        else:
            exp = df_dash.loc[df_dash["tx_type"] == "expense", "amount"].sum()
            inc = df_dash.loc[df_dash["tx_type"] == "income", "amount"].sum()
            net = inc - exp

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expense", money_fmt(currency, float(exp)))
            m2.metric("Income", money_fmt(currency, float(inc)))
            m3.metric("Net", money_fmt(currency, float(net)))
            m4.metric("Transactions", f"{len(df_dash):,}")

            st.divider()

            dff = df_dash.copy()
            dff["month"] = pd.to_datetime(dff["tx_date"]).dt.to_period("M").astype(str)
            monthly = dff.groupby(["month", "tx_type"], as_index=False)["amount"].sum()
            monthly["month"] = pd.to_datetime(monthly["month"])

            chart = (
                alt.Chart(monthly)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("amount:Q", title=f"Amount ({currency})"),
                    color="tx_type:N",
                    tooltip=["tx_type:N", alt.Tooltip("amount:Q", format=",.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")

            expenses = dff[dff["tx_type"] == "expense"].copy()
            if not expenses.empty:
                by_cat = (
                    expenses.groupby("category", as_index=False)["amount"]
                    .sum()
                    .sort_values("amount", ascending=False)
                )
                by_cat["category"] = by_cat["category"].fillna("Uncategorized")

                bar = (
                    alt.Chart(by_cat)
                    .mark_bar()
                    .encode(
                        x=alt.X("amount:Q", title=f"Expense ({currency})"),
                        y=alt.Y("category:N", sort="-x", title="Category"),
                        tooltip=["category:N", alt.Tooltip("amount:Q", format=",.2f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(bar, width="stretch")

            st.subheader("Recent")
            st.dataframe(df_dash.head(30), hide_index=True, width="stretch")

    # ----- Add -----
    with tabs[1]:
        st.subheader("Add transaction")
        colL, colR = st.columns([1.25, 1.0])

        with colL:
            with st.form("add_tx", clear_on_submit=True):
                tx_date = st.date_input("Date", value=date.today(), key="add_date")
                tx_type = st.selectbox("Type", TX_TYPES, index=0, key="add_type")
                amount = st.number_input("Amount", value=0.0, step=10.0, format="%.2f", key="add_amount")

                cat_options = list(cat_label_to_id.keys())
                cat_pick = st.selectbox("Category", ["(Uncategorized)"] + cat_options, key="add_cat")
                category_id = None if cat_pick == "(Uncategorized)" else cat_label_to_id.get(cat_pick)

                acc_options = list(acc_label_to_id.keys())
                acc_pick = st.selectbox("Account", ["(None)"] + acc_options, key="add_acc")
                account_id = None if acc_pick == "(None)" else acc_label_to_id.get(acc_pick)

                merchant = st.text_input("Merchant (optional)", value="", key="add_merchant")
                note = st.text_input("Note (optional)", value="", key="add_note")

                split = st.toggle("Split into multiple categories", value=False, key="add_split")
                splits_df = None
                if split:
                    st.caption("Add split lines (each line becomes a separate transaction).")
                    default_cat = cat_options[0] if cat_options else "Other"
                    splits_df = st.data_editor(
                        pd.DataFrame(
                            [{"category": default_cat, "amount": float(amount), "note": ""}]
                        ),
                        num_rows="dynamic",
                        hide_index=True,
                        width="stretch",
                        key="add_split_editor",
                    )

                submitted = st.form_submit_button("Add", width="stretch")

            if submitted:
                if not split:
                    insert_transaction(conn, tx_date, tx_type, float(amount), category_id, account_id, merchant, note, None)
                    st.success("Added.")
                else:
                    group_id = str(uuid.uuid4())
                    inserted_n = 0
                    if splits_df is not None and not splits_df.empty:
                        for _, r in splits_df.iterrows():
                            c_name = str(r.get("category", "")).strip()
                            try:
                                a_amt = float(r.get("amount", ""))
                            except Exception:
                                continue
                            c_id = cat_label_to_id.get(c_name) if c_name in cat_label_to_id else None
                            n_note = str(r.get("note", "")).strip()
                            insert_transaction(conn, tx_date, tx_type, a_amt, c_id, account_id, merchant, n_note or note, group_id)
                            inserted_n += 1
                    st.success(f"Added {inserted_n} split line(s).")

        with colR:
            st.subheader("Quick create")
            with st.expander("Add category"):
                new_cat = st.text_input("New category name", value="", key="qc_new_cat")
                if st.button("Create category", key="qc_add_cat", width="stretch"):
                    if new_cat.strip():
                        exec1(conn, "INSERT OR IGNORE INTO categories(name, created_at) VALUES (?,?)", (new_cat.strip(), now_iso()))
                        st.success("Created.")
                        st.rerun()

            with st.expander("Add account"):
                new_acc = st.text_input("New account name", value="", key="qc_new_acc")
                if st.button("Create account", key="qc_add_acc", width="stretch"):
                    if new_acc.strip():
                        exec1(conn, "INSERT OR IGNORE INTO accounts(name, created_at) VALUES (?,?)", (new_acc.strip(), now_iso()))
                        st.success("Created.")
                        st.rerun()

    # ----- Transactions -----
    with tabs[2]:
        st.subheader("Transactions (edit + delete)")
        today = date.today()

        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.6])
        with c1:
            start_d = st.date_input("Start", value=today - timedelta(days=30), key="tx_start")
        with c2:
            end_d = st.date_input("End", value=today, key="tx_end")
        with c3:
            types = st.multiselect("Types", TX_TYPES, default=TX_TYPES, key="tx_types")
        with c4:
            search = st.text_input("Search (note/merchant)", value="", key="tx_search")

        c5, c6, c7 = st.columns([1.4, 1.2, 1.2])
        with c5:
            picked_cats = st.multiselect("Categories", list(cat_label_to_id.keys()), default=list(cat_label_to_id.keys()), key="tx_cats")
        with c6:
            picked_accs = st.multiselect("Accounts", list(acc_label_to_id.keys()), default=list(acc_label_to_id.keys()), key="tx_accs")
        with c7:
            show_deleted = st.toggle("Show deleted (Trash)", value=False, key="tx_show_deleted")

        category_ids = [cat_label_to_id[n] for n in picked_cats] if picked_cats else None
        account_ids = [acc_label_to_id[n] for n in picked_accs] if picked_accs else None

        df_tx = fetch_transactions(
            conn,
            start_d,
            end_d,
            types=types if types else None,
            category_ids=category_ids,
            account_ids=account_ids,
            search=search,
            include_deleted=show_deleted,
        )

        if df_tx.empty:
            st.info("No transactions found.")
        else:
            view = df_tx.copy()
            view.insert(0, "select", False)

            edited = st.data_editor(
                view,
                hide_index=True,
                width="stretch",
                column_config={
                    "select": st.column_config.CheckboxColumn("Select"),
                    "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                    "tx_date": st.column_config.DateColumn("Date"),
                },
                disabled=["id", "created_at", "updated_at", "deleted_at", "group_id", "recurring_id"],
                key="tx_editor",
            )

            selected_ids = edited.loc[edited["select"] == True, "id"].astype(str).tolist()

            b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
            with b1:
                if st.button("Save edits", key="tx_save", width="stretch"):
                    to_save = edited.drop(columns=["select"]).copy()
                    for _, r in to_save.iterrows():
                        tid = str(r["id"])
                        d = safe_to_date(r.get("tx_date"))
                        if d is None:
                            continue

                        cat_name = r.get("category", None)
                        acc_name = r.get("account", None)
                        cat_id = cat_label_to_id.get(cat_name) if isinstance(cat_name, str) and cat_name in cat_label_to_id else None
                        acc_id = acc_label_to_id.get(acc_name) if isinstance(acc_name, str) and acc_name in acc_label_to_id else None

                        exec1(
                            conn,
                            """
                            UPDATE transactions
                            SET tx_date=?, tx_type=?, amount=?, category_id=?, account_id=?, merchant=?, note=?, updated_at=?
                            WHERE id=?
                            """,
                            (
                                fmt_yyyy_mm_dd(d),
                                str(r.get("tx_type", "expense")).lower(),
                                float(r.get("amount", 0.0)),
                                cat_id,
                                acc_id,
                                (str(r.get("merchant", "")).strip() or None),
                                (str(r.get("note", "")).strip() or None),
                                now_iso(),
                                tid,
                            ),
                        )
                    st.success("Saved.")
                    st.rerun()

            with b2:
                confirm_del = st.checkbox("Confirm delete", value=False, key="tx_confirm_delete")
                if st.button("Delete selected", key="tx_delete", width="stretch", disabled=not confirm_del):
                    n = soft_delete_transactions(conn, selected_ids)
                    st.success(f"Deleted {n} row(s). (Moved to Trash)")
                    st.rerun()

            with b3:
                if st.button("Restore selected", key="tx_restore", width="stretch"):
                    n = restore_transactions(conn, selected_ids)
                    st.success(f"Restored {n} row(s).")
                    st.rerun()

            with b4:
                confirm_purge = st.checkbox("Confirm PERMANENT delete", value=False, key="tx_confirm_purge")
                if st.button("Purge selected (PERMANENT)", key="tx_purge", width="stretch", disabled=not confirm_purge):
                    n = purge_transactions(conn, selected_ids)
                    st.success(f"Purged {n} row(s).")
                    st.rerun()

    # ----- Budgets -----
    with tabs[3]:
        st.subheader("Budgets (monthly per category)")
        bdf = get_budget_table(conn)
        cats_live = get_categories(conn, include_hidden=False)

        if cats_live.empty:
            st.info("Create categories first.")
        else:
            col1, col2 = st.columns([1.2, 1])
            with col1:
                cat_pick = st.selectbox("Category", cats_live["name"].tolist(), key="bud_cat")
            with col2:
                limit = st.number_input("Monthly limit", value=0.0, step=100.0, format="%.2f", key="bud_limit")
            if st.button("Set budget", key="bud_set", width="stretch"):
                cid = int(cats_live.loc[cats_live["name"] == cat_pick, "id"].iloc[0])
                upsert_budget(conn, cid, float(limit))
                st.success("Budget set.")
                st.rerun()

            st.divider()
            if bdf.empty:
                st.info("No budgets yet.")
            else:
                st.dataframe(bdf, hide_index=True, width="stretch")

            st.divider()
            st.markdown("**This month progress**")
            start_m = date.today().replace(day=1)
            txm = fetch_transactions(conn, start_m, date.today(), types=["expense"])
            if txm.empty:
                st.caption("No expenses this month yet.")
            else:
                txm["category"] = txm["category"].fillna("Uncategorized")
                spent = txm.groupby("category", as_index=False)["amount"].sum()
                merged = spent.merge(
                    bdf[["category", "monthly_limit"]] if not bdf.empty else pd.DataFrame(columns=["category", "monthly_limit"]),
                    on="category",
                    how="left",
                )
                merged["monthly_limit"] = merged["monthly_limit"].fillna(0.0)
                merged = merged.sort_values("amount", ascending=False)

                for _, r in merged.iterrows():
                    cat = r["category"]
                    amt = float(r["amount"])
                    lim = float(r["monthly_limit"])
                    if lim > 0:
                        st.write(f"{cat}: {money_fmt(currency, amt)} / {money_fmt(currency, lim)}")
                        st.progress(min(1.0, amt / lim))
                    else:
                        st.write(f"{cat}: {money_fmt(currency, amt)} (no budget set)")

    # ----- Recurring -----
    with tabs[4]:
        st.subheader("Recurring rules")
        cats_live = get_categories(conn, include_hidden=False)
        accs_live = get_accounts(conn)

        with st.form("add_recurring", clear_on_submit=True):
            name = st.text_input("Name", value="Rent", key="rr_name")
            tx_type = st.selectbox("Type", TX_TYPES, index=0, key="rr_type")
            amount = st.number_input("Amount", value=0.0, step=10.0, format="%.2f", key="rr_amt")

            colA, colB = st.columns(2)
            with colA:
                cat_name = st.selectbox("Category", ["(Uncategorized)"] + cats_live["name"].tolist(), key="rr_cat")
            with colB:
                acc_name = st.selectbox("Account", ["(None)"] + accs_live["name"].tolist(), key="rr_acc")

            start_date = st.date_input("Start date", value=date.today(), key="rr_start")
            frequency = st.selectbox("Frequency", ["monthly", "weekly", "daily"], index=0, key="rr_freq")
            interval_n = st.number_input("Every N", min_value=1, value=1, step=1, key="rr_int")

            has_end = st.checkbox("Has end date", value=False, key="rr_has_end")
            end_date = (
                st.date_input("End date", value=start_date + relativedelta(months=+12), key="rr_end")
                if has_end
                else None
            )

            note = st.text_input("Note (optional)", value="", key="rr_note")

            create = st.form_submit_button("Create recurring rule", width="stretch")

        if create:
            cat_id = None if cat_name == "(Uncategorized)" else int(cats_live.loc[cats_live["name"] == cat_name, "id"].iloc[0])
            acc_id = None if acc_name == "(None)" else int(accs_live.loc[accs_live["name"] == acc_name, "id"].iloc[0])

            exec1(
                conn,
                """
                INSERT INTO recurring_rules(
                    name, tx_type, amount, category_id, account_id,
                    start_date, frequency, interval_n, next_run_date, end_date,
                    note, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    name.strip(),
                    tx_type,
                    float(amount),
                    cat_id,
                    acc_id,
                    fmt_yyyy_mm_dd(start_date),
                    frequency,
                    int(interval_n),
                    fmt_yyyy_mm_dd(start_date),
                    fmt_yyyy_mm_dd(end_date) if end_date else None,
                    note.strip() or None,
                    now_iso(),
                    now_iso(),
                ),
            )
            st.success("Recurring rule created.")
            st.rerun()

        st.divider()
        rules = q(
            conn,
            """
            SELECT rr.id, rr.name, rr.tx_type, rr.amount, c.name AS category, a.name AS account,
                   rr.start_date, rr.frequency, rr.interval_n, rr.next_run_date, rr.end_date,
                   rr.note, rr.is_active, rr.updated_at
            FROM recurring_rules rr
            LEFT JOIN categories c ON c.id=rr.category_id
            LEFT JOIN accounts a ON a.id=rr.account_id
            ORDER BY rr.is_active DESC, rr.next_run_date ASC
            """,
        )
        rdf = pd.DataFrame([dict(r) for r in rules])
        if rdf.empty:
            st.info("No recurring rules yet.")
        else:
            st.dataframe(rdf, hide_index=True, width="stretch")
            if st.button("Run recurring now", key="rr_run", width="stretch"):
                n = run_recurring_catchup(conn, date.today())
                st.success(f"Generated {n} transaction(s).")
                st.rerun()

    # ----- Import / Export -----
    with tabs[5]:
        st.subheader("Import / Export CSV")
        today = date.today()

        c1, c2 = st.columns(2)
        with c1:
            exp_start = st.date_input("Export start", value=today - timedelta(days=365), key="exp_start")
        with c2:
            exp_end = st.date_input("Export end", value=today, key="exp_end")
        exp_df = fetch_transactions(conn, exp_start, exp_end, include_deleted=False)
        st.download_button(
            "Download CSV (transactions)",
            data=df_to_csv_bytes(exp_df),
            file_name=f"transactions_{fmt_yyyy_mm_dd(exp_start)}_to_{fmt_yyyy_mm_dd(exp_end)}.csv",
            mime="text/csv",
            width="stretch",
            key="exp_download",
        )

        st.divider()

        # ---- Import from legacy CSV path (old app) ----
        with st.expander("Import from legacy CSV (old app)", expanded=False):
            legacy_path_str = st.text_input(
                "Legacy CSV path",
                value=str(LEGACY_CSV_DEFAULT),
                key="legacy_csv_path",
            )
            legacy_path = Path(legacy_path_str).expanduser()
            if not legacy_path.exists():
                st.caption("File not found. Adjust the path to your old spendings.csv.")
            else:
                st.caption(f"Found file: {legacy_path}")
                if st.button("Import from this CSV", key="legacy_import_btn", width="stretch"):
                    inserted, skipped = import_legacy_csv(conn, legacy_path)
                    st.success(f"Imported {inserted} row(s). Skipped {skipped} row(s).")
                    st.rerun()

        st.divider()

        # ---- Generic CSV import via browser upload ----
        up = st.file_uploader("Upload CSV to import (custom mapping)", type=["csv"], key="imp_upload")
        if up is not None:
            imp = pd.read_csv(up, encoding_errors="ignore")
            st.dataframe(imp.head(25), hide_index=True, width="stretch")

            cols = imp.columns.tolist()
            mapping = {
                "date": st.selectbox("Column: date", cols, key="imp_date"),
                "amount": st.selectbox("Column: amount", cols, key="imp_amount"),
                "type": st.selectbox("Column: type", cols, key="imp_type"),
                "category": st.selectbox("Column: category", cols, key="imp_cat"),
                "account": st.selectbox("Column: account", cols, key="imp_acc"),
                "note": st.selectbox("Column: note", cols, key="imp_note"),
                "merchant": st.selectbox("Column: merchant", cols, key="imp_merchant"),
            }
            if st.button("Import rows", key="imp_run", width="stretch"):
                inserted, skipped = import_csv_transactions(conn, imp, mapping)
                st.success(f"Imported {inserted} row(s). Skipped {skipped} row(s).")
                st.rerun()

    # ----- Categories -----
    with tabs[6]:
        st.subheader("Categories")
        cats_all = get_categories(conn, include_hidden=True)
        st.dataframe(cats_all, hide_index=True, width="stretch")

        new_cat = st.text_input("New category name", value="", key="cats_new_cat")
        if st.button("Add category", key="cats_add", width="stretch"):
            if new_cat.strip():
                exec1(conn, "INSERT OR IGNORE INTO categories(name, created_at) VALUES (?,?)", (new_cat.strip(), now_iso()))
                st.success("Added.")
                st.rerun()

    # ----- Backup -----
    with tabs[7]:
        st.subheader("Backup / safety")
        st.write(f"DB file: `{db_path}`")
        st.write(f"Backups folder: `{BACKUP_DIR}`")
        if st.button("Create backup now", key="backup_btn", width="stretch"):
            if db_path.exists():
                out = create_backup(db_path)
                st.success(f"Backup created: {out}")
            else:
                st.warning("Database file not found yet.")

    st.caption(f"Local DB: {db_path}")


if __name__ == "__main__":
    main()
