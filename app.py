import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_cookies_controller import CookieController

DB = Path(__file__).parent / "inventory.db"

st.set_page_config(
    page_title="Mercury Inventory BI",
    page_icon="📦",
    layout="wide"
)

cookies = CookieController()

saved_token = cookies.get("mercury_auth")
if saved_token == "authenticated" and "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="background:#c8102e;color:white;padding:25px;
                border-radius:12px;text-align:center">
        <h1 style="color:white">💊 Inventory BI System</h1>
        <p>Predictive Supply Chain & Inventory Optimization</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            valid_user = st.secrets["login"]["username"]
            valid_password = st.secrets["login"]["password"]

            if username == valid_user and password == valid_password:
                st.session_state["authenticated"] = True
                cookies.set("mercury_auth", "authenticated")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        except (KeyError, FileNotFoundError):
            st.error("Configure your login credentials in secrets.toml.")

    st.stop()


@contextmanager
def get_db():
    con = sqlite3.connect(DB)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize():
    with get_db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS products (
            sku TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            supplier TEXT,
            stock INTEGER NOT NULL DEFAULT 0,
            unit_cost REAL DEFAULT 0,
            lead_days INTEGER DEFAULT 7,
            expiry TEXT
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_date TEXT NOT NULL,
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            delivery_date TEXT,
            sku TEXT,
            quantity INTEGER,
            supplier TEXT
        )
        """)

        count = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]

        if count == 0:
            demo_products = [
                ("MED001", "Paracetamol 500mg", "OTC", "Supplier A", 180, 1.50, 5, 420),
                ("MED002", "Cetirizine 10mg", "OTC", "Supplier B", 42, 2.25, 7, 45),
                ("MED003", "Oral Rehydration Salts", "OTC", "Supplier A", 75, 8.00, 4, 200),
                ("MED004", "Vitamin C 500mg", "Vitamins", "Supplier C", 95, 3.50, 8, 300),
                ("MED005", "Amoxicillin 500mg", "Prescription", "Supplier B", 25, 6.75, 6, 180),
                ("MED006", "Antacid Tablets", "OTC", "Supplier C", 20, 4.00, 5, 35),
            ]

            for p in demo_products:
                con.execute("""
                INSERT INTO products
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p[0], p[1], p[2], p[3], p[4],
                    p[5], p[6],
                    (date.today() + timedelta(days=p[7])).isoformat()
                ))

            rng = np.random.default_rng(42)
            demand = [8, 3, 4, 5, 2, 2]

            for p, avg in zip(demo_products, demand):
                for days_ago in range(89, -1, -1):
                    d = date.today() - timedelta(days=days_ago)
                    qty = int(rng.poisson(avg))
                    con.execute("""
                    INSERT INTO sales(sale_date, sku, quantity)
                    VALUES (?, ?, ?)
                    """, (d.isoformat(), p[0], qty))


initialize()


@st.cache_data
def read_products():
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query("SELECT * FROM products ORDER BY name", con)
    finally:
        con.close()


@st.cache_data
def read_sales():
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query("SELECT * FROM sales ORDER BY sale_date", con)
    finally:
        con.close()


def analyze(products, sales):
    results = []

    for _, p in products.iterrows():
        history = sales[sales.sku == p.sku].copy()

        if history.empty:
            daily = pd.Series([0.0] * 90)
        else:
            history["sale_date"] = pd.to_datetime(history["sale_date"])
            daily = (
                history.groupby("sale_date")["quantity"]
                .sum()
                .asfreq("D", fill_value=0)
            )

        recent = daily.tail(28)
        previous = daily.tail(56).head(28)

        if len(previous) > 0:
            avg_day = 0.65 * recent.mean() + 0.35 * previous.mean()
        else:
            avg_day = recent.mean()

        avg_day = max(0, float(avg_day))

        lead = max(1, int(p.lead_days))
        variation = float(daily.tail(56).std(ddof=0) or 0)

        safety = int(np.ceil(1.65 * variation * np.sqrt(lead)))
        reorder_point = int(np.ceil(avg_day * lead + safety))
        forecast30 = int(np.ceil(avg_day * 30))
        order_qty = max(0, forecast30 + safety - int(p.stock))

        days_expiry = (
            (date.fromisoformat(p.expiry) - date.today()).days
            if p.expiry else None
        )

        if p.stock <= reorder_point:
            status = "REORDER"
        elif days_expiry is not None and days_expiry <= 60:
            status = "EXPIRY WATCH"
        else:
            status = "HEALTHY"

        results.append({
            "SKU": p.sku,
            "Product": p["name"],
            "Supplier": p.supplier,
            "Stock": int(p.stock),
            "Daily Demand": round(avg_day, 2),
            "Forecast 7 Days": int(np.ceil(avg_day * 7)),
            "Forecast 30 Days": forecast30,
            "Forecast 90 Days": int(np.ceil(avg_day * 90)),
            "Safety Stock": safety,
            "Reorder Point": reorder_point,
            "Suggested Order": order_qty,
            "Unit Cost": float(p.unit_cost),
            "Inventory Value": int(p.stock) * float(p.unit_cost),
            "Expiry": p.expiry,
            "Days to Expiry": days_expiry,
            "Status": status
        })

    return pd.DataFrame(results)


st.markdown("""
<style>
:root { --mercury-red: #c8102e; --mercury-deep: #991b32; --ink: #202633; --muted: #697386; }
.stApp { background: #f6f7fb; }
[data-testid="stHeader"] { background: rgba(246,247,251,.88); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1600px; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaf0; }
[data-testid="stSidebar"] * { color: #202633; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 { color: var(--mercury-red); }
div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e7e9f0; padding: 18px 18px 14px; border-radius: 16px; box-shadow: 0 3px 12px rgba(24,34,54,.035); }
div[data-testid="stMetricLabel"] { color: #697386; font-size: .88rem; }
div[data-testid="stMetricValue"] { color: #202633; font-weight: 750; }
div.stButton > button, div.stFormSubmitButton > button, a[data-testid="stLinkButton"] { background: var(--mercury-red); color: #fff; border: 0; border-radius: 10px; font-weight: 650; min-height: 2.6rem; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { background: var(--mercury-deep); color: #fff; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #e7e9f0; border-radius: 12px; overflow: hidden; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e7e9f0; border-radius: 16px; padding: 10px; }
h1, h2, h3 { letter-spacing: -.025em; color: #202633; }
small, .stCaption { color: #697386; }
</style>
""", unsafe_allow_html=True)

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown(
        "<div style='font-size:12px;font-weight:700;letter-spacing:2px;color:#c8102e'>INVENTORY • ANALYTICS • FORECASTING</div>",
        unsafe_allow_html=True)
    st.markdown("<h1 style='margin:4px 0 0;font-size:2rem'>💊 Mercury Drugs Inventory BI</h1>", unsafe_allow_html=True)
    st.caption("Predictive Supply Chain & Inventory Optimization")
with head_right:
    st.markdown(
        f"<div style='text-align:right;padding-top:8px;color:#697386;font-size:13px'>ACADEMIC PROTOTYPE<br><b style='color:#202633'>{date.today().strftime('%B %d, %Y')}</b></div>",
        unsafe_allow_html=True)
st.caption("Student project only. Not affiliated with or connected to the actual Mercury Drug company.")

products = read_products()
sales = read_sales()
report = analyze(products, sales)

# FIX: Removed "Products" from navigation since it's inside Inventory Forecast
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Inventory Forecast",
    "Record Sale",
    "Receive Delivery",
    "Sales Analysis",
    "Export Reports"
])

st.sidebar.divider()
st.sidebar.caption("Signed in as admin")

if st.sidebar.button("🚪 Log out", use_container_width=True):
    cookies.remove("mercury_auth")
    st.session_state["authenticated"] = False
    st.rerun()

if page == "Dashboard":
    location_col, map_col = st.columns([3, 1])
    with location_col:
        st.markdown("""
        <div style="background:#fff;border:1px solid #e7e9f0;border-radius:14px;padding:17px 20px;">
          <div style="font-size:12px;color:#c8102e;font-weight:750;letter-spacing:1px">STORE LOCATION</div>
          <div style="font-size:15px;font-weight:650;color:#202633;margin-top:6px">QVGM+G73, Allen Avenue, Catbalogan City Proper</div>
          <div style="font-size:12px;color:#697386;margin-top:3px">City of Catbalogan, 6700 Samar</div>
        </div>
        """, unsafe_allow_html=True)
    with map_col:
        st.link_button("📍 Open Google Maps",
                       "https://www.google.com/maps/search/?api=1&query=QVGM%2BG73%2C%20Allen%20Avenue%2C%20Catbalogan%20City%20Proper%2C%206700%20Samar",
                       use_container_width=True)

    st.markdown("### Overview")
    a, b, c, d = st.columns(4)
    reorder_count = int((report.Status == "REORDER").sum())
    expiry_count = int((report["Days to Expiry"].fillna(999) <= 60).sum())
    a.metric("Total products", f"{len(products):,}", help="Number of products registered in the inventory.")
    b.metric("Reorder alerts", f"{reorder_count:,}", delta="Needs review" if reorder_count else "Stock levels OK",
             delta_color="inverse" if reorder_count else "normal")
    c.metric("Inventory value", f"₱{report['Inventory Value'].sum():,.2f}",
             help="Current stock multiplied by unit cost.")
    d.metric("Expiry watch", f"{expiry_count:,}", delta="Review dates" if expiry_count else "No upcoming alerts",
             delta_color="inverse" if expiry_count else "normal")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Stock health")
        stock_counts = report["Status"].value_counts().reindex(["HEALTHY", "REORDER", "EXPIRY WATCH"],
                                                               fill_value=0).reset_index()
        stock_counts.columns = ["Status", "Products"]
        fig_health = px.pie(stock_counts, names="Status", values="Products", hole=.62,
                            color="Status",
                            color_discrete_map={"HEALTHY": "#16a34a", "REORDER": "#c8102e", "EXPIRY WATCH": "#d97706"})
        fig_health.update_layout(margin=dict(t=20, b=10, l=10, r=10), legend_title_text="",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_health, use_container_width=True)
    with right:
        st.markdown("### Demand forecast · 30 days")
        fig = px.bar(report.sort_values("Forecast 30 Days", ascending=False), x="Product", y="Forecast 30 Days",
                     color_discrete_sequence=["#c8102e"])
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), xaxis_title="", yaxis_title="Estimated units",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Priority Action Center")

    actions = report.copy()
    actions["Stock Alert"] = np.where(actions["Stock"] <= actions["Reorder Point"], "REORDER", "OK")
    actions["Expiry Alert"] = np.where(actions["Days to Expiry"].notna() & (actions["Days to Expiry"] <= 60),
                                       "EXPIRY WATCH", "OK")

    actions = actions[(actions["Stock Alert"] != "OK") | (actions["Expiry Alert"] != "OK")].copy()

    if actions.empty:
        st.success("No stock or expiry alerts found.")
    else:
        def get_priority(row):
            if row["Stock Alert"] == "REORDER" and row["Expiry Alert"] == "EXPIRY WATCH":
                return "HIGH - BOTH"
            elif row["Stock Alert"] == "REORDER":
                return "HIGH - REORDER"
            return "REVIEW - EXPIRY"

        actions["Priority"] = actions.apply(get_priority, axis=1)

        st.dataframe(
            actions[[
                "Priority", "SKU", "Product", "Stock", "Reorder Point",
                "Suggested Order", "Expiry", "Days to Expiry",
                "Stock Alert", "Expiry Alert",
            ]],
            use_container_width=True,
            hide_index=True
        )

    with st.expander("📍 Store information and project notes"):
        st.write(
            "This is an academic prototype using demonstration data. Forecasts are estimates from recorded daily unit sales and should be reviewed before making real purchasing decisions.")

elif page == "Inventory Forecast":
    st.subheader("Predictive inventory recommendations")
    st.write("Forecasts are estimates based on historical daily sales. They are not guaranteed predictions.")

    st.dataframe(report, use_container_width=True, hide_index=True)

    st.markdown("### Why is this product recommended?")

    selected_sku = st.selectbox(
        "Select a product",
        report["SKU"].tolist(),
        format_func=lambda sku: f"{sku} — {report.loc[report['SKU'] == sku, 'Product'].iloc[0]}"
    )

    item = report.loc[report["SKU"] == selected_sku].iloc[0]

    st.markdown(f"#### {item['Product']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current stock", int(item["Stock"]))
    col2.metric("Reorder point", int(item["Reorder Point"]))
    col3.metric("Suggested order", int(item["Suggested Order"]))

    st.write(f"Estimated daily demand: {item['Daily Demand']:.2f} units")
    st.write(f"Estimated demand for the next 30 days: {item['Forecast 30 Days']} units")

    if item["Stock"] <= item["Reorder Point"]:
        st.warning(
            f"Replenishment is recommended because current stock "
            f"({int(item['Stock'])}) is at or below the reorder point "
            f"({int(item['Reorder Point'])})."
        )
        st.info(
            f"The calculated suggested order quantity is "
            f"{int(item['Suggested Order'])} units. "
            "Review this estimate before placing an order."
        )
    else:
        st.success("Current stock is above the calculated reorder point.")

    expiry_days = item["Days to Expiry"]
    if pd.notna(expiry_days) and expiry_days <= 60:
        st.warning(f"Expiry alert: this product has {int(expiry_days)} days until its recorded expiry date.")

    st.subheader("Add a product")

    with st.form("new_product"):
        sku = st.text_input("SKU / product code")
        name = st.text_input("Product name")
        category = st.selectbox("Category", ["OTC", "Prescription", "Vitamins", "Medical Supplies", "Other"])
        supplier = st.text_input("Supplier")
        stock = st.number_input("Opening stock", min_value=0, step=1)
        cost = st.number_input("Unit cost (PHP)", min_value=0.0, step=0.5)
        lead = st.number_input("Supplier lead time (days)", min_value=1, value=7)
        expiry = st.date_input("Nearest expiry date", value=date.today() + timedelta(days=180))

        save = st.form_submit_button("Save product")

        if save:
            if not sku.strip() or not name.strip():
                st.error("Enter both SKU and product name.")
            else:
                try:
                    with get_db() as con:
                        con.execute("""
                        INSERT INTO products
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sku.strip(), name.strip(), category,
                            supplier.strip(), int(stock), float(cost),
                            int(lead), expiry.isoformat()
                        ))
                    st.cache_data.clear()
                    st.success("Product added.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("That SKU already exists.")

    st.subheader("Product Master")

    search_query = st.text_input("🔎 Search products", placeholder="Search by SKU, name, supplier, or category...")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        category_options = ["All categories"] + sorted(products["category"].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by category", category_options)

    with filter_col2:
        stock_filter = st.selectbox("Filter by stock status", ["All products", "Needs reorder", "Above reorder point"])

    product_view = products.merge(
        report[["SKU", "Reorder Point", "Status"]],
        left_on="sku",
        right_on="SKU",
        how="left"
    )

    if search_query.strip():
        searchable_columns = ["sku", "name", "category", "supplier"]
        matches = product_view[searchable_columns].astype(str).apply(
            lambda column: column.str.contains(search_query.strip(), case=False, na=False, regex=False)
        ).any(axis=1)
        product_view = product_view[matches]

    if selected_category != "All categories":
        product_view = product_view[product_view["category"] == selected_category]

    if stock_filter == "Needs reorder":
        product_view = product_view[product_view["stock"] <= product_view["Reorder Point"]]
    elif stock_filter == "Above reorder point":
        product_view = product_view[product_view["stock"] > product_view["Reorder Point"]]

    st.caption(f"Products displayed: {len(product_view)}")

    st.dataframe(
        product_view[[
            "sku", "name", "category", "supplier", "stock",
            "unit_cost", "lead_days", "expiry", "Reorder Point", "Status"
        ]],
        use_container_width=True,
        hide_index=True
    )

elif page == "Record Sale":
    st.subheader("Record a sale")

    names = dict(zip(products.sku, products["name"]))

    with st.form("sale_form"):
        sku = st.selectbox("Product", products.sku.tolist(), format_func=lambda x: f"{x} — {names[x]}")
        sale_date = st.date_input("Sale date", value=date.today())
        quantity = st.number_input("Quantity sold", min_value=1, step=1)
        save = st.form_submit_button("Record sale")

        if save:
            with get_db() as con:
                row = con.execute("SELECT stock FROM products WHERE sku=?", (sku,)).fetchone()

                if quantity > row[0]:
                    st.error(f"Insufficient stock. Available: {row[0]}")
                else:
                    con.execute("""
                    INSERT INTO sales(sale_date, sku, quantity)
                    VALUES (?, ?, ?)
                    """, (sale_date.isoformat(), sku, int(quantity)))

                    con.execute(
                        "UPDATE products SET stock=stock-? WHERE sku=?",
                        (int(quantity), sku)
                    )

                    st.cache_data.clear()
                    st.success("Sale recorded and stock updated.")
                    st.rerun()

elif page == "Receive Delivery":
    st.subheader("Record incoming stock")

    names = dict(zip(products.sku, products["name"]))

    with st.form("delivery_form"):
        sku = st.selectbox("Product", products.sku.tolist(), format_func=lambda x: f"{x} — {names[x]}")
        quantity = st.number_input("Quantity received", min_value=1, step=1)
        supplier = st.text_input("Supplier / delivery reference")
        delivery_date = st.date_input("Delivery date", value=date.today())
        save = st.form_submit_button("Receive stock")

        if save:
            with get_db() as con:
                con.execute(
                    "UPDATE products SET stock=stock+? WHERE sku=?",
                    (int(quantity), sku)
                )
                con.execute("""
                INSERT INTO deliveries (delivery_date, sku, quantity, supplier)
                VALUES (?, ?, ?, ?)
                """, (delivery_date.isoformat(), sku, int(quantity), supplier))

            st.cache_data.clear()
            st.success("Delivery recorded and inventory updated.")
            st.rerun()

elif page == "Sales Analysis":
    st.subheader("Sales Performance Analytics")
    st.caption("Analyze recorded unit sales and product demand. Demonstration data is synthetic.")

    if sales.empty:
        st.info("No sales history is available.")
    else:
        sales_analysis = sales.copy()
        sales_analysis["sale_date"] = pd.to_datetime(sales_analysis["sale_date"])

        total_units = int(sales_analysis["quantity"].sum())
        products_sold = int(sales_analysis.loc[sales_analysis["quantity"] > 0, "sku"].nunique())
        sales_records = len(sales_analysis)

        by_product = sales_analysis.groupby("sku")["quantity"].sum().reset_index()
        by_product = by_product.merge(products[["sku", "name"]], on="sku", how="left")
        by_product = by_product.sort_values("quantity", ascending=False)

        best_product = by_product.iloc[0]["name"] if not by_product.empty else "No data"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Units Sold", f"{total_units:,}")
        col2.metric("Products Sold", f"{products_sold:,}")
        col3.metric("Sales Records", f"{sales_records:,}")

        st.markdown("### Sales Trends")
        daily_sales = sales_analysis.groupby("sale_date")["quantity"].sum().reset_index().sort_values("sale_date")

        fig_daily = px.line(
            daily_sales, x="sale_date", y="quantity",
            title="Daily Units Sold", markers=True,
            labels={"sale_date": "Date", "quantity": "Units Sold"}
        )
        fig_daily.update_layout(hovermode="x unified", margin=dict(t=55, b=20, l=20, r=20))
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("### Product Performance")
        top_products = by_product.head(10).sort_values("quantity", ascending=True)

        fig_products = px.bar(
            top_products, x="quantity", y="name", orientation="h",
            title="Top 10 Products by Units Sold",
            labels={"quantity": "Units Sold", "name": "Product"},
            color_discrete_sequence=["#c8102e"]
        )
        fig_products.update_layout(yaxis_title="", xaxis_title="Total Units Sold", margin=dict(t=55, b=20, l=20, r=20))
        st.plotly_chart(fig_products, use_container_width=True)

        st.info(
            f"Highest-selling product in the recorded dataset: {best_product}. This ranking is based on unit quantities, not revenue or profit.")

        with st.expander("View detailed product sales"):
            st.dataframe(
                by_product.rename(columns={"sku": "SKU", "name": "Product", "quantity": "Total Units Sold"}),
                use_container_width=True,
                hide_index=True
            )

elif page == "Export Reports":
    st.subheader("Download reports")

    st.download_button(
        "Download inventory forecast CSV",
        report.to_csv(index=False).encode("utf-8"),
        "inventory_forecast.csv",
        "text/csv"
    )

    st.download_button(
        "Download sales history CSV",
        sales.to_csv(index=False).encode("utf-8"),
        "sales_history.csv",
        "text/csv"
    )

    con = sqlite3.connect(DB)
    try:
        deliveries = pd.read_sql_query("SELECT * FROM deliveries ORDER BY delivery_date DESC", con)
    finally:
        con.close()

    st.download_button(
        "Download delivery history CSV",
        deliveries.to_csv(index=False).encode("utf-8"),
        "delivery_history.csv",
        "text/csv"
    )
