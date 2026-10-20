import sqlite3
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

if not cookies.ready():
    st.stop()

saved_token = cookies.get("mercury_auth")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = (
        saved_token == "authenticated"
    )
        
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


def connect():
    return sqlite3.connect(DB)


def initialize():
    with connect() as con:
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

        count = con.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]

        if count == 0:
            demo_products = [
                ("MED001", "Paracetamol 500mg", "OTC",
                 "Supplier A", 180, 1.50, 5, 420),
                ("MED002", "Cetirizine 10mg", "OTC",
                 "Supplier B", 42, 2.25, 7, 45),
                ("MED003", "Oral Rehydration Salts", "OTC",
                 "Supplier A", 75, 8.00, 4, 200),
                ("MED004", "Vitamin C 500mg", "Vitamins",
                 "Supplier C", 95, 3.50, 8, 300),
                ("MED005", "Amoxicillin 500mg", "Prescription",
                 "Supplier B", 25, 6.75, 6, 180),
                ("MED006", "Antacid Tablets", "OTC",
                 "Supplier C", 20, 4.00, 5, 35),
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

            # SYNTHETIC demo sales only
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


def read_products():
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM products ORDER BY name", con
        )


def read_sales():
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM sales ORDER BY sale_date", con
        )


def analyze(products, sales):
    results = []

    for _, p in products.iterrows():
        history = sales[sales.sku == p.sku].copy()

        if history.empty:
            daily = pd.Series([0.0] * 90)
        else:
            history["sale_date"] = pd.to_datetime(
                history["sale_date"]
            )
            daily = (
                history.groupby("sale_date")["quantity"]
                .sum()
                .asfreq("D", fill_value=0)
            )

        recent = daily.tail(28)
        previous = daily.tail(56).head(28)

        if len(previous) > 0:
            avg_day = (
                    0.65 * recent.mean()
                    + 0.35 * previous.mean()
            )
        else:
            avg_day = recent.mean()

        avg_day = max(0, float(avg_day))

        lead = max(1, int(p.lead_days))
        variation = float(daily.tail(56).std(ddof=0) or 0)

        safety = int(np.ceil(
            1.65 * variation * np.sqrt(lead)
        ))

        reorder_point = int(np.ceil(
            avg_day * lead + safety
        ))

        forecast30 = int(np.ceil(avg_day * 30))
        order_qty = max(
            0, forecast30 + safety - int(p.stock)
        )

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


st.title("📦 Predictive Supply Chain & Inventory BI")
st.caption(
    "Proposed Mercury Drugs project | Allen Avenue, "
    "Catbalogan City Proper"
)

st.markdown("""
<div style="
    background: linear-gradient(115deg, #b5122b, #e51b35);
    padding: 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 18px;
">
    <div style="font-size:13px;letter-spacing:2px;">
        INVENTORY • ANALYTICS • FORECASTING
    </div>
    <h1 style="color:white;margin:10px 0;">
        💊 Mercury Drugs Inventory BI
    </h1>
    <p style="margin:0;">
        Predictive Supply Chain & Inventory Optimization
    </p>
    <p style="font-size:12px;margin-top:12px;">
        Academic Prototype • Synthetic Demonstration Data
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Student project only. Not affiliated with or connected "
    "to the actual Mercury Drug company."
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #fff7f8;
    color: #111111;
}
/* This ensures all text, labels, and radio button options inside the sidebar are dark and visible */
[data-testid="stSidebar"] * {
    color: #111111 !important;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #f0d5da;
    padding: 14px;
    border-radius: 10px;
}
div.stButton > button {
    background-color: #c8102e;
    color: white;
    border-radius: 8px;
    border: none;
}
div.stButton > button:hover {
    background-color: #a50d24;
    color: white;
}
</style>
""", unsafe_allow_html=True)

products = read_products()
sales = read_sales()
report = analyze(products, sales)

page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Inventory Forecast",
    "Products",
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
    st.markdown("""
    <div style="
        background:white;
        border:1px solid #eadde0;
        border-left:6px solid #c8102e;
        padding:18px;
        border-radius:10px;
        margin-bottom:20px;
    ">
        <div style="color:#b5122b;font-size:18px;
                    font-weight:bold;">
            📍 Store Location
        </div>
        <div style="color:#333;margin-top:10px;">
            QVGM+G73, Allen Avenue, Catbalogan City Proper,
            City of Catbalogan, 6700 Samar
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.link_button(
        "📍 Open in Google Maps",
        "https://www.google.com/maps/search/?api=1&query=QVGM%2BG73%2C%20Allen%20Avenue%2C%20Catbalogan%20City%20Proper%2C%20City%20of%20Catbalogan%2C%206700%20Samar"
    )

    st.write("")
    a, b, c, d = st.columns(4)

    a, b, c, d = st.columns(4)

    a.metric("Products", len(products))
    b.metric(
        "Reorder alerts",
        int((report.Status == "REORDER").sum())
    )
    c.metric(
        "Inventory value",
        f"₱{report['Inventory Value'].sum():,.2f}"
    )
    d.metric(
        "Expiry alerts",
        int((report["Days to Expiry"].fillna(999) <= 60).sum())
    )

    st.subheader("Current stock vs reorder point")
    chart = report.set_index("Product")[
        ["Stock", "Reorder Point"]
    ]
    st.bar_chart(chart)

    st.subheader("Inventory action list")
    st.dataframe(
        report[report.Status != "HEALTHY"],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Forecast demand: next 30 days")
    fig = px.bar(
        report, x="Product", y="Forecast 30 Days",
        title="Estimated units required"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Inventory Forecast":
    st.subheader("Predictive inventory recommendations")

    st.write(
        "Forecasts are estimates based on historical daily sales. "
        "They are not guaranteed predictions."
    )

    st.dataframe(report, use_container_width=True, hide_index=True)

elif page == "Products":
    st.subheader("Add a product")

    with st.form("new_product"):
        sku = st.text_input("SKU / product code")
        name = st.text_input("Product name")
        category = st.selectbox(
            "Category",
            ["OTC", "Prescription", "Vitamins",
             "Medical Supplies", "Other"]
        )
        supplier = st.text_input("Supplier")
        stock = st.number_input(
            "Opening stock", min_value=0, step=1
        )
        cost = st.number_input(
            "Unit cost (PHP)", min_value=0.0, step=0.5
        )
        lead = st.number_input(
            "Supplier lead time (days)",
            min_value=1, value=7
        )
        expiry = st.date_input(
            "Nearest expiry date",
            value=date.today() + timedelta(days=180)
        )

        save = st.form_submit_button("Save product")

        if save:
            if not sku.strip() or not name.strip():
                st.error("Enter both SKU and product name.")
            else:
                try:
                    with connect() as con:
                        con.execute("""
                        INSERT INTO products
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sku.strip(), name.strip(), category,
                            supplier.strip(), int(stock), float(cost),
                            int(lead), expiry.isoformat()
                        ))
                    st.success("Product added.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("That SKU already exists.")

    st.subheader("Product master")
    st.dataframe(products, use_container_width=True, hide_index=True)

elif page == "Record Sale":
    st.subheader("Record a sale")

    names = dict(zip(products.sku, products["name"]))

    with st.form("sale_form"):
        sku = st.selectbox(
            "Product",
            products.sku.tolist(),
            format_func=lambda x: f"{x} — {names[x]}"
        )
        sale_date = st.date_input("Sale date", value=date.today())
        quantity = st.number_input(
            "Quantity sold", min_value=1, step=1
        )
        save = st.form_submit_button("Record sale")

        if save:
            with connect() as con:
                row = con.execute(
                    "SELECT stock FROM products WHERE sku=?",
                    (sku,)
                ).fetchone()

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

                    st.success("Sale recorded and stock updated.")
                    st.rerun()

elif page == "Receive Delivery":
    st.subheader("Record incoming stock")

    names = dict(zip(products.sku, products["name"]))

    with st.form("delivery_form"):
        sku = st.selectbox(
            "Product", products.sku.tolist(),
            format_func=lambda x: f"{x} — {names[x]}"
        )
        quantity = st.number_input(
            "Quantity received", min_value=1, step=1
        )
        supplier = st.text_input("Supplier / delivery reference")
        delivery_date = st.date_input(
            "Delivery date", value=date.today()
        )
        save = st.form_submit_button("Receive stock")

        if save:
            with connect() as con:
                con.execute(
                    "UPDATE products SET stock=stock+? WHERE sku=?",
                    (int(quantity), sku)
                )
                con.execute("""
                INSERT INTO deliveries
                (delivery_date, sku, quantity, supplier)
                VALUES (?, ?, ?, ?)
                """, (
                    delivery_date.isoformat(), sku,
                    int(quantity), supplier
                ))

            st.success("Delivery recorded and inventory updated.")
            st.rerun()

elif page == "Sales Analysis":
    st.subheader("Sales performance")

    if sales.empty:
        st.info("No sales history.")
    else:
        sales["sale_date"] = pd.to_datetime(sales["sale_date"])
        daily_sales = (
            sales.groupby("sale_date")["quantity"]
            .sum().reset_index()
        )

        fig = px.line(
            daily_sales, x="sale_date", y="quantity",
            title="Daily units sold"
        )
        st.plotly_chart(fig, use_container_width=True)

        by_product = (
            sales.groupby("sku")["quantity"]
            .sum().reset_index()
        )
        by_product = by_product.merge(
            products[["sku", "name"]], on="sku", how="left"
        )

        st.plotly_chart(
            px.bar(
                by_product, x="name", y="quantity",
                title="Total units sold by product"
            ),
            use_container_width=True
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

    with connect() as con:
        deliveries = pd.read_sql_query(
            "SELECT * FROM deliveries ORDER BY delivery_date DESC",
            con
        )

    st.download_button(
        "Download delivery history CSV",
        deliveries.to_csv(index=False).encode("utf-8"),
        "delivery_history.csv",
        "text/csv"
    )
