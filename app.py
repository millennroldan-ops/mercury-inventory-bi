
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Predictive Supply Chain & Inventory BI")
st.subheader("Mercury Drugs — Academic Prototype")

st.success("Your Python environment is working!")

data = pd.DataFrame({
    "Product": ["Paracetamol", "Cetirizine", "Vitamin C"],
    "Stock": [180, 42, 95],
    "Reorder Point": [65, 50, 40]
})

st.dataframe(data, use_container_width=True)

fig = px.bar(
    data,
    x="Product",
    y=["Stock", "Reorder Point"],
    barmode="group",
    title="Inventory Overview"
)

st.plotly_chart(fig, use_container_width=True)
