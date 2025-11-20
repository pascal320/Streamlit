import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Explorer Dashboard")

# Let users upload an Excel file
excel_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if excel_file:
    # Read the uploaded file into a DataFrame
    data = pd.read_excel(excel_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())  # display first few rows

    st.subheader("Basic Statistics")
    st.write(data.describe())  # summary statistics

    # Simple filter setup
    st.subheader("Apply Filter")
    cols = data.columns.tolist()
    filter_col = st.selectbox("Select a column to filter", cols)
    options = data[filter_col].dropna().unique()
    chosen_value = st.selectbox("Select value", options)

    # Create filtered view
    subset = data[data[filter_col] == chosen_value]
    st.write(subset)

    # Keep plots active between interactions
    if "plot_one" not in st.session_state:
        st.session_state.plot_one = None
    if "plot_two" not in st.session_state:
        st.session_state.plot_two = None

    # Plot 1 section
    st.subheader("Plot 1")
    x1 = st.selectbox("x-axis", cols, key="x1")
    y1 = st.selectbox("y-axis", cols, key="y1")

    if st.button("Show Plot 1", key="btn1"):
        st.session_state.plot_one = subset[[x1, y1]]

    if st.session_state.plot_one is not None:
        st.line_chart(st.session_state.plot_one.set_index(x1)[y1])

    # Plot 2 section
    st.subheader("Plot 2")
    x2 = st.selectbox("x-axis", cols, key="x2")
    y2 = st.selectbox("y-axis", cols, key="y2")

    if st.button("Show Plot 2", key="btn2"):
        st.session_state.plot_two = subset[[x2, y2]]

    if st.session_state.plot_two is not None:
        st.line_chart(st.session_state.plot_two.set_index(x2)[y2])

    # Vendor-specific data lookup
    st.subheader("Vendor Lookup")
    vendor_ids = data["ID"].unique() if "ID" in data.columns else []
    vendor_choice = st.selectbox("Choose vendor ID", vendor_ids)
    vendor_rows = subset[subset["ID"] == vendor_choice]
    st.dataframe(vendor_rows)

else:
    st.info("Upload an Excel file to begin exploring your data.")