import os

import pandas as pd
import requests
import streamlit as st

# Backend address. Inside Docker Compose the backend is reachable by its service name.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip("/")
TIMEOUT = 30

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")
st.title("SuperKart Sales Prediction")
st.write("Enter the product and store details below to predict the product-store sales total.")


def call_api(method, path, **kwargs):
    """Call the backend and return (json, error_message, response_headers)."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as exc:
        return None, f"Unable to reach the prediction API at {BACKEND_URL}: {exc}", {}
    try:
        body = resp.json()
    except ValueError:
        body = None
    if resp.status_code == 200 and body is not None:
        return body, None, resp.headers
    detail = body.get("error") if isinstance(body, dict) else resp.text[:200]
    return None, f"Prediction API returned {resp.status_code}: {detail}", {}


with st.sidebar:
    st.subheader("Service status")
    health, err, _ = call_api("GET", "/health")
    if err:
        st.error("Backend unavailable")
        st.caption(err)
    else:
        st.success("Backend online")

# ------------------------------------------------------------ single prediction
st.subheader("Single Prediction")
col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.01, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input(
        "Product Allocated Area (ratio 0-1)", min_value=0.001, max_value=1.0, value=0.027, format="%.3f"
    )
    Product_MRP = st.number_input("Product MRP", min_value=0.01, value=117.08)
    Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    Store_Type = st.selectbox(
        "Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1
    )
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16, step=1)

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": int(Store_Age_Years),
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict", type="primary"):
    with st.spinner("Calling prediction API..."):
        result, error, _ = call_api("POST", "/v1/predict", json=product_data)
    if error:
        st.error(error)
    else:
        st.success(f"Predicted Product Store Sales Total: {result['Sales']:,.2f}")

# ------------------------------------------------------------ batch prediction
st.subheader("Batch Prediction")
st.caption("Upload a CSV containing the columns: " + ", ".join(product_data.keys()))
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.button("Predict for Batch", type="primary"):
    with st.spinner("Calling prediction API..."):
        results, error, headers = call_api(
            "POST",
            "/v1/predictbatch",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
        )
    if error:
        st.error(error)
    else:
        out = pd.DataFrame(
            {"Row": [int(k) for k in results.keys()], "Predicted_Sales": list(results.values())}
        ).sort_values("Row")
        skipped = headers.get("X-Skipped-Rows", "")
        scored = int(out["Predicted_Sales"].notna().sum())
        st.success(f"Predictions completed for {scored} of {len(out)} rows.")
        if skipped:
            st.warning(
                f"Rows skipped (value not seen in training, e.g. an unsupported Store_Type): {skipped}"
            )
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="superkart_predictions.csv",
            mime="text/csv",
        )
