import sys
import traceback

print("[DIAG 01] app.py entered", flush=True)

import streamlit as st
print("[DIAG 02] streamlit imported", flush=True)

import pandas as pd
print("[DIAG 03] pandas imported", flush=True)

import numpy as np
print("[DIAG 04] numpy imported", flush=True)

import plotly
print("[DIAG 05] plotly imported", flush=True)

import yfinance as yf
print("[DIAG 06] yfinance imported", flush=True)

import openpyxl
print("[DIAG 07] openpyxl imported", flush=True)

engine_error = None
try:
    print("[DIAG 08] importing quant_engine...", flush=True)
    import quant_engine as qe
    print("[DIAG 09] quant_engine imported", flush=True)
except Exception as e:
    engine_error = f"{type(e).__name__}: {e}"
    print("[DIAG ERROR] quant_engine:", engine_error, flush=True)
    traceback.print_exc()

st.set_page_config(page_title="AI Stock Hunter Diagnostic", page_icon="🧪", layout="centered")
print("[DIAG 10] set_page_config completed", flush=True)

st.title("🧪 AI Stock Hunter — Startup Diagnostic")
st.success("Streamlit reached app.py successfully.")
st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)
st.write("Pandas:", pd.__version__)
st.write("NumPy:", np.__version__)
st.write("Plotly:", plotly.__version__)
st.write("yfinance:", yf.__version__)

if engine_error:
    st.error("quant_engine import failed: " + engine_error)
else:
    st.success("quant_engine.py imported successfully.")
    st.write("Engine version:", getattr(qe, "ENGINE_VERSION", "unknown"))
    st.write("Engine build:", getattr(qe, "ENGINE_BUILD_ID", "unknown"))

st.info("If you can see this page, the Streamlit/Python environment is healthy and the startup problem is inside the full app.py rather than the deployment platform.")
print("[DIAG 11] page rendered", flush=True)
