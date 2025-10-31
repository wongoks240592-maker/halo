# streamlit_valve_player_app.py — Cyberpunk Neon Edition (with Upload Support)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Valve Player Dashboard",
    page_icon="🎮",
    layout="wide",
)

# --- Cyberpunk Neon Theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        color: #e0eaff;
        background: radial-gradient(circle at top left, #0f0f1e 0%, #000010 100%);
    }

    .main-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00e5ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5em;
    }

    .subtext {
        text-align: center;
        color: #aaa;
        margin-bottom: 2em;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 15px;
        padding: 1.5em;
        text-align: center;
        transition: 0.3s;
    }

    .metric-card:hover {
        background: rgba(0,255,255,0.1);
        transform: scale(1.02);
    }

    .footer {
        text-align: center;
        color: #888;
        margin-top: 3em;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- File Upload or Default Load ---
uploaded_file = st.sidebar.file_uploader("📁 Upload your Valve_Player_Data.csv", type=["csv"])

@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('Valve_Player_Data.csv')
    except FileNotFoundError:
        return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Using uploaded CSV file.")
else:
    df = load_default_data()
    if df is None:
        st.error("⚠️ No data found. Please upload a CSV file in the sidebar.")
        st.stop()
    else:
        st.sidebar.info("📂 Using local 'Valve_Player_Data.csv' file.")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# --- Header ---
st.markdown('<div class="main-title">🎮 Valve Player Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Cyberpunk Neon Edition — Upload your data or explore defaults ⚡</div>', unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs(["🏠 Overview", "📊 Visuals", "🔍 Analysis", "📥 Download"])

# --- Overview Tab ---
with tabs[0]:
    st.markdown("### ⚙️ Quick Summary")
    n_rows, n_cols = df.shape
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Rows</h3><h2>{n_rows}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Columns</h3><h2>{n_cols}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Numeric Columns</h3><h2>{len(numeric_cols)}</h2></div>', unsafe_allow_html=True)

    if 'score' in df.columns:
        st.markdown("---")
        st.markdown("#### 🏆 Top 10 Players by Score")
        st.dataframe(df.sort_values('score', ascending=False).head(10))

# --- Visuals Tab ---
with tabs[1]:
    st.markdown("### 🌈 Visual Exploration")

    if numeric_cols:
        col = st.selectbox("Choose a numeric column", numeric_cols)
        fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=['#00FFFF'])
        fig.update_layout(template='plotly_dark', title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        xcol = st.selectbox("X-axis", numeric_cols, index=0)
        ycol = st.selectbox("Y-axis", numeric_cols, index=1)
        color = st.selectbox("Color by", [None] + cat_cols, index=0)
        fig2 = px.scatter(df, x=xcol, y=ycol, color=color, color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig2.update_layout(template='plotly_dark', title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig2, use_container_width=True)

# --- Analysis Tab ---
with tabs[2]:
    st.markdown("### 🔬 Correlation & PCA")

    if len(numeric_cols) >= 3:
        corr = df[numeric_cols[:10]].corr()
        fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Magma'))
        fig3.update_layout(title='Correlation Heatmap', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 🌀 PCA Projection")
        X = df[numeric_cols].dropna()
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(Xs)
        proj_df = pd.DataFrame(proj, columns=['PC1','PC2'])
        if cat_cols:
            proj_df[cat_cols[0]] = df[cat_cols[0]].dropna().reset_index(drop=True)
        fig4 = px.scatter(proj_df, x='PC1', y='PC2', color=cat_cols[0] if cat_cols else None,
                          color_discrete_sequence=['#00FFFF', '#FF00FF', '#00FFAA'])
        fig4.update_layout(template='plotly_dark', title='PCA Projection')
        st.plotly_chart(fig4, use_container_width=True)

# --- Download Tab ---
with tabs[3]:
    st.markdown("### 📥 Download Filtered Data")

    @st.cache_data
    def convert_df_to_csv(d):
        return d.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df)
    st.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name='valve_player_data.csv',
        mime='text/csv'
    )

# --- Footer ---
st.markdown('<div class="footer">Made with 💜 Streamlit + Plotly | Cyberpunk Neon Theme | Upload Ready | 2025</div>', unsafe_allow_html=True)
