# streamlit_valve_player_app.py
# Streamlit app for exploring Valve_Player_Data.csv
# Place this file in the same repository as your CSV or upload the CSV in the app.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Valve Player Explorer",
    page_icon="🎮",
    layout="wide",
)

# --- Custom CSS for personality ---
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #001e3c 100%); color: #e6eef8; }
    .header { font-family: 'Segoe UI', Roboto, sans-serif; }
    .card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; }
    .metric { color: #bfe9ff }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper functions ---
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

@st.cache_data
def categorical_cols(df):
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# --- Data load section ---
with st.sidebar:
    st.title("Valve Player Explorer")
    st.write("Upload a CSV or use the default dataset provided.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"] )
    use_sample = st.checkbox("Use bundled CSV path (/mnt/data/Valve_Player_Data.csv)", value=True)
    if uploaded and use_sample:
        st.info("Using uploaded file (uploaded file takes precedence).")
    st.markdown("---")
    st.write("App controls")
    show_raw = st.checkbox("Show raw table", value=False)
    selected_theme = st.selectbox("Layout theme (affects visuals)", ["neon","dust","clean"], index=0)

# determine data source
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        if use_sample:
            df = load_data('/mnt/data/Valve_Player_Data.csv')
        else:
            st.error("No file selected. Upload a CSV or enable the bundled CSV path in the sidebar.")
            st.stop()
    except Exception as e:
        st.error(f"Could not load /mnt/data/Valve_Player_Data.csv: {e}")
        st.stop()

st.title("🎮 Valve Player Explorer — 개성있는 데이터 놀이터")
st.caption("직관적 필터, 상호작용 차트, PCA/상관관계 분석을 제공합니다.")

# Quick info
n_rows, n_cols = df.shape
with st.container():
    col1, col2, col3 = st.columns([1,1,2])
    col1.metric("Rows", n_rows)
    col2.metric("Columns", n_cols)
    numeric = numeric_cols(df)
    cat = categorical_cols(df)
    col3.write(f"Numeric: {len(numeric)} | Categorical: {len(cat)}")

# Show raw if asked
if show_raw:
    st.subheader("Raw data")
    st.dataframe(df)

# --- Filters ---
st.sidebar.subheader("데이터 필터링")
filters = {}
for c in cat[:6]:
    vals = df[c].dropna().unique().tolist()[:100]
    if len(vals) > 1:
        sel = st.sidebar.multiselect(f"{c}", options=vals, default=vals[:6])
        if sel:
            filters[c] = sel

for c in numeric[:6]:
    min_v, max_v = float(df[c].min()), float(df[c].max())
    r = st.sidebar.slider(f"{c} range", min_v, max_v, (min_v, max_v))
    filters[c] = r

# Apply filters
df_filtered = df.copy()
for k, v in filters.items():
    if k in cat:
        df_filtered = df_filtered[df_filtered[k].isin(v)]
    elif k in numeric:
        df_filtered = df_filtered[(df_filtered[k] >= v[0]) & (df_filtered[k] <= v[1])]

st.markdown("---")

# --- Main visualizations ---
st.subheader("데이터 요약 & 시각화")

# Top KPIs
with st.container():
    a,b,c = st.columns(3)
    a.metric("Filtered rows", len(df_filtered))
    if 'player_id' in df.columns:
        b.metric("Unique players", df['player_id'].nunique())
    else:
        b.metric("Unique values (sample col)", df.iloc[:,0].nunique())
    if numeric:
        c.metric("Avg of first numeric",
                 round(df_filtered[numeric[0]].mean(),2))

# Distribution of a chosen numeric column
if numeric:
    st.markdown("**숫자형 분포 시각화**")
    col = st.selectbox("Choose numeric for distribution", numeric, index=0)
    fig = px.histogram(df_filtered, x=col, nbins=40, marginal='box', title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Scatter with two numeric
if len(numeric) >= 2:
    st.markdown("**상관/산점도**")
    xcol = st.selectbox("X", numeric, index=0, key='xcol')
    ycol = st.selectbox("Y", numeric, index=1, key='ycol')
    color_col = st.selectbox("Color by (optional)", [None]+cat, index=0)
    fig2 = px.scatter(df_filtered, x=xcol, y=ycol, color=color_col, hover_data=df_filtered.columns, title=f"{ycol} vs {xcol}")
    st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
if numeric:
    st.markdown("**상관관계 히트맵 (상위 10 numeric)**")
    top_num = numeric[:10]
    corr = df_filtered[top_num].corr()
    fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=top_num, y=top_num, colorscale='Viridis'))
    fig3.update_layout(height=500, margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig3, use_container_width=True)

# PCA projection
if len(numeric) >= 3:
    st.markdown("**PCA로 2D 투영 (간단한 군집 감지)**")
    n_components = 2
    scaler = StandardScaler()
    X = df_filtered[numeric].dropna()
    try:
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(Xs)
        proj_df = pd.DataFrame(proj, columns=['PC1','PC2'])
        if cat:
            color_choice = cat[0]
            proj_df[color_choice] = df_filtered[color_choice].dropna().reset_index(drop=True)
        fig4 = px.scatter(proj_df, x='PC1', y='PC2', color=color_choice if cat else None, title='PCA projection')
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"PCA failed: {e}")

# Show top players or top rows
st.markdown("---")
st.subheader("데이터 뷰 & 다운로드")
if 'score' in df.columns:
    st.markdown("Top 10 by 'score'")
    st.dataframe(df.sort_values('score', ascending=False).head(10))
else:
    st.dataframe(df.head(10))

# Allow user to download filtered data
@st.cache_data
def convert_df_to_csv(d):
    return d.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Filtered CSV Download", data=csv, file_name='filtered_valve_player_data.csv', mime='text/csv')

# Footer / credits
st.markdown("---")
with st.container():
    col1, col2 = st.columns([3,1])
    col1.markdown("Made with ❤️ — customize colors, fonts, or add game-specific tabs. Click the GitHub icon in your repo to edit this file.")
    col2.markdown("Version: 1.0")

# Small personalization controls
if st.button('Apply playful theme'):
    st.balloons()
    st.success('Theme applied — enjoy!')

# End of file
