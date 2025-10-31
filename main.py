# streamlit_valve_player_app_v2.py — Cyberpunk Neon Enhanced Edition
# 스트림릿 + Plotly 기반 Valve Player Dashboard
# Cyberpunk Neon 테마 + 업로드 + PCA + 상관계수 + 다운로드 기능 포함

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="Valve Player Dashboard",  # 브라우저 탭 제목
    page_icon="🎮",                      # 탭 아이콘
    layout="wide",                        # 넓은 레이아웃
)

# --- Cyberpunk Neon CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
    color: #e0eaff; /* 글자색 */
    background: radial-gradient(circle at top left, #0f0f1e 0%, #000010 100%); /* 배경 */
}
.main-title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #00e5ff, #ff00ff); /* 글자 그라데이션 */
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
    transform: scale(1.02); /* 카드 확대 효과 */
}
.footer {
    text-align: center;
    color: #888;
    margin-top: 3em;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --- 사이드바: 파일 업로드 & 컬럼 선택 ---
uploaded_file = st.sidebar.file_uploader("📁 Valve_Player_Data.csv 업로드", type=["csv"])

@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('Valve_Player_Data.csv')  # 로컬 CSV 로드
    except FileNotFoundError:
        return pd.DataFrame()  # 없으면 빈 DataFrame 반환

if uploaded_file:
    df = pd.read_csv(uploaded_file)  
    st.sidebar.success("✅ 업로드한 CSV 사용")
else:
    df = load_default_data()
    if df.empty:
        st.error("⚠️ 데이터가 없습니다. CSV 파일을 업로드해주세요.")
        st.stop()
    st.sidebar.info("📂 로컬 'Valve_Player_Data.csv' 사용")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 숫자 컬럼
cat_cols = df.select_dtypes(include=['object']).columns.tolist()       # 범주형 컬럼

# --- 사이드바 필터 ---
st.sidebar.markdown("### ⚡ 필터 & 옵션")
selected_numeric = st.sidebar.multiselect("분석할 숫자 컬럼 선택", numeric_cols, default=numeric_cols[:5])
selected_cat = st.sidebar.selectbox("PCA 색상 기준 범주형 컬럼 선택", [None] + cat_cols)

# --- 헤더 ---
st.markdown('<div class="main-title">🎮 Valve Player Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Cyberpunk Neon Enhanced Edition — 업로드하거나 탐색 ⚡</div>', unsafe_allow_html=True)

# --- 탭 ---
tabs = st.tabs(["🏠 개요", "📊 시각화", "🔍 분석", "📥 다운로드"])

# --- 개요 탭 ---
with tabs[0]:
    st.markdown("### ⚙️ 빠른 요약")
    n_rows, n_cols = df.shape
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>행 개수</h3><h2>{n_rows}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>열 개수</h3><h2>{n_cols}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>숫자형 컬럼</h3><h2>{len(numeric_cols)}</h2></div>', unsafe_allow_html=True)

    if 'score' in df.columns:
        st.markdown("---")
        st.markdown("#### 🏆 상위 10 플레이어 (score 기준)")
        st.dataframe(df.sort_values('score', ascending=False).head(10))

# --- 시각화 탭 ---
with tabs[1]:
    st.markdown("### 🌈 시각적 탐색")
    if selected_numeric:
        col = st.selectbox("히스토그램 대상 컬럼 선택", selected_numeric)
        fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=['#00FFFF'])
        fig.update_layout(template='plotly_dark', title=f"{col} 분포")
        st.plotly_chart(fig, use_container_width=True)

    if len(selected_numeric) > 1:
        xcol = st.selectbox("X축", selected_numeric, index=0)
        ycol = st.selectbox("Y축", selected_numeric, index=1)
        color = st.selectbox("색상 기준", [None] + cat_cols, index=0)
        fig2 = px.scatter(df, x=xcol, y=ycol, color=color,
                          color_discrete_sequence=['#00FFFF','#FF00FF','#00FFAA'])
        fig2.update_traces(marker=dict(size=12, line=dict(width=1, color='#ffffff')))
        fig2.update_layout(template='plotly_dark', title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig2, use_container_width=True)

# --- 분석 탭 ---
with tabs[2]:
    st.markdown("### 🔬 상관계수 & PCA")
    if len(selected_numeric) >= 3:
        corr = df[selected_numeric].fillna(df[selected_numeric].mean()).corr()  # NaN 평균 채움
        fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Magma'))
        fig3.update_layout(title='상관계수 히트맵', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 🌀 PCA 2차원 투영")
        X = df[selected_numeric].fillna(df[selected_numeric].mean())
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(Xs)
        proj_df = pd.DataFrame(proj, columns=['PC1','PC2'])
        if selected_cat and selected_cat in df.columns:
            proj_df[selected_cat] = df[selected_cat].dropna().reset_index(drop=True)
        fig4 = px.scatter(proj_df, x='PC1', y='PC2', color=selected_cat,
                          color_discrete_sequence=['#00FFFF','#FF00FF','#00FFAA'])
        fig4.update_traces(marker=dict(size=12, line=dict(width=1, color='#ffffff')))
        fig4.update_layout(template='plotly_dark', title='PCA 투영')
        st.plotly_chart(fig4, use_container_width=True)

# --- 다운로드 탭 ---
with tabs[3]:
    st.markdown("### 📥 데이터 다운로드")
    @st.cache_data
    def convert_df_to_csv(d): 
        return d.to_csv(index=False).encode('utf-8')
    csv = convert_df_to_csv(df)
    st.download_button("💾 CSV 다운로드", data=csv, file_name='valve_player_data.csv', mime='text/csv')

# --- 푸터 ---
st.markdown('<div class="footer">Made with 💜 Streamlit + Plotly | Cyberpunk Neon Enhanced | 2025</div>', unsafe_allow_html=True)
