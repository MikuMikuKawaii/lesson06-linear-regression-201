import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균기온 분석 및 예측 앱")
st.markdown("서울 기온 데이터를 바탕으로 연도별 평균기온의 추세를 분석하고, 회귀 직선을 이용해 미래의 기온을 예측합니다.")

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding='utf-8')
    return df

# 데이터 불러오기 예외 처리
try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 데이터 전처리
df['날짜'] = pd.to_datetime(df['날짜'])
df['연도'] = df['날짜'].dt.year

# 1. 2025년까지만 필터링
df_filtered = df[df['연도'] <= 2025].copy()

# 결측치 제거
df_filtered = df_filtered.dropna(subset=['평균기온'])

# 2. 관측일이 300일 이상인 해만 필터링
yearly_counts = df_filtered.groupby('연도').size()
valid_years = yearly_counts[yearly_counts >= 300].index
df_filtered = df_filtered[df_filtered['연도'].isin(valid_years)]

# 연도별 평균기온 계산
yearly_avg = df_filtered.groupby('연도')['평균기온'].mean().reset_index()

# 회귀 분석 모델 학습
X = yearly_avg[['연도']]
y = yearly_avg['평균기온']

model = LinearRegression()
model.fit(X, y)
yearly_avg['회귀예측'] = model.predict(X)

# 상관계수 계산
correlation = yearly_avg['연도'].corr(yearly_avg['평균기온'])

# 메타데이터 정보 추출
num_years = len(yearly_avg)
start_year = int(yearly_avg['연도'].min())
end_year = int(yearly_avg['연도'].max())

# 화면 상단 분석 정보 표시
st.subheader("📊 데이터 분석 요약")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("회귀 직선에 사용된 해의 개수", f"{num_years}개")
with col2:
    st.metric("시작 연도", f"{start_year}년")
with col3:
    st.metric("끝 연도", f"{end_year}년")
with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.markdown("---")

# Plotly 그래프 생성
st.subheader("📈 서울 연평균기온 산점도 및 회귀 직선")
fig = px.scatter(
    yearly_avg, 
    x='연도', 
    y='평균기온', 
    labels={'연도': '연도', '평균기온': '연평균기온 (°C)'},
    title="연도별 연평균기온 변화 추이"
)
fig.update_traces(marker=dict(size=8, color='royalblue'))

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=yearly_avg['연도'], 
        y=yearly_avg['회귀예측'], 
        mode='lines', 
        name='회귀 직선',
        line=dict(color='crimson', width=3)
    )
)
fig.update_layout(template='plotly_white', hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 슬라이더를 통한 기온 예측
st.subheader("🔮 특정 연도 예상 기온 확인")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026)

pred_temp = model.predict([[selected_year]])[0]

# 선택한 해의 예상 기온 강조 표시
st.markdown(f"""
<div style="padding: 25px; border-radius: 12px; background-color: #f0f2f6; text-align: center; border: 1px solid #d6d9df;">
    <h3 style="margin-bottom: 10px; color: #31333F;">📌 {selected_year}년 서울 예상 평균기온</h3>
    <h1 style="color: #ff4b4b; font-size: 3.5rem; margin-top: 0;">{pred_temp:.2f} °C</h1>
</div>
""", unsafe_allow_html=True)
