"""
역헤드앤숄더 패턴 스캐너 - Streamlit 웹앱 (모바일 카드형 UI)
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# 출력 디렉토리 경로 설정
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"

# 페이지 설정
st.set_page_config(
    page_title="패턴 스캐너",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"  # 모바일에서 사이드바 숨김
)

# ========== 모바일 친화적 CSS ==========
st.markdown("""
<style>
/* 전체 폰트 크기 증가 */
html, body, [class*="css"] {
    font-size: 16px;
}

/* 카드 스타일 */
.stock-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.stock-card:hover {
    border-color: #e94560;
    box-shadow: 0 6px 20px rgba(233, 69, 96, 0.2);
}

/* 카드 헤더 */
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.stock-name {
    font-size: 1.4rem;
    font-weight: bold;
    color: #ffffff;
}

.stock-price {
    font-size: 1.3rem;
    color: #00d9ff;
    font-weight: bold;
}

/* 상태 배지 */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.95rem;
    font-weight: bold;
    margin-right: 8px;
}

.status-early {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: #000;
}

.status-rising {
    background: linear-gradient(135deg, #0984e3, #74b9ff);
    color: #000;
}

.status-breakout {
    background: linear-gradient(135deg, #fdcb6e, #f39c12);
    color: #000;
}

/* 가격 정보 그리드 */
.price-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 16px 0;
}

.price-box {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}

.price-label {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 4px;
}

.price-value {
    font-size: 1.2rem;
    font-weight: bold;
}

.price-profit {
    color: #00b894;
}

.price-loss {
    color: #e17055;
}

/* 상승 여력 강조 */
.upside-highlight {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 12px 0;
}

.upside-label {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.8);
}

.upside-value {
    font-size: 2rem;
    font-weight: bold;
    color: #fff;
}

/* 매매 가이드 상단 바 */
.guide-bar {
    background: linear-gradient(90deg, #2d3436 0%, #636e72 100%);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.guide-item {
    text-align: center;
}

.guide-label {
    font-size: 0.75rem;
    color: #b2bec3;
}

.guide-value {
    font-size: 1.1rem;
    font-weight: bold;
    color: #fff;
}

/* 버튼 스타일 (터치 친화적) */
.stButton > button {
    min-height: 48px !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
}

/* Expander 스타일 */
.streamlit-expanderHeader {
    font-size: 1.1rem !important;
    min-height: 48px !important;
}

/* 모바일 반응형 */
@media (max-width: 768px) {
    .stock-name {
        font-size: 1.2rem;
    }
    .stock-price {
        font-size: 1.1rem;
    }
    .price-grid {
        grid-template-columns: 1fr 1fr;
    }
    .guide-bar {
        flex-direction: column;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)


# 세션 상태 초기화
if "results" not in st.session_state:
    st.session_state.results = None
if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = None
if "chart_files" not in st.session_state:
    st.session_state.chart_files = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def load_chart_files():
    """차트 파일 목록 캐싱"""
    chart_files = {}
    if CHART_DIR.exists():
        for f in CHART_DIR.iterdir():
            if f.suffix.lower() == ".png":
                name = f.stem
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and len(parts[1]) == 6:
                    ticker = parts[1]
                    chart_files[ticker] = str(f)
    return chart_files


def load_existing_results():
    """기존 결과 파일 로드"""
    result_path = OUTPUT_DIR / "results.csv"
    if result_path.exists():
        df = pd.read_csv(result_path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        st.session_state.chart_files = load_chart_files()
        return df
    return None


def apply_filters(df, min_head_depth, min_symmetry, pattern_states):
    """필터 적용"""
    if df is None or len(df) == 0:
        return df

    filtered = df.copy()
    filtered = filtered[filtered["머리깊이"] >= min_head_depth]
    filtered = filtered[filtered["어깨대칭성"] >= min_symmetry]

    if pattern_states:
        filtered = filtered[filtered["패턴상태"].isin(pattern_states)]

    return filtered.reset_index(drop=True)


def get_chart_image(ticker: str) -> str:
    """차트 이미지 경로 반환"""
    if ticker in st.session_state.chart_files:
        path = st.session_state.chart_files[ticker]
        if os.path.exists(path):
            return path

    if CHART_DIR.exists():
        for f in CHART_DIR.iterdir():
            if f.suffix.lower() == ".png" and f.stem.endswith(f"_{ticker}"):
                return str(f)

    return None


def get_status_emoji(state):
    """상태별 이모지 반환"""
    if state == "초기진입":
        return "🎯"
    elif state == "상승중":
        return "📈"
    elif state == "돌파임박":
        return "⚡"
    return "📍"


def get_status_class(state):
    """상태별 CSS 클래스 반환"""
    if state == "초기진입":
        return "status-early"
    elif state == "상승중":
        return "status-rising"
    elif state == "돌파임박":
        return "status-breakout"
    return "status-rising"


def display_stock_card(row, idx):
    """개별 종목 카드 표시"""
    ticker = str(row["종목코드"]).zfill(6)
    name = row["종목명"]
    current_price = int(row["현재가"])
    state = row["패턴상태"]
    upside = row.get("넥라인상승여력", 0)
    head_rise = row.get("머리대비상승", 0)
    expected_return = row["예상수익률"] / 100

    # B전략 익절/손절 계산
    take_profit = int(current_price * (1 + expected_return * 0.5))
    stop_loss = int(current_price * 0.9)

    emoji = get_status_emoji(state)
    status_class = get_status_class(state)

    # 카드 HTML
    st.markdown(f"""
    <div class="stock-card">
        <div class="card-header">
            <span class="stock-name">{emoji} {idx+1}. {name}</span>
            <span class="stock-price">{current_price:,}원</span>
        </div>
        <div>
            <span class="status-badge {status_class}">{state}</span>
            <span style="color: #888; font-size: 0.9rem;">머리↗ +{head_rise:.0f}%</span>
        </div>
        <div class="upside-highlight">
            <div class="upside-label">넥라인까지 상승여력</div>
            <div class="upside-value">+{upside:.0f}%</div>
        </div>
        <div class="price-grid">
            <div class="price-box">
                <div class="price-label">🎯 익절가</div>
                <div class="price-value price-profit">{take_profit:,}원</div>
            </div>
            <div class="price-box">
                <div class="price-label">🛑 손절가</div>
                <div class="price-value price-loss">{stop_loss:,}원</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 차트 보기 버튼 (Expander)
    chart_path = get_chart_image(ticker)
    with st.expander(f"📊 차트 보기 - {name}", expanded=False):
        if chart_path:
            st.image(chart_path, use_container_width=True)
        else:
            st.warning("차트 이미지가 없습니다.")

        # 상세 정보
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**가격 정보**")
            st.write(f"넥라인: {int(row['넥라인가격']):,}원")
            st.write(f"목표가: {int(row['목표가']):,}원")
        with col2:
            st.markdown("**패턴 정보**")
            st.write(f"신뢰도: {row['신뢰도점수']:.0f}점")
            st.write(f"대칭성: {row['어깨대칭성']:.0f}%")

        st.caption(f"⏰ 최대 보유: 60일 | 종목코드: {ticker}")


def display_card_list(df):
    """카드형 종목 리스트 표시"""
    if df is None or len(df) == 0:
        st.info("표시할 종목이 없습니다.")
        return

    # 상위 10개 카드 표시
    for idx in range(min(len(df), 10)):
        row = df.iloc[idx]
        display_stock_card(row, idx)

    # 10개 초과 시 더보기
    if len(df) > 10:
        with st.expander(f"➕ 나머지 {len(df) - 10}개 종목 더보기"):
            for idx in range(10, len(df)):
                row = df.iloc[idx]
                display_stock_card(row, idx)


# ========== 시작 시 자동 로드 ==========
if not st.session_state.initialized:
    results = load_existing_results()
    if results is not None:
        st.session_state.results = results
        st.session_state.initialized = True


# ========== 사이드바 (필터) ==========
with st.sidebar:
    st.title("⚙️ 필터 설정")

    if st.button("🔄 새로고침", type="primary", use_container_width=True):
        results = load_existing_results()
        if results is not None:
            st.session_state.results = results
            st.success(f"{len(results)}개 로드")
            st.rerun()

    st.divider()

    min_head_depth = st.slider("최소 머리 깊이 (%)", 0.0, 50.0, 10.0, 1.0)
    min_symmetry = st.slider("최소 대칭성 (%)", 80.0, 100.0, 90.0, 1.0)

    pattern_states = st.multiselect(
        "패턴 상태",
        ["초기진입", "상승중", "돌파임박"],
        default=["초기진입", "상승중"]
    )

    # 필터 적용
    if st.session_state.results is not None:
        st.session_state.filtered_results = apply_filters(
            st.session_state.results,
            min_head_depth,
            min_symmetry,
            pattern_states
        )

    st.divider()

    if st.session_state.results is not None:
        total = len(st.session_state.results)
        filtered = len(st.session_state.filtered_results) if st.session_state.filtered_results is not None else 0
        st.metric("탐지 종목", f"{filtered}/{total}개")


# ========== 메인 영역 ==========

# 헤더
st.markdown("## 🎯 상승 직전 종목")

# 매매 가이드 상단 바
st.markdown("""
<div class="guide-bar">
    <div class="guide-item">
        <div class="guide-label">전략</div>
        <div class="guide-value">B (중립)</div>
    </div>
    <div class="guide-item">
        <div class="guide-label">3년 수익률</div>
        <div class="guide-value" style="color: #00b894;">+9.9%</div>
    </div>
    <div class="guide-item">
        <div class="guide-label">승률</div>
        <div class="guide-value">36.1%</div>
    </div>
    <div class="guide-item">
        <div class="guide-label">손익비</div>
        <div class="guide-value">2.8:1</div>
    </div>
    <div class="guide-item">
        <div class="guide-label">최대보유</div>
        <div class="guide-value">60일</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 새로고침 버튼
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 결과 새로고침", use_container_width=True, type="primary"):
        results = load_existing_results()
        if results is not None:
            st.session_state.results = results
            # 필터 적용
            st.session_state.filtered_results = apply_filters(
                results, 10.0, 90.0, ["초기진입", "상승중"]
            )
            st.rerun()

st.markdown("---")

# 카드형 종목 리스트
df = st.session_state.filtered_results

if df is None or len(df) == 0:
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #888;">
        <p style="font-size: 1.3rem;">📌 '결과 새로고침' 버튼을 눌러주세요</p>
        <p style="font-size: 0.9rem;">또는 사이드바에서 필터를 조정하세요</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # 종목 개수 표시
    st.markdown(f"**{len(df)}개 종목** | 🎯초기진입 📈상승중 ⚡돌파임박")

    # 카드 리스트 표시
    display_card_list(df)

# 푸터
st.markdown("---")
st.caption("역헤드앤숄더 패턴 스캐너 v3.0 (모바일 카드 UI) | B전략 +9.9% | 매일 16:30 업데이트")
