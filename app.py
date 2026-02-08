"""
역헤드앤숄더 패턴 스캐너 - Streamlit 웹앱 (결과 뷰어 전용)
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# 출력 디렉토리 경로 설정 (절대 경로 사용)
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"

# 페이지 설정
st.set_page_config(
    page_title="역헤드앤숄더 패턴 스캐너",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = None
if "chart_files" not in st.session_state:
    st.session_state.chart_files = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def load_chart_files():
    """차트 파일 목록 캐싱 (종목코드 기반)"""
    chart_files = {}
    if CHART_DIR.exists():
        # iterdir 사용 (glob보다 안정적)
        for f in CHART_DIR.iterdir():
            if f.suffix.lower() == ".png":
                # 파일명에서 종목코드 추출 (마지막 _XXXXXX.png)
                name = f.stem  # 확장자 제외
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and len(parts[1]) == 6:
                    ticker = parts[1]
                    chart_files[ticker] = str(f)  # 문자열로 저장
    return chart_files


def load_existing_results():
    """기존 결과 파일 로드"""
    result_path = OUTPUT_DIR / "results.csv"
    if result_path.exists():
        df = pd.read_csv(result_path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        # 차트 파일 목록도 함께 로드
        st.session_state.chart_files = load_chart_files()
        return df
    return None


def apply_filters(df, min_head_depth, min_symmetry, pattern_states):
    """필터 적용"""
    if df is None or len(df) == 0:
        return df

    filtered = df.copy()

    # 머리 깊이 필터
    filtered = filtered[filtered["머리깊이"] >= min_head_depth]

    # 대칭성 필터
    filtered = filtered[filtered["어깨대칭성"] >= min_symmetry]

    # 패턴 상태 필터
    if pattern_states:
        filtered = filtered[filtered["패턴상태"].isin(pattern_states)]

    return filtered.reset_index(drop=True)


def get_chart_image(ticker: str) -> str:
    """차트 이미지 경로 반환 (종목코드 기반)"""
    # 캐시된 차트 파일에서 찾기
    if ticker in st.session_state.chart_files:
        path = st.session_state.chart_files[ticker]
        if os.path.exists(path):
            return path

    # 직접 검색 (fallback)
    if CHART_DIR.exists():
        for f in CHART_DIR.iterdir():
            if f.suffix.lower() == ".png" and f.stem.endswith(f"_{ticker}"):
                return str(f)

    return None


def display_table_row(df, idx):
    """테이블 행 1개 표시"""
    row = df.iloc[idx]
    cols = st.columns([0.4, 1.5, 1, 0.8, 0.9, 0.9, 0.8])

    # 번호 버튼 (클릭 가능)
    with cols[0]:
        btn_type = "primary" if idx == st.session_state.selected_idx else "secondary"
        if st.button(f"{idx+1}", key=f"row_btn_{idx}", type=btn_type):
            st.session_state.selected_idx = idx
            st.rerun()

    # 데이터 표시
    cols[1].write(row["종목명"])
    cols[2].write(f"{int(row['현재가']):,}원")

    # 패턴상태 (이모지로 간결하게) - A방식
    state = row["패턴상태"]
    if state == "초기진입":
        cols[3].write("🎯")  # 최우선 타겟
    elif state == "상승중":
        cols[3].write("📈")
    elif state == "돌파임박":
        cols[3].write("⚡")
    else:
        cols[3].write("📍")

    # 머리대비상승, 넥라인상승여력 표시
    head_rise = row.get("머리대비상승", 0)
    upside = row.get("넥라인상승여력", 0)
    cols[4].write(f"+{head_rise:.0f}%")
    cols[5].write(f"**{upside:.0f}%**")
    cols[6].write(f"{row['예상수익률']:.0f}%")


def display_stock_table(df):
    """종목 테이블 표시 (클릭 가능한 번호 포함)"""
    if df is None or len(df) == 0:
        return

    st.caption("번호(#)를 클릭하면 해당 종목 차트로 이동합니다 | 🎯초기진입 📈상승중 ⚡돌파임박")

    # 헤더
    header_cols = st.columns([0.4, 1.5, 1, 0.8, 0.9, 0.9, 0.8])
    headers = ["#", "종목명", "현재가", "상태", "머리↗", "넥라인↗", "수익률"]
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    # 상위 10개 표시
    for idx in range(min(len(df), 10)):
        display_table_row(df, idx)

    # 11개 이상이면 나머지는 접기
    if len(df) > 10:
        with st.expander(f"➕ 나머지 {len(df) - 10}개 종목 보기"):
            for idx in range(10, len(df)):
                display_table_row(df, idx)


def display_chart_detail(df, idx):
    """차트 상세 표시"""
    if df is None or len(df) == 0 or idx >= len(df):
        return

    row = df.iloc[idx]
    ticker = str(row["종목코드"]).zfill(6)
    name = row["종목명"]

    # 차트 이미지 가져오기
    chart_path = get_chart_image(ticker)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 {name} ({ticker})")
        if chart_path:
            st.image(chart_path, use_container_width=True)
        else:
            st.error(f"🖼️ 차트 이미지 없음")
            st.caption(f"종목코드: {ticker}")
            if st.session_state.chart_files:
                st.caption(f"사용 가능한 차트: {len(st.session_state.chart_files)}개")
            else:
                st.caption("차트 폴더가 비어있거나 찾을 수 없습니다.")

    with col2:
        st.subheader("📋 패턴 상세 정보")

        # 패턴 상태 배지 (A방식)
        state = row["패턴상태"]
        if state == "초기진입":
            st.success(f"🎯 {state} (최우선)")
        elif state == "상승중":
            st.info(f"📈 {state}")
        elif state == "돌파임박":
            st.warning(f"⚡ {state}")
        else:
            st.info(f"📍 {state}")

        # 핵심 지표: 상승 여력
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            head_rise = row.get("머리대비상승", 0)
            st.metric("머리 대비", f"+{head_rise:.1f}%")
        with col_m2:
            upside = row.get("넥라인상승여력", 0)
            st.metric("넥라인까지", f"+{upside:.1f}%")

        st.divider()

        # B전략 매매 가이드
        st.markdown("**💰 B전략 매매 가이드**")
        current_price = int(row["현재가"])
        expected_return = row["예상수익률"] / 100  # 퍼센트를 소수로 변환
        take_profit_price = int(current_price * (1 + expected_return * 0.5))
        stop_loss_price = int(current_price * 0.9)

        col_tp, col_sl = st.columns(2)
        with col_tp:
            st.success(f"🎯 익절가\n**{take_profit_price:,}원**")
        with col_sl:
            st.error(f"🛑 손절가\n**{stop_loss_price:,}원**")

        st.caption(f"⏰ 최대 보유: 60일")

        st.divider()

        # 가격 정보
        st.markdown("**가격 정보**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"현재가: **{current_price:,}원**")
            st.write(f"넥라인: {int(row['넥라인가격']):,}원")
        with col_b:
            st.write(f"목표가: **{int(row['목표가']):,}원**")
            st.write(f"패턴손절: {int(row['손절가']):,}원")

        st.divider()

        # 어깨 가격
        st.markdown("**패턴 구성**")
        st.write(f"오른쪽어깨: **{int(row['오른쪽어깨가격']):,}원**")
        st.write(f"왼쪽어깨: {int(row['왼쪽어깨가격']):,}원")
        st.write(f"머리: {int(row['머리가격']):,}원")

        st.divider()

        # 지표
        st.markdown("**신뢰도 지표**")
        st.write(f"신뢰도 점수: {row['신뢰도점수']:.0f}점")
        st.write(f"어깨 대칭성: {row['어깨대칭성']:.0f}%")
        st.write(f"예상 수익률: **{row['예상수익률']:.0f}%**")


def display_gallery(df, top_n=10):
    """갤러리 뷰 표시"""
    if df is None or len(df) == 0:
        st.info("표시할 차트가 없습니다.")
        return

    top_df = df.head(top_n)

    # 2열 그리드
    cols = st.columns(2)

    displayed = 0
    for idx, row in top_df.iterrows():
        ticker = str(row["종목코드"]).zfill(6)
        name = row["종목명"]

        col = cols[displayed % 2]

        with col:
            chart_path = get_chart_image(ticker)

            upside = row.get("넥라인상승여력", 0)
            st.markdown(f"**{idx+1}. {name}** ({row['패턴상태']}) - 넥라인까지 +{upside:.0f}%")

            if chart_path:
                st.image(chart_path, use_container_width=True)
            else:
                st.warning(f"🖼️ 이미지 없음 ({ticker})")

            # 클릭하면 메인으로 이동
            if st.button(f"상세보기", key=f"gallery_{idx}"):
                st.session_state.selected_idx = idx
                st.rerun()

            st.divider()
            displayed += 1


# ========== 시작 시 자동 로드 ==========
if not st.session_state.initialized:
    results = load_existing_results()
    if results is not None:
        st.session_state.results = results
        st.session_state.initialized = True


# ========== 사이드바 ==========
with st.sidebar:
    st.title("🔍 패턴 스캐너")

    st.divider()

    # 안내 문구
    st.info("📌 **결과 뷰어 전용**\n\n매일 오후 4:30 자동 업데이트\n\n수동 스캔: 로컬 PC에서 `python main.py`")

    st.divider()

    # 기존 결과 로드
    if st.button("🔄 결과 새로고침", type="primary", use_container_width=True):
        results = load_existing_results()
        if results is not None:
            st.session_state.results = results
            st.success(f"{len(results)}개 종목 로드됨")
            st.caption(f"차트 파일: {len(st.session_state.chart_files)}개")
            st.rerun()
        else:
            st.warning("저장된 결과가 없습니다.")

    st.divider()

    # 필터 설정
    st.subheader("⚙️ 필터 조건")

    min_head_depth = st.slider(
        "최소 머리 깊이 (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    min_symmetry = st.slider(
        "최소 어깨 대칭성 (%)",
        min_value=80.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    pattern_options = ["초기진입", "상승중", "돌파임박"]
    pattern_states = st.multiselect(
        "패턴 상태",
        options=pattern_options,
        default=["초기진입", "상승중"]  # 초기진입 최우선
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

    # 매매 가이드 (B전략)
    st.subheader("💰 매매 가이드")
    st.markdown("**전략: B (중립)**")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.metric("3년 수익률", "+9.9%")
    with col_g2:
        st.metric("승률", "36.1%")

    st.caption("익절: 예상수익률×50% | 손절: -10%")
    st.caption("최대보유: 60일 | 손익비: 2.8:1")

    st.divider()

    # 스캔 정보
    st.subheader("📊 결과 정보")

    if st.session_state.results is not None:
        total = len(st.session_state.results)
        filtered = len(st.session_state.filtered_results) if st.session_state.filtered_results is not None else 0
        charts = len(st.session_state.chart_files)
        st.write(f"총 탐지: {total}개")
        st.write(f"필터 후: {filtered}개")
        st.write(f"차트 파일: {charts}개")


# ========== 메인 영역 ==========
st.title("🎯 상승 직전 종목 스캐너")
st.caption("역헤드앤숄더 패턴 기반 | for ranny")

# 탭 구성
tab1, tab2 = st.tabs(["📋 종목 리스트", "🖼️ 갤러리"])

with tab1:
    df = st.session_state.filtered_results

    if df is None or len(df) == 0:
        st.info("👈 사이드바에서 '결과 새로고침'을 클릭하세요.")
    else:
        # 종목 테이블 (번호 클릭 가능)
        st.subheader(f"🏆 탐지 종목 ({len(df)}개)")
        display_stock_table(df)

        st.divider()

        # 이전/다음 네비게이션
        selected_idx = st.session_state.selected_idx
        if selected_idx >= len(df):
            selected_idx = 0
            st.session_state.selected_idx = 0

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⬅️ 이전", use_container_width=True, disabled=(selected_idx == 0)):
                st.session_state.selected_idx = max(0, selected_idx - 1)
                st.rerun()

        with col2:
            st.markdown(f"<h4 style='text-align: center;'>{selected_idx + 1} / {len(df)}</h4>", unsafe_allow_html=True)

        with col3:
            if st.button("다음 ➡️", use_container_width=True, disabled=(selected_idx >= len(df) - 1)):
                st.session_state.selected_idx = min(len(df) - 1, selected_idx + 1)
                st.rerun()

        # 선택된 종목 차트 표시
        display_chart_detail(df, selected_idx)

with tab2:
    st.subheader("🖼️ 상위 10개 종목 차트")

    df = st.session_state.filtered_results

    if df is None or len(df) == 0:
        st.info("👈 사이드바에서 '결과 새로고침'을 클릭하세요.")
    else:
        display_gallery(df)


# 푸터
st.divider()
st.caption("역헤드앤숄더 패턴 스캐너 v2.1 (B전략) | 3년 백테스트 +9.9% | 매일 16:30 자동 업데이트")
