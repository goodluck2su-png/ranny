"""
역헤드앤숄더 패턴 스캐너 - Streamlit 웹앱
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
from pathlib import Path
import io

# src 디렉토리를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import OUTPUT_DIR, CHART_DIR, MIN_HEAD_DEPTH, SHOULDER_PRICE_TOLERANCE
from data_collector import filter_stocks_fast
from pattern_detector import scan_stocks
from chart_visualizer import generate_top_charts, ensure_dirs, draw_pattern_chart

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
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = None


def load_existing_results():
    """기존 결과 파일 로드"""
    result_path = OUTPUT_DIR / "results.csv"
    if result_path.exists():
        df = pd.read_csv(result_path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        return df
    return None


def run_scanner():
    """패턴 스캐너 실행"""
    with st.spinner("종목 필터링 중..."):
        filtered_stocks = filter_stocks_fast(verbose=False)

    if len(filtered_stocks) == 0:
        st.error("필터링된 종목이 없습니다.")
        return None

    with st.spinner(f"{len(filtered_stocks)}개 종목 패턴 스캔 중..."):
        results = scan_stocks(filtered_stocks, verbose=False)

    if len(results) == 0:
        st.warning("패턴이 발견된 종목이 없습니다.")
        return None

    # 결과 저장
    ensure_dirs()
    result_path = OUTPUT_DIR / "results.csv"
    results.to_csv(result_path, index=False, encoding="utf-8-sig")

    # 차트 생성
    with st.spinner("차트 생성 중..."):
        generate_top_charts(results)

    return results


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


def get_chart_image(ticker: str, name: str, pattern_data: dict) -> Path:
    """차트 이미지 경로 반환 또는 생성"""
    ensure_dirs()

    # 기존 차트 파일 찾기
    for chart_file in CHART_DIR.glob(f"*_{name}_{ticker}.png"):
        return chart_file

    # 없으면 생성
    output_path = CHART_DIR / f"temp_{name}_{ticker}.png"
    success = draw_pattern_chart(ticker, name, pattern_data, output_path)

    if success:
        return output_path
    return None


def display_stock_table(df):
    """종목 테이블 표시"""
    if df is None or len(df) == 0:
        st.info("표시할 종목이 없습니다.")
        return

    # 표시할 컬럼 선택
    display_cols = ["종목명", "현재가", "패턴상태", "신뢰도점수", "머리깊이", "어깨대칭성", "예상수익률"]
    display_df = df[display_cols].copy()

    # 포맷팅
    display_df["현재가"] = display_df["현재가"].apply(lambda x: f"{int(x):,}원")
    display_df["신뢰도점수"] = display_df["신뢰도점수"].apply(lambda x: f"{x:.1f}점")
    display_df["머리깊이"] = display_df["머리깊이"].apply(lambda x: f"{x:.1f}%")
    display_df["어깨대칭성"] = display_df["어깨대칭성"].apply(lambda x: f"{x:.1f}%")
    display_df["예상수익률"] = display_df["예상수익률"].apply(lambda x: f"{x:.1f}%")

    # 테이블 표시 (클릭 가능)
    st.dataframe(
        display_df,
        width="stretch",
        height=300,
        hide_index=False
    )


def display_chart_detail(df, idx):
    """차트 상세 표시"""
    if df is None or len(df) == 0 or idx >= len(df):
        return

    row = df.iloc[idx]
    ticker = str(row["종목코드"]).zfill(6)
    name = row["종목명"]

    # 차트 이미지 가져오기
    chart_path = get_chart_image(ticker, name, row.to_dict())

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 {name} ({ticker})")
        if chart_path and chart_path.exists():
            st.image(str(chart_path), width="stretch")
        else:
            st.error("차트를 생성할 수 없습니다.")

    with col2:
        st.subheader("📋 패턴 상세 정보")

        # 패턴 상태 배지
        state = row["패턴상태"]
        if state == "돌파임박":
            st.success(f"🔥 {state}")
        elif state == "넥라인근접":
            st.warning(f"⚡ {state}")
        else:
            st.info(f"📍 {state}")

        st.metric("신뢰도 점수", f"{row['신뢰도점수']:.1f}점")

        st.divider()

        # 가격 정보
        st.markdown("**가격 정보**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"현재가: **{int(row['현재가']):,}원**")
            st.write(f"넥라인: {int(row['넥라인가격']):,}원")
        with col_b:
            st.write(f"목표가: **{int(row['목표가']):,}원**")
            st.write(f"손절가: {int(row['손절가']):,}원")

        st.divider()

        # 어깨 가격
        st.markdown("**패턴 구성**")
        st.write(f"왼쪽어깨: {int(row['왼쪽어깨가격']):,}원")
        st.write(f"머리: {int(row['머리가격']):,}원")
        st.write(f"오른쪽어깨: {int(row['오른쪽어깨가격']):,}원")

        st.divider()

        # 지표
        st.markdown("**신뢰도 지표**")
        st.write(f"머리 깊이: {row['머리깊이']:.1f}%")
        st.write(f"어깨 대칭성: {row['어깨대칭성']:.1f}%")
        st.write(f"시간 대칭성: {row['시간대칭성']:.1f}%")
        st.write(f"예상 수익률: **{row['예상수익률']:.1f}%**")


def display_gallery(df, top_n=10):
    """갤러리 뷰 표시"""
    if df is None or len(df) == 0:
        st.info("표시할 차트가 없습니다.")
        return

    top_df = df.head(top_n)

    # 2열 그리드
    cols = st.columns(2)

    for idx, row in top_df.iterrows():
        ticker = str(row["종목코드"]).zfill(6)
        name = row["종목명"]

        col = cols[idx % 2]

        with col:
            chart_path = get_chart_image(ticker, name, row.to_dict())

            if chart_path and chart_path.exists():
                st.markdown(f"**{idx+1}. {name}** ({row['패턴상태']}) - {row['신뢰도점수']:.1f}점")
                st.image(str(chart_path), width="stretch")

                # 클릭하면 메인으로 이동
                if st.button(f"상세보기", key=f"gallery_{idx}"):
                    st.session_state.selected_idx = idx
                    st.rerun()

                st.divider()


# ========== 사이드바 ==========
with st.sidebar:
    st.title("🔍 패턴 스캐너")

    st.divider()

    # 스캔 실행 버튼
    if st.button("🚀 스캔 실행", type="primary", width="stretch"):
        results = run_scanner()
        if results is not None:
            st.session_state.results = results
            st.session_state.last_scan_time = datetime.now()
            st.session_state.selected_idx = 0
            st.rerun()

    # 기존 결과 로드
    if st.button("📂 기존 결과 로드", width="stretch"):
        results = load_existing_results()
        if results is not None:
            st.session_state.results = results
            st.success(f"{len(results)}개 종목 로드됨")
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

    pattern_options = ["돌파임박", "넥라인근접", "바닥형성"]
    pattern_states = st.multiselect(
        "패턴 상태",
        options=pattern_options,
        default=pattern_options
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

    # 스캔 정보
    st.subheader("📊 스캔 정보")

    if st.session_state.last_scan_time:
        st.write(f"마지막 스캔: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M')}")

    if st.session_state.results is not None:
        total = len(st.session_state.results)
        filtered = len(st.session_state.filtered_results) if st.session_state.filtered_results is not None else 0
        st.write(f"총 탐지: {total}개")
        st.write(f"필터 후: {filtered}개")


# ========== 메인 영역 ==========
st.title("📈 역헤드앤숄더 패턴 스캐너")
st.caption("for ranny")

# 탭 구성
tab1, tab2 = st.tabs(["📋 종목 리스트", "🖼️ 갤러리"])

with tab1:
    df = st.session_state.filtered_results

    if df is None or len(df) == 0:
        st.info("👈 사이드바에서 '스캔 실행' 또는 '기존 결과 로드'를 클릭하세요.")
    else:
        # 상단: 종목 테이블
        st.subheader(f"🏆 탐지 종목 ({len(df)}개)")

        # 종목 선택
        selected_idx = st.selectbox(
            "종목 선택",
            options=range(len(df)),
            format_func=lambda x: f"{x+1}. {df.iloc[x]['종목명']} - {df.iloc[x]['신뢰도점수']:.1f}점 ({df.iloc[x]['패턴상태']})",
            index=st.session_state.selected_idx
        )
        st.session_state.selected_idx = selected_idx

        # 테이블 표시
        display_stock_table(df)

        st.divider()

        # 하단: 차트 상세
        # 이전/다음 버튼
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⬅️ 이전", width="stretch", disabled=(selected_idx == 0)):
                st.session_state.selected_idx = max(0, selected_idx - 1)
                st.rerun()

        with col2:
            st.markdown(f"<h4 style='text-align: center;'>{selected_idx + 1} / {len(df)}</h4>", unsafe_allow_html=True)

        with col3:
            if st.button("다음 ➡️", width="stretch", disabled=(selected_idx >= len(df) - 1)):
                st.session_state.selected_idx = min(len(df) - 1, selected_idx + 1)
                st.rerun()

        # 차트 표시
        display_chart_detail(df, selected_idx)

with tab2:
    st.subheader("🖼️ 상위 10개 종목 차트")

    df = st.session_state.filtered_results

    if df is None or len(df) == 0:
        st.info("👈 사이드바에서 '스캔 실행' 또는 '기존 결과 로드'를 클릭하세요.")
    else:
        display_gallery(df)


# 푸터
st.divider()
st.caption("역헤드앤숄더 패턴 스캐너 v1.0 | KOSPI/KOSDAQ")
