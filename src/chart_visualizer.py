"""
차트 시각화 모듈 - 상위 종목 캔들차트 + 패턴 표시
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from pathlib import Path

from pykrx import stock
from config import OUTPUT_DIR, CHART_DIR, TOP_N_CHARTS, OHLCV_PERIOD_DAYS


# 한글 폰트 설정
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# mplfinance 타이틀용 폰트 설정
TITLE_FONT = fm.FontProperties(family='Malgun Gothic', size=12)


def ensure_dirs():
    """출력 디렉토리 생성"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)


def get_ohlcv_for_chart(ticker: str, days: int = None) -> pd.DataFrame:
    """차트용 OHLCV 데이터 수집 (영문 컬럼명)"""
    if days is None:
        days = OHLCV_PERIOD_DAYS

    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.5))

    try:
        df = stock.get_market_ohlcv_by_date(
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            ticker
        )

        if df.empty:
            return None

        # mplfinance용 컬럼명 변경
        df = df.rename(columns={
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume"
        })

        # 인덱스를 DatetimeIndex로 변환
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"[{ticker}] OHLCV 수집 실패: {e}")
        return None


def detect_pattern_indices(df: pd.DataFrame, pattern_result: dict) -> dict:
    """패턴 결과에서 인덱스 정보 추출 (차트 표시용)"""
    from scipy.signal import argrelextrema
    from config import EXTREMA_ORDER

    prices = df["Low"].values

    # 저점 탐지
    minima_idx = argrelextrema(prices, np.less_equal, order=EXTREMA_ORDER)[0]

    if len(minima_idx) < 3:
        return None

    # 패턴 찾기 (pattern_detector와 동일한 로직)
    best_pattern = None
    best_score = 0

    for i in range(len(minima_idx) - 2):
        for j in range(i + 1, len(minima_idx) - 1):
            for k in range(j + 1, len(minima_idx)):
                left_idx = minima_idx[i]
                head_idx = minima_idx[j]
                right_idx = minima_idx[k]

                left_price = prices[left_idx]
                head_price = prices[head_idx]
                right_price = prices[right_idx]

                # 머리가 양 어깨보다 낮아야 함
                if not (head_price < left_price and head_price < right_price):
                    continue

                # 어깨 대칭성
                shoulder_diff = abs(left_price - right_price)
                shoulder_avg = (left_price + right_price) / 2
                symmetry = shoulder_diff / shoulder_avg

                if symmetry <= 0.10:  # 10% 이내
                    score = (1 - symmetry) * 100 + (right_idx / len(df) * 50)
                    if score > best_score:
                        best_score = score
                        best_pattern = {
                            "left_idx": left_idx,
                            "head_idx": head_idx,
                            "right_idx": right_idx,
                            "left_price": left_price,
                            "head_price": head_price,
                            "right_price": right_price
                        }

    return best_pattern


def draw_pattern_chart(
    ticker: str,
    name: str,
    pattern_result: dict,
    output_path: Path
) -> bool:
    """패턴 차트 생성"""

    df = get_ohlcv_for_chart(ticker)
    if df is None or len(df) < 60:
        print(f"  [{name}] 데이터 부족")
        return False

    # 저장된 패턴 인덱스 사용 (pattern_detector에서 검증된 값)
    if "왼쪽어깨idx" not in pattern_result:
        print(f"  [{name}] 패턴 인덱스 정보 없음")
        return False

    left_idx = int(pattern_result["왼쪽어깨idx"])
    head_idx = int(pattern_result["머리idx"])
    right_idx = int(pattern_result["오른쪽어깨idx"])

    # 인덱스 범위 검증
    if right_idx >= len(df):
        print(f"  [{name}] 인덱스 범위 초과 (right_idx={right_idx}, len={len(df)})")
        return False

    # 넥라인은 저장된 값 사용
    neckline_price = pattern_result["넥라인가격"]

    # 마커 위치 (날짜 기준)
    dates = df.index.tolist()

    # 패턴 포인트 (저점 3개) - 저장된 가격 사용
    pattern_points = [
        (dates[left_idx], pattern_result["왼쪽어깨가격"], "왼쪽어깨"),
        (dates[head_idx], pattern_result["머리가격"], "머리"),
        (dates[right_idx], pattern_result["오른쪽어깨가격"], "오른쪽어깨")
    ]

    # mplfinance 스타일
    mc = mpf.make_marketcolors(
        up='red', down='blue',
        edge='inherit',
        wick='inherit',
        volume='inherit'
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='lightgray')

    # 패턴 라인 (어깨-머리-어깨 연결)
    pattern_line = pd.Series(index=df.index, dtype=float)
    pattern_line.iloc[left_idx] = pattern_result["왼쪽어깨가격"]
    pattern_line.iloc[head_idx] = pattern_result["머리가격"]
    pattern_line.iloc[right_idx] = pattern_result["오른쪽어깨가격"]
    pattern_line = pattern_line.interpolate()

    # 넥라인 (수평선)
    neckline = pd.Series(index=df.index, data=neckline_price)

    # 목표가 라인
    target_price = pattern_result["목표가"]
    target_line = pd.Series(index=df.index, data=target_price)

    # 손절가 라인
    stop_loss = pattern_result["손절가"]
    stop_line = pd.Series(index=df.index, data=stop_loss)

    # 추가 플롯
    apds = [
        mpf.make_addplot(neckline, color='green', linestyle='--', width=1.5, label='넥라인'),
        mpf.make_addplot(target_line, color='red', linestyle=':', width=1, label='목표가'),
        mpf.make_addplot(stop_line, color='blue', linestyle=':', width=1, label='손절가'),
    ]

    # 차트 제목
    title = f"{name} ({ticker}) - {pattern_result['패턴상태']} | 신뢰도: {pattern_result['신뢰도점수']}점"

    # 차트 생성
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        volume=True,
        addplot=apds,
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True
    )

    # 한글 제목 설정
    ax = axes[0]
    ax.set_title(title, fontproperties=TITLE_FONT, fontsize=12, pad=10)

    # 패턴 포인트 표시 (scatter)
    for date, price, label in pattern_points:
        ax.scatter(df.index.get_loc(date), price, s=100, marker='o', color='purple', zorder=5)
        ax.annotate(label, (df.index.get_loc(date), price),
                   xytext=(0, -20), textcoords='offset points',
                   ha='center', fontsize=9, color='purple', fontproperties=TITLE_FONT)

    # 범례 추가
    ax.axhline(y=neckline_price, color='green', linestyle='--', linewidth=1.5, label=f'넥라인: {int(neckline_price):,}원')
    ax.axhline(y=target_price, color='red', linestyle=':', linewidth=1, label=f'목표가: {int(target_price):,}원')
    ax.axhline(y=stop_loss, color='blue', linestyle=':', linewidth=1, label=f'손절가: {int(stop_loss):,}원')

    # 정보 텍스트
    info_text = (
        f"현재가: {pattern_result['현재가']:,}원\n"
        f"예상수익률: {pattern_result['예상수익률']}%\n"
        f"어깨대칭성: {pattern_result['어깨대칭성']}%"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontproperties=TITLE_FONT,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [{name}] 차트 저장: {output_path.name}")
    return True


def generate_top_charts(results_df: pd.DataFrame, top_n: int = None) -> list:
    """상위 N개 종목 차트 생성"""
    if top_n is None:
        top_n = TOP_N_CHARTS

    ensure_dirs()

    # 신뢰도 순으로 정렬 (이미 정렬되어 있지만 확인)
    df = results_df.sort_values("신뢰도점수", ascending=False).head(top_n)

    print(f"\n상위 {len(df)}개 종목 차트 생성 중...")
    print("=" * 50)

    generated = []
    for idx, row in df.iterrows():
        ticker = str(row["종목코드"]).zfill(6)
        name = row["종목명"]

        output_path = CHART_DIR / f"{idx+1:02d}_{name}_{ticker}.png"

        success = draw_pattern_chart(ticker, name, row.to_dict(), output_path)
        if success:
            generated.append(output_path)

    print("=" * 50)
    print(f"차트 생성 완료: {len(generated)}/{len(df)}개")
    print(f"저장 위치: {CHART_DIR}")

    return generated


if __name__ == "__main__":
    print("=" * 50)
    print("차트 시각화 테스트")
    print("=" * 50)

    # 결과 파일 로드
    try:
        results = pd.read_csv("pattern_results.csv", dtype={"종목코드": str})
        results["종목코드"] = results["종목코드"].str.zfill(6)
        print(f"패턴 결과 로드: {len(results)}개 종목")
    except FileNotFoundError:
        print("pattern_results.csv 파일이 없습니다.")
        print("먼저 pattern_detector.py를 실행하세요.")
        exit()

    # 차트 생성
    charts = generate_top_charts(results)

    if charts:
        print(f"\n생성된 차트 파일:")
        for chart in charts:
            print(f"  - {chart.name}")
