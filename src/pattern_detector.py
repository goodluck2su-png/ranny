"""
역헤드앤숄더 패턴 탐지 모듈
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from tqdm import tqdm

from pykrx import stock
from config import (
    PATTERN_MIN_DAYS, PATTERN_MAX_DAYS, MIN_TROUGH_INTERVAL,
    EXTREMA_ORDER, SHOULDER_PRICE_TOLERANCE, SHOULDER_TIME_TOLERANCE,
    PRIOR_DECLINE_PCT, MIN_TRADING_VALUE, OHLCV_PERIOD_DAYS,
    VOLUME_BONUS_MULTIPLIER, VOLUME_BONUS_SCORE, PATTERN_STATES,
    MIN_HEAD_DEPTH, NECKLINE_SLOPE_THRESHOLD, NECKLINE_SLOPE_PENALTY
)


def get_ohlcv_with_avg_volume(ticker: str, days: int = None) -> tuple:
    """OHLCV 데이터 수집 + 20일 평균 거래대금 계산"""
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
            return None, 0

        # 거래대금 컬럼이 없으면 계산 (종가 * 거래량)
        if "거래대금" not in df.columns:
            df["거래대금"] = df["종가"] * df["거래량"]

        # 20일 평균 거래대금 계산
        avg_trading_value = df["거래대금"].tail(20).mean()

        return df, avg_trading_value

    except Exception as e:
        return None, 0


def find_local_minima(prices: np.ndarray, order: int = None) -> np.ndarray:
    """저점(local minima) 탐지 - scipy argrelextrema 사용"""
    if order is None:
        order = EXTREMA_ORDER

    # argrelextrema로 저점 인덱스 찾기
    minima_idx = argrelextrema(prices, np.less_equal, order=order)[0]

    return minima_idx


def find_local_maxima(prices: np.ndarray, order: int = None) -> np.ndarray:
    """고점(local maxima) 탐지"""
    if order is None:
        order = EXTREMA_ORDER

    maxima_idx = argrelextrema(prices, np.greater_equal, order=order)[0]

    return maxima_idx


def check_prior_decline(df: pd.DataFrame, pattern_start_idx: int) -> bool:
    """선행 하락 조건 확인 (패턴 시작 전 고점 대비 -20% 이상)"""
    if pattern_start_idx < 20:
        return False

    # 패턴 시작 전 구간
    prior_data = df.iloc[:pattern_start_idx]

    if len(prior_data) < 20:
        return False

    # 이전 고점
    prior_high = prior_data["고가"].max()
    # 패턴 시작점 가격
    pattern_start_price = df.iloc[pattern_start_idx]["저가"]

    # 하락률 계산
    decline_pct = (pattern_start_price - prior_high) / prior_high

    return decline_pct <= PRIOR_DECLINE_PCT


def validate_inverse_head_and_shoulders(
    df: pd.DataFrame,
    left_shoulder_idx: int,
    head_idx: int,
    right_shoulder_idx: int
) -> dict:
    """역헤드앤숄더 패턴 유효성 검증"""

    prices = df["저가"].values
    closes = df["종가"].values
    highs = df["고가"].values

    left_shoulder_price = prices[left_shoulder_idx]
    head_price = prices[head_idx]
    right_shoulder_price = prices[right_shoulder_idx]

    # 1. 머리가 양 어깨보다 낮아야 함
    if not (head_price < left_shoulder_price and head_price < right_shoulder_price):
        return None

    # 2. 머리 깊이 검증 (어깨 평균 대비 최소 10% 낮아야 함)
    shoulder_avg = (left_shoulder_price + right_shoulder_price) / 2
    head_depth = (shoulder_avg - head_price) / shoulder_avg

    if head_depth < MIN_HEAD_DEPTH:
        return None

    # 3. 어깨 가격 대칭성 (±10%)
    shoulder_diff = abs(left_shoulder_price - right_shoulder_price)
    shoulder_symmetry = shoulder_diff / shoulder_avg

    if shoulder_symmetry > SHOULDER_PRICE_TOLERANCE:
        return None

    # 4. 시간 대칭성 (50% 이내)
    left_to_head = head_idx - left_shoulder_idx
    head_to_right = right_shoulder_idx - head_idx

    if left_to_head == 0 or head_to_right == 0:
        return None

    time_ratio = min(left_to_head, head_to_right) / max(left_to_head, head_to_right)

    if time_ratio < (1 - SHOULDER_TIME_TOLERANCE):
        return None

    # 5. 저점 간격 최소 15거래일
    if (head_idx - left_shoulder_idx) < MIN_TROUGH_INTERVAL:
        return None
    if (right_shoulder_idx - head_idx) < MIN_TROUGH_INTERVAL:
        return None

    # 6. 패턴 기간 (60~200거래일)
    pattern_days = right_shoulder_idx - left_shoulder_idx
    if pattern_days < PATTERN_MIN_DAYS or pattern_days > PATTERN_MAX_DAYS:
        return None

    # 7. 선행 하락 확인
    if not check_prior_decline(df, left_shoulder_idx):
        return None

    # 넥라인 계산 (왼쪽 어깨와 머리 사이 고점 + 머리와 오른쪽 어깨 사이 고점)
    left_neckline_idx = left_shoulder_idx + np.argmax(highs[left_shoulder_idx:head_idx])
    right_neckline_idx = head_idx + np.argmax(highs[head_idx:right_shoulder_idx])

    left_neckline_price = highs[left_neckline_idx]
    right_neckline_price = highs[right_neckline_idx]

    # 넥라인 기울기 계산 (우하향 = 음수)
    # (오른쪽 고점 - 왼쪽 고점) / 왼쪽 고점
    neckline_slope = (right_neckline_price - left_neckline_price) / left_neckline_price

    # 8. 우하향 넥라인 제외 (-10% 이상 하락 시)
    if neckline_slope < NECKLINE_SLOPE_THRESHOLD:
        return None

    # 넥라인 (두 고점의 평균)
    neckline_price = (left_neckline_price + right_neckline_price) / 2

    # 현재가
    current_price = closes[-1]

    # 패턴 상태 판별
    head_to_neckline = neckline_price - head_price

    if current_price < head_price + head_to_neckline * 0.3:
        pattern_state = "바닥형성"
    elif current_price < neckline_price * 0.95:
        pattern_state = "넥라인근접"
    else:
        pattern_state = "돌파임박"

    # 목표가 (넥라인 + 패턴 높이)
    pattern_height = neckline_price - head_price
    target_price = neckline_price + pattern_height

    # 손절가 (머리 저점)
    stop_loss = head_price

    # 예상 수익률
    expected_return = (target_price - current_price) / current_price * 100

    # 어깨 대칭성 점수 (100% - 차이%)
    symmetry_score = (1 - shoulder_symmetry) * 100

    return {
        "left_shoulder_idx": left_shoulder_idx,
        "head_idx": head_idx,
        "right_shoulder_idx": right_shoulder_idx,
        "left_shoulder_price": left_shoulder_price,
        "head_price": head_price,
        "right_shoulder_price": right_shoulder_price,
        "neckline_price": neckline_price,
        "current_price": current_price,
        "pattern_state": pattern_state,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "expected_return": expected_return,
        "symmetry_score": symmetry_score,
        "head_depth": head_depth,           # 머리 깊이 (0.10 = 10%)
        "time_symmetry": time_ratio,         # 시간 대칭성 (1.0 = 완벽 대칭)
        "pattern_days": pattern_days,
        "neckline_slope": neckline_slope     # 넥라인 기울기 (음수 = 우하향)
    }


def detect_pattern(df: pd.DataFrame, debug: bool = False) -> dict:
    """단일 종목에서 역헤드앤숄더 패턴 탐지"""

    if df is None or len(df) < PATTERN_MIN_DAYS:
        if debug:
            print(f"    - 데이터 부족: {len(df) if df is not None else 0}일")
        return None

    prices = df["저가"].values

    # 저점 탐지
    minima_idx = find_local_minima(prices, EXTREMA_ORDER)

    if len(minima_idx) < 3:
        if debug:
            print(f"    - 저점 부족: {len(minima_idx)}개")
        return None

    if debug:
        print(f"    - 저점 {len(minima_idx)}개 발견")

    # 가장 최근 패턴 우선 탐색 (역순으로 검색)
    # k(오른쪽어깨)가 가장 최근인 것부터, 유효한 패턴 발견 시 즉시 반환
    best_pattern = None
    checked = 0

    # 역순 탐색: 가장 최근 조합부터 검사
    for k in range(len(minima_idx) - 1, 1, -1):  # 오른쪽어깨 (최근→과거)
        for j in range(k - 1, 0, -1):  # 머리
            for i in range(j - 1, -1, -1):  # 왼쪽어깨
                left_idx = minima_idx[i]
                head_idx = minima_idx[j]
                right_idx = minima_idx[k]

                pattern = validate_inverse_head_and_shoulders(
                    df, left_idx, head_idx, right_idx
                )
                checked += 1

                if pattern:
                    # 가장 최근 유효 패턴 발견 → 즉시 반환
                    if debug:
                        print(f"    - {checked}개 조합 검사 후 패턴 발견")
                    return pattern

    if debug:
        print(f"    - {checked}개 조합 검사, 유효 패턴 없음")

    return None


def calculate_reliability_score(pattern: dict, avg_volume: float, current_volume: float) -> float:
    """
    신뢰도 점수 계산 (100점 만점)

    - 기본: 50점
    - 머리 깊이: 10%당 +5점 (최대 20점)
    - 어깨 대칭성: (대칭성% - 90) × 1점 (최대 10점)
    - 시간 대칭성: 대칭일수록 가점 (최대 10점)
    - 패턴 상태: 돌파임박 +10점
    - 거래량 가점: +10점
    """
    score = 50  # 기본 점수

    # 1. 머리 깊이 가점 (10%당 +5점, 최대 20점)
    # head_depth 0.10 → +5점, 0.20 → +10점, 0.40+ → +20점
    head_depth_score = min(pattern["head_depth"] * 50, 20)
    score += head_depth_score

    # 2. 어깨 대칭성 가점 ((대칭성% - 90) × 1점, 최대 10점)
    # symmetry_score 90% → 0점, 95% → 5점, 100% → 10점
    symmetry_bonus = max(0, (pattern["symmetry_score"] - 90))
    score += min(symmetry_bonus, 10)

    # 3. 시간 대칭성 가점 (최대 10점)
    # time_symmetry 1.0 → +10점, 0.5 → +5점
    time_symmetry_score = pattern["time_symmetry"] * 10
    score += time_symmetry_score

    # 4. 패턴 상태 가점 (돌파임박만 +10점)
    if pattern["pattern_state"] == "돌파임박":
        score += 10

    # 5. 거래량 가점 (20일 평균 대비 1.5배 초과 시 +10점)
    if avg_volume > 0 and current_volume > avg_volume * VOLUME_BONUS_MULTIPLIER:
        score += VOLUME_BONUS_SCORE

    # 6. 넥라인 기울기 감점 (우하향 시 -5점)
    if pattern.get("neckline_slope", 0) < 0:
        score -= NECKLINE_SLOPE_PENALTY

    return max(min(score, 100), 0)  # 0~100 범위


def scan_stocks(filtered_stocks: pd.DataFrame, verbose: bool = True, debug_count: int = 0) -> pd.DataFrame:
    """필터링된 종목에서 패턴 스캔"""

    results = []
    debug_idx = 0
    skipped_volume = 0
    skipped_data = 0

    iterator = tqdm(filtered_stocks.iterrows(), total=len(filtered_stocks), desc="패턴 스캔") if verbose else filtered_stocks.iterrows()

    for idx, row in iterator:
        ticker = row["종목코드"]
        name = row["종목명"]
        market = row["시장"]

        # 1. OHLCV 수집 + 20일 평균 거래대금
        df, avg_trading_value = get_ohlcv_with_avg_volume(ticker)

        if df is None or len(df) < PATTERN_MIN_DAYS:
            skipped_data += 1
            continue

        # 2. 20일 평균 거래대금 10억 미만 제외 (2차 필터)
        if avg_trading_value < MIN_TRADING_VALUE:
            skipped_volume += 1
            continue

        # 3. 패턴 탐지 (처음 N개 종목만 디버그)
        debug_mode = debug_idx < debug_count
        if debug_mode:
            print(f"\n  [{name}] 분석 중...")

        pattern = detect_pattern(df, debug=debug_mode)
        debug_idx += 1

        if pattern is None:
            continue

        # 4. 현재 거래량
        current_volume = df["거래대금"].iloc[-1]

        # 5. 신뢰도 점수
        reliability = calculate_reliability_score(pattern, avg_trading_value, current_volume)

        results.append({
            "종목명": name,
            "종목코드": ticker,
            "현재가": int(pattern["current_price"]),
            "시가총액": row["시가총액"],
            "패턴상태": pattern["pattern_state"],
            "어깨대칭성": round(pattern["symmetry_score"], 1),
            "넥라인가격": int(pattern["neckline_price"]),
            "목표가": int(pattern["target_price"]),
            "손절가": int(pattern["stop_loss"]),
            "예상수익률": round(pattern["expected_return"], 1),
            "신뢰도점수": round(reliability, 1),
            "일평균거래대금": int(avg_trading_value),
            "패턴기간": pattern["pattern_days"],
            # 신뢰도 점수 구성 요소
            "머리깊이": round(pattern["head_depth"] * 100, 1),  # 퍼센트로 표시
            "시간대칭성": round(pattern["time_symmetry"] * 100, 1),  # 퍼센트로 표시
            "넥라인기울기": round(pattern["neckline_slope"] * 100, 1),  # 퍼센트로 표시
            # 차트 표시용 인덱스 저장
            "왼쪽어깨idx": pattern["left_shoulder_idx"],
            "머리idx": pattern["head_idx"],
            "오른쪽어깨idx": pattern["right_shoulder_idx"],
            "왼쪽어깨가격": int(pattern["left_shoulder_price"]),
            "머리가격": int(pattern["head_price"]),
            "오른쪽어깨가격": int(pattern["right_shoulder_price"])
        })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        result_df = result_df.sort_values("신뢰도점수", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n스캔 통계:")
        print(f"  - 데이터 부족 제외: {skipped_data}개")
        print(f"  - 거래대금 10억 미만 제외: {skipped_volume}개")
        print(f"  - 패턴 발견: {len(result_df)}개 종목")

    return result_df


if __name__ == "__main__":
    print("=" * 50)
    print("역헤드앤숄더 패턴 탐지 테스트")
    print("=" * 50)

    # 필터링된 종목 로드 (종목코드를 문자열로)
    try:
        filtered_stocks = pd.read_csv("filtered_stocks.csv", dtype={"종목코드": str})
        # 6자리 패딩 (005930 형식)
        filtered_stocks["종목코드"] = filtered_stocks["종목코드"].str.zfill(6)
        print(f"필터링된 종목 로드: {len(filtered_stocks)}개")
        print(f"종목코드 예시: {filtered_stocks['종목코드'].iloc[0]}")
    except:
        print("filtered_stocks.csv 파일이 없습니다.")
        print("먼저 data_collector.py를 실행하세요.")
        exit()

    # 패턴 스캔 (처음 5개 종목 디버그 출력)
    print("\n[디버그] 처음 5개 종목 상세 분석:")
    results = scan_stocks(filtered_stocks, debug_count=5)

    # 결과 출력
    if len(results) > 0:
        print("\n" + "=" * 50)
        print("탐지된 패턴")
        print("=" * 50)
        print(results.to_string())

        # 저장
        results.to_csv("pattern_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n저장 완료: pattern_results.csv ({len(results)}개 종목)")
    else:
        print("\n패턴이 발견된 종목이 없습니다.")
