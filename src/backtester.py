"""
역헤드앤숄더 패턴 백테스팅 모듈

과거 데이터로 패턴의 실제 수익성 검증
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

from pykrx import stock
from config import (
    PATTERN_MIN_DAYS, PATTERN_MAX_DAYS, MIN_TROUGH_INTERVAL,
    EXTREMA_ORDER, SHOULDER_PRICE_TOLERANCE, SHOULDER_TIME_TOLERANCE,
    MIN_HEAD_DEPTH, NECKLINE_SLOPE_THRESHOLD, PRIOR_DECLINE_PCT,
    OUTPUT_DIR, MIN_TRADING_VALUE
)
from pattern_detector import find_local_minima, check_prior_decline
from data_collector import filter_stocks_fast


# ========== 백테스팅 설정 ==========
BACKTEST_START_DATE = "20250801"  # 패턴 탐지 시점 (6개월 전)
BACKTEST_END_DATE = "20260207"    # 현재 (결과 추적 종료)
STOP_LOSS_PCT = -0.10             # 손절 기준 -10%
HOLD_DAYS_MAX = 120               # 최대 보유 기간 (약 6개월)


def get_historical_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """특정 기간의 OHLCV 데이터 수집"""
    try:
        df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None


def validate_pattern_for_backtest(
    df: pd.DataFrame,
    left_shoulder_idx: int,
    head_idx: int,
    right_shoulder_idx: int
) -> dict:
    """백테스트용 패턴 유효성 검증 (상승 직전 필터 없이)"""

    prices = df["저가"].values
    closes = df["종가"].values
    highs = df["고가"].values

    left_shoulder_price = prices[left_shoulder_idx]
    head_price = prices[head_idx]
    right_shoulder_price = prices[right_shoulder_idx]

    # 1. 머리가 양 어깨보다 낮아야 함
    if not (head_price < left_shoulder_price and head_price < right_shoulder_price):
        return None

    # 2. 머리 깊이 검증
    shoulder_avg = (left_shoulder_price + right_shoulder_price) / 2
    head_depth = (shoulder_avg - head_price) / shoulder_avg
    if head_depth < MIN_HEAD_DEPTH:
        return None

    # 3. 어깨 가격 대칭성
    shoulder_diff = abs(left_shoulder_price - right_shoulder_price)
    shoulder_symmetry = shoulder_diff / shoulder_avg
    if shoulder_symmetry > SHOULDER_PRICE_TOLERANCE:
        return None

    # 4. 시간 대칭성
    left_to_head = head_idx - left_shoulder_idx
    head_to_right = right_shoulder_idx - head_idx
    if left_to_head == 0 or head_to_right == 0:
        return None
    time_ratio = min(left_to_head, head_to_right) / max(left_to_head, head_to_right)
    if time_ratio < (1 - SHOULDER_TIME_TOLERANCE):
        return None

    # 5. 저점 간격
    if (head_idx - left_shoulder_idx) < MIN_TROUGH_INTERVAL:
        return None
    if (right_shoulder_idx - head_idx) < MIN_TROUGH_INTERVAL:
        return None

    # 6. 패턴 기간
    pattern_days = right_shoulder_idx - left_shoulder_idx
    if pattern_days < PATTERN_MIN_DAYS or pattern_days > PATTERN_MAX_DAYS:
        return None

    # 7. 선행 하락
    if not check_prior_decline(df, left_shoulder_idx):
        return None

    # 넥라인 계산
    left_neckline_idx = left_shoulder_idx + np.argmax(highs[left_shoulder_idx:head_idx])
    right_neckline_idx = head_idx + np.argmax(highs[head_idx:right_shoulder_idx])
    left_neckline_price = highs[left_neckline_idx]
    right_neckline_price = highs[right_neckline_idx]

    # 넥라인 기울기
    neckline_slope = (right_neckline_price - left_neckline_price) / left_neckline_price
    if neckline_slope < NECKLINE_SLOPE_THRESHOLD:
        return None

    neckline_price = (left_neckline_price + right_neckline_price) / 2

    # 목표가
    pattern_height = neckline_price - head_price
    target_price = neckline_price + pattern_height

    # 탐지 시점 가격 (오른쪽 어깨 이후 마지막 가격)
    detection_price = closes[right_shoulder_idx]

    return {
        "right_shoulder_idx": right_shoulder_idx,
        "right_shoulder_price": right_shoulder_price,
        "neckline_price": neckline_price,
        "target_price": target_price,
        "head_price": head_price,
        "detection_price": detection_price,
        "head_depth": head_depth,
        "symmetry_score": (1 - shoulder_symmetry) * 100
    }


def detect_pattern_for_backtest(df: pd.DataFrame) -> dict:
    """백테스트용 패턴 탐지"""
    if df is None or len(df) < PATTERN_MIN_DAYS:
        return None

    prices = df["저가"].values
    minima_idx = find_local_minima(prices, EXTREMA_ORDER)

    if len(minima_idx) < 3:
        return None

    # 역순 탐색
    for k in range(len(minima_idx) - 1, 1, -1):
        for j in range(k - 1, 0, -1):
            for i in range(j - 1, -1, -1):
                left_idx = minima_idx[i]
                head_idx = minima_idx[j]
                right_idx = minima_idx[k]

                pattern = validate_pattern_for_backtest(
                    df, left_idx, head_idx, right_idx
                )

                if pattern:
                    return pattern

    return None


def backtest_single_stock(ticker: str, name: str, pattern: dict,
                          future_df: pd.DataFrame) -> dict:
    """단일 종목 백테스트 실행"""

    if future_df is None or len(future_df) < 5:
        return None

    neckline = pattern["neckline_price"]
    target = pattern["target_price"]
    head_price = pattern["head_price"]
    detection_price = pattern["detection_price"]

    closes = future_df["종가"].values
    highs = future_df["고가"].values
    lows = future_df["저가"].values
    dates = future_df.index.tolist()

    # === A방식: 오른쪽 어깨 부근 매수 (탐지 시점) ===
    entry_a = detection_price
    exit_a = None
    exit_reason_a = None
    exit_date_a = None
    max_price_a = entry_a
    min_price_a = entry_a

    for i, (high, low, close) in enumerate(zip(highs, lows, closes)):
        max_price_a = max(max_price_a, high)
        min_price_a = min(min_price_a, low)

        # 손절 체크 (-10%)
        if low <= entry_a * (1 + STOP_LOSS_PCT):
            exit_a = entry_a * (1 + STOP_LOSS_PCT)
            exit_reason_a = "손절"
            exit_date_a = dates[i]
            break

        # 목표가 도달
        if high >= target:
            exit_a = target
            exit_reason_a = "목표달성"
            exit_date_a = dates[i]
            break

        # 최대 보유 기간
        if i >= HOLD_DAYS_MAX - 1:
            exit_a = close
            exit_reason_a = "기간종료"
            exit_date_a = dates[i]
            break

    if exit_a is None:
        exit_a = closes[-1]
        exit_reason_a = "기간종료"
        exit_date_a = dates[-1]

    return_a = (exit_a - entry_a) / entry_a * 100

    # === B방식: 넥라인 돌파 후 매수 ===
    entry_b = None
    exit_b = None
    exit_reason_b = None
    exit_date_b = None
    breakout_idx = None

    # 넥라인 돌파 시점 찾기
    for i, (high, close) in enumerate(zip(highs, closes)):
        if close > neckline:  # 종가 기준 돌파
            entry_b = close
            breakout_idx = i
            break

    if entry_b is not None:
        max_price_b = entry_b
        for i in range(breakout_idx + 1, len(closes)):
            high = highs[i]
            low = lows[i]
            close = closes[i]
            max_price_b = max(max_price_b, high)

            # 손절
            if low <= entry_b * (1 + STOP_LOSS_PCT):
                exit_b = entry_b * (1 + STOP_LOSS_PCT)
                exit_reason_b = "손절"
                exit_date_b = dates[i]
                break

            # 목표가
            if high >= target:
                exit_b = target
                exit_reason_b = "목표달성"
                exit_date_b = dates[i]
                break

            # 최대 보유
            if (i - breakout_idx) >= HOLD_DAYS_MAX - 1:
                exit_b = close
                exit_reason_b = "기간종료"
                exit_date_b = dates[i]
                break

        if exit_b is None:
            exit_b = closes[-1]
            exit_reason_b = "기간종료"
            exit_date_b = dates[-1]

        return_b = (exit_b - entry_b) / entry_b * 100
    else:
        return_b = None
        exit_reason_b = "돌파실패"

    # 넥라인 돌파 여부
    neckline_breakout = any(c > neckline for c in closes)

    return {
        "종목명": name,
        "종목코드": ticker,
        "탐지가": int(detection_price),
        "넥라인": int(neckline),
        "목표가": int(target),
        "머리깊이": round(pattern["head_depth"] * 100, 1),
        "대칭성": round(pattern["symmetry_score"], 1),
        # 결과
        "넥라인돌파": "O" if neckline_breakout else "X",
        # A방식 결과
        "A_진입가": int(entry_a),
        "A_청산가": int(exit_a) if exit_a else None,
        "A_수익률": round(return_a, 1),
        "A_결과": exit_reason_a,
        # B방식 결과
        "B_진입가": int(entry_b) if entry_b else None,
        "B_청산가": int(exit_b) if exit_b else None,
        "B_수익률": round(return_b, 1) if return_b is not None else None,
        "B_결과": exit_reason_b
    }


def run_backtest(verbose: bool = True):
    """백테스트 메인 실행"""

    print("=" * 60)
    print("역헤드앤숄더 패턴 백테스팅")
    print(f"탐지 시점: {BACKTEST_START_DATE} / 추적 종료: {BACKTEST_END_DATE}")
    print("=" * 60)

    # 1. 과거 시점의 종목 목록 (현재 기준으로 필터링)
    print("\n[1/4] 종목 필터링...")
    filtered_stocks = filter_stocks_fast(verbose=False)
    print(f"  필터링된 종목: {len(filtered_stocks)}개")

    # 2. 과거 시점 데이터로 패턴 탐지
    print("\n[2/4] 과거 시점 패턴 탐지...")

    # 패턴 탐지용 데이터 기간 (탐지 시점 기준 1년)
    pattern_start = datetime.strptime(BACKTEST_START_DATE, "%Y%m%d") - timedelta(days=365)
    pattern_start_str = pattern_start.strftime("%Y%m%d")

    detected_patterns = []

    iterator = tqdm(filtered_stocks.iterrows(), total=len(filtered_stocks),
                   desc="패턴 탐지") if verbose else filtered_stocks.iterrows()

    for idx, row in iterator:
        ticker = row["종목코드"]
        name = row["종목명"]

        # 탐지 시점까지의 데이터
        df = get_historical_ohlcv(ticker, pattern_start_str, BACKTEST_START_DATE)
        if df is None or len(df) < PATTERN_MIN_DAYS:
            continue

        # 거래대금 체크
        if "거래대금" not in df.columns:
            df["거래대금"] = df["종가"] * df["거래량"]
        avg_trading = df["거래대금"].tail(20).mean()
        if avg_trading < MIN_TRADING_VALUE:
            continue

        # 패턴 탐지
        pattern = detect_pattern_for_backtest(df)
        if pattern:
            detected_patterns.append({
                "ticker": ticker,
                "name": name,
                "pattern": pattern
            })

    print(f"  패턴 발견: {len(detected_patterns)}개 종목")

    if len(detected_patterns) == 0:
        print("\n패턴이 발견된 종목이 없습니다.")
        return

    # 3. 이후 주가 추적
    print("\n[3/4] 이후 주가 추적 및 수익 계산...")

    results = []

    iterator = tqdm(detected_patterns, desc="백테스트") if verbose else detected_patterns

    for item in iterator:
        ticker = item["ticker"]
        name = item["name"]
        pattern = item["pattern"]

        # 탐지 시점 이후 데이터
        future_df = get_historical_ohlcv(ticker, BACKTEST_START_DATE, BACKTEST_END_DATE)

        result = backtest_single_stock(ticker, name, pattern, future_df)
        if result:
            results.append(result)

    if len(results) == 0:
        print("\n백테스트 가능한 종목이 없습니다.")
        return

    result_df = pd.DataFrame(results)

    # 4. 통계 계산 및 출력
    print("\n[4/4] 결과 분석...")

    # 넥라인 돌파율
    breakout_count = (result_df["넥라인돌파"] == "O").sum()
    breakout_rate = breakout_count / len(result_df) * 100

    # A방식 통계
    a_returns = result_df["A_수익률"]
    a_wins = (a_returns > 0).sum()
    a_win_rate = a_wins / len(result_df) * 100
    a_avg_return = a_returns.mean()
    a_winners = a_returns[a_returns > 0]
    a_losers = a_returns[a_returns <= 0]
    a_avg_win = a_winners.mean() if len(a_winners) > 0 else 0
    a_avg_loss = a_losers.mean() if len(a_losers) > 0 else 0
    a_target_count = (result_df["A_결과"] == "목표달성").sum()
    a_target_rate = a_target_count / len(result_df) * 100

    # B방식 통계 (돌파 성공 종목만)
    b_valid = result_df[result_df["B_수익률"].notna()]
    if len(b_valid) > 0:
        b_returns = b_valid["B_수익률"]
        b_wins = (b_returns > 0).sum()
        b_win_rate = b_wins / len(b_valid) * 100
        b_avg_return = b_returns.mean()
        b_winners = b_returns[b_returns > 0]
        b_losers = b_returns[b_returns <= 0]
        b_avg_win = b_winners.mean() if len(b_winners) > 0 else 0
        b_avg_loss = b_losers.mean() if len(b_losers) > 0 else 0
        b_target_count = (b_valid["B_결과"] == "목표달성").sum()
        b_target_rate = b_target_count / len(b_valid) * 100
    else:
        b_win_rate = 0
        b_avg_return = 0
        b_avg_win = 0
        b_avg_loss = 0
        b_target_rate = 0

    # 결과 출력
    print("\n" + "=" * 60)
    print("백테스트 결과 요약")
    print("=" * 60)
    print(f"\n총 탐지 종목: {len(result_df)}개")
    print(f"넥라인 돌파: {breakout_count}개 ({breakout_rate:.1f}%)")

    print(f"\n{'='*30}")
    print("A방식 (오른쪽 어깨 부근 매수)")
    print(f"{'='*30}")
    print(f"  승률: {a_win_rate:.1f}% ({a_wins}/{len(result_df)})")
    print(f"  목표달성률: {a_target_rate:.1f}% ({a_target_count}/{len(result_df)})")
    print(f"  평균 수익률: {a_avg_return:+.1f}%")
    print(f"  평균 수익 (승): {a_avg_win:+.1f}%")
    print(f"  평균 손실 (패): {a_avg_loss:+.1f}%")

    print(f"\n{'='*30}")
    print("B방식 (넥라인 돌파 후 매수)")
    print(f"{'='*30}")
    if len(b_valid) > 0:
        print(f"  적용 가능: {len(b_valid)}개 (돌파 성공)")
        print(f"  승률: {b_win_rate:.1f}% ({b_wins}/{len(b_valid)})")
        print(f"  목표달성률: {b_target_rate:.1f}%")
        print(f"  평균 수익률: {b_avg_return:+.1f}%")
        print(f"  평균 수익 (승): {b_avg_win:+.1f}%")
        print(f"  평균 손실 (패): {b_avg_loss:+.1f}%")
    else:
        print("  적용 가능한 종목 없음")

    # 파일 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    csv_path = OUTPUT_DIR / "backtest_report.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n종목별 결과: {csv_path}")

    # 요약 텍스트 저장
    summary_path = OUTPUT_DIR / "backtest_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("역헤드앤숄더 패턴 백테스트 결과\n")
        f.write(f"탐지 시점: {BACKTEST_START_DATE}\n")
        f.write(f"추적 종료: {BACKTEST_END_DATE}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"총 탐지 종목: {len(result_df)}개\n")
        f.write(f"넥라인 돌파: {breakout_count}개 ({breakout_rate:.1f}%)\n\n")

        f.write("[A방식: 오른쪽 어깨 부근 매수]\n")
        f.write(f"  승률: {a_win_rate:.1f}%\n")
        f.write(f"  목표달성률: {a_target_rate:.1f}%\n")
        f.write(f"  평균 수익률: {a_avg_return:+.1f}%\n")
        f.write(f"  평균 수익 (승): {a_avg_win:+.1f}%\n")
        f.write(f"  평균 손실 (패): {a_avg_loss:+.1f}%\n\n")

        f.write("[B방식: 넥라인 돌파 후 매수]\n")
        if len(b_valid) > 0:
            f.write(f"  적용 가능: {len(b_valid)}개\n")
            f.write(f"  승률: {b_win_rate:.1f}%\n")
            f.write(f"  목표달성률: {b_target_rate:.1f}%\n")
            f.write(f"  평균 수익률: {b_avg_return:+.1f}%\n")
            f.write(f"  평균 수익 (승): {b_avg_win:+.1f}%\n")
            f.write(f"  평균 손실 (패): {b_avg_loss:+.1f}%\n")
        else:
            f.write("  적용 가능한 종목 없음\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("상위 수익 종목 (A방식)\n")
        f.write("=" * 60 + "\n")
        top5 = result_df.nlargest(5, "A_수익률")[["종목명", "A_수익률", "A_결과", "넥라인돌파"]]
        f.write(top5.to_string(index=False))

        f.write("\n\n" + "=" * 60 + "\n")
        f.write("하위 수익 종목 (A방식)\n")
        f.write("=" * 60 + "\n")
        bottom5 = result_df.nsmallest(5, "A_수익률")[["종목명", "A_수익률", "A_결과", "넥라인돌파"]]
        f.write(bottom5.to_string(index=False))

    print(f"요약 리포트: {summary_path}")

    # 상위/하위 종목 출력
    print("\n" + "=" * 60)
    print("상위 수익 종목 (A방식)")
    print("=" * 60)
    print(result_df.nlargest(5, "A_수익률")[["종목명", "A_수익률", "A_결과", "넥라인돌파"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("하위 수익 종목 (A방식)")
    print("=" * 60)
    print(result_df.nsmallest(5, "A_수익률")[["종목명", "A_수익률", "A_결과", "넥라인돌파"]].to_string(index=False))

    return result_df


if __name__ == "__main__":
    run_backtest()
