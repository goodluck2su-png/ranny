"""
데이터 수집 및 종목 필터링 모듈 (최적화 버전)
"""
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock
from tqdm import tqdm
import time

from config import (
    MARKETS, MIN_MARKET_CAP, MIN_PRICE, MIN_TRADING_VALUE,
    EXCLUDE_PREFERRED, EXCLUDE_ETF_ETN, EXCLUDE_SPAC,
    EXCLUDE_NEW_LISTING_MONTHS, OHLCV_PERIOD_DAYS,
    EXCLUDE_ADMIN, EXCLUDE_WARNING
)


def get_recent_trading_date() -> str:
    """최근 거래일 조회"""
    today = datetime.now()
    for i in range(10):
        check_date = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_ticker(check_date, market="KOSPI")
            if df["종가"].sum() > 0:  # 데이터가 있으면
                print(f"  최근 거래일: {check_date}")
                return check_date
        except:
            continue
    return today.strftime("%Y%m%d")


def get_all_market_data() -> pd.DataFrame:
    """전체 종목 데이터 일괄 수집 (최적화)"""
    trading_date = get_recent_trading_date()

    all_data = []

    for market in MARKETS:
        print(f"  {market} 데이터 수집 중...")

        # 전체 종목 시가총액 일괄 조회
        cap_df = stock.get_market_cap_by_ticker(trading_date, market=market)

        # 전체 종목 OHLCV 일괄 조회
        ohlcv_df = stock.get_market_ohlcv_by_ticker(trading_date, market=market)

        # 컬럼명 확인 및 매핑
        print(f"    OHLCV 컬럼: {ohlcv_df.columns.tolist()}")
        print(f"    CAP 컬럼: {cap_df.columns.tolist()}")

        # 종목명 매핑
        tickers = cap_df.index.tolist()

        for ticker in tickers:
            try:
                name = stock.get_market_ticker_name(ticker)

                # 컬럼명 유연하게 처리
                close_col = "종가" if "종가" in ohlcv_df.columns else ohlcv_df.columns[3]
                volume_col = "거래량" if "거래량" in ohlcv_df.columns else ohlcv_df.columns[4]

                row = {
                    "종목코드": ticker,
                    "종목명": name,
                    "시장": market,
                    "현재가": ohlcv_df.loc[ticker, close_col] if ticker in ohlcv_df.index else 0,
                    "시가총액": cap_df.loc[ticker, "시가총액"] if ticker in cap_df.index else 0,
                    "거래량": ohlcv_df.loc[ticker, volume_col] if ticker in ohlcv_df.index else 0,
                    "거래대금": ohlcv_df.loc[ticker, "거래대금"] if ticker in ohlcv_df.index else 0,
                }
                all_data.append(row)
            except:
                continue

    return pd.DataFrame(all_data)


def is_preferred_stock(name: str) -> bool:
    """우선주 여부 확인"""
    if pd.isna(name):
        return False
    return name.endswith("우") or name.endswith("우B") or "우선" in name


def is_etf_etn(name: str) -> bool:
    """ETF/ETN 여부 확인"""
    if pd.isna(name):
        return False
    etf_keywords = ["ETF", "ETN", "레버리지", "인버스", "선물", "액티브", "합성"]
    return any(kw in name for kw in etf_keywords)


def is_spac(name: str) -> bool:
    """스팩 여부 확인"""
    if pd.isna(name):
        return False
    spac_keywords = ["스팩", "SPAC", "기업인수"]
    return any(kw in name for kw in spac_keywords)


def get_administrative_stocks() -> set:
    """관리종목/투자경고 종목 조회 (KRX 정보데이터시스템 기반)"""
    admin_tickers = set()

    try:
        # pykrx에서 제공하는 관리종목 조회 시도
        from pykrx import stock

        # 관리종목은 종목명에 표시되지 않으므로
        # 거래정지 또는 이상 종목 확인
        # 현재 pykrx에서 직접 지원하지 않아 빈 set 반환
        # 추후 KRX API 연동 시 업데이트 필요

    except Exception as e:
        print(f"  관리종목 조회 실패: {e}")

    return admin_tickers


def check_new_listing_bulk(tickers: list, months: int = 6) -> dict:
    """신규상장 여부 일괄 확인"""
    cutoff_date = (datetime.now() - timedelta(days=months * 30)).strftime("%Y%m%d")
    result = {}

    print(f"  신규상장 체크 중... (기준: {cutoff_date})")

    for market in MARKETS:
        try:
            # 과거 시점 종목 리스트
            old_tickers = stock.get_market_ticker_list(cutoff_date, market=market)
            for ticker in old_tickers:
                result[ticker] = False  # 신규상장 아님
        except:
            continue

    # 리스트에 없으면 신규상장
    for ticker in tickers:
        if ticker not in result:
            result[ticker] = True  # 신규상장

    return result


def get_avg_trading_value_bulk(tickers: list, days: int = 20) -> dict:
    """평균 거래대금 일괄 계산"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)

    result = {}

    print(f"  평균 거래대금 계산 중...")

    for market in MARKETS:
        try:
            # 기간 거래대금 일괄 조회
            df = stock.get_market_ohlcv_by_ticker(
                end_date.strftime("%Y%m%d"),
                market=market
            )

            # 최근 20일 데이터 수집
            trading_data = {}
            for i in range(days):
                check_date = (end_date - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    daily_df = stock.get_market_ohlcv_by_ticker(check_date, market=market)
                    for ticker in daily_df.index:
                        if ticker not in trading_data:
                            trading_data[ticker] = []
                        trading_data[ticker].append(daily_df.loc[ticker, "거래대금"])
                except:
                    continue

            # 평균 계산
            for ticker, values in trading_data.items():
                if values:
                    result[ticker] = sum(values) / len(values)

        except Exception as e:
            print(f"  {market} 거래대금 조회 실패: {e}")
            continue

    return result


def filter_stocks_fast(verbose: bool = True) -> pd.DataFrame:
    """종목 필터링 (최적화 버전)"""

    # 1. 전체 데이터 일괄 수집
    print("\n[1/4] 전체 종목 데이터 수집...")
    all_stocks = get_all_market_data()
    print(f"  전체 종목: {len(all_stocks)}개")

    # 2. 기본 필터 (이름 기반)
    print("\n[2/4] 기본 필터 적용...")
    df = all_stocks.copy()

    before = len(df)
    if EXCLUDE_PREFERRED:
        df = df[~df["종목명"].apply(is_preferred_stock)]
        print(f"  우선주 제외: {before} → {len(df)}")

    before = len(df)
    if EXCLUDE_ETF_ETN:
        df = df[~df["종목명"].apply(is_etf_etn)]
        print(f"  ETF/ETN 제외: {before} → {len(df)}")

    before = len(df)
    if EXCLUDE_SPAC:
        df = df[~df["종목명"].apply(is_spac)]
        print(f"  스팩 제외: {before} → {len(df)}")

    # 3. 수치 필터
    print("\n[3/4] 수치 조건 필터...")

    before = len(df)
    df = df[df["시가총액"] >= MIN_MARKET_CAP]
    print(f"  시총 1,500억↑: {before} → {len(df)}")

    before = len(df)
    df = df[df["현재가"] >= MIN_PRICE]
    print(f"  현재가 3만원↑: {before} → {len(df)}")

    # 4. 신규상장 체크
    print("\n[4/4] 신규상장 체크...")
    new_listing = check_new_listing_bulk(df["종목코드"].tolist(), EXCLUDE_NEW_LISTING_MONTHS)

    before = len(df)
    df = df[~df["종목코드"].apply(lambda x: new_listing.get(x, True))]
    print(f"  신규상장 제외: {before} → {len(df)}")

    # 5. 거래대금 필터 (일평균 10억 이상, 당일 기준 5억으로 완화)
    before = len(df)
    df = df[df["거래대금"] >= MIN_TRADING_VALUE / 2]
    print(f"  거래대금 5억↑: {before} → {len(df)}")

    # 결과 정리
    result = df[["종목코드", "종목명", "시장", "현재가", "시가총액", "거래대금"]].copy()
    result = result.sort_values("시가총액", ascending=False).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"최종 필터링 결과: {len(result)}개 종목")
    print(f"{'='*50}")

    return result


def get_ohlcv_data(ticker: str, days: int = None) -> pd.DataFrame:
    """일봉 OHLCV 데이터 수집"""
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
        return df
    except Exception as e:
        print(f"[{ticker}] OHLCV 수집 실패: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 50)
    print("종목 필터링 (최적화 버전)")
    print("=" * 50)

    # 필터링 실행
    filtered_stocks = filter_stocks_fast()

    # 결과 출력
    print("\n[필터링된 종목 TOP 20]")
    print(filtered_stocks.head(20).to_string())

    # 저장
    filtered_stocks.to_csv("filtered_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: filtered_stocks.csv ({len(filtered_stocks)}개 종목)")
