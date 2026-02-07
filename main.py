"""
역헤드앤숄더 패턴 스캐너 - 메인 실행 파일
"""
import sys
import os

# src 디렉토리를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime

from config import OUTPUT_DIR, RESULT_FILE
from data_collector import filter_stocks_fast
from pattern_detector import scan_stocks
from chart_visualizer import generate_top_charts, ensure_dirs


def main():
    print("=" * 60)
    print("역헤드앤숄더 패턴 스캐너")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 종목 필터링
    print("\n[STEP 1] 종목 필터링")
    print("-" * 40)
    filtered_stocks = filter_stocks_fast()

    if len(filtered_stocks) == 0:
        print("필터링된 종목이 없습니다.")
        return

    # 2. 패턴 스캔
    print("\n[STEP 2] 패턴 스캔")
    print("-" * 40)
    results = scan_stocks(filtered_stocks)

    if len(results) == 0:
        print("패턴이 발견된 종목이 없습니다.")
        return

    # 3. 결과 저장
    ensure_dirs()
    result_path = OUTPUT_DIR / RESULT_FILE
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {result_path}")

    # 4. 결과 요약
    print("\n[STEP 3] 결과 요약")
    print("-" * 40)
    print(f"패턴 발견 종목: {len(results)}개")
    print(f"\n패턴 상태별 분포:")
    print(results["패턴상태"].value_counts().to_string())

    print(f"\n신뢰도 TOP 10:")
    top10 = results.head(10)[["종목명", "현재가", "패턴상태", "예상수익률", "신뢰도점수"]]
    print(top10.to_string())

    # 5. 차트 생성
    print("\n[STEP 4] 차트 생성")
    print("-" * 40)
    charts = generate_top_charts(results)

    # 6. 완료
    print("\n" + "=" * 60)
    print("스캔 완료!")
    print(f"  - 결과 파일: {result_path}")
    print(f"  - 차트 파일: {len(charts)}개 생성")
    print("=" * 60)


if __name__ == "__main__":
    main()
