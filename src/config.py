"""
역헤드앤숄더 패턴 스캐너 설정
"""
from pathlib import Path

# ========== 경로 설정 ==========
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"

# ========== 패턴 조건 ==========
PATTERN_MIN_DAYS = 60      # 패턴 최소 기간 (3개월 * 20거래일) - 완화
PATTERN_MAX_DAYS = 200     # 패턴 최대 기간 (10개월 * 20거래일) - 완화
MIN_TROUGH_INTERVAL = 15   # 저점 간 최소 간격 (3주) - 완화

# 저점 탐지
EXTREMA_ORDER = 5          # scipy argrelextrema order - 완화 (더 많은 저점 탐지)

# 어깨 조건
SHOULDER_PRICE_TOLERANCE = 0.10    # 어깨 가격 오차 ±10%
SHOULDER_TIME_TOLERANCE = 0.50     # 어깨 시간 대칭 50% 이내 (수정)

# 머리 깊이 조건
MIN_HEAD_DEPTH = 0.10              # 머리가 어깨 평균 대비 최소 10% 낮아야 함

# 넥라인 기울기 조건
NECKLINE_SLOPE_THRESHOLD = -0.10   # 우하향 넥라인 -10% 이상 하락 시 제외
NECKLINE_SLOPE_PENALTY = 5         # 우하향 넥라인 감점 (5점)

# 선행 하락
PRIOR_DECLINE_PCT = -0.15          # 패턴 시작 전 고점 대비 -15%

# ========== 필터링 조건 ==========
MARKETS = ["KOSPI", "KOSDAQ"]
MIN_MARKET_CAP = 1500_0000_0000    # 시총 1,500억 이상
MIN_PRICE = 30000                   # 현재가 3만원 이상
MIN_TRADING_VALUE = 10_0000_0000   # 일평균 거래대금 10억 이상

# 제외 조건
EXCLUDE_ADMIN = True               # 관리종목 제외
EXCLUDE_WARNING = True             # 투자경고 제외
EXCLUDE_PREFERRED = True           # 우선주 제외
EXCLUDE_ETF_ETN = True             # ETF/ETN 제외
EXCLUDE_SPAC = True                # 스팩 제외
EXCLUDE_NEW_LISTING_MONTHS = 6     # 신규상장 6개월 이내 제외

# ========== 거래량 가점 ==========
VOLUME_BONUS_MULTIPLIER = 1.5      # 20일 평균 대비 배수
VOLUME_BONUS_SCORE = 10            # 가점 점수

# ========== 상승 직전 종목 필터 (A방식: 백테스트 검증) ==========
MIN_UPSIDE_TO_NECKLINE = 0.20      # 넥라인까지 최소 20% 상승여력
MAX_RISE_FROM_HEAD = 0.30          # 머리(바닥) 대비 최대 30% 상승
EXCLUDE_NECKLINE_BREAKOUT = True   # 넥라인 돌파 종목 제외

# ========== 진입 시점 상태 (A방식) ==========
PATTERN_STATES = {
    "EARLY_ENTRY": "초기진입",       # 머리 대비 15% 이내 상승 ★최우선
    "RISING": "상승중",              # 머리 대비 15~30% 상승
    "NEAR_BREAKOUT": "돌파임박"      # 넥라인까지 10% 이내
}

# ========== 출력 설정 ==========
TOP_N_CHARTS = 10                  # 상위 차트 출력 개수
RESULT_FILE = "results.csv"

# ========== 데이터 수집 ==========
OHLCV_PERIOD_DAYS = 250            # 약 1년치 데이터 수집
