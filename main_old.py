import pandas as pd
from robots.intelligence_robot import IntelligenceRobot
from robots.choice_robot import ChoiceRobot
from robots.trade_robot import TradeRobot
import configparser
import os
import datetime
import sqlite3  # SQLite DB 확인용

# conda activate stock_robots # 이전에 생성한 가상 환경 활성화
# pip install konlpy
# pip install pandas-ta # 기술적 지표 계산용

# pip install pandas, finance-datareader, konlpy, pandas-ta, sqlite3

# KoNLPy 설치 시 주의사항:
# KoNLPy는 자바(Java) 런타임 환경(JRE/JDK)이 필요합니다.
# 먼저 OpenJDK 또는 Oracle JDK를 설치하고, 시스템 환경 변수에 JAVA_HOME을 설정해야 할 수도 있습니다.
# 자세한 설치 방법은 KoNLPy 공식 문서를 참고해주세요.

# conda activate stock_robots # 이전에 생성한 가상 환경 활성화
# pip install Flask
# pip install matplotlib

def display_menu():
    """메인 메뉴를 출력합니다."""
    print("\n" + "=" * 60)
    print("        SW 로봇 기반 주식 투자 시스템 v0.3")
    print("       (고급 분석 및 알고리즘 강화, DB 연동)")
    print("=" * 60)
    print("1. Market Intelligence Robot 실행 (뉴스 수집 및 감성 분석)")
    print("2. Stock Analysis & Choice Robot 실행 (주가/재무/기술적/퀀트 분석)")
    print("3. Trade Robot (모의 거래) 실행")
    print("4. 현재 계좌 현황 및 거래 로그 조회")
    print("5. DB에 저장된 데이터 조회")  # DB 조회 메뉴 추가
    print("0. 종료")
    print("=" * 60)


def main():
    # 설정 파일 로드
    config = configparser.ConfigParser()
    config_file_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.ini')
    config.read(config_file_path, encoding='utf-8')

    # 로봇 인스턴스 생성
    mi_robot = IntelligenceRobot(config_path=config_file_path)
    sa_robot = ChoiceRobot(config_path=config_file_path)
    tr_robot = TradeRobot(config_path=config_file_path)

    # 로봇 간 데이터 전달을 위한 변수
    last_market_sentiment = "중립"  # Market Intelligence -> Stock Analysis
    last_recommended_stock_code = None
    last_recommended_stock_name = None
    last_recommended_stock_price = None  # Stock Analysis -> Trade

    while True:
        display_menu()
        choice = input("메뉴를 선택하세요: ")

        if choice == '1':
            # Market Intelligence Robot
            print("\n[알림] Market Intelligence Robot은 네이버 뉴스에서 데이터를 수집합니다.")
            print("      (크롤링 간격으로 인해 다소 시간이 소요될 수 있습니다.)")
            print("      (KoNLPy 기반의 형태소 분석 및 사전 기반 감성 분석을 수행합니다.)")
            user_input_keyword = input("탐색할 키워드를 입력하세요 (생략 시 기본값 중 랜덤 선택): ").strip()

            keyword_for_robot = None
            keyword_for_filename = "random_keyword"  # Default for filename if no specific keyword is used

            if user_input_keyword:
                keyword_for_robot = user_input_keyword
                keyword_for_filename = "".join(
                    char for char in user_input_keyword if char.isalnum() or char.isspace() or char == '_')
                keyword_for_filename = keyword_for_filename.replace(" ", "_").strip('_')
                if not keyword_for_filename:
                    keyword_for_filename = "default_keyword"

            time_period = input("검색 기간을 입력하세요 (예: 최근 1일, 최근 1주, 최근 1개월 / 생략 시 '최근 1주'): ").strip()
            if not time_period:
                time_period = '최근 1주'

            last_market_sentiment, _ = mi_robot.run_intelligence(keyword_for_robot, time_period, keyword_for_filename)
            print(
                f"\n[시스템 메시지] Market Intelligence 결과가 Stock Analysis Robot에 전달되었습니다. (시장 감성: {last_market_sentiment})")
            print(f"수집된 뉴스 데이터는 '{mi_robot.db_path}'의 'news_data' 테이블과 해당 경로에 저장됩니다.")

        elif choice == '2':
            # Stock Analysis & Choice Robot
            print(f"\n[Stock Analysis Robot] 현재 시장 감성: {last_market_sentiment}")
            print("      (KRX 상장 종목 데이터 및 시뮬레이션 재무/기술적 데이터를 사용합니다.)")
            print("      (DART API 키를 config/settings.ini에 설정하면 실제 재무 데이터 연동 가능합니다.)")

            print("\n적용할 투자 전략을 선택하세요:")
            print("1. 가치 투자 (저PER, 고ROE, 저부채)")
            print("2. 모멘텀 투자 (최근 주가 상승률)")
            print("3. 기술적 분석 (이평선, RSI, MACD 신호)")
            strategy_choice = input("선택 (1/2/3, 생략 시 1): ").strip()
            strategy_type = "가치투자"
            if strategy_choice == '2':
                strategy_type = "모멘텀"
            elif strategy_choice == '3':
                strategy_type = "기술적분석"

            try:
                min_per_str = input(f"최대 PER을 입력하세요 (현재 기본값: {sa_robot.min_per_default} / 생략 시 기본값): ").strip()
                min_per = int(min_per_str) if min_per_str else None

                max_debt_ratio_str = input(
                    f"최대 부채비율(%)을 입력하세요 (현재 기본값: {sa_robot.max_debt_ratio_default} / 생략 시 기본값): ").strip()
                max_debt_ratio = int(max_debt_ratio_str) if max_debt_ratio_str else None

                min_roe_str = input(f"최소 ROE(%)를 입력하세요 (현재 기본값: {sa_robot.min_roe_default} / 생략 시 기본값): ").strip()
                min_roe = int(min_roe_str) if min_roe_str else None

            except ValueError:
                print("-> 오류: 입력값은 숫자만 입력해주세요. 기본값이 적용됩니다.")
                min_per = None
                max_debt_ratio = None
                min_roe = None

            code, name, price, _ = sa_robot.analyze_and_recommend(
                last_market_sentiment,
                min_per=min_per,
                max_debt_ratio=max_debt_ratio,
                min_roe=min_roe,
                strategy_type=strategy_type  # 전략 유형 전달
            )

            if code:
                last_recommended_stock_code = code
                last_recommended_stock_name = name
                last_recommended_stock_price = price
                print(
                    f"\n[시스템 메시지] Stock Analysis 결과가 Trade Robot에 전달되었습니다. (추천 종목: {last_recommended_stock_name}, 현재가: {last_recommended_stock_price:,}원)")
                print(f"추천 종목 데이터는 '{sa_robot.db_path}'의 'stock_recommendations' 테이블과 해당 경로에 저장됩니다.")
            else:
                print("\n[시스템 메시지] Stock Analysis 로봇이 추천 종목을 찾지 못했습니다.")
                last_recommended_stock_code = None
                last_recommended_stock_name = None
                last_recommended_stock_price = None

        elif choice == '3':
            # Trade Robot (모의 거래)
            if last_recommended_stock_code:
                print(
                    f"\n[Trade Robot] 추천 종목: {last_recommended_stock_name} ({last_recommended_stock_code}), 현재가: {last_recommended_stock_price:,}원")
                symbol_code = last_recommended_stock_code
                symbol_name = last_recommended_stock_name
                current_price_for_trade = last_recommended_stock_price
            else:
                print("\n[Trade Robot] 추천 종목이 없습니다. 직접 입력하거나 Stock Analysis Robot을 먼저 실행하세요.")
                symbol_name = input("거래할 종목명을 입력하세요: ").strip()
                symbol_code = input("거래할 종목 코드를 입력하세요: ").strip()
                try:
                    current_price_for_trade_str = input("현재 시장가(시뮬레이션)를 입력하세요: ").strip()
                    current_price_for_trade = int(current_price_for_trade_str)
                except ValueError:
                    print("-> 오류: 현재가는 숫자로 입력해주세요. 기본값을 사용합니다.")
                    current_price_for_trade = 10000  # 임의의 기본값

            order_type = input("주문 유형 (매수/매도): ").strip()
            price_type = input("가격 유형 (시장가/지정가): ").strip()

            order_price = 0
            if price_type == '지정가':
                try:
                    order_price_str = input("지정가격을 입력하세요: ").strip()
                    order_price = int(order_price_str)
                except ValueError:
                    print("-> 오류: 지정가격은 숫자로 입력해주세요. 시장가로 처리됩니다.")
                    price_type = '시장가'

            try:
                quantity_str = input("수량을 입력하세요: ").strip()
                quantity = int(quantity_str)
            except ValueError:
                print("-> 오류: 수량은 숫자로 입력해주세요. 주문 실패.")
                quantity = 0

            if order_type in ['매수', '매도'] and quantity > 0:
                tr_robot.execute_order(
                    symbol_code, symbol_name, order_type, price_type,
                    order_price, quantity, current_market_price=current_price_for_trade
                )
            else:
                print("-> 오류: 유효하지 않은 주문 유형 또는 수량입니다.")

        elif choice == '4':
            # 현재 계좌 현황 및 거래 로그 조회
            tr_robot.display_portfolio()
            print("\n[전체 거래 로그]")
            trade_df = tr_robot.get_trade_log()
            if not trade_df.empty:
                print(trade_df.to_string(index=False))
                print(f"-> 거래 로그는 '{tr_robot.db_path}'의 'trade_log' 테이블과 해당 경로에 저장됩니다.")
            else:
                print("-> 기록된 거래 로그가 없습니다.")

        elif choice == '5':
            # DB에 저장된 데이터 조회 (편의 기능)
            print("\n--- [데이터베이스 조회] ---")
            print("조회할 테이블을 선택하세요:")
            print("1. news_data (뉴스 데이터)")
            print("2. stock_recommendations (추천 종목 데이터)")
            print("3. trade_log (거래 로그)")
            db_choice = input("선택 (1/2/3): ").strip()

            conn = None
            try:
                conn = sqlite3.connect(mi_robot.db_path)  # 모든 로봇이 동일한 DB 파일을 사용
                if db_choice == '1':
                    df_db = pd.read_sql_query("SELECT * FROM news_data ORDER BY crawled_at DESC LIMIT 10", conn)
                    print("\n[뉴스 데이터 (최신 10건)]")
                elif db_choice == '2':
                    df_db = pd.read_sql_query(
                        "SELECT * FROM stock_recommendations ORDER BY recommend_date DESC LIMIT 10", conn)
                    print("\n[추천 종목 데이터 (최신 10건)]")
                elif db_choice == '3':
                    df_db = pd.read_sql_query("SELECT * FROM trade_log ORDER BY trade_datetime DESC LIMIT 10", conn)
                    print("\n[거래 로그 (최신 10건)]")
                else:
                    print("-> 잘못된 테이블 선택입니다.")
                    df_db = pd.DataFrame()

                if not df_db.empty:
                    print(df_db.to_string())
                else:
                    print("-> 데이터가 없습니다.")
            except Exception as e:
                print(f"-> DB 조회 중 오류 발생: {e}")
            finally:
                if conn:
                    conn.close()

        elif choice == '0':
            print("SW 로봇 시스템을 종료합니다. 안녕히 계세요!")
            break
        else:
            print("잘못된 메뉴 선택입니다. 다시 시도해주세요.")

        input("\n계속하려면 Enter 키을 누르세요...")


if __name__ == "__main__":
    # 프로젝트 폴더 구조 확인 및 생성 (없으면)
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, 'data')
    config_dir = os.path.join(project_root, 'config')
    robots_dir = os.path.join(project_root, 'robots')

    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    if not os.path.exists(robots_dir): os.makedirs(robots_dir)

    # settings.ini 파일이 없으면 기본값으로 생성
    config_file_path_for_init = os.path.join(config_dir, 'settings.ini')
    if not os.path.exists(config_file_path_for_init):
        print("config/settings.ini 파일이 없어 기본값을 생성합니다.")
        default_config = configparser.ConfigParser()
        default_config['MARKET_INTELLIGENCE'] = {
            'NEWS_SOURCES': "['네이버뉴스', '다음금융']",
            'KEYWORD_LIST': "['AI 반도체', '2차전지', '인플레이션', '미중 갈등', '금리인상', '테마주']",
            'NAVER_NEWS_BASE_URL': "https://search.naver.com/search.naver?where=news&sm=tab_pge&query={keyword}&sort=0&photo=0&field=0&pd=0&ds=&de=&clusterid=&mynews=0&office_type=0&office_section_code=0&seM=false&res_fr=0&res_to=0&ie=utf8&spq=0&start={start_page}",
            'POSITIVE_WORDS': "호재, 상승, 증가, 강세, 기대, 혁신, 성장, 수혜, 긍정, 확대, 회복, 개선, 돌파, 최고, 낙관, 유리, 강화",
            'NEGATIVE_WORDS': "악재, 하락, 감소, 약세, 우려, 규제, 부정, 축소, 위기, 둔화, 조정, 경고, 압박, 불리, 감소, 불안, 난관"
        }
        default_config['STOCK_ANALYSIS'] = {
            'MIN_PER_DEFAULT': '20',
            'MAX_DEBT_RATIO_DEFAULT': '100',
            'MIN_ROE_DEFAULT': '10'
        }
        default_config['TRADE'] = {
            'VIRTUAL_INITIAL_CASH': '10000000'
        }
        default_config['API_KEYS'] = {
            'DART_API_KEY': ''
        }
        with open(config_file_path_for_init, 'w', encoding='utf-8') as configfile:
            default_config.write(configfile)

    # robots 폴더에 __init__.py 파일이 없으면 생성
    init_file_path = os.path.join(robots_dir, '__init__.py')
    if not os.path.exists(init_file_path):
        with open(init_file_path, 'w') as f:
            f.write('')  # 비어있는 __init__.py 파일 생성

    main()