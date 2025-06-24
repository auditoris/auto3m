# stock_robot_project/app.py

# v1.0 - 완성 및 최적화
# pip install dart-fss

# v0.5 - 자동화 및 실시간 시스템
# conda activate stock_robots # 이전에 생성한 가상 환경 활성화
# pip install APScheduler

# v0.4 - 사용자 UI 및 시각화
# conda activate stock_robots # 이전에 생성한 가상 환경 활성화
# pip install Flask
# pip install matplotlib
# http://127.0.0.1:5000/

# v0.3 -고급 분석 및 알고리즘 강화
# pip install pandas, finance-datareader, konlpy, pandas-ta, sqlite3
# KoNLPy 설치 시 주의사항:
# KoNLPy는 자바(Java) 런타임 환경(JRE/JDK)이 필요합니다.
# 먼저 OpenJDK 또는 Oracle JDK를 설치하고, 시스템 환경 변수에 JAVA_HOME을 설정해야 할 수도 있습니다.
# 자세한 설치 방법은 KoNLPy 공식 문서를 참고해주세요.

# stock_robot_project/app.py

# stock_robot_project/app.py

from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import datetime
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import configparser
import sqlite3
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import random  # Market Intelligence 키워드 랜덤 선택용

# 로봇 클래스 임포트
from robots.intelligence_robot import IntelligenceRobot
from robots.choice_robot import ChoiceRobot
from robots.trade_robot import TradeRobot

app = Flask(__name__)

# 로깅 설정
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(log_dir, 'robot_system.log'), encoding='utf-8'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Matplotlib 한글 폰트 설정 (이전과 동일)
try:
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    malgun_font = next((font for font in font_path if 'malgun' in font.lower()), None)
    if malgun_font:
        fm.fontManager.addfont(malgun_font)
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        logger.warning("경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.error(f"폰트 설정 오류: {e}. 기본 폰트로 진행합니다.")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# 로봇 인스턴스 초기화
project_root = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(project_root, 'config', 'settings.ini')

mi_robot = IntelligenceRobot(config_path=config_file_path)
sa_robot = ChoiceRobot(config_path=config_file_path)
tr_robot = TradeRobot(config_path=config_file_path)

# 전역 변수 (또는 Flask Application Context에 저장, 여기서는 간단하게 전역 사용)
# 로봇 실행 상태 및 최신 결과 저장
robot_status = {
    'mi_status': '대기 중', 'mi_last_run': 'N/A', 'mi_last_sentiment': '중립',
    'sa_status': '대기 중', 'sa_last_run': 'N/A', 'sa_last_recommendation': '없음',
    'tr_status': '대기 중', 'tr_last_run': 'N/A', 'tr_last_action': '없음',
    'current_market_sentiment': '중립',  # MI 로봇이 업데이트할 종합 시장 감성
    'last_recommended_stock_code': None,
    'last_recommended_stock_name': None,
    'last_recommended_stock_price': None
}


def update_robot_status(robot_key, status, last_run, **kwargs):
    """로봇 상태 업데이트 헬퍼 함수"""
    robot_status[f'{robot_key}_status'] = status
    robot_status[f'{robot_key}_last_run'] = last_run.strftime('%Y-%m-%d %H:%M:%S')
    for key, value in kwargs.items():
        robot_status[key] = value
    logger.info(f"로봇 상태 업데이트: {robot_key} - {status} at {last_run}")


def run_market_intelligence_scheduled():
    """스케줄러에 의해 자동 실행될 시장 인텔리전스 로봇 함수"""
    current_time = datetime.datetime.now()
    update_robot_status('mi', '실행 중', current_time)
    logger.info("-> Market Intelligence Robot 자동 실행 시작.")
    try:
        config = configparser.ConfigParser()
        config.read(config_file_path, encoding='utf-8')
        keyword_list_str = config['MARKET_INTELLIGENCE'].get('KEYWORD_LIST', "['AI 반도체']")
        keyword_list = eval(keyword_list_str)

        keyword_to_use = random.choice(keyword_list)
        filename_keyword = "".join(char for char in keyword_to_use if char.isalnum() or char.isspace() or char == '_')
        filename_keyword = filename_keyword.replace(" ", "_").strip('_')
        if not filename_keyword:
            filename_keyword = "default_keyword"

        sentiment, news_data_df = mi_robot.run_intelligence(keyword=keyword_to_use, time_period='최근 1일',
                                                            filename_keyword=filename_keyword)

        update_robot_status('mi', '완료', current_time,
                            mi_last_sentiment=sentiment,
                            current_market_sentiment=sentiment)  # 전역 시장 감성 업데이트
        logger.info(f"-> Market Intelligence Robot 실행 완료. 종합 시장 감성: {sentiment}")
    except Exception as e:
        update_robot_status('mi', '오류 발생', current_time)
        logger.error(f"Market Intelligence Robot 자동 실행 중 오류 발생: {e}")


def run_stock_analysis_scheduled():
    """스케줄러에 의해 자동 실행될 주식 분석 로봇 함수"""
    current_time = datetime.datetime.now()
    update_robot_status('sa', '실행 중', current_time)
    logger.info("-> Stock Analysis Robot 자동 실행 시작.")
    try:
        # 시장 감성을 기반으로 주식 분석 및 추천 (mi 로봇이 업데이트한 최신 감성 사용)
        code, name, price, _ = sa_robot.analyze_and_recommend(robot_status['current_market_sentiment'])
        if code:
            update_robot_status('sa', '완료', current_time,
                                sa_last_recommendation=f"{name} ({code})",
                                last_recommended_stock_code=code,
                                last_recommended_stock_name=name,
                                last_recommended_stock_price=price)
            logger.info(f"-> Stock Analysis Robot 실행 완료. 추천 종목: {name} ({code}), 현재가: {price}")
        else:
            update_robot_status('sa', '완료 (추천 없음)', current_time, sa_last_recommendation='없음')
            logger.info("-> Stock Analysis Robot이 자동 실행에서 추천 종목을 찾지 못했습니다.")
            # 추천 종목 초기화
            robot_status['last_recommended_stock_code'] = None
            robot_status['last_recommended_stock_name'] = None
            robot_status['last_recommended_stock_price'] = None
    except Exception as e:
        update_robot_status('sa', '오류 발생', current_time)
        logger.error(f"Stock Analysis Robot 자동 실행 중 오류 발생: {e}")


def run_trade_robot_scheduled():
    """스케줄러에 의해 자동 실행될 모의 거래 로봇 함수 (고도화된 자동 매매 로직)"""
    current_time = datetime.datetime.now()
    update_robot_status('tr', '실행 중', current_time)
    logger.info("-> Trade Robot 자동 실행 시작 (고도화된 자동 매매 로직).")
    action_taken = "없음"
    try:
        current_sentiment = robot_status['current_market_sentiment']
        rec_code = robot_status['last_recommended_stock_code']
        rec_name = robot_status['last_recommended_stock_name']
        rec_price = robot_status['last_recommended_stock_price']

        # --- 자동 매매 로직 개선 (예시) ---
        # 1. 긍정적 시장 감성 + 추천 종목이 있으면 매수 고려
        if current_sentiment == "긍정적" and rec_code and rec_price:
            # 이미 보유 중인지 확인
            is_holding = not tr_robot.portfolio[tr_robot.portfolio['종목코드'] == rec_code].empty
            if not is_holding:
                # 가상 현금의 10%를 사용하여 매수 (예시)
                amount_to_invest = tr_robot.virtual_cash * 0.1
                quantity_to_buy = int(amount_to_invest / rec_price)
                if quantity_to_buy > 0:
                    tr_robot.execute_order(rec_code, rec_name, '매수', '시장가', rec_price, quantity_to_buy, rec_price)
                    action_taken = f"매수: {rec_name} {quantity_to_buy}주"
                    logger.info(f"   자동 매매: 시장 긍정적, 신규 추천 종목 {rec_name} {quantity_to_buy}주 매수 완료.")
                else:
                    logger.info(f"   자동 매매: 매수할 수량이 부족합니다. 현금: {tr_robot.virtual_cash:,}원")
            else:
                logger.info(f"   자동 매매: {rec_name} 이미 보유 중, 추가 매수 보류.")

        # 2. 부정적 시장 감성 또는 포트폴리오 평가 (손절/익절 고려)
        elif current_sentiment == "부정적" or not tr_robot.portfolio.empty:
            if current_sentiment == "부정적":
                logger.info("   자동 매매: 시장이 부정적이어서 포트폴리오 정리 고려.")

            # 보유 종목 중 손실이 큰 종목부터 매도 (예시)
            if not tr_robot.portfolio.empty:
                # 현재 포트폴리오를 업데이트하고 평가손익 계산 (실시간 가격 반영 필요)
                # 여기서는 단순화를 위해 보유 종목 중 첫 번째 종목을 대상으로 함
                # 실제로는 각 종목의 현재가를 조회하여 평가 손익을 계산하고, 전략에 따라 매도 결정.
                # (V1.0에서는 FinaDnceDataReader로 실시간 주가 조회를 통해 평가손익 업데이트 로직을 TradeRobot에 추가할 수 있음)

                # 임시로 portfolio DataFrame을 복사하고 실시간 가격을 업데이트한다고 가정
                temp_portfolio = tr_robot.portfolio.copy()

                # 각 종목의 현재가 업데이트 (FinanceDataReader 사용, 시간 지연 가능)
                updated_portfolio_data = []
                for idx, row in temp_portfolio.iterrows():
                    stock_code = row['종목코드']
                    try:
                        latest_price_df = sa_robot._get_stock_price_data(stock_code, period='1d')  # 최신 1일 데이터만
                        if latest_price_df is not None and not latest_price_df.empty:
                            latest_price = latest_price_df['Close'].iloc[-1]
                            row['현재가'] = latest_price
                            row['평가손익'] = (latest_price - row['매입단가']) * row['보유수량']
                            row['수익률'] = (latest_price / row['매입단가'] - 1) * 100
                        else:
                            logger.warning(f"   자동 매매: {stock_code}의 최신 가격을 가져올 수 없습니다.")
                    except Exception as err:
                        logger.warning(f"   자동 매매: {stock_code} 가격 업데이트 중 오류: {err}")
                    updated_portfolio_data.append(row)

                temp_portfolio = pd.DataFrame(updated_portfolio_data)

                # 손실이 5% 이상인 종목 또는 시장이 부정적일 때 모든 종목 매도 (예시 전략)
                stocks_to_sell = temp_portfolio[(temp_portfolio['수익률'] <= -5) | (current_sentiment == "부정적")]

                if not stocks_to_sell.empty:
                    for idx, stock_to_sell in stocks_to_sell.iterrows():
                        sell_code = stock_to_sell['종목코드']
                        sell_name = stock_to_sell['종목명']
                        sell_quantity = stock_to_sell['보유수량']
                        sell_price_at_action = stock_to_sell['현재가']  # 업데이트된 현재가 사용

                        if sell_quantity > 0:
                            tr_robot.execute_order(sell_code, sell_name, '매도', '시장가', sell_price_at_action,
                                                   sell_quantity, sell_price_at_action)
                            action_taken = f"매도: {sell_name} {sell_quantity}주 (수익률: {stock_to_sell['수익률']:.2f}%)"
                            logger.info(
                                f"   자동 매매: {sell_name} {sell_quantity}주 시장가 매도 주문 완료. (수익률: {stock_to_sell['수익률']:.2f}%)")
                else:
                    logger.info("   자동 매매: 매도할 종목이 없거나, 매도 조건에 맞는 종목이 없습니다.")
            else:
                logger.info("   자동 매매: 현재 보유 종목이 없습니다.")

    except Exception as e:
        action_taken = f"오류 발생: {e}"
        update_robot_status('tr', '오류 발생', current_time)
        logger.error(f"Trade Robot 자동 실행 중 오류 발생: {e}")
    finally:
        update_robot_status('tr', '완료', current_time, tr_last_action=action_taken)


# 스케줄러 초기화
scheduler = BackgroundScheduler()

# 로봇 실행 스케줄 정의 (V0.5와 동일)
# 실제 운영 시에는 시장 개장 시간 등을 고려하여 스케줄을 조정합니다.
scheduler.add_job(run_market_intelligence_scheduled, 'interval', minutes=60, id='mi_robot_job')
scheduler.add_job(run_stock_analysis_scheduled, 'interval', minutes=120, id='sa_robot_job')
scheduler.add_job(run_trade_robot_scheduled, 'interval', minutes=180, id='tr_robot_job')


# Flask 앱이 시작될 때 스케줄러 시작
@app.before_first_request
def start_scheduler():
    logger.info("-> 스케줄러 시작.")
    scheduler.start()
    # 앱 시작 시 초기 로봇 상태 설정 (UI 표시용)
    robot_status['mi_last_run'] = 'N/A'
    robot_status['sa_last_run'] = 'N/A'
    robot_status['tr_last_run'] = 'N/A'


# Flask 앱이 종료될 때 스케줄러 종료
atexit.register(lambda: scheduler.shutdown(wait=False))
logger.info("-> 스케줄러 종료 핸들러 등록 완료.")


# --- 웹 UI 라우트 (robot_status를 템플릿에 전달하도록 수정) ---

@app.route('/')
def index():
    """홈 페이지"""
    return render_template('index.html', robot_status=robot_status)


@app.route('/market_intelligence', methods=['GET', 'POST'])
def market_intelligence():
    """시장 인텔리전스 로봇 페이지"""
    news_data = pd.DataFrame()
    sentiment_graph_url = None

    if request.method == 'POST':
        keyword = request.form.get('keyword', '').strip()
        time_period = request.form.get('time_period', '최근 1주').strip()

        filename_keyword = "random_keyword"
        if keyword:
            filename_keyword = "".join(char for char in keyword if char.isalnum() or char.isspace() or char == '_')
            filename_keyword = filename_keyword.replace(" ", "_").strip('_')
            if not filename_keyword:
                filename_keyword = "default_keyword"

        current_time = datetime.datetime.now()
        update_robot_status('mi', '실행 중 (수동)', current_time)
        try:
            current_sentiment, news_data = mi_robot.run_intelligence(keyword, time_period, filename_keyword)
            update_robot_status('mi', '완료 (수동)', current_time,
                                mi_last_sentiment=current_sentiment,
                                current_market_sentiment=current_sentiment)  # 전역 시장 감성 업데이트
            logger.info(f"-> Market Intelligence Robot 수동 실행 완료. 종합 시장 감성: {current_sentiment}")
        except Exception as e:
            update_robot_status('mi', '오류 발생 (수동)', current_time)
            logger.error(f"Market Intelligence Robot 수동 실행 중 오류 발생: {e}")

        # 감성 분포 그래프 생성
        if not news_data.empty:
            sentiment_counts = news_data['감성'].value_counts()
            if not sentiment_counts.empty:
                plt.figure(figsize=(8, 6))
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                                      colors=['#ff9999', '#66b3ff', '#99ff99'])
                plt.title(f'{keyword} 관련 뉴스 감성 분포')
                plt.ylabel('')

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                sentiment_graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()

    # DB에서 최신 데이터 조회 (자동/수동 실행 결과 모두 반영)
    conn = sqlite3.connect(mi_robot.db_path)
    news_data_db = pd.read_sql_query("SELECT * FROM news_data ORDER BY crawled_at DESC LIMIT 20", conn)
    conn.close()
    news_data = news_data_db.rename(columns={
        'title': '제목', 'link': '링크', 'sentiment': '감성',
        'source': '출처', 'news_date': '날짜', 'keyword': '키워드',
        'crawled_at': '크롤링_시간'
    })

    return render_template('market_intelligence.html',
                           robot_status=robot_status,  # 상태 전달
                           last_market_sentiment=robot_status['current_market_sentiment'],
                           news_data=news_data.to_html(classes='table table-striped table-bordered',
                                                       escape=False) if not news_data.empty else None,
                           sentiment_graph_url=sentiment_graph_url)


@app.route('/stock_analysis', methods=['GET', 'POST'])
def stock_analysis():
    """주식 분석 및 추천 로봇 페이지"""
    recommendations_df = pd.DataFrame()
    stock_chart_url = None

    # DB에서 현재 추천 종목 데이터 로드
    conn = sqlite3.connect(sa_robot.db_path)
    recommendations_db = pd.read_sql_query("SELECT * FROM stock_recommendations ORDER BY recommend_date DESC LIMIT 5",
                                           conn)
    conn.close()
    if not recommendations_db.empty:
        recommendations_df = recommendations_db.rename(columns={
            'stock_code': '종목코드', 'stock_name': '추천종목', 'current_price': '현재가',
            'per': 'PER', 'roe': 'ROE', 'debt_ratio': '부채비율',
            'sector': '업종', 'recommend_reason': '추천사유', 'expected_return': '예상수익률(시뮬)',
            'recommend_date': '추천일자'
        })

    if request.method == 'POST':
        strategy_type = request.form.get('strategy_type', '가치투자')
        min_per = request.form.get('min_per')
        max_debt_ratio = request.form.get('max_debt_ratio')
        min_roe = request.form.get('min_roe')

        min_per = int(min_per) if min_per else None
        max_debt_ratio = int(max_debt_ratio) if max_debt_ratio else None
        min_roe = int(min_roe) if min_roe else None

        current_time = datetime.datetime.now()
        update_robot_status('sa', '실행 중 (수동)', current_time)
        try:
            code, name, price, new_recommendations_df = sa_robot.analyze_and_recommend(
                robot_status['current_market_sentiment'],  # 전역 변수에서 최신 시장 감성 사용
                min_per=min_per,
                max_debt_ratio=max_debt_ratio,
                min_roe=min_roe,
                strategy_type=strategy_type
            )
            if code:
                update_robot_status('sa', '완료 (수동)', current_time,
                                    sa_last_recommendation=f"{name} ({code})",
                                    last_recommended_stock_code=code,
                                    last_recommended_stock_name=name,
                                    last_recommended_stock_price=price)
                logger.info(f"-> Stock Analysis Robot 수동 실행 완료. 추천 종목: {name} ({code}), 현재가: {price}")
            else:
                update_robot_status('sa', '완료 (추천 없음 - 수동)', current_time, sa_last_recommendation='없음')
                logger.info("-> Stock Analysis Robot이 추천 종목을 찾지 못했습니다 (수동 실행).")
                robot_status['last_recommended_stock_code'] = None
                robot_status['last_recommended_stock_name'] = None
                robot_status['last_recommended_stock_price'] = None

            conn = sqlite3.connect(sa_robot.db_path)
            recommendations_db = pd.read_sql_query(
                "SELECT * FROM stock_recommendations ORDER BY recommend_date DESC LIMIT 5", conn)
            conn.close()
            recommendations_df = recommendations_db.rename(columns={
                'stock_code': '종목코드', 'stock_name': '추천종목', 'current_price': '현재가',
                'per': 'PER', 'roe': 'ROE', 'debt_ratio': '부채비율',
                'sector': '업종', 'recommend_reason': '추천사유', 'expected_return': '예상수익률(시뮬)',
                'recommend_date': '추천일자'
            })

            # 추천된 첫 번째 종목의 주가 차트 생성
            if robot_status['last_recommended_stock_code']:
                df_price = sa_robot._get_stock_price_data(robot_status['last_recommended_stock_code'], start_date=(
                            datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d'))
                if df_price is not None and not df_price.empty:
                    df_price_with_ta = sa_robot._calculate_technical_indicators(df_price.copy())
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_price_with_ta.index, df_price_with_ta['Close'], label='종가', color='blue')
                    plt.plot(df_price_with_ta.index, df_price_with_ta['MA_20'], label='MA(20)', color='red')
                    plt.plot(df_price_with_ta.index, df_price_with_ta['MA_60'], label='MA(60)', color='green')

                    if 'BB_upper' in df_price_with_ta.columns and df_price_with_ta['BB_upper'].any():
                        plt.plot(df_price_with_ta.index, df_price_with_ta['BB_upper'], label='BB 상단', color='purple',
                                 linestyle='--')
                        plt.plot(df_price_with_ta.index, df_price_with_ta['BB_lower'], label='BB 하단', color='orange',
                                 linestyle='--')

                    plt.title(
                        f"{robot_status['last_recommended_stock_name']} ({robot_status['last_recommended_stock_code']}) 주가 및 기술적 지표")
                    plt.xlabel('날짜')
                    plt.ylabel('가격')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    stock_chart_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
        except Exception as e:
            update_robot_status('sa', '오류 발생 (수동)', current_time)
            logger.error(f"Stock Analysis Robot 수동 실행 중 오류 발생: {e}")

    return render_template('stock_analysis.html',
                           robot_status=robot_status,  # 상태 전달
                           last_market_sentiment=robot_status['current_market_sentiment'],
                           recommendations=recommendations_df.to_html(classes='table table-striped table-bordered',
                                                                      index=False,
                                                                      escape=False) if not recommendations_df.empty else None,
                           stock_chart_url=stock_chart_url,
                           min_per_default=sa_robot.min_per_default,
                           max_debt_ratio_default=sa_robot.max_debt_ratio_default,
                           min_roe_default=sa_robot.min_roe_default)


@app.route('/trade', methods=['GET', 'POST'])
def trade():
    """모의 거래 로봇 페이지"""

    portfolio_df = tr_robot.portfolio.copy()
    cash_balance = tr_robot.virtual_cash
    trade_log_df = tr_robot.get_trade_log()

    if request.method == 'POST':
        symbol_code = request.form.get('symbol_code', '').strip()
        symbol_name = request.form.get('symbol_name', '').strip()
        order_type = request.form.get('order_type', '').strip()
        price_type = request.form.get('price_type', '').strip()
        order_price_str = request.form.get('order_price', '').strip()
        quantity_str = request.form.get('quantity', '').strip()

        order_price = int(order_price_str) if order_price_str.isdigit() else 0
        quantity = int(quantity_str) if quantity_str.isdigit() else 0

        current_market_price = robot_status['last_recommended_stock_price'] if robot_status[
            'last_recommended_stock_price'] else order_price
        if current_market_price == 0:
            current_market_price = 50000

        current_time = datetime.datetime.now()
        update_robot_status('tr', '실행 중 (수동)', current_time)
        action_taken_manual = "없음"
        try:
            if order_type and quantity > 0 and symbol_code and symbol_name:
                tr_robot.execute_order(symbol_code, symbol_name, order_type, price_type, order_price, quantity,
                                       current_market_price)
                action_taken_manual = f"수동 {order_type}: {symbol_name} {quantity}주"
                logger.info(f"-> Trade Robot 수동 실행 완료. {action_taken_manual}")
                portfolio_df = tr_robot.portfolio.copy()
                cash_balance = tr_robot.virtual_cash
                trade_log_df = tr_robot.get_trade_log()
            else:
                logger.warning("-> 유효하지 않은 수동 주문 요청입니다.")
        except Exception as e:
            logger.error(f"Trade Robot 수동 실행 중 오류 발생: {e}")
            update_robot_status('tr', '오류 발생 (수동)', current_time)
        finally:
            update_robot_status('tr', '완료 (수동)', current_time, tr_last_action=action_taken_manual)

    total_asset_value = cash_balance
    if not portfolio_df.empty:
        # 포트폴리오 최신 가격 업데이트 및 평가액 재계산 (UI 표시용)
        updated_portfolio_data = []
        for idx, row in portfolio_df.iterrows():
            stock_code = row['종목코드']
            try:
                latest_price_df = sa_robot._get_stock_price_data(stock_code, period='1d')
                if latest_price_df is not None and not latest_price_df.empty:
                    latest_price = latest_price_df['Close'].iloc[-1]
                    row['현재가'] = latest_price
                    row['평가손익'] = (latest_price - row['매입단가']) * row['보유수량']
                    row['수익률'] = (latest_price / row['매입단가'] - 1) * 100
                else:
                    logger.warning(f"-> UI 표시용: {stock_code}의 최신 가격을 가져올 수 없습니다.")
            except Exception as err:
                logger.warning(f"-> UI 표시용: {stock_code} 가격 업데이트 중 오류: {err}")
            updated_portfolio_data.append(row)

        portfolio_df = pd.DataFrame(updated_portfolio_data)
        total_asset_value += (portfolio_df['현재가'] * portfolio_df['보유수량']).sum()

    asset_chart_url = None
    if not trade_log_df.empty:
        trade_log_df['trade_datetime_date'] = pd.to_datetime(trade_log_df['trade_datetime']).dt.date

        daily_balances = trade_log_df.groupby('trade_datetime_date')['balance'].last()

        plt.figure(figsize=(10, 6))
        plt.plot(daily_balances.index, daily_balances.values, marker='o', linestyle='-', color='blue')
        plt.title('가상 현금 잔고 변화')
        plt.xlabel('날짜')
        plt.ylabel('잔고 (원)')
        plt.grid(True)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        asset_chart_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

    return render_template('trade.html',
                           robot_status=robot_status,  # 상태 전달
                           cash_balance=f"{int(cash_balance):,}",
                           total_asset_value=f"{int(total_asset_value):,}",
                           portfolio=portfolio_df.to_html(classes='table table-striped table-bordered', index=False,
                                                          escape=False) if not portfolio_df.empty else None,
                           trade_log=trade_log_df.to_html(classes='table table-striped table-bordered', index=False,
                                                          escape=False) if not trade_log_df.empty else None,
                           last_recommended_stock_code=robot_status['last_recommended_stock_code'],
                           last_recommended_stock_name=robot_status['last_recommended_stock_name'],
                           last_recommended_stock_price=f"{int(robot_status['last_recommended_stock_price']):,}" if
                           robot_status['last_recommended_stock_price'] else None,
                           asset_chart_url=asset_chart_url)


@app.route('/db_data')
def db_data():
    """DB에 저장된 데이터 조회 페이지"""
    conn = sqlite3.connect(mi_robot.db_path)

    news_data_db = pd.read_sql_query("SELECT * FROM news_data ORDER BY crawled_at DESC LIMIT 10", conn)
    recommendations_db = pd.read_sql_query("SELECT * FROM stock_recommendations ORDER BY recommend_date DESC LIMIT 10",
                                           conn)
    trade_log_db = pd.read_sql_query("SELECT * FROM trade_log ORDER BY trade_datetime DESC LIMIT 10", conn)

    conn.close()

    news_data = news_data_db.rename(columns={
        'title': '제목', 'link': '링크', 'sentiment': '감성',
        'source': '출처', 'news_date': '날짜', 'keyword': '키워드',
        'crawled_at': '크롤링_시간'
    })
    recommendations = recommendations_db.rename(columns={
        'stock_code': '종목코드', 'stock_name': '추천종목', 'current_price': '현재가',
        'per': 'PER', 'roe': 'ROE', 'debt_ratio': '부채비율',
        'sector': '업종', 'recommend_reason': '추천사유', 'expected_return': '예상수익률(시뮬)',
        'recommend_date': '추천일자'
    })
    trade_log = trade_log_db.rename(columns={
        'trade_datetime': '거래일시', 'stock_code': '종목코드', 'stock_name': '종목명',
        'trade_type': '거래유형', 'price_type': '가격유형', 'trade_price': '거래가격',
        'quantity': '거래수량', 'trade_amount': '거래금액', 'fee': '수수료',
        'tax': '세금', 'net_trade_amount': '실제거래금액', 'balance': '잔고'
    })

    return render_template('db_data.html',
                           robot_status=robot_status,  # 상태 전달
                           news_data=news_data.to_html(classes='table table-striped table-bordered',
                                                       escape=False) if not news_data.empty else None,
                           recommendations=recommendations.to_html(classes='table table-striped table-bordered',
                                                                   index=False,
                                                                   escape=False) if not recommendations.empty else None,
                           trade_log=trade_log.to_html(classes='table table-striped table-bordered', index=False,
                                                       escape=False) if not trade_log.empty else None)


if __name__ == '__main__':
    data_dir = os.path.join(project_root, 'data')
    config_dir = os.path.join(project_root, 'config')
    robots_dir = os.path.join(project_root, 'robots')
    logs_dir = os.path.join(project_root, 'logs')

    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    if not os.path.exists(robots_dir): os.makedirs(robots_dir)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)

    # settings.ini 파일이 없으면 기본값으로 생성
    config_file_path_for_init = os.path.join(config_dir, 'settings.ini')
    if not os.path.exists(config_file_path_for_init):
        logger.info("config/settings.ini 파일이 없어 기본값을 생성합니다.")
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
            'MIN_ROE_DEFAULT': '10',
            'DART_API_KEY': '80c0229e5fa4cf75ac80cce218d32404c7de0842'  # DART API 키 추가 (비워둘 수 있음)
        }
        default_config['TRADE'] = {
            'VIRTUAL_INITIAL_CASH': '10000000',
            'TRADE_FEE_RATE': '0.00015',  # 매매 수수료율 (0.015%)
            'TRADE_TAX_RATE': '0.0025'  # 거래세율 (0.25%)
        }
        default_config['API_KEYS'] = {
            'DART_API_KEY': '80c0229e5fa4cf75ac80cce218d32404c7de0842'  # DART API 키 섹션 (이 섹션에 직접 입력)
        }
        with open(config_file_path_for_init, 'w', encoding='utf-8') as configfile:
            default_config.write(configfile)

    init_file_path = os.path.join(robots_dir, '__init__.py')
    if not os.path.exists(init_file_path):
        with open(init_file_path, 'w') as f:
            f.write('')

    app.run(debug=True)
