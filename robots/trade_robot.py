# robots/trade_robot.py

import pandas as pd
import datetime
import sqlite3
import configparser
import logging

logger = logging.getLogger(__name__)


class TradeRobot:
    def __init__(self_obj, config_path='config/settings.ini'):
        self_obj.config = configparser.ConfigParser()
        self_obj.config.read(config_path, encoding='utf-8')

        self_obj.db_path = 'data/robot_data.db'

        # 1. 먼저 virtual_cash 속성을 초기화합니다.
        self_obj.virtual_cash = int(self_obj.config['TRADE'].get('VIRTUAL_INITIAL_CASH', '10000000'))  # 초기 가상 현금
        self_obj.trade_fee_rate = float(self_obj.config['TRADE'].get('TRADE_FEE_RATE', '0.00015'))  # 매매 수수료율 (0.015%)
        self_obj.trade_tax_rate = float(self_obj.config['TRADE'].get('TRADE_TAX_RATE', '0.0025'))  # 거래세율 (0.25%)

        # 2. virtual_cash가 정의된 후에 _init_db()를 호출하여 DB를 초기화하고 초기 현금 잔고를 삽입합니다.
        self_obj._init_db()

        # 3. DB에서 최신 현금 잔고와 포트폴리오를 로드합니다.
        self_obj._load_portfolio_from_db()  # DB에서 포트폴리오 로드
        self_obj._update_virtual_cash_from_db()  # DB에서 최신 현금 잔고 로드

    def _init_db(self_obj):
        """거래 기록 및 포트폴리오 데이터를 저장할 SQLite 테이블 초기화."""
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()

        # 거래 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_datetime TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                trade_type TEXT NOT NULL, -- '매수' 또는 '매도'
                price_type TEXT NOT NULL, -- '시장가' 또는 '지정가'
                trade_price REAL NOT NULL, -- 실제 거래된 가격
                quantity INTEGER NOT NULL,
                trade_amount REAL NOT NULL, -- 수수료/세금 제외 금액
                fee REAL NOT NULL,
                tax REAL NOT NULL,
                net_trade_amount REAL NOT NULL, -- 수수료/세금 포함 실제 현금 증감액
                balance REAL NOT NULL, -- 거래 후 현금 잔고
                memo TEXT
            )
        ''')

        # 포트폴리오 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                매입단가 REAL NOT NULL,
                보유수량 INTEGER NOT NULL,
                매입금액 REAL NOT NULL,
                현재가 REAL,         -- 최신 업데이트된 현재가
                평가손익 REAL,
                수익률 REAL,
                last_updated TEXT
            )
        ''')

        # 가상 현금 잔고 테이블 (단일 레코드)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS virtual_cash (
                id INTEGER PRIMARY KEY DEFAULT 1,
                balance REAL NOT NULL
            )
        ''')
        # 최초 실행 시 가상 현금 잔고 삽입
        cursor.execute("INSERT OR IGNORE INTO virtual_cash (id, balance) VALUES (1, ?)", (self_obj.virtual_cash,))

        conn.commit()
        conn.close()

    def _load_portfolio_from_db(self_obj):
        """DB에서 포트폴리오 정보를 로드합니다."""
        conn = sqlite3.connect(self_obj.db_path)
        self_obj.portfolio = pd.read_sql_query("SELECT * FROM portfolio", conn)
        conn.close()
        if self_obj.portfolio.empty:
            self_obj.portfolio = pd.DataFrame(columns=[
                'stock_code', 'stock_name', '매입단가', '보유수량', '매입금액',
                '현재가', '평가손익', '수익률', 'last_updated'
            ])
            logger.info("포트폴리오가 비어있습니다.")
        else:
            logger.info(f"포트폴리오 {len(self_obj.portfolio)}개 종목 로드 완료.")

    def _update_portfolio_to_db(self_obj):
        """포트폴리오 정보를 DB에 저장합니다."""
        conn = sqlite3.connect(self_obj.db_path)
        self_obj.portfolio.to_sql('portfolio', conn, if_exists='replace', index=False)
        conn.close()
        logger.debug("포트폴리오 정보가 DB에 업데이트되었습니다.")

    def _update_virtual_cash_from_db(self_obj):
        """DB에서 최신 가상 현금 잔고를 로드합니다."""
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM virtual_cash WHERE id = 1")
        result = cursor.fetchone()
        conn.close()
        if result:
            self_obj.virtual_cash = result[0]
            logger.info(f"DB에서 가상 현금 잔고 로드 완료: {self_obj.virtual_cash:,}원")
        else:
            logger.warning("가상 현금 잔고를 DB에서 찾을 수 없습니다. 초기값으로 설정합니다.")
            self_obj._save_virtual_cash_to_db()  # 초기값 저장

    def _save_virtual_cash_to_db(self_obj):
        """가상 현금 잔고를 DB에 저장합니다."""
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()
        cursor.execute("REPLACE INTO virtual_cash (id, balance) VALUES (1, ?)", (self_obj.virtual_cash,))
        conn.commit()
        conn.close()
        logger.debug(f"가상 현금 잔고 DB에 저장 완료: {self_obj.virtual_cash:,}원")

    def get_trade_log(self_obj):
        """DB에서 거래 로그를 가져옵니다."""
        conn = sqlite3.connect(self_obj.db_path)
        trade_log_df = pd.read_sql_query("SELECT * FROM trade_log ORDER BY trade_datetime DESC", conn)
        conn.close()
        return trade_log_df

    def execute_order(self_obj, stock_code, stock_name, order_type, price_type, order_price, quantity,
                      current_market_price):
        """
        주식 매수/매도 주문을 실행하고 포트폴리오 및 현금 잔고를 업데이트합니다.

        Args:
            stock_code (str): 종목 코드
            stock_name (str): 종목명
            order_type (str): '매수' 또는 '매도'
            price_type (str): '시장가' 또는 '지정가'
            order_price (float): 지정가 주문 시 가격 (시장가 시 무시)
            quantity (int): 수량
            current_market_price (float): 현재 시장가 (자동 매매 시 사용)
        """
        trade_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        actual_trade_price = current_market_price if price_type == '시장가' else order_price

        if actual_trade_price <= 0 or quantity <= 0:
            logger.error("거래 가격 또는 수량이 유효하지 않습니다.")
            return

        trade_amount_raw = actual_trade_price * quantity
        fee = trade_amount_raw * self_obj.trade_fee_rate
        tax = 0  # 매수 시 세금 없음
        net_trade_amount = trade_amount_raw + fee  # 매수 시 총 지불액

        memo = ""

        if order_type == '매수':
            if self_obj.virtual_cash >= net_trade_amount:
                self_obj.virtual_cash -= net_trade_amount

                # 포트폴리오 업데이트
                if stock_code in self_obj.portfolio['stock_code'].values:
                    # 기존 보유 종목: 평단가 및 수량 업데이트
                    idx = self_obj.portfolio[self_obj.portfolio['stock_code'] == stock_code].index[0]
                    old_quantity = self_obj.portfolio.loc[idx, '보유수량']
                    old_buy_amount = self_obj.portfolio.loc[idx, '매입금액']

                    new_total_quantity = old_quantity + quantity
                    new_total_buy_amount = old_buy_amount + trade_amount_raw  # 수수료는 현금에서 빠져나감

                    self_obj.portfolio.loc[idx, '매입단가'] = new_total_buy_amount / new_total_quantity
                    self_obj.portfolio.loc[idx, '보유수량'] = new_total_quantity
                    self_obj.portfolio.loc[idx, '매입금액'] = new_total_buy_amount
                    self_obj.portfolio.loc[idx, 'last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # 신규 보유 종목
                    new_stock = pd.DataFrame([{
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        '매입단가': actual_trade_price,
                        '보유수량': quantity,
                        '매입금액': trade_amount_raw,
                        '현재가': actual_trade_price,  # 매수 시 현재가는 매수 가격으로 초기화
                        '평가손익': 0,
                        '수익률': 0,
                        'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }])
                    self_obj.portfolio = pd.concat([self_obj.portfolio, new_stock], ignore_index=True)

                memo = f"{stock_name} {quantity}주 매수 완료"
                logger.info(f"매수 성공: {memo}. 잔고: {self_obj.virtual_cash:,}원")
                self_obj._update_portfolio_to_db()
                self_obj._save_virtual_cash_to_db()

            else:
                memo = f"매수 실패: 현금 부족 ({net_trade_amount:,}원 필요, {self_obj.virtual_cash:,}원 보유)"
                logger.warning(memo)
                return

        elif order_type == '매도':
            if stock_code in self_obj.portfolio['stock_code'].values:
                idx = self_obj.portfolio[self_obj.portfolio['stock_code'] == stock_code].index[0]
                current_holdings = self_obj.portfolio.loc[idx, '보유수량']

                if current_holdings >= quantity:
                    tax = trade_amount_raw * self_obj.trade_tax_rate  # 매도 시 거래세 적용
                    net_trade_amount = trade_amount_raw - fee - tax  # 매도 시 총 수령액

                    self_obj.virtual_cash += net_trade_amount

                    # 포트폴리오 업데이트
                    if current_holdings == quantity:
                        # 전량 매도
                        self_obj.portfolio = self_obj.portfolio.drop(idx).reset_index(drop=True)
                    else:
                        # 일부 매도
                        self_obj.portfolio.loc[idx, '보유수량'] -= quantity
                        self_obj.portfolio.loc[idx, '매입금액'] -= (self_obj.portfolio.loc[idx, '매입단가'] * quantity)
                        self_obj.portfolio.loc[idx, 'last_updated'] = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S')

                    memo = f"{stock_name} {quantity}주 매도 완료"
                    logger.info(f"매도 성공: {memo}. 잔고: {self_obj.virtual_cash:,}원")
                    self_obj._update_portfolio_to_db()
                    self_obj._save_virtual_cash_to_db()

                else:
                    memo = f"매도 실패: 보유 수량 부족 ({quantity}주 요청, {current_holdings}주 보유)"
                    logger.warning(memo)
                    return
            else:
                memo = f"매도 실패: {stock_name} ({stock_code}) 종목을 보유하고 있지 않습니다."
                logger.warning(memo)
                return

        # 거래 로그 DB에 저장
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO trade_log (trade_datetime, stock_code, stock_name, trade_type, price_type, trade_price, quantity, trade_amount, fee, tax, net_trade_amount, balance, memo) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_datetime, stock_code, stock_name, order_type, price_type, actual_trade_price, quantity,
             trade_amount_raw, fee, tax, net_trade_amount, self_obj.virtual_cash, memo)
        )
        conn.commit()
        conn.close()
        logger.debug("거래 로그가 DB에 저장되었습니다.")
