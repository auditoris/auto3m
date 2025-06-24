# robots/choice_robot.py

import os
import pandas as pd
import random
import configparser
import FinanceDataReader as fdr
import datetime
import json
import pandas_ta as ta
import sqlite3
import dart_fss as dfs  # DART 연동 라이브러리 추가
import logging

logger = logging.getLogger(__name__)

# DB 파일 경로 (financial_data_updater.py와 동일하게 설정)
DB_FILE = os.path.join(os.getcwd(), 'data', 'financial_data.db')

class ChoiceRobot:
    def __init__(self_obj, config_path='config/settings.ini'):
        self_obj.config = configparser.ConfigParser()
        self_obj.config.read(config_path, encoding='utf-8')

        self_obj.dart_api_key = self_obj.config['API_KEYS'].get('DART_API_KEY', '')  # DART API 키 로드
        if self_obj.dart_api_key:
            try:
                dfs.set_api_key(api_key=self_obj.dart_api_key)
                logger.info("DART API 키가 설정되었습니다.")
            except Exception as e:
                logger.error(f"DART API 키 설정 중 오류 발생: {e}")
                self_obj.dart_api_key = ''  # 오류 발생 시 키 무효화

        # 파일이 저장될 경로 (예: 프로젝트 루트에 'data' 폴더 생성)
        self_obj.data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(self_obj.data_dir, exist_ok=True) # data 폴더가 없으면 생성
        self_obj.stock_list = self_obj._load_filtered_stock_list()

        self_obj.db_path = 'data/robot_data.db'
        self_obj._init_db()

        self_obj.min_per_default = int(self_obj.config['STOCK_ANALYSIS'].get('MIN_PER_DEFAULT', '20'))
        self_obj.max_debt_ratio_default = int(self_obj.config['STOCK_ANALYSIS'].get('MAX_DEBT_RATIO_DEFAULT', '100'))
        self_obj.min_roe_default = int(self_obj.config['STOCK_ANALYSIS'].get('MIN_ROE_DEFAULT', '10'))

    def _load_filtered_stock_list(self):
        """
        KRX CSV 파일에서 KOSPI 200과 KOSDAQ 150 종목만 가져와 반환합니다.
        사전에 KRX 정보데이터시스템에서 해당 CSV 파일을 다운로드하여
        프로젝트 내 'data' 폴더에 저장해야 합니다.
        예: 'data/KOSPI200_구성종목.csv', 'data/KOSDAQ150_구성종목.csv'
        """
        logger.info("KOSPI 200 및 KOSDAQ 150 종목 목록 로드 중 (KRX CSV 사용)...")

        kospi200_file = os.path.join(self.data_dir, 'KOSPI200_구성종목.csv')
        kosdaq150_file = os.path.join(self.data_dir, 'KOSDAQ150_구성종목.csv')

        all_stocks = pd.DataFrame()

        try:
            # KOSPI 200 종목 로드
            if os.path.exists(kospi200_file):
                # KRX CSV 파일의 실제 컬럼명을 확인해야 합니다.
                # '종목코드', '종목명' 또는 '단축코드', '한글종목명' 등
                # 여기서는 가장 흔한 '종목코드'와 '종목명'을 가정합니다.
                kospi200_df = pd.read_csv(kospi200_file, encoding='cp949')  # 한글 인코딩 주의
                # 필요에 따라 컬럼명 변경
                kospi200_df = kospi200_df.rename(columns={'종목코드': 'Symbol', '종목명': 'CompanyName'})
                kospi200_df['Symbol'] = kospi200_df['Symbol'].astype(str).str.zfill(6)  # 종목코드를 6자리 문자열로
                all_stocks = pd.concat([all_stocks, kospi200_df[['Symbol', 'CompanyName']]])
                logger.info(f"KOSPI 200 종목 수: {len(kospi200_df)}")
            else:
                logger.warning(f"KOSPI 200 구성 종목 파일이 없습니다: {kospi200_file}. 수동 다운로드 필요.")

            # KOSDAQ 150 종목 로드
            if os.path.exists(kosdaq150_file):
                kosdaq150_df = pd.read_csv(kosdaq150_file, encoding='cp949')  # 한글 인코딩 주의
                kosdaq150_df = kosdaq150_df.rename(columns={'종목코드': 'Symbol', '종목명': 'CompanyName'})
                kosdaq150_df['Symbol'] = kosdaq150_df['Symbol'].astype(str).str.zfill(6)  # 종목코드를 6자리 문자열로
                all_stocks = pd.concat([all_stocks, kosdaq150_df[['Symbol', 'CompanyName']]])
                logger.info(f"KOSDAQ 150 종목 수: {len(kosdaq150_df)}")
            else:
                logger.warning(f"KOSDAQ 150 구성 종목 파일이 없습니다: {kosdaq150_file}. 수동 다운로드 필요.")

            # 합쳐진 종목 리스트에서 중복 제거 및 인덱스 초기화
            filtered_stocks = all_stocks.drop_duplicates(subset=['Symbol']).reset_index(drop=True)
            logger.info(f"총 필터링된 종목 수 (KOSPI 200 + KOSDAQ 150): {len(filtered_stocks)}개")
            return filtered_stocks

        except Exception as e:
            logger.error(f"KOSPI/KOSDAQ 벤치마크 종목 로드 중 오류 발생: {e}")
            logger.warning("모든 상장 종목 목록을 가져오는 기본 방식으로 대체합니다.")
            # 오류 발생 시 모든 종목을 가져오는 기존 방식으로 대체 (최후의 수단)
            return fdr.StockListing('KRX')  # KRX 전체 종목 목록

    def _init_db(self_obj):
        """주식 추천 데이터를 저장할 SQLite 테이블 초기화."""
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                current_price REAL,
                per REAL,
                roe REAL,
                debt_ratio REAL,
                sector TEXT,
                recommend_reason TEXT,
                expected_return REAL,
                recommend_date TEXT,
                crawled_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _get_stock_list(self_obj):
        """한국거래소 상장 종목 리스트를 가져옵니다."""
        try:
            df_krx = fdr.StockListing('KRX')
            return df_krx
        except Exception as e:
            logger.error(f"주식 종목 리스트를 가져오는 중 오류 발생: {e}")
            return pd.DataFrame()

    def _get_stock_price_data(self_obj, stock_code, start_date=None, end_date=None, period=None):
        """주가 데이터를 가져옵니다."""
        try:
            if start_date is None:
                start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.date.today().strftime('%Y-%m-%d')

            if period == '1d':  # 최신 1일 데이터만 필요한 경우
                df_price = fdr.DataReader(stock_code, start=end_date, end=end_date)
            else:
                df_price = fdr.DataReader(stock_code, start=start_date, end=end_date)

            if df_price.empty:
                logger.warning(f"'{stock_code}' 주가 데이터를 찾을 수 없습니다.")
                return None
            return df_price
        except Exception as e:
            logger.error(f"'{stock_code}' 주가 데이터를 가져오는 중 오류 발생: {e}")
            return None

    # def _get_financial_data(self_obj, stock_code, corp_name=None):
    #     """
    #     DART API(dart-fss)를 사용하여 재무 데이터를 가져옵니다.
    #     API 키가 없거나 dart-fss에서 오류 발생 시 시뮬레이션 데이터 사용.
    #     """
    #     financial_info = {
    #         'per': round(random.uniform(5, 50), 2),
    #         'roe': round(random.uniform(5, 30), 2),
    #         'debt_ratio': round(random.uniform(30, 200), 2),
    #         'sector': '시뮬레이션 업종'  # 기본값 설정
    #     }
    #
    #     if self_obj.dart_api_key:
    #         try:
    #             dfs.set_api_key(self_obj.dart_api_key)
    #
    #             corp_code = None
    #             target_sector = '업종정보없음'
    #
    #             # 1. 회사 고유 번호(corp_code) 및 섹터 정보 찾기
    #             if corp_name:
    #                 raw_corp_list_obj = dfs.get_corp_list()  # CorpList 객체 반환
    #
    #                 # --- !!! 이 부분이 다시 핵심 수정입니다 !!! ---
    #                 # CorpList 객체의 .corps 속성(Corp 객체들의 리스트)을 DataFrame으로 변환합니다.
    #                 # 각 Corp 객체는 'corp_name', 'corp_code', 'stock_code', 'sector' 등의 속성을 가집니다.
    #                 extracted_corp_data = []
    #                 for corp in raw_corp_list_obj.corps:
    #                     extracted_corp_data.append({
    #                         'corp_name': getattr(corp, 'corp_name', None),
    #                         'corp_code': getattr(corp, 'corp_code', None),
    #                         'stock_code': getattr(corp, 'stock_code', None),
    #                         'sector': getattr(corp, 'sector', '정보없음')
    #                     })
    #
    #                 corp_df = pd.DataFrame(extracted_corp_data)  # 추출된 딕셔너리 리스트로 DataFrame 생성
    #
    #                 if corp_df.empty:
    #                     logger.error("DART API: pd.DataFrame(extracted_corp_data)가 비어있습니다. 기업 목록 로드 실패.")
    #                     raise ValueError("Failed to convert CorpList.corps to DataFrame.")
    #                 logger.info(f"DART API: corp_df 로드 성공. 총 {len(corp_df)}개 기업. 컬럼: {corp_df.columns.tolist()}")
    #
    #                 # 이제 corp_df DataFrame에서 회사 이름으로 필터링
    #                 filtered_corp = corp_df[corp_df['corp_name'] == corp_name]
    #
    #                 if not filtered_corp.empty:  # 필터링된 결과가 있다면
    #                     corp_code = filtered_corp['corp_code'].iloc[0]
    #
    #                     # DataFrame에서 직접 sector 정보 가져오기
    #                     if 'sector' in filtered_corp.columns and pd.notna(filtered_corp['sector'].iloc[0]):
    #                         target_sector = filtered_corp['sector'].iloc[0]
    #                     else:
    #                         logger.warning(f"DART API: '{corp_name}'의 DataFrame에서 'sector' 정보를 찾을 수 없습니다.")
    #
    #                 else:
    #                     logger.warning(f"DART API: '{corp_name}'에 해당하는 회사 정보를 찾을 수 없습니다. (corp_df 필터링 실패)")
    #                     return financial_info  # 회사 정보를 찾지 못했으니 DART API 시도 중단
    #
    #             if corp_code:  # corp_code를 찾았을 경우에만 DART-FSS 재무 데이터 호출
    #                 logger.info(f"DART API: '{corp_name}'({stock_code})의 corp_code '{corp_code}'로 재무 데이터 요청.")
    #
    #                 # 2. 재무 데이터 가져오기 (dfs.fs.extract 사용)
    #                 fs_all = dfs.fs.extract(corp_code=corp_code,
    #                                         bgn_de='20200101',
    #                                         end_de=datetime.date.today().strftime('%Y%m%d'))
    #
    #                 if fs_all is None:
    #                     logger.warning(f"DART-FSS: '{corp_name}'({stock_code})의 재무 데이터(dfs.fs.extract)가 None을 반환했습니다.")
    #                     return financial_info
    #
    #                 if fs_all.empty:
    #                     logger.warning(
    #                         f"DART-FSS: '{corp_name}'({stock_code})의 재무 데이터(dfs.fs.extract) DataFrame이 비어있습니다.")
    #                     return financial_info
    #
    #                 logger.info(
    #                     f"DART-FSS: '{corp_name}'({stock_code})의 finstate DataFrame 상위 5개 행:\n{fs_all.head().to_string()}")
    #                 logger.info(
    #                     f"DART-FSS: '{corp_name}'({stock_code})의 finstate DataFrame 컬럼: {fs_all.columns.tolist()}")
    #
    #                 # 가장 최신 보고서 선택
    #                 if 'rcept_dt' in fs_all.columns:
    #                     latest_fs = fs_all.sort_values(by='rcept_dt', ascending=False).iloc[0]
    #                 else:
    #                     logger.warning(f"DART-FSS: '{corp_name}' 재무 데이터에 'rcept_dt' 컬럼이 없습니다. 첫 번째 행 사용.")
    #                     latest_fs = fs_all.iloc[0]
    #
    #                 # PER, ROE, 부채비율 계산 로직은 여전히 직접 구현해야 합니다.
    #                 # financial_info['per'] = ...
    #                 # financial_info['roe'] = ...
    #                 # financial_info['debt_ratio'] = ...
    #
    #                 financial_info['sector'] = target_sector
    #
    #                 logger.info(f"DART API로 '{corp_name}'({stock_code}) 재무 데이터 가져오기 성공.")
    #                 return financial_info
    #             else:
    #                 logger.warning(f"DART API로 '{corp_name}'({stock_code})의 재무 데이터를 가져오지 못했습니다 (corp_code 없음).")
    #                 return financial_info
    #
    #         except Exception as e:
    #             logger.error(f"DART API(dart-fss)로 '{corp_name}'({stock_code}) 재무 데이터를 가져오는 중 오류 발생: {e}")
    #             return financial_info
    #
    #     return financial_info
    def _get_financial_data(self_obj, stock_code, corp_name=None):
        """
        SQLite DB에서 종목의 재무 데이터를 가져옵니다.
        DB에 데이터가 없으면 시뮬레이션 데이터를 반환합니다.
        """
        financial_info = {
            'per': round(random.uniform(5, 50), 2),
            'roe': round(random.uniform(5, 30), 2),
            'debt_ratio': round(random.uniform(30, 200), 2),
            'sector': '시뮬레이션 업종'  # 기본값 설정
        }

        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # 종목 코드 (Symbol)로 데이터 조회
            cursor.execute("""
                SELECT per, roe, debt_ratio, sector 
                FROM financial_data 
                WHERE symbol = ?
            """, (stock_code,))

            result = cursor.fetchone()
            conn.close()

            if result:
                # DB에서 가져온 데이터로 financial_info 업데이트
                financial_info['per'] = result[0] if result[0] is not None else financial_info['per']
                financial_info['roe'] = result[1] if result[1] is not None else financial_info['roe']
                financial_info['debt_ratio'] = result[2] if result[2] is not None else financial_info['debt_ratio']
                financial_info['sector'] = result[3] if result[3] is not None else financial_info['sector']

                logger.info(f"SQLite DB에서 '{corp_name}'({stock_code}) 재무 데이터 로드 성공.")
                return financial_info
            else:
                logger.warning(f"SQLite DB에 '{corp_name}'({stock_code}) 재무 데이터가 없습니다. 시뮬레이션 값 반환.")
                return financial_info

        except sqlite3.Error as e:
            logger.error(f"SQLite DB에서 '{corp_name}'({stock_code}) 재무 데이터 로드 중 오류 발생: {e}")
            return financial_info  # DB 오류 시 시뮬레이션 값 반환
        except Exception as e:
            logger.error(f"재무 데이터 로드 중 예상치 못한 오류: {e}")
            return financial_info

    def _calculate_technical_indicators(self_obj, df_price):
        """주가 데이터에 기술적 지표를 추가합니다."""
        df_price['MA_20'] = ta.sma(df_price['Close'], length=20)
        df_price['MA_60'] = ta.sma(df_price['Close'], length=60)
        df_price['RSI'] = ta.rsi(df_price['Close'], length=14)

        macd = ta.macd(df_price['Close'])
        if macd is not None and not macd.empty:
            df_price['MACD'] = macd['MACD_12_26_9']
            df_price['MACDh'] = macd['MACDh_12_26_9']
            df_price['MACDs'] = macd['MACDs_12_26_9']
        else:
            df_price['MACD'] = None
            df_price['MACDh'] = None
            df_price['MACDs'] = None

        bbands_df = ta.bbands(df_price['Close'], length=20)
        upper_col_ta_name = f'BBU_20_2.0'
        middle_col_ta_name = f'BBM_20_2.0'
        lower_col_ta_name = f'BBL_20_2.0'

        if all(col in bbands_df.columns for col in [upper_col_ta_name, middle_col_ta_name, lower_col_ta_name]):
            df_price['BB_upper'] = bbands_df[upper_col_ta_name]
            df_price['BB_middle'] = bbands_df[middle_col_ta_name]
            df_price['BB_lower'] = bbands_df[lower_col_ta_name]
        else:
            logger.warning("-> 경고: 볼린저 밴드 컬럼을 예상한 이름으로 찾을 수 없습니다. pandas_ta 버전 또는 설정 확인 필요.")
            df_price['BB_upper'] = None
            df_price['BB_middle'] = None
            df_price['BB_lower'] = None

        return df_price

    def analyze_and_recommend(self_obj, market_sentiment, **kwargs):
        """
        시장 감성, 재무 지표, 기술적 지표를 종합하여 종목을 분석하고 추천합니다.

        kwargs:
            min_per (int): 최대 PER (기본값: settings.ini)
            max_debt_ratio (int): 최대 부채비율 (기본값: settings.ini)
            min_roe (int): 최소 ROE (기본값: settings.ini)
            strategy_type (str): '가치투자', '모멘텀', '기술적분석'
        """
        min_per = kwargs.get('min_per', self_obj.min_per_default)
        max_debt_ratio = kwargs.get('max_debt_ratio', self_obj.max_debt_ratio_default)
        min_roe = kwargs.get('min_roe', self_obj.min_roe_default)
        strategy_type = kwargs.get('strategy_type', '가치투자')

        logger.info(f"주식 분석 및 추천 시작. 시장 감성: {market_sentiment}, 전략: {strategy_type}")

        # df_krx = self_obj._get_stock_list()
        # if df_krx.empty:
        #     logger.error("종목 리스트를 가져올 수 없어 종목 분석을 진행할 수 없습니다.")
        #     return None, None, None, pd.DataFrame()
        #
        # eligible_stocks = []
        # # KRX 종목 리스트에서 상위 200개 종목을 대상으로 분석 (너무 많으면 시간 오래 걸림)
        # # 실제 운영에서는 코스피/코스닥 200 등 특정 종목군 대상으로 하는 것이 효율적
        # # 여기서는 테스트를 위해 상위 200개 또는 랜덤 50개 종목으로 제한
        # sample_size = min(200, len(df_krx))
        # sampled_stocks = df_krx.sample(sample_size) if len(df_krx) > 200 else df_krx
        #
        # for idx, row in sampled_stocks.iterrows():
        #     stock_code = row['Code']  # ['Symbol']
        #     stock_name = row['Name']
        #     sector = row['Sector']  # FinanceDataReader에서 가져오는 섹터 정보
        #
        #     financial_data = self_obj._get_financial_data(stock_code, corp_name=stock_name)
        #     if not financial_data:
        #         logger.warning(f"{stock_name}({stock_code})의 재무 데이터를 가져오기 실패, 건너_습니다.")
        #         continue
        #
        #     per = financial_data.get('per')
        #     roe = financial_data.get('roe')
        #     debt_ratio = financial_data.get('debt_ratio')
        df_krx = self_obj._get_stock_list()
        if df_krx.empty:
            logger.error("종목 리스트를 가져올 수 없어 종목 분석을 진행할 수 없습니다.")
            return None, None, None, pd.DataFrame()

        # 새로 확인된 df_krx 컬럼 헤더를 바탕으로 'Code'와 'Name' 사용
        # df_krx.columns: ['Code', 'ISU_CD', 'Name', ..., 'MarketId']
        logger.info(f"df_krx 컬럼: {df_krx.columns.tolist()}")  # 이 부분은 확인용으로 잠시 두거나 삭제해도 됨.
        logger.info(f"df_krx 상위 5개 행:\n{df_krx.head().to_string()}")  # 이 부분도 확인용으로 잠시 두거나 삭제해도 됨.

        eligible_stocks = []
        sample_size = min(200, len(df_krx))
        sampled_stocks = df_krx.sample(sample_size) if len(df_krx) > 200 else df_krx.copy()

        for idx, row in sampled_stocks.iterrows():
            stock_code = row['Code']  # 'Code' 컬럼 사용
            stock_name = row['Name']  # 'Name' 컬럼 사용

            financial_data = self_obj._get_financial_data(stock_code, corp_name=stock_name)
            if not financial_data:
                logger.warning(f"{stock_name}({stock_code})의 재무 데이터를 가져오기 실패, 건너뜁니다.")
                continue

            per = financial_data.get('per')
            roe = financial_data.get('roe')
            debt_ratio = financial_data.get('debt_ratio')

            # --- 핵심 변경 사항 ---
            # financial_data 딕셔너리에서 'sector' 정보를 가져옵니다.
            # _get_financial_data 내부에서 DART API나 fdr.StockInfo()로부터 이 정보를 가져오려 시도합니다.
            sector = financial_data.get('sector', '업종정보없음')  # 만약 정보가 없으면 '업종정보없음'으로 설정

            # 1. 재무 기준 필터링 (가치투자 기본 조건)
            if not (per is not None and per <= min_per and \
                    roe is not None and roe >= min_roe and \
                    debt_ratio is not None and debt_ratio <= max_debt_ratio):
                continue

            # 주가 데이터 및 기술적 지표 계산
            df_price = self_obj._get_stock_price_data(stock_code)
            if df_price is None or df_price.empty:
                logger.warning(f"{stock_name}({stock_code})의 주가 데이터를 가져올 수 없어 기술적 분석을 건너뜁니다.")
                continue

            df_price_with_ta = self_obj._calculate_technical_indicators(df_price.copy())

            current_price = df_price_with_ta['Close'].iloc[-1] if not df_price_with_ta.empty else 0
            if current_price == 0: continue

            recommend_reason = []

            # 2. 시장 감성 기반 추천 조정
            if market_sentiment == "긍정적" and random.random() < 0.7:  # 긍정적 시장일 때 추천 확률 높임
                recommend_reason.append("긍정적 시장 감성")
            elif market_sentiment == "부정적" and random.random() < 0.3:  # 부정적 시장일 때 추천 확률 낮춤
                continue  # 부정적 시장에서는 굳이 추천하지 않음

            # 3. 전략별 추가 조건
            if strategy_type == '가치투자':
                # 재무 필터링은 이미 위에서 적용됨
                recommend_reason.append("가치투자 기준 충족 (저PER, 고ROE, 저부채)")

            elif strategy_type == '모멘텀':
                # 최근 3개월 (약 60거래일) 상승률 계산
                if len(df_price_with_ta) >= 60:
                    past_price = df_price_with_ta['Close'].iloc[-60]
                    momentum = (current_price / past_price - 1) * 100
                    if momentum > 10:  # 최근 3개월 10% 이상 상승
                        recommend_reason.append(f"강한 모멘텀 (3개월 상승률: {momentum:.2f}%)")
                    else:
                        continue  # 모멘텀 부족
                else:
                    continue  # 데이터 부족

            elif strategy_type == '기술적분석':
                ma20 = df_price_with_ta['MA_20'].iloc[-1]
                ma60 = df_price_with_ta['MA_60'].iloc[-1]
                rsi = df_price_with_ta['RSI'].iloc[-1]
                macd_h = df_price_with_ta['MACDh'].iloc[-1]
                bb_lower = df_price_with_ta['BB_lower'].iloc[-1]

                # 골든 크로스 (단기 이평선이 장기 이평선 상향 돌파)
                if ma20 > ma60 and df_price_with_ta['MA_20'].iloc[-2] < df_price_with_ta['MA_60'].iloc[-2]:
                    recommend_reason.append("MA 골든 크로스 발생")

                # RSI 과매도 구간 탈출 (RSI 30 이하에서 상승 전환)
                if rsi > 30 and df_price_with_ta['RSI'].iloc[-2] <= 30:
                    recommend_reason.append("RSI 과매도 구간 탈출")

                # MACD 시그널선 상향 돌파 또는 MACD Histogram 양전환
                if macd_h > 0 and df_price_with_ta['MACDh'].iloc[-2] <= 0:
                    recommend_reason.append("MACD 시그널 상향 돌파 (MACD 히스토그램 양전환)")

                # 볼린저 밴드 하단선 터치 후 반등
                if current_price > bb_lower and df_price_with_ta['Close'].iloc[-2] <= bb_lower:
                    recommend_reason.append("볼린저 밴드 하단선 지지 후 반등")

                if not recommend_reason:  # 기술적 분석 조건에 하나도 해당 안 되면 제외
                    continue

            if recommend_reason:  # 최종 추천 사유가 있다면 추가
                # 예상 수익률은 시뮬레이션 (전략에 따라 달라질 수 있음)
                expected_return = random.uniform(5, 25)

                eligible_stocks.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'current_price': current_price,
                    'per': per,
                    'roe': roe,
                    'debt_ratio': debt_ratio,
                    'sector': sector,  # financial_data에서 가져온 sector 값을 사용
                    'recommend_reason': ", ".join(recommend_reason),
                    'expected_return': expected_return,
                    'recommend_date': datetime.date.today().strftime('%Y-%m-%d'),
                    'crawled_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        df_recommendations = pd.DataFrame(eligible_stocks)

        if not df_recommendations.empty:
            # 예상 수익률 기준으로 상위 1개만 추천 (또는 상위 5개 등)
            # 여기서는 편의상 가장 먼저 찾은 종목을 반환하거나, 랜덤으로 1개 선택
            # 또는 'market_sentiment'에 따라 긍정적일 때는 여러 개, 부정적일 때는 1개 등 조절
            df_recommendations = df_recommendations.sort_values(by='expected_return', ascending=False).reset_index(
                drop=True)

            # DB에 저장
            conn = sqlite3.connect(self_obj.db_path)
            df_recommendations.to_sql('stock_recommendations', conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"총 {len(df_recommendations)}개의 추천 종목이 'stock_recommendations' 테이블에 저장되었습니다.")

            # 가장 높은 예상 수익률 종목 하나 반환
            top_recommendation = df_recommendations.iloc[0]
            return top_recommendation['stock_code'], top_recommendation['stock_name'], top_recommendation[
                'current_price'], df_recommendations
        else:
            logger.info("추천할 종목을 찾지 못했습니다.")
            return None, None, None, pd.DataFrame()

# 테스트 코드 (직접 실행 시)
if __name__ == "__main__":
    sa_robot = ChoiceRobot()
    recommended_code, recommended_name, recommended_price, recommendations_df = sa_robot.analyze_and_recommend(
        market_sentiment="중립",
        min_per=25,
        max_debt_ratio=150,
        min_roe=5,
        strategy_type="가치투자"  # 전략 유형 지정 가능
    )
    if recommended_code:
        print(f"\n최종 Stock Analysis 추천: {recommended_name} ({recommended_code}), 현재가: {recommended_price:,}원")

    # DB에서 데이터 조회 테스트
    conn = sqlite3.connect(sa_robot.db_path)
    df_db_test = pd.read_sql_query("SELECT * FROM stock_recommendations ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    print("\n[DB에서 조회한 최신 추천 종목 데이터]")
    print(df_db_test)