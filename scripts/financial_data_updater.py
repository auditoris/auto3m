# financial_data_updater.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import time
import datetime
import os
import logging
import FinanceDataReader as fdr # FinanceDataReader 임포트
import random

# 로깅 설정 (콘솔 출력 및 파일 저장)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 데이터베이스 파일 경로
DB_FILE = 'financial_data.db'

# 프로젝트 루트 폴더에 'data' 폴더를 만들어 DB를 저장할 경우
DB_FILE = os.path.join(os.getcwd(), '../data', 'financial_data.db')
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True) # 폴더 생성

def get_stock_list_from_fdr():
    """
    FinanceDataReader를 사용하여 KOSPI, KOSDAQ 전체 종목 리스트를 가져옵니다.
    """
    logger.info("FinanceDataReader로 KOSPI 및 KOSDAQ 전체 종목 목록 로드 중...")

    try:
        # KOSPI 전체 종목 가져오기
        # 'Code' 컬럼을 'Symbol'로 변경합니다.
        kospi_stocks = fdr.StockListing('KOSPI')[['Code', 'Name']].rename(
            columns={'Code': 'Symbol', 'Name': 'CompanyName'})
        logger.info(f"FinanceDataReader로 KOSPI 전체 종목 수: {len(kospi_stocks)}")

        # KOSDAQ 전체 종목 가져오기
        # 'Code' 컬럼을 'Symbol'로 변경합니다.
        kosdaq_stocks = fdr.StockListing('KOSDAQ')[['Code', 'Name']].rename(
            columns={'Code': 'Symbol', 'Name': 'CompanyName'})
        logger.info(f"FinanceDataReader로 KOSDAQ 전체 종목 수: {len(kosdaq_stocks)}")

        # 두 DataFrame을 합치고 중복 제거 (Symbol 기준)
        all_stocks = pd.concat([kospi_stocks, kosdaq_stocks]).drop_duplicates(subset=['Symbol']).reset_index(drop=True)

        # 종목코드를 6자리 문자열로 포맷팅 (예: '005930')
        all_stocks['Symbol'] = all_stocks['Symbol'].astype(str).str.zfill(6)

        logger.info(f"총 로드된 KOSPI/KOSDAQ 전체 종목 수: {len(all_stocks)}개")
        return all_stocks

    except Exception as e:
        logger.error(f"FinanceDataReader로 KOSPI/KOSDAQ 전체 종목 로드 중 오류 발생: {e}. 빈 DataFrame 반환.")
        return pd.DataFrame(columns=['Symbol', 'CompanyName'])

def get_stock_list_350_from_fdr():
    """
    FinanceDataReader를 사용하여 KOSPI 200과 KOSDAQ 150 종목 리스트를 가져옵니다.
    """
    logger.info("FinanceDataReader로 KOSPI 200 및 KOSDAQ 150 종목 목록 로드 중...")

    all_stocks_df = pd.DataFrame()

    try:
        # KOSPI 200 종목 가져오기
        # 'KRX/INDEX/STOCK/1028'은 KOSPI 200의 구성 종목을 가져오는 SnapDataReader 코드입니다.
        kospi200_df = fdr.SnapDataReader('KRX/INDEX/STOCK/1028')
        logger.info(f"FinanceDataReader로 KOSPI 200 종목 수: {len(kospi200_df)}")

        # 컬럼명 확인 및 조정: FinanceDataReader의 SnapDataReader 결과 컬럼명은 'Code'와 'Name'일 가능성이 높습니다.
        if 'Code' in kospi200_df.columns and 'Name' in kospi200_df.columns:
            kospi200_df = kospi200_df.rename(columns={'Code': 'Symbol', 'Name': 'CompanyName'})
            # 종목코드를 6자리 문자열로 포맷팅
            kospi200_df['Symbol'] = kospi200_df['Symbol'].astype(str).str.zfill(6)
            all_stocks_df = pd.concat([all_stocks_df, kospi200_df[['Symbol', 'CompanyName']]])
        else:
            logger.warning("KOSPI 200 데이터의 예상 컬럼('Code', 'Name')이 없습니다. FinanceDataReader 버전을 확인하세요.")

        # KOSDAQ 150 종목 가져오기
        # KOSDAQ 150의 SnapDataReader 코드는 'KRX/INDEX/STOCK/2028'로 추정됩니다.
        # 만약 오류가 발생하면 fdr.SnapDataReader('KRX/INDEX/LIST')를 통해 정확한 코드를 찾아야 합니다.
        kosdaq150_df = fdr.SnapDataReader('KRX/INDEX/STOCK/2028')
        logger.info(f"FinanceDataReader로 KOSDAQ 150 종목 수: {len(kosdaq150_df)}")

        if 'Code' in kosdaq150_df.columns and 'Name' in kosdaq150_df.columns:
            kosdaq150_df = kosdaq150_df.rename(columns={'Code': 'Symbol', 'Name': 'CompanyName'})
            kosdaq150_df['Symbol'] = kosdaq150_df['Symbol'].astype(str).str.zfill(6)
            all_stocks_df = pd.concat([all_stocks_df, kosdaq150_df[['Symbol', 'CompanyName']]])
        else:
            logger.warning("KOSDAQ 150 데이터의 예상 컬럼('Code', 'Name')이 없습니다. FinanceDataReader 버전을 확인하세요.")

        # 합쳐진 종목 리스트에서 중복 제거 및 인덱스 초기화
        filtered_stocks = all_stocks_df.drop_duplicates(subset=['Symbol']).reset_index(drop=True)
        logger.info(f"총 필터링된 종목 수 (KOSPI 200 + KOSDAQ 150): {len(filtered_stocks)}개")
        return filtered_stocks

    except Exception as e:
        logger.error(f"FinanceDataReader로 벤치마크 종목 로드 중 오류 발생: {e}. 모든 상장 종목을 가져오는 방식으로 대체합니다.")
        # 오류 발생 시 모든 종목을 가져오는 기존 방식(KOSPI + KOSDAQ)으로 대체 (최후의 수단)
        try:
            kospi_stocks = fdr.StockListing('KOSPI')[['Symbol', 'Name']].rename(columns={'Name': 'CompanyName'})
            kosdaq_stocks = fdr.StockListing('KOSDAQ')[['Symbol', 'Name']].rename(columns={'Name': 'CompanyName'})
            return pd.concat([kospi_stocks, kosdaq_stocks]).drop_duplicates(subset=['Symbol']).reset_index(drop=True)
        except Exception as e_fdr_fallback:
            logger.error(f"FinanceDataReaderFallback: 모든 종목 로드 중 오류 발생: {e_fdr_fallback}. 빈 DataFrame 반환.")
            return pd.DataFrame(columns=['Symbol', 'CompanyName'])

def create_db_table():
    """SQLite 데이터베이스 테이블을 생성합니다."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            symbol TEXT PRIMARY KEY,
            company_name TEXT,
            per REAL,
            roe REAL,
            debt_ratio REAL,
            sector TEXT,
            last_updated TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"데이터베이스 테이블이 {DB_FILE}에 생성되거나 이미 존재합니다.")


def get_financial_data_from_naver(stock_code):
    """
    네이버 금융에서 특정 종목의 재무 데이터와 섹터 정보를 크롤링합니다.
    """
    url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP 오류가 발생하면 예외 발생
        soup = BeautifulSoup(response.text, 'html.parser')

        per = roe = debt_ratio = None
        sector = "정보없음"

        # PER, ROE 추출
        # 네이버 금융 메인 페이지의 투자지표 테이블 구조에 따라 변경될 수 있음
        # 현재 (2025년 6월 기준) div.section.invest_info > table.type2 > tbody > tr
        # td 값들을 순회하며 찾습니다.
        try:
            invest_table = soup.select_one('div.section.invest_info table.type2 tbody')
            if invest_table:
                # PER과 ROE를 포함하는 행을 찾기 (텍스트 기반)
                for tr in invest_table.find_all('tr'):
                    th_text = tr.find('th').get_text(strip=True) if tr.find('th') else ''
                    td_value = tr.find('em').get_text(strip=True).replace(',', '') if tr.find('em') else ''

                    if 'PER' in th_text and td_value.replace('.', '', 1).isdigit():
                        per = float(td_value)
                    elif 'ROE' in th_text and td_value.replace('.', '', 1).isdigit():
                        roe = float(td_value)
        except Exception as e:
            logger.warning(f"[{stock_code}] PER/ROE 추출 오류: {e}")

        # 부채비율 추출 (재무제표 탭으로 이동해야 할 수도 있음. 일단 메인 페이지 시도)
        # 네이버 금융은 메인 페이지에 주요 지표만 있고, 부채비율은 재무제표 탭에 있는 경우가 많음.
        # 여기서는 간단히 PER, ROE만 시도하고, 부채비율은 별도 로직으로 추가하거나 제외
        # 만약 재무제표 탭으로 가야 한다면 URL 변경 후 동일한 BS4 로직 필요
        # 예시로 '0'으로 설정하거나, 재무제표 탭에서 추출하는 로직 추가 필요
        debt_ratio = 0.0  # 기본값. 실제 구현 시 재무제표 탭에서 추출 로직 추가 권장

        # 섹터 정보 추출 (종목 개요 섹션)
        # 보통 "업종" 또는 "산업군" 등으로 표시됨
        try:
            # 네이버 금융에서 섹터(업종) 정보는 종목 메인 페이지 상단에 있습니다.
            # <a href="/sise/sise_group_detail.naver?type=upjong&no=101">서비스업</a> 이런 형태
            sector_element = soup.select_one('a[href*="sise_group_detail.naver?type=upjong"]')
            if sector_element:
                sector = sector_element.get_text(strip=True)
            else:
                logger.warning(f"[{stock_code}] 섹터 정보 링크를 찾을 수 없습니다.")
        except Exception as e:
            logger.warning(f"[{stock_code}] 섹터 추출 오류: {e}")

        return {
            'per': per,
            'roe': roe,
            'debt_ratio': debt_ratio,  # 실제 크롤링 로직 추가 필요
            'sector': sector
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[{stock_code}] 네이버 금융 접속 오류: {e}")
        return None
    except Exception as e:
        logger.error(f"[{stock_code}] 네이버 금융 데이터 파싱 중 예상치 못한 오류: {e}")
        return None



def update_financial_data_to_db():
    """
    네이버 금융에서 재무 데이터를 크롤링하여 SQLite DB에 업데이트합니다.
    """
    create_db_table()
    # 종목 리스트를 FinanceDataReader에서 자동으로 가져오도록 변경
    stock_list_df = get_stock_list_from_fdr()

    if stock_list_df.empty:
        logger.error("업데이트할 종목 목록이 비어 있습니다. 작업을 중단합니다.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    total_stocks = len(stock_list_df)
    for i, row in stock_list_df.iterrows():
        symbol = row['Symbol']
        company_name = row['CompanyName']

        logger.info(f"[{i + 1}/{total_stocks}] {company_name} ({symbol}) 재무 데이터 크롤링 시작...")
        data = get_financial_data_from_naver(symbol)

        if data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO financial_data (symbol, company_name, per, roe, debt_ratio, sector, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, company_name, data['per'], data['roe'], data['debt_ratio'], data['sector'], current_time))
                conn.commit()
                logger.info(f"  -> {company_name} ({symbol}) 데이터 DB에 저장 완료.")
            except sqlite3.Error as e:
                logger.error(f"  -> {company_name} ({symbol}) DB 저장 오류: {e}")
        else:
            logger.warning(f"  -> {company_name} ({symbol}) 재무 데이터 크롤링 실패 또는 데이터 없음. DB에 저장하지 않음.")

        time.sleep(random.uniform(1, 3))  # IP 차단 방지를 위해 랜덤 시간 지연

    conn.close()
    logger.info("재무 데이터 업데이트 완료.")


if __name__ == "__main__":
    update_financial_data_to_db()
