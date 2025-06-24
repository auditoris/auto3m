import dart_fss as dfs
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DART API 키 설정 (본인의 키로 변경)
# 실제 앱에서는 settings.ini에서 가져오겠지만, 테스트를 위해 여기에 직접 넣어봅니다.
YOUR_DART_API_KEY = "80c0229e5fa4cf75ac80cce218d32404c7de0842" # 실제 DART API 키로 변경하세요!
dfs.set_api_key(YOUR_DART_API_KEY)

# 모든 상장된 기업 리스트 불러오기
corp_list = dfs.get_corp_list()

# 삼성전자를 이름으로 찾기 ( 리스트 반환 )
# samsung = corp_list.find_by_name('삼성전자', exactly=True)[0]

# 증권 코드를 이용한 찾기
# samsung = corp_list.find_by_stock_code('005930')

# 다트에서 사용하는 회사코드를 이용한 찾기
# samsung = corp_list.find_by_corp_code('00126380')

# "삼성"을 포함한 모든 공시 대상 찾기
# corps = corp_list.find_by_name('삼성')

# "삼성"을 포함한 모든 공시 대상중 코스피 및 코스닥 시장에 상장된 공시 대상 검색(Y: 코스피, K: 코스닥, N:코넥스, E:기타)
# corps = corp_list.find_by_name('삼성', market=['Y','K']) # 아래와 동일
# corps = corp_list.find_by_name('삼성', market='YK')

# "휴대폰" 생산품과 연관된 공시 대상
corps = corp_list.find_by_product('휴대폰')
if not isinstance(corps, pd.DataFrame):
    logger.info(f"corp_list.find_by_product 반환 타입: {type(corps)}")
    corps_df = pd.DataFrame(corps)
logger.info(corps_df.head())

# "휴대폰" 생산품과 연관된 공시 대상 중 코스피 시장에 상장된 대상만 검색
# corps = corp_list.find_by_product('휴대폰', market='Y')

# 섹터 리스트 확인
logger.info(corp_list.sectors)

# "텔레비전 방송업" 섹터 검색
corps = corp_list.find_by_sector('1차 철강 제조업')
if not isinstance(corps, pd.DataFrame):
    logger.info(f"corp_list.find_by_sector 반환 타입: {type(corps)}")
    corps_df = pd.DataFrame(corps)
logger.info(corps_df.head())

# 2012년 1월 1일부터 현재까지 분기 연결재무제표 검색 (연간보고서, 반기보고서 포함)
# 삼성전자
corp_code = '00126380'

cinfo = dfs.corp.Corp(corp_code=corp_code).to_dict()
# if not isinstance(cinfo, pd.DataFrame):
#     logger.info(f"company 반환 타입: {type(cinfo)}")
#     cinfo = pd.DataFrame(cinfo)
logger.info(cinfo)

# 2012년 01월 01일 부터 연결재무제표 검색
# samsung = corp_list.find_by_corp_code(corp_code=corp_code)
# fs = samsung.extract_fs(bgn_de='20120101') 와 동일
fs_df = dfs.fs.extract(corp_code=corp_code, bgn_de='20250601')
if not isinstance(fs_df, pd.DataFrame):
    logger.info(f"extract_fs 반환 타입: {type(fs_df)}")
    fs_df = pd.DataFrame(fs_df)
logger.info(fs_df.head())

# -----------------------------------------------------------
# try:
#     logger.info("dfs.get_corp_list() 호출 중...")
#     corp_list_obj = dfs.get_corp_list()
#
#     logger.info(f"dfs.get_corp_list() 반환 타입: {type(corp_list_obj)}")
#     logger.info(f"corp_list_obj 객체 정보:\n{help(corp_list_obj)}") # 이 부분에서 중요한 정보가 나옵니다!
#
#     # 반환된 객체가 DataFrame인지 확인
#     if isinstance(corp_list_obj, pd.DataFrame):
#         logger.info("반환된 객체는 이미 DataFrame입니다.")
#         corp_list_df = corp_list_obj
#     # .to_df() 메서드가 있는지 확인 (이전 시도)
#     elif hasattr(corp_list_obj, 'to_df'):
#         logger.info("반환된 객체에 .to_df() 메서드가 있습니다. 이를 사용하여 DataFrame으로 변환합니다.")
#         corp_list_df = corp_list_obj.to_df()
#     # 직접 pd.DataFrame()으로 변환 가능한지 확인 (직전 시도)
#     else:
#         logger.info("반환된 객체를 pd.DataFrame()으로 직접 변환 시도합니다.")
#         # corp_list_df = pd.DataFrame(corp_list_obj)
#         corp_list_df = pd.DataFrame(corp_list_obj.corps)
#         logger.info(f"corp_list_df 성공적으로 변환. 총 {len(corp_list_df)}개 기업.")
#         logger.info(f"corp_list_df 컬럼: {corp_list_df.columns.tolist()}")
#         logger.info(corp_list_df.head(5))
#
#     # DataFrame이 비어 있는지 확인 (오류가 없어야 함)
#     if corp_list_df.empty:
#         logger.warning("변환된 corp_list_df가 비어 있습니다.")
#     else:
#         logger.info(f"corp_list_df 성공적으로 변환. 총 {len(corp_list_df)}개 기업.")
#         logger.info(f"corp_list_df 컬럼: {corp_list_df.columns.tolist()}")
#         logger.info(f"corp_list_df 상위 5개 행:\n{corp_list_df.head().to_string()}")
#
# except Exception as e:
#     logger.error(f"dfs.get_corp_list() 테스트 중 오류 발생: {e}")

# ------------------------------------------------------------------------------
# import FinanceDataReader as fdr
# import datetime
#
# try:
#     # 삼성전자 1일치 주가 데이터 가져오기 (비교적 가벼운 요청)
#     df_test = fdr.DataReader('005930', datetime.date.today() - datetime.timedelta(days=7))
#     print("삼성전자 주가 데이터 로드 성공:")
#     print(df_test.head())
# except Exception as e:
#     print(f"삼성전자 주가 데이터 로드 실패: {e}")
#
# try:
#     # 미국 시장 종목 리스트 가져오기 (KRX와 다른 API 엔드포인트 사용)
#     df_us_test = fdr.StockListing('NASDAQ')
#     print("\nNASDAQ 종목 리스트 로드 성공:")
#     print(df_us_test.head())
# except Exception as e:
#     print(f"NASDAQ 종목 리스트 로드 실패: {e}")

# ----------------------------------------------------------------------
# import FinanceDataReader as fdr
#
# try:
#     stock_info = fdr.StockInfo("005930")  # 예시: 삼성전자 종목 코드
#     print(stock_info)
# except Exception as e:
#     print(f"오류 발생: {e}")


# ------------------------------------------------------------------------
# import FinanceDataReader as fdr
# import pandas as pd
# import logging
#
# # 로깅 설정 (콘솔에 출력되도록 설정)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# def get_and_print_fdr_columns():
#     """
#     FinanceDataReader를 사용하여 KRX 상장 종목 리스트를 가져오고
#     DataFrame의 컬럼 헤더와 상위 몇 개 행을 출력하는 함수.
#     """
#     try:
#         logger.info("FinanceDataReader를 사용하여 KRX 상장 종목 리스트 가져오는 중...")
#         # KRX 상장 종목 전체 리스트 가져오기
#         df_krx = fdr.StockListing('KRX')
#
#         if df_krx.empty:
#             logger.warning("FinanceDataReader에서 KRX 상장 종목 리스트를 가져오지 못했습니다. DataFrame이 비어 있습니다.")
#             return
#
#         logger.info(f"성공적으로 KRX 상장 종목 리스트를 가져왔습니다. 총 {len(df_krx)}개 종목.")
#
#         # 컬럼 헤더 출력
#         logger.info(f"DataFrame 컬럼 헤더: {df_krx.columns.tolist()}")
#
#         # 상위 5개 행 출력 (데이터 형태 확인용)
#         logger.info("DataFrame 상위 5개 행:\n" + df_krx.head().to_string())
#
#     except Exception as e:
#         logger.error(f"FinanceDataReader 데이터 가져오기 중 오류 발생: {e}")
#
# if __name__ == "__main__":
#     get_and_print_fdr_columns()