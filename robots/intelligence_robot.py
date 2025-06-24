# robots/intelligence_robot.py

import pandas as pd
import random
import datetime
import configparser
import requests
from bs4 import BeautifulSoup
import time
import re
from konlpy.tag import Okt  # KoNLPy의 Okt 형태소 분석기 사용
import sqlite3  # SQLite 데이터베이스 연동


class IntelligenceRobot:
    def __init__(self_obj, config_path='config/settings.ini'):
        self_obj.config = configparser.ConfigParser()
        self_obj.config.read(config_path, encoding='utf-8')
        self_obj.news_sources = [s.strip() for s in
                                 self_obj.config['MARKET_INTELLIGENCE']['NEWS_SOURCES'].strip('[]').replace("'",
                                                                                                            "").split(
                                     ',')]
        self_obj.keyword_list = [k.strip() for k in
                                 self_obj.config['MARKET_INTELLIGENCE']['KEYWORD_LIST'].strip('[]').replace("'",
                                                                                                            "").split(
                                     ',')]
        self_obj.naver_news_base_url = self_obj.config['MARKET_INTELLIGENCE']['NAVER_NEWS_BASE_URL']

        self_obj.positive_words = set(self_obj.config['MARKET_INTELLIGENCE']['POSITIVE_WORDS'].split(', '))
        self_obj.negative_words = set(self_obj.config['MARKET_INTELLIGENCE']['NEGATIVE_WORDS'].split(', '))

        self_obj.okt = Okt()  # Okt 형태소 분석기 초기화

        self_obj.db_path = 'data/robot_data.db'  # SQLite DB 경로
        self_obj._init_db()  # DB 초기화 메서드 호출

    def _init_db(self_obj):
        """뉴스 데이터를 저장할 SQLite 테이블 초기화."""
        conn = sqlite3.connect(self_obj.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT,
                sentiment TEXT,
                source TEXT,
                news_date TEXT,
                keyword TEXT,
                crawled_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _get_news_sentiment(self_obj, text):
        """
        뉴스 제목/내용 기반의 사전 기반 감성 분석.
        KoNLPy를 사용하여 형태소 분석 후 긍정/부정 단어 매칭.
        """
        tokens = self_obj.okt.morphs(text)  # 형태소 분석

        positive_score = 0
        negative_score = 0

        for token in tokens:
            if token in self_obj.positive_words:
                positive_score += 1
            elif token in self_obj.negative_words:
                negative_score += 1

        if positive_score > negative_score:
            return '긍정'
        elif negative_score > positive_score:
            return '부정'
        else:
            return '중립'

    def _crawl_naver_news(self_obj, keyword, num_pages=2):
        """네이버 뉴스에서 특정 키워드 뉴스를 크롤링합니다."""
        crawled_news = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        for i in range(num_pages):
            start_page = i * 10 + 1
            url = self_obj.naver_news_base_url.format(keyword=keyword, start_page=start_page)

            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                news_items = soup.select('div.news_wrap.api_ani_send')

                for item in news_items:
                    title_tag = item.select_one('a.news_tit')
                    source_tag = item.select_one('a.dsc_thumb')
                    date_tag = item.select_one('div.dsc_wrap div.info_group span.info')

                    title = title_tag.get_text(strip=True) if title_tag else '제목 없음'
                    link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else '링크 없음'
                    source = source_tag['aria-label'] if source_tag and 'aria-label' in source_tag.attrs else '출처 없음'

                    date_str = date_tag.get_text(strip=True) if date_tag else '날짜 없음'
                    if "시간 전" in date_str or "분 전" in date_str:
                        news_date = datetime.date.today().strftime('%Y-%m-%d')
                    elif "일 전" in date_str:
                        days_ago = int(date_str.replace("일 전", "").strip())
                        news_date = (datetime.date.today() - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
                    else:
                        news_date = date_str.replace('.', '-')[:-1]

                    sentiment = self_obj._get_news_sentiment(title)  # 형태소 분석 기반 감성 분석

                    crawled_news.append({
                        '제목': title,
                        '링크': link,
                        '감성': sentiment,
                        '출처': source,
                        '날짜': news_date,
                        '키워드': keyword,  # 어떤 키워드로 크롤링했는지 저장
                        '크롤링_시간': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                time.sleep(random.uniform(1, 2))
            except requests.exceptions.RequestException as e:
                print(f"-> 뉴스 크롤링 오류 발생: {e}")
                continue
            except Exception as e:
                print(f"-> 뉴스 파싱 중 오류 발생: {e}")
                continue
        return crawled_news

    def run_intelligence(self_obj, keyword=None, time_period='최근 1주', filename_keyword=None):
        """
        시장 정보를 탐색하고 분석 결과를 반환합니다.
        :param keyword: 탐색할 키워드 (None이면 설정된 키워드 중 랜덤 선택)
        :param time_period: 검색 기간 (현재는 네이버 크롤링에 직접 반영되지 않음)
        :param filename_keyword: 파일명으로 사용할 클린된 키워드 (main.py에서 전달)
        :return: 시장 감성 (str) 및 분석된 뉴스 데이터 (DataFrame)
        """
        if keyword is None:
            keyword = random.choice(self_obj.keyword_list)
            if filename_keyword is None:  # 랜덤 키워드 선택 시 파일명용 키워드 생성
                filename_keyword = "".join(char for char in keyword if char.isalnum() or char.isspace() or char == '_')
                filename_keyword = filename_for_keyword.replace(" ", "_").strip('_')
                if not filename_keyword:
                    filename_keyword = "default_keyword"

        print(f"\n--- [Market Intelligence Robot] 실행: '{keyword}' ({time_period} 검색) ---")

        crawled_data = self_obj._crawl_naver_news(keyword)
        if not crawled_data:
            print("-> 뉴스 데이터를 수집하지 못했습니다. 시뮬레이션 데이터로 대체합니다.")
            news_titles_sim = [
                f"속보: {keyword} 관련 산업 동향 변화 감지 (시뮬)",
                f"{keyword} 테마주 급등세 지속될까? (시뮬)",
            ]
            sentiment_labels_sim = ['긍정', '중립', '부정']
            results_sim = []
            for i in range(random.randint(5, 10)):
                results_sim.append({
                    '제목': random.choice(news_titles_sim),
                    '링크': '링크 없음',
                    '감성': random.choice(sentiment_labels_sim),
                    '출처': random.choice(self_obj.news_sources),
                    '날짜': (datetime.date.today() - datetime.timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
                    '키워드': keyword,
                    '크롤링_시간': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            df_news = pd.DataFrame(results_sim)
        else:
            df_news = pd.DataFrame(crawled_data)

        positive_count = df_news[df_news['감성'] == '긍정'].shape[0]
        negative_count = df_news[df_news['감성'] == '부정'].shape[0]

        overall_sentiment = "중립"
        if positive_count > negative_count * 1.5:
            overall_sentiment = "긍정적"
        elif negative_count > positive_count * 1.5:
            overall_sentiment = "부정적"

        print(f"-> 탐색 키워드: '{keyword}'")
        print(f"-> 총 {df_news.shape[0]}건의 뉴스 기사 분석 완료.")
        print(f"-> 종합 시장 감성: {overall_sentiment}")
        print("\n[분석된 뉴스 요약]")
        print(df_news[['제목', '출처', '날짜', '감성']].to_string())

        # SQLite DB 컬럼명에 맞게 DataFrame 컬럼명 변경
        df_news_for_db = df_news.rename(columns={
            '제목': 'title',
            '링크': 'link',
            '감성': 'sentiment',
            '출처': 'source',
            '날짜': 'news_date',
            '키워드': 'keyword',
            '크롤링_시간': 'crawled_at'
        })

        # SQLite DB에 저장
        # to_sql 함수는 DataFrame의 컬럼 순서와 DB 테이블의 컬럼 순서가 달라도 컬럼명이 일치하면 자동으로 매핑합니다.
        # 따라서, rename만 해주면 됩니다.
        conn = sqlite3.connect(self_obj.db_path)
        df_news_for_db.to_sql('news_data', conn, if_exists='append', index=False)  # 변경된 DataFrame 사용
        conn.close()
        print(f"-> 분석된 뉴스 데이터가 '{self_obj.db_path}'의 'news_data' 테이블에 저장되었습니다.")

        # CSV로도 저장 (선택 사항, 필요 없으면 제거 가능)
        # CSV 저장은 한글 컬럼명으로 해도 되므로 df_news 원본을 사용
        df_news.to_csv(f'data/mi_news_data_{datetime.date.today().strftime("%Y%m%d")}_{filename_keyword}.csv',
                       index=False, encoding='utf-8-sig')
        print(
            f"-> 분석된 뉴스 데이터가 'data/mi_news_data_{datetime.date.today().strftime('%Y%m%d')}_{filename_keyword}.csv' 에도 저장되었습니다.")

        return overall_sentiment, df_news


# 테스트 코드 (직접 실행 시)
if __name__ == "__main__":
    mi_robot = IntelligenceRobot()
    sentiment, news_data = mi_robot.run_intelligence(keyword="반도체", filename_keyword="반도체")
    print(f"\n최종 Market Intelligence 결과 (감성): {sentiment}")
    print(news_data.head())

    # DB에서 데이터 조회 테스트
    conn = sqlite3.connect(mi_robot.db_path)
    df_db_test = pd.read_sql_query("SELECT * FROM news_data ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    print("\n[DB에서 조회한 최신 뉴스 데이터]")
    print(df_db_test)
    