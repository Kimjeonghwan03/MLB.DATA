import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time


def crawl_mlb_articles(target_count=50):
    print("브라우저를 실행하여 MLB.com 보안 우회를 시도합니다...")

    # 1. 차단 우회용 크롬 드라이버 설정
    options = uc.ChromeOptions()
    # 처음에는 브라우저가 움직이는 것을 눈으로 확인하기 위해 Headless(숨김) 모드를 사용하지 않습니다.
    driver = uc.Chrome(options=options, version_main=149)

    # 2. 기사 URL을 수집할 리스트
    article_urls = []

    try:
        # 뉴스 메인 페이지 접속
        driver.get("https://www.mlb.com/news")
        print("MLB 메인 뉴스 페이지 로딩 중...")
        time.sleep(5)  # 페이지가 완전히 그려질 때까지 넉넉히 대기

        # 3. URL 수집 로직 (스크롤을 내리며 기사 링크 찾기)
        print("기사 링크 수집을 시작합니다...")
        last_height = driver.execute_script("return document.body.scrollHeight")

        while len(article_urls) < target_count:
            # 현재 화면에 있는 기사 링크(a 태그 중 href에 '/news/나 /article/'이 포함된 것) 찾기
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '/news/') or contains(@href, '/article/')]")

            for link in links:
                url = link.get_attribute('href')
                if url and url not in article_urls and "mlb.com" in url:
                    article_urls.append(url)
                    if len(article_urls) >= target_count:
                        break

            print(f"현재 찾은 URL 개수: {len(article_urls)} / {target_count}")
            if len(article_urls) >= target_count:
                break

            # 화면 맨 아래로 스크롤 내려서 새로운 기사 로딩 유도
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("더 이상 스크롤할 수 없습니다.")
                break
            last_height = new_height

        print(f"\n총 {len(article_urls)}개의 기사 URL 수집 완료! 이제 본문을 추출합니다.")

        # 4. 수집된 URL에 각각 접속하여 텍스트(본문) 추출
        scraped_data = []
        for i, url in enumerate(article_urls, 1):
            print(f"[{i}/{len(article_urls)}] 기사 추출 중: {url}")
            driver.get(url)
            time.sleep(3)  # 본문 로딩 대기

            paragraphs = driver.find_elements(By.TAG_NAME, 'p')
            article_text = ""
            for p in paragraphs:
                text = p.text.strip()
                # 의미 있는 길이의 영문 문장만 추가
                if len(text) > 30:
                    article_text += text + " "

            if article_text:
                scraped_data.append(article_text.strip())

            # 데이터 유실을 막기 위해 10개마다 중간 저장 (Checkpoint)
            if i % 10 == 0:
                temp_df = pd.DataFrame(scraped_data, columns=['text'])
                temp_df.to_csv("mlb_data_backup.csv", index=False, encoding="utf-8")

    except Exception as e:
        print(f"크롤링 중 에러 발생: {e}")

    finally:
        # 모든 작업이 끝나면 브라우저 닫기
        driver.quit()

    # 최종 결과를 데이터프레임으로 반환
    df = pd.DataFrame(scraped_data, columns=['text'])
    return df


# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 우선 테스트로 10개만 수집해 봅니다. 작동이 확인되면 이 숫자를 늘리세요.
    test_count = 10
    df_mlb = crawl_mlb_articles(target_count=test_count)

    if not df_mlb.empty:
        df_mlb.to_csv("mlb_raw_data_selenium.csv", index=False, encoding="utf-8")
        print("\n🎉 크롤링 성공! mlb_raw_data_selenium.csv 파일이 저장되었습니다.")
    else:
        print("\n수집된 데이터가 없습니다.")