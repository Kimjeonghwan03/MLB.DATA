import cloudscraper
from bs4 import BeautifulSoup
import time


def get_mlb_urls_from_sitemap(target_count=50000):
    urls = []
    sitemap_index_url = 'https://www.mlb.com'

    # requests 대신 cloudscraper를 사용하여 봇 차단을 우회합니다.
    scraper = cloudscraper.create_scraper()

    print("메인 사이트맵에 접속 중...")
    response = scraper.get(sitemap_index_url)

    # 만약 여기서도 200이 안 나오면 우회가 실패한 것입니다.
    if response.status_code != 200:
        print(f"접속 실패! 응답 코드: {response.status_code}")
        return urls

    soup = BeautifulSoup(response.content, 'xml')

    sitemap_tags = soup.find_all('loc')

    for tag in sitemap_tags:
        child_sitemap_url = tag.text

        # 기사 관련 사이트맵 찾기 (news, article, 혹은 sitemap-news 등)
        if 'news' in child_sitemap_url or 'article' in child_sitemap_url:
            print(f"하위 사이트맵 탐색 중: {child_sitemap_url}")
            child_resp = scraper.get(child_sitemap_url)
            child_soup = BeautifulSoup(child_resp.content, 'xml')

            article_tags = child_soup.find_all('loc')
            for article in article_tags:
                url = article.text
                if '/news/' in url or '/article/' in url:
                    urls.append(url)

                    if len(urls) >= target_count:
                        print(f"\n🎉 목표 URL {target_count}개 수집 완료!")
                        return urls

            time.sleep(1)
            print(f"현재까지 수집된 URL: {len(urls)}개")

    print(f"\n탐색 완료! 총 {len(urls)}개의 URL을 찾았습니다.")
    return urls


print("크롤링을 시작합니다. 화면에 진행 상황이 표시됩니다...")
target_number = 100  # 우선 100개로 테스트
mlb_article_urls = get_mlb_urls_from_sitemap(target_number)

if len(mlb_article_urls) > 0:
    with open('mlb_urls.txt', 'w', encoding='utf-8') as f:
        for item in mlb_article_urls:
            f.write(f"{item}\n")
    print(f"\n프로젝트 폴더에 [mlb_urls.txt] 파일이 생성되고 {len(mlb_article_urls)}개의 주소가 저장되었습니다!")
else:
    print("\n여전히 URL을 찾지 못했습니다. 사이트맵 구조가 변경되었을 수 있습니다.")