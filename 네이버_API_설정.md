# 네이버 검색 API 설정 가이드

네이버 검색 API를 사용하면 더 많은 검색 결과를 얻을 수 있습니다.

## 1. 네이버 개발자 센터에서 API 키 발급

1. [네이버 개발자 센터](https://developers.naver.com/)에 접속
2. 로그인 후 "Application" → "애플리케이션 등록" 클릭
3. 애플리케이션 정보 입력:
   - 애플리케이션 이름: 원하는 이름
   - 사용 API: **검색** 선택
   - 비로그인 오픈 API 서비스 환경: **WEB 설정** 선택
   - 서비스 URL: `http://localhost:8080` (또는 사용하는 도메인)
4. 등록 후 **Client ID**와 **Client Secret** 확인

## 2. .env 파일에 추가

프로젝트 루트의 `.env` 파일에 다음을 추가하세요:

```bash
NAVER_CLIENT_ID=your_naver_client_id_here
NAVER_CLIENT_SECRET=your_naver_client_secret_here
```

## 3. 사용 효과

네이버 API를 설정하면:
- **Google CSE**: 최대 50개 결과
- **네이버 일반 검색**: 최대 30개 결과
- **네이버 블로그 검색**: 최대 30개 결과
- **총 최대 110개**의 검색 결과를 수집할 수 있습니다!

## 4. API 사용량 제한

- 네이버 검색 API는 **일일 25,000건**까지 무료로 사용 가능합니다
- 일반적으로 충분한 사용량입니다

## 5. 선택 사항

네이버 API 키가 없어도 Google CSE만으로도 작동합니다.
다만 네이버 블로그와 네이버 검색 결과를 추가로 얻을 수 없습니다.

