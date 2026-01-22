# 🚀 실행 방법

## 더블클릭으로 실행하기 (가장 간단!)

### macOS 사용자

1. **`서버실행.command`** 파일을 더블클릭하세요
2. 터미널 창이 열리며 자동으로 서버가 시작됩니다
3. 터미널에 표시된 주소로 접속하세요 (기본: `http://localhost:8080`)

### 주의사항

- 처음 실행 시 의존성 설치에 시간이 걸릴 수 있습니다 (1-2분)
- `.env` 파일이 없으면 경고 메시지가 표시됩니다
- 서버를 종료하려면 터미널 창에서 `Ctrl+C`를 누르세요

## 다른 실행 방법

### 터미널에서 실행

```bash
# 쉘 스크립트
./run_server.sh

# Python 스크립트
python3 run_server.py
```

### Windows 사용자

`run_server.bat` 파일을 더블클릭하세요.

## 환경 변수 설정

프로젝트 폴더에 `.env` 파일을 만들고 다음 내용을 추가하세요:

```
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

## 문제 해결

### "권한이 거부되었습니다" 오류
터미널에서 다음 명령어를 실행하세요:
```bash
chmod +x 서버실행.command
```

### Python을 찾을 수 없습니다
Python 3가 설치되어 있는지 확인하세요:
```bash
python3 --version
```

### 포트가 이미 사용 중입니다
- 기본 포트는 8080입니다 (5000번 포트는 macOS의 AirPlay Receiver가 사용할 수 있습니다)
- 포트가 사용 중이면 자동으로 다른 포트를 찾아 사용합니다
- 터미널에 표시된 주소를 확인하세요
- 특정 포트를 사용하려면 `.env` 파일에 `PORT=원하는포트번호` 추가

