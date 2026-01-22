#!/bin/bash

# 근방 여행지 추천 웹 서버 실행 스크립트

echo "🚀 근방 여행지 추천 웹 서버를 시작합니다..."
echo ""

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo "⚠️  경고: .env 파일이 없습니다."
    echo "   .env 파일을 생성하고 다음 환경 변수를 설정해주세요:"
    echo "   - GEMINI_API_KEY"
    echo "   - GOOGLE_MAPS_API_KEY"
    echo ""
    read -p "계속하시겠습니까? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Python 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경을 생성합니다..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경을 활성화합니다..."
source venv/bin/activate

# 의존성 설치 확인
if [ ! -f "venv/.installed" ]; then
    echo "📥 의존성을 설치합니다..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# 서버 실행
echo ""
echo "✅ 서버를 시작합니다..."
echo "   브라우저에서 표시된 주소로 접속하세요 (기본: http://localhost:8080)"
echo "   종료하려면 Ctrl+C를 누르세요"
echo ""

python app.py

