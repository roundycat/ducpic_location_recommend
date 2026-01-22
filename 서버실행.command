#!/bin/bash

# 근방 여행지 추천 웹 서버 실행 스크립트
# 더블클릭으로 실행 가능

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# 터미널 창 제목 설정
echo -e "\033]0;근방 여행지 추천 서버\007"

clear
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║        🗺️  근방 여행지 추천 웹 서버                   ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo "⚠️  경고: .env 파일이 없습니다."
    echo ""
    echo "   .env 파일을 생성하고 다음 환경 변수를 설정해주세요:"
    echo "   - GEMINI_API_KEY"
    echo "   - GOOGLE_MAPS_API_KEY"
    echo ""
    echo "   예시:"
    echo "   GEMINI_API_KEY=your_key_here"
    echo "   GOOGLE_MAPS_API_KEY=your_key_here"
    echo ""
    read -p "계속하시겠습니까? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "프로그램을 종료합니다."
        sleep 2
        exit 1
    fi
fi

# Python 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경을 생성합니다..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ 가상환경 생성 실패. Python3가 설치되어 있는지 확인해주세요."
        echo ""
        read -p "계속하려면 Enter를 누르세요..."
        exit 1
    fi
fi

# 가상환경 활성화
echo "🔧 가상환경을 활성화합니다..."
source venv/bin/activate

# Python 경로 확인
if ! command -v python &> /dev/null; then
    echo "❌ Python을 찾을 수 없습니다."
    read -p "계속하려면 Enter를 누르세요..."
    exit 1
fi

# 의존성 설치 확인
if [ ! -f "venv/.installed" ]; then
    echo "📥 의존성을 설치합니다..."
    echo "   (처음 실행 시 시간이 걸릴 수 있습니다)"
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 의존성 설치 실패."
        echo ""
        read -p "계속하려면 Enter를 누르세요..."
        exit 1
    fi
    touch venv/.installed
    echo "✅ 의존성 설치 완료!"
    echo ""
fi

# Flask 설치 확인
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask가 설치되지 않았습니다. 설치합니다..."
    python -m pip install flask flask-cors
    if [ $? -ne 0 ]; then
        echo "❌ Flask 설치 실패."
        read -p "계속하려면 Enter를 누르세요..."
        exit 1
    fi
fi

# 서버 실행
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║  ✅ 서버를 시작합니다...                              ║"
echo "║                                                       ║"
echo "║  🌐 브라우저에서 표시된 주소로 접속하세요            ║"
echo "║     (기본: http://localhost:8080)                    ║"
echo "║                                                       ║"
echo "║  ⏹️  종료하려면 Ctrl+C를 누르세요                    ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

python app.py

# 서버 종료 후
echo ""
echo "서버가 종료되었습니다."
echo ""
read -p "이 창을 닫으려면 Enter를 누르세요..."
