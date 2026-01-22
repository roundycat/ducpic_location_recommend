#!/usr/bin/env python3
"""
근방 여행지 추천 웹 서버 실행 스크립트
크로스 플랫폼 지원 (Windows, macOS, Linux)
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_step(message):
    """단계별 메시지 출력"""
    print(f"\n{'='*50}")
    print(f"  {message}")
    print(f"{'='*50}")

def check_env_file():
    """환경 변수 파일 확인"""
    env_path = Path(".env")
    if not env_path.exists():
        print("\n⚠️  경고: .env 파일이 없습니다.")
        print("   .env 파일을 생성하고 다음 환경 변수를 설정해주세요:")
        print("   - GEMINI_API_KEY")
        print("   - GOOGLE_MAPS_API_KEY")
        print()
        response = input("계속하시겠습니까? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            sys.exit(1)

def setup_venv():
    """가상환경 설정"""
    venv_path = Path("venv")
    is_windows = platform.system() == "Windows"
    
    if not venv_path.exists():
        print_step("📦 가상환경을 생성합니다...")
        python_cmd = "python" if is_windows else "python3"
        subprocess.run([python_cmd, "-m", "venv", "venv"], check=True)
    
    # 가상환경 활성화 경로
    if is_windows:
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return python_exe, pip_exe

def install_dependencies(pip_exe):
    """의존성 설치"""
    installed_marker = Path("venv") / ".installed"
    
    if not installed_marker.exists():
        print_step("📥 의존성을 설치합니다...")
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
        installed_marker.touch()
    else:
        print("✅ 의존성이 이미 설치되어 있습니다.")

def run_server(python_exe):
    """서버 실행"""
    print_step("✅ 서버를 시작합니다...")
    print("\n   🌐 브라우저에서 표시된 주소로 접속하세요 (기본: http://localhost:8080)")
    print("   ⏹️  종료하려면 Ctrl+C를 누르세요\n")
    
    try:
        subprocess.run([str(python_exe), "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 서버를 종료합니다. 안녕히 가세요!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)

def main():
    """메인 함수"""
    print("\n" + "="*50)
    print("  🗺️  근방 여행지 추천 웹 서버")
    print("="*50)
    
    # 현재 디렉토리 확인
    if not Path("app.py").exists():
        print("❌ 오류: app.py 파일을 찾을 수 없습니다.")
        print("   이 스크립트를 프로젝트 루트 디렉토리에서 실행해주세요.")
        sys.exit(1)
    
    # 환경 변수 파일 확인
    check_env_file()
    
    # 가상환경 설정
    python_exe, pip_exe = setup_venv()
    
    # 의존성 설치
    install_dependencies(pip_exe)
    
    # 서버 실행
    run_server(python_exe)

if __name__ == "__main__":
    main()

