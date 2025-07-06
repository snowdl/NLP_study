import sys
import os
import logging
from pathlib import Path
import ipynbname

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def set_project_root(levels_up=None):
    """
    Automatically sets the working directory to the project root.

    It first tries to move 2 levels up from the current file or notebook.
    If the path does not exist or fails, it tries 3 levels up.
    This ensures compatibility across different script or notebook locations.

    This function helps maintain consistent relative paths in a shared project.
    """
    levels_up = levels_up or [2, 3]  # 기본값 [2, 3]을 사용
    
    for level in levels_up:
        try:
            # For Jupyter Notebook
            try:
                notebook_path = Path(ipynbname.path())
                root = notebook_path.parents[level]
            except Exception:
                # For .py scripts
                root = Path(__file__).resolve().parents[level]
            
            # 루트 디렉토리 설정
            if root.exists():
                # 현재 디렉토리가 루트와 다를 경우에만 디렉토리 변경
                if os.getcwd() != str(root):
                    os.chdir(root)
                    logging.info(f"✅ Project root set to ({level} levels up): {os.getcwd()}")
                else:
                    logging.info("⚠️ No need to change the working directory.")
                return  # 루트를 찾으면 바로 종료

        except Exception as e:
            # 예외 처리 강화: 루트 디렉토리 찾지 못한 경우 메시지 출력
            logging.error(f"❌ Failed to find project root (Level: {level}, Exception: {e})")
    
    # 프로젝트 루트를 찾지 못한 경우
    logging.error("❌ Could not determine project root directory.")
import os
from pathlib import Path

def set_project_root():
    """
    Automatically sets the working directory to the project root.

    It first tries to move 2 levels up from the current file or notebook.
    If the path does not exist or fails, it tries 3 levels up.
    This ensures compatibility across different script or notebook locations.

    This function helps maintain consistent relative paths in a shared project.
    """
    levels = [2, 3]  # First try 2 levels up, then 3 levels up
    for level in levels:
        try:
            # For Jupyter Notebook
            import ipynbname
            notebook_path = Path(ipynbname.path())
            root = notebook_path.parents[level]
        except Exception:
            # For .py scripts
            root = Path(__file__).resolve().parents[level]
        
        if root.exists():
            os.chdir(root)
            print(f"✅ Project root set to ({level} levels up): {os.getcwd()}")
            return
        
    print("❌ Could not determine project root directory.")

