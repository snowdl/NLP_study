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

