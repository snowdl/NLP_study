import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f'Project root added to sys.path: {project_root}')
