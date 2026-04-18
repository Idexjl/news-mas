import os
import sys

# Ensure project root is importable so `from src.X import Y` works without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
