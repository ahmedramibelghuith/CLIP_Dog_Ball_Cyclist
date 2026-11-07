"""Point d'entrée local : python main.py --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from clip_project.cli import main

if __name__ == "__main__":
    main()
