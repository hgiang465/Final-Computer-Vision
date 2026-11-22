from pathlib import Path
from runpy import run_path


def run_script(name):
    script = Path(__file__).resolve().with_name(f"{name}.py")
    run_path(str(script), run_name="__main__")
