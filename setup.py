"""Setup script for Adaptive Bayesian Closed-Loop Framework."""
import subprocess
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS = ROOT_DIR / "requirements.txt"


def check_python_version():
    version = sys.version_info[:2]
    if version < REQUIRED_PYTHON:
        print(f"ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
              f"found {version[0]}.{version[1]}")
        sys.exit(1)
    print(f"[OK] Python {version[0]}.{version[1]}.{sys.version_info[2]}")


def install_dependencies():
    print("\nInstalling dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
        print("[OK] Dependencies installed")
    except subprocess.CalledProcessError:
        print("[WARN] Some dependencies failed. Run: pip install -r requirements.txt")


def create_dirs():
    dirs = ["data/raw", "data/processed", "results/figures", "paper/figures"]
    for d in dirs:
        (ROOT_DIR / d).mkdir(parents=True, exist_ok=True)
    print("[OK] Directories created")


def verify_imports():
    print("\nVerifying installation...")
    required = {
        "numpy": "np",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "sklearn": "sklearn",
        "pandas": "pandas",
        "seaborn": "seaborn",
    }
    all_ok = True
    for pkg, name in required.items():
        try:
            __import__(name)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg}")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("Adaptive Bayesian Closed-Loop Framework - Setup")
    print("=" * 60)
    check_python_version()
    install_dependencies()
    create_dirs()
    all_ok = verify_imports()
    print()
    if all_ok:
        print("[OK] Setup complete!")
    else:
        print("[FAIL] Setup incomplete. Check errors above.")


if __name__ == "__main__":
    main()
