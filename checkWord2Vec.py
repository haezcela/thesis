import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Please install it manually.")
        sys.exit(1)

def check_installation(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing...")
        install_package(package)

def verify_installation(package):
    try:
        module = __import__(package)
        print(f"{package} version: {module.__version__}")
    except ImportError:
        print(f"Failed to import {package} after installation.")

def main():
    # Check and install pandas
    check_installation("pandas")
    verify_installation("pandas")

    # Check and install gensim
    check_installation("gensim")
    verify_installation("gensim")

if __name__ == "__main__":
    main()
