import subprocess
import sys

# 启动Streamlit应用
subprocess.run([sys.executable, "-m", "streamlit", "run", "qr_rs_visualizer.py"])
