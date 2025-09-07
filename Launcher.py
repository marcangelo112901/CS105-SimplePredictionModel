import os
import subprocess

script = os.path.abspath("App.py")
subprocess.run(["streamlit", "run", script])