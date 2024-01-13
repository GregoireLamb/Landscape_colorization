import sys
import os

from src.frontend import main

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(script_dir, 'src'))

os.chdir(script_dir)
print("Starting application...")
main()
