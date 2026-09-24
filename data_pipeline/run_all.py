# run_all.py
# just a lazy shortcut so I don't have to type 4 separate commands every
# time I want to test the whole pipeline. Runs scrape -> clean -> build_db
# -> queries in order.
#
# using sys.executable instead of just "python" because on my windows
# machine plain "python" wasn't recognized (had to use "py" instead) -
# sys.executable just means "whatever python is running THIS script"
# so it avoids that problem

import os
import sys

python_cmd = sys.executable

print("running scrape.py...")
os.system(f'"{python_cmd}" scrape.py')

print("running clean.py...")
os.system(f'"{python_cmd}" clean.py')

print("running build_db.py...")
os.system(f'"{python_cmd}" build_db.py')

print("running queries.py...")
os.system(f'"{python_cmd}" queries.py')

print("all done!")
