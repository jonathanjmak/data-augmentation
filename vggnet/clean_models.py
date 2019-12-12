import os
from glob import glob

for fn in glob("run_logs/*/best_model.path"):
  os.remove(fn)