from pathlib import Path
import os
root = Path("/media/data/GitHub/simple_hwr/RESULTS/ONLINE_PREDS/RESUME_model/new_experiment33/imgs")

for i in root.rglob("*"):
    new = i.parent / i.name[2:]
    os.rename(i,new)