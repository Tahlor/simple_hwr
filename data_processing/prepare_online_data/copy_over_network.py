import shutil
import os
from pathlib import Path

source = Path(f"/media/data/GitHub/simple_hwr/data/online_coordinate_data/")
dests = ["/fslg_hwr/hw_data/strokes/online_coordinate_data", "/home/taylor/shares/brodie/github/simple_hwr/data/online_coordinate_data"]

for dest in dests:
    for var in "random", "normal":
        subvar = f"MAX_stroke_vBoosted2_{var}"
        try:
            (Path(dest) / subvar).mkdir(exist_ok=True, parents=True)
            shutil.copy(source / subvar / "train_online_coords.json", dest)
        except Exception as e:
            print(e)
