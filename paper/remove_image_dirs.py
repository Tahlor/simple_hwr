import re, os
from pathlib import Path
import shutil

root = "/fslhome/tarch/fsl_groups/fslg_hwr/compute/taylor_simple_hwr/results"
root = "/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results"

files_deleted = 0
for dir in Path(root).rglob("imgs"):
    print(dir.parent.name)
    if dir.is_dir() and dir.name=="imgs":
        subs = [s for s in Path(dir).glob('*') if re.search(r'[0-9]*', str(s)) and s.is_dir()]
        order = [int(x.name) for x in subs if x.name.isdigit()]
        if len(order)>5:
            subs = [x for _,x in sorted(zip(order,subs))]
            print([x.name for x in subs])
            for d in subs[:-5]:
                #print(f"Deleting {d}")
                files_deleted += len(list(d.rglob('*')))
                shutil.rmtree(d)
        print(f"Files deleted: {files_deleted}")
