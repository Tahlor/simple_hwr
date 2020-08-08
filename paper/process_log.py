from pathlib import Path
import re

path="/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_adaptive_no_truncation/20200508_190735-dtw_adaptive_no_truncation/05-08-2020.log"
path="/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_adaptive_no_truncation/20200508_190735-dtw_adaptive_no_truncation/05-08-2020.log"
path="/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_adaptive_no_truncation/20200508_190735-dtw_adaptive_no_truncation/05-08-2020.log"
#path="/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/stroke_config/ver8/super/20200405_225449-dtw_adaptive_new2_restartLR/04-08-2020.log"

prev_line = ""
output = {}
with Path(path).open() as f:
    for line in f:
        #print(line)
        if line.find("adaptive GT changes")>0:
            print(line)
            adaptive_gt_changes = re.search("(Made )([0-9]+)", line)
            update = re.search("(, )([0-9]+)(,)", prev_line)            
            output[int(update[2])] = int(adaptive_gt_changes[2])
        if re.search(", [0-9]+,", line):
            prev_line = line

print(output)

for x in output.items():
    print(*x)
