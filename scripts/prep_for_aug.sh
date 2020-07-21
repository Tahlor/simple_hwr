#!/bin/bash
# replace /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/slurm_scripts/scripts/stroke_config/ver11_proper_eos/log_dtw_adaptive_no_truncation_default64v2.slurm log.slurm .sh

original_path=$1
# Copy
cp -r $original_path $original_path_NOA

# Change to no adaptation
cd $original_path_NOA
replace $original_path $original_path_NOA .sh
Y
replace $original_path $original_path_NOA .yaml
Y
sed -i 's/    //g' resume.sh
# sed ':a;N;$!ba;s|\n#!/bin/bash|#!/bin/bash|g' resume.sh # test
sed -i ':a;N;$!ba;s|\n#!/bin/bash|#!/bin/bash|g' resume.sh

cp all_stats.json all_stats_backup.json


# Change to adaption
cd ../$original_path
#replace 'name: dtw/\n' 'name: dtw_adaptive'  .yaml
sed -i ':a;N;$!ba;s/name: dtw\n/name: dtw_adaptive\n/g' RESUME.yaml
sed -i ':a;N;$!ba;s|\n#!/bin/bash|#!/bin/bash|g' resume.sh

# LR
# find . -type f -name "*$3" -exec sed -i "s@$1@$2@g" {} \;
sed -i 's/.*reset_LR.*/reset_LR: true/' RESUME.yaml
sed -i 's/.*learning_rate.*/learning_rate: 1e-05/' RESUME.yaml
sed -i 's/.*scheduler_gamma:.*/scheduler_gamma: 0.97/' RESUME.yaml

sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/with_adaptation/2020|' RESUME.yaml
sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/with_adaptation/2020|' resume.sh


cp all_stats.json all_stats_backup.json

sed -i 's/    //g' resume.sh
