#!/bin/bash
# replace /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/slurm_scripts/scripts/stroke_config/ver11_proper_eos/log_dtw_adaptive_no_truncation_default64v2.slurm log.slurm .sh

original_path=$1
# Copy
cp -r $original_path $original_path_NOA

# Change to no adaptation
cd $original_path_NOA
find . -type f -name "resume.sh"   -exec sed -i 's/$original_path/$original_path_NOA/' {} \;
find . -type f -name "RESUME.yaml" -exec sed -i 's/$original_path/$original_path_NOA/' {} \;
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
# Recursive replace
find . -type f -name "RESUME.yaml" -exec sed -i 's/.*reset_LR.*/reset_LR: true/' {} \;
find . -type f -name "RESUME.yaml" -exec sed -i 's/.*learning_rate.*/learning_rate: 1e-05/' {} \;
find . -type f -name "RESUME.yaml" -exec sed -i 's/.*scheduler_gamma:.*/scheduler_gamma: 0.97/'  {} \;
find . -type f -name "RESUME.yaml" -exec sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/no_adaptation/2020|'  {} \;

find . -type f -name "resume.sh" -exec sed -i ':a;N;$!ba;s|\n#!/bin/bash|#!/bin/bash|g' {} \;
find . -type f -name "resume.sh" -exec sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/no_adaptation/2020|' {} \;

#sed -i 's/.*reset_LR.*/reset_LR: true/' RESUME.yaml
#sed -i 's/.*learning_rate.*/learning_rate: 1e-05/' RESUME.yaml
#sed -i 's/.*scheduler_gamma:.*/scheduler_gamma: 0.97/' RESUME.yaml
#
#sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/with_adaptation/2020|' RESUME.yaml
#sed -i 's|dtw_no_truncation/2020|dtw_no_truncation/with_adaptation/2020|' resume.sh


cp all_stats.json all_stats_backup.json

sed -i 's/    //g' resume.sh
