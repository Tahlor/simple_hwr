# Copy models around
 BRODIE TO GALOIS
 cp ~/shares/brodie/github/simple_hwr/RESULTS/ver8/20200405_035941-dtw_adaptive_new/dtw_adaptive_new_model.pt /media/data/GitHub/simple_hwr/RESULTS/pretrained/

 BRODIE TO SUPER
 cp ~/shares/brodie/github/simple_hwr/RESULTS/ver8/20200405_035941-dtw_adaptive_new/dtw_adaptive_new_model.pt /simple_hwr/RESULTS/pretrained/

SUPER TO BRODIE
 new_home=adapted_v2
 brodie_home=~/shares/brodie/github/simple_hwr/RESULTS/pretrained/$new_home
 super_home=/simple_hwr/RESULTS/pretrained/$new_home
 galois_home=/media/data/GitHub/simple_hwr/RESULTS/pretrained/$new_home
 source_path=/simple_hwr/RESULTS/ver8/super/20200405_225449-dtw_adaptive_new2_restartLR
 mkdir $brodie_home
 mkdir $super_home
 mkdir $galois_home

 ln -s $source_path/dtw_adaptive_new2_restartLR_model.pt $galois_home/dtw_adaptive_new2_restartLR_model.pt
 ln -s $source_path/training_dataset.npy $galois_home/training_dataset.npy

 ln -s $source_path/dtw_adaptive_new2_restartLR_model.pt $brodie_home/dtw_adaptive_new2_restartLR_model.pt
 ln -s $source_path/training_dataset.npy $brodie_home/training_dataset.npy

 ln -s $source_path/dtw_adaptive_new2_restartLR_model.pt $super_home/dtw_adaptive_new2_restartLR_model.pt
 ln -s $source_path/training_dataset.npy $super_home/training_dataset.npy


 #cp $source_path/*.npy $brodie_home
 #cp $source_path/*.pt  $brodie_home




 # Synthetic Data Created FOR STROKE RECOVERY
mv *normal* normal & mv *random* random
cp /synth/checkpoints/gen_training_data/*.json /media/data/GitHub/simple_hwr/data/synthetic_online/boosted2

## Copy Model
cp /media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_adaptive_new_model.pt /simple_hwr/RESULTS/pretrained/

# COPY DATA - also in copy_over_network.py
source = Path(f"/media/data/GitHub/simple_hwr/data/online_coordinate_data/")
dests = ["/fslg_hwr/hw_data/strokes/online_coordinate_data", "/home/taylor/shares/brodie/github/simple_hwr/data/online_coordinate_data"]

### Copy training set
cp /media/data/GitHub/simple_hwr/RESULTS/ver8/20200405_203648-dtw_adaptive_new2/training_dataset.npy /simple_hwr/RESULTS/pretrained/


sudo ln -s /media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr /simple_hwr