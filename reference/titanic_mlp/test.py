from hwr_utils.hw_dataset import HwDataset

config_path = r"./configs/online_config_iam.json" # sys.argv[1]

data_config = config['data']
augment_config = config['augmentation']
network_config = config['network']

train_dataset = HwDataset(data_config['training_set_path'], char_to_idx,
                          img_height=network_config['input_height'],
                          data_root=data_config['image_root_directory'],
                          warp=config['warp'],
                          augment_path=augment_config['training_set_path'],
                          augment_root=augment_config['image_root_directory'])
