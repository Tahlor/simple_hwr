# BYU ML Lab Deep Integration of LM into HWR

## Prerequisites

To work with this project effectively, supercomputer access is highly
recommended.  Sign up [here](https://rc.byu.edu/account/create/).

Next, request group access from Taylor Archibald.

After logging in, install Anaconda 3:

``` sh
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
```

## Environment

Install this from the repo:
https://github.com/SeanNaren/warp-ctc

### Configuration

All configurations are stored in the `config` folder as `.yaml` files.

#### Adding a new configuration option

1. Add option to a config file
2. Modify `hw_dataset.py` class to accept new option
3. Modify `train.py` to read option from config file and pass to `HwDataset` class
4. Modify `hwr_utils.py` at `defaults` to include a default parameter in case a config file does not specify your new option.

### Creating and Activating Conda Environment

``` sh
conda env create -f environment.yaml
conda activate hwr
```

## Execution

### Downloading/Preparing Datasets

Ensure that you have an IAM Handwriting Database access account ([register](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php)), and IAM On-Line Handwriting Database access account ([register](http://www.fki.inf.unibe.ch/DBs/iamOnDB/iLogin/index.php)), then:

``` bash
cd data
./generate-all-datasets.sh
```

For the first IAM prompt, use your username and password for IAM Handwriting DB, then for the second IAM prompt, use your username and password for IAM On-Line Handwriting DB.

### Train

To train, run `train.py` with one of the configurations found in the `configs` folder.  For example:

``` sh
python train.py --config ./configs/baseline.yaml
```

### Recognize

``` sh
python recognize.py sample_config.json prepare_font_data/output/0.png
```
or 

``` sh
python recognize.py sample_config_iam.json prepare_IAM_Lines/lines/r06/r06-000/r06-000-00.png
```
