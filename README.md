# BYU ML Lab Deep Integration of LM into HWR

## Aanaconda
After logging in, install Anaconda 3:

``` sh
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
```

## Environment

Create the environment defined in `environment.yaml`.
```
conda env create -f environment.yaml --name hwr
conda activate hwr
```

### Configuration

All configurations are stored in the `config` folder as `.yaml` files.

## Execution

### Downloading/Preparing Datasets

Ensure that you have an IAM Handwriting Database access account ([register](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php)), and IAM On-Line Handwriting Database access account ([register](http://www.fki.inf.unibe.ch/DBs/iamOnDB/iLogin/index.php)), then:

``` bash
cd data
./generate-all-datasets.sh
```

For the first IAM prompt, use your username and password for IAM Handwriting DB, then for the second IAM prompt, use your username and password for IAM On-Line Handwriting DB. This script should download/extract/setup the IAM data.

## Trajectory Recovery

#### Modifying/updating the config files
To use existing config options, just modify the config file directly. See `config/DEBUG.yaml` for an example configuaration with some descriptions (though it's not guaranteed to work). The `example_weights/example.conf` is working with the model weights in the `example_weights` folder. To add new options:

1. Add option to a config file
2. Modify `./hwr_utils/stroke_dataset.py.py` class to accept new option
3. Modify `train_stroke_recovery.py` to read the option from the config file and pass to `StrokeRecoveryDataset` class
4. Modify `hwr_utils.py` at `defaults` to include a default parameter in case a config file does not specify your new option.

### Training
Once the data is downloaded and the environment setup, setup a config file. You should then be able to train the model:

```
python train_stroke_recovery.py --config PATH_TO_CONFIG
```

### Evaluation
An example config with a model and weights can be run for offline data (though you may need to configure where your offline data is within the script).

```
python stroke_recovery_offline.py
```

Also see `python stroke_recovery_online.py`, which is similar but for online data.

## Handwriting Recognition

#### Modifying/updating the config files
To use existing config options, just modify the config file directly. To add new options:

1. Add option to a config file
2. Modify `hw_dataset.py` class to accept new option
3. Modify `train.py` to read the option from the config file and pass to `HwDataset` class
4. Modify `hwr_utils.py` at `defaults` to include a default parameter in case a config file does not specify your new option.

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

## Fulton Super Computer Prerequisites

If you are a BYU student, consider requesting access to the supercomputer. Sign up [here](https://rc.byu.edu/account/create/).

Next, request group access from Taylor Archibald.
