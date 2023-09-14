# GADE

This repository provides the code proposed in 'EMOFAKE: AN INITIAL DATASET FOR EMOTION FAKE AUDIO DETECTION'

### Data preparation
We train GADE using the ASVspoof 2019 logical access(https://datashare.ed.ac.uk/handle/10283/3336) dataset.
You can also use EmoFake(https://drive.google.com/file/d/1aYPNVCVIBs6c9er_bhT3U8YzClF-EDsR/viewusp=sharing) for training.

### Pre-trained SER
https://github.com/Vincent-ZHQ/CA-MSER You can use any emtoion datasets to train CA-MSER.
We provided the checkpoint trained with ESD.

### Feature extraction
```
python extract_features.py
```
Please change the wav2vec model path and the data path to your own path.

### Training 
The `main.py` includes train and evaluation.

To train :
```
python main.py --config ./config/AASIST.conf
```
Please change the data path and the pre-trained ser model path to your own path.

### Evaluation
To evaluate:
```
python main.py --eval --config ./config/AASIST.conf --pick='0' --trained_models=["data/exp/GADE/models/epoch_10_0.000.pth",
                                 "data/exp/GADE/models/epoch_15_0.000.pth",
                                 "data/exp/GADE/models/epoch_20_0.000.pth"]
```
Please change the trained model path and the data path to your own path.

To evaluate by emotion(EmoFake):
```
python main.py --eval --config ./config/AASIST.conf --pick='1' --emotion='Sad' --trained_models=["data/exp/GADE/models/epoch_10_0.000.pth",
                                 "data/exp/GADE/models/epoch_15_0.000.pth",
                                 "data/exp/GADE/models/epoch_20_0.000.pth"]
```
Please change the emotion you need..

To evaluate by systems(EmoFake):
```
python main.py --eval --config ./config/AASIST.conf --pick='2' --system='S3' --trained_models=["data/exp/GADE/models/epoch_10_0.000.pth",
                                 "data/exp/GADE/models/epoch_15_0.000.pth",
                                 "data/exp/GADE/models/epoch_20_0.000.pth"]
```
Please change the system you need.
