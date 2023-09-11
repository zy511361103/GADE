import os
import pickle

from torch import Tensor
from sklearn.preprocessing import MinMaxScaler
import gc
import librosa
import numpy as np

from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import Wav2Vec2Processor
from torchvision import transforms
from PIL import Image


def genSpoof_list(db_name, dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):  # train
        if db_name == 'ASV2019':
            for line in l_meta:
                _, key, _, _, label = line.strip().split(' ')
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        if db_name == 'EmoFake' or db_name == 'ADD':
            for line in l_meta:
                key, label = line.strip().split(' ')
                key = key.replace('.wav', '')
                file_list.append(key)
                d_meta[key] = 1 if label == 'genuine' or label == 'bonafide' else 0
        return d_meta, file_list


    elif (is_eval):  # test
        if db_name == 'ASV2019':
            for line in l_meta:
                _, key, _, _, label = line.strip().split(' ')
                file_list.append(key)
        if db_name == 'EmoFake' or db_name == 'ADD':
            for line in l_meta:
                key, label = line.strip().split(' ')
                file_list.append(key.replace('.wav', ''))
        if db_name == 'ASV2021':
            for line in l_meta:
                _, key, _, _, _, label, _, _ = line.strip().split(' ')
                file_list.append(key)
        return file_list

    else:  # dev
        if db_name == 'ASV2019':
            for line in l_meta:
                _, key, _, _, label = line.strip().split(' ')
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        if db_name == 'EmoFake' or db_name == 'ADD':
            for line in l_meta:
                key, label = line.strip().split(' ')
                key = key.replace('.wav', '')
                file_list.append(key)
                d_meta[key] = 1 if label == 'genuine' or label == 'bonafide' else 0

        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def extract_logspec(x, sr, window, win_length_, hop_length_, ndft, nfreq):
    # unpack params
    win_length = int((win_length_ / 1000) * sr)
    hop_length = int((hop_length_ / 1000) * sr)

    # calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft, hop_length=hop_length,
                               win_length=win_length,
                               window=window))

    spec = librosa.amplitude_to_db(spec, ref=np.max)

    # extract the required frequency bins
    spec = spec[:nfreq]

    # Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec, 0)

    return spec


def segment_nd_features(input_values, mfcc, data, cut_len, sr, wav2vec_path):
    segment_size = int(int(cut_len / sr) * 100)
    segment_size_wav = int(segment_size * sr / 100)

    data = data.transpose(0, 2, 1)
    time = data.shape[1]
    time_wav = input_values.shape[0]
    nch = data.shape[0]
    start, end = 0, segment_size
    start_wav, end_wav = 0, segment_size_wav
    # num_segs = math.ceil(time / segment_size)  # number of segments of each utterance
    num_segs = 1
    mfcc_tot = []
    audio_tot = []
    data_tot = []

    processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)

    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        if end_wav > time_wav:
            end_wav = time_wav
            start_wav = max(0, end_wav - segment_size_wav)

        # Do padding
        mfcc_pad = np.pad(
            mfcc[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")

        audio_pad = np.pad(input_values[start_wav:end_wav], ((segment_size_wav - (end_wav - start_wav)), 0),
                           mode="constant")

        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
            data_pad.append(data_ch)

        data_pad = np.array(data_pad)

        # Stack
        mfcc_tot.append(mfcc_pad)
        data_tot.append(data_pad)

        audio_pad_np = np.array(audio_pad)
        audio_pad_pt = processor(audio_pad_np, sampling_rate=sr, return_tensors="pt").input_values
        audio_pad_pt = audio_pad_pt.view(-1)
        audio_pad_pt_np = audio_pad_pt.cpu().detach().numpy()
        audio_tot.append(audio_pad_pt_np)

        # Update variables
        start = end
        end = min(time, end + segment_size)
        start_wav = end_wav
        end_wav = min(time_wav, end_wav + segment_size_wav)

    mfcc_tot = np.stack(mfcc_tot)
    data_tot = np.stack(data_tot)
    audio_tot = np.stack(audio_tot)

    # Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0, 1, 3, 2)

    return (data_tot, mfcc_tot, audio_tot)


def extract_train_features(basedir, list_fn, label, cut_len, wav2vec_path, ext, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for wav_name in tqdm(list_fn):
        # Read wave data
        x, sr = librosa.load(os.path.join(basedir, wav_name + ext), sr=None)

        X_pad = pad_random(x, cut_len)
        x_inp = Tensor(X_pad)

        # Apply pre-emphasis filter
        x = librosa.effects.preemphasis(x, zi=[0.0])

        # Extract required features into (C,F,T)
        features_data = extract_logspec(x, sr, window='hamming', win_length_=40, hop_length_=10, ndft=800,
                                        nfreq=200)

        hop_length = 160  # hop_length smaller, seq_len larger
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T  # (seq_len, 20)

        # Segment features into (N,C,F,T)
        features_segmented = segment_nd_features(x, mfcc, features_data, cut_len, sr, wav2vec_path)

        # Collect all the segments
        out_dir = os.path.join(output_path, wav_name + '.pkl')
        audio_features = {"X_inp": x_inp, "label": label[wav_name],
                          "seg_spec": features_segmented[0].astype(np.float32),
                          "seg_mfcc": np.vstack(features_segmented[1]).astype(np.float32),
                          "seg_audio": np.vstack(features_segmented[2]).astype(np.float32)}
        with open(out_dir, "wb") as fout:
            pickle.dump(audio_features, fout)
        fout.close()


def extract_eval_features(basedir, list_fn, cut_len, wav2vec_path, ext, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("wav2vec model will load at: ", wav2vec_path)
    for wav_name in tqdm(list_fn):
        # Read wave data
        x, sr = librosa.load(os.path.join(basedir, wav_name + ext), sr=None)

        X_pad = pad_random(x, cut_len)
        x_inp = Tensor(X_pad)

        # Apply pre-emphasis filter
        x = librosa.effects.preemphasis(x, zi=[0.0])

        # Extract required features into (C,F,T)
        features_data = extract_logspec(x, sr, window='hamming', win_length_=40, hop_length_=10, ndft=800,
                                        nfreq=200)

        hop_length = 160  # hop_length smaller, seq_len larger
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T  # (seq_len, 20)

        # Segment features into (N,C,F,T)
        features_segmented = segment_nd_features(x, mfcc, features_data, cut_len, sr, wav2vec_path)

        out_dir = os.path.join(output_path, wav_name + '.pkl')
        audio_features = {"X_inp": x_inp, "seg_spec": features_segmented[0].astype(np.float32),
                          "seg_mfcc": np.vstack(features_segmented[1]).astype(np.float32),
                          "seg_audio": np.vstack(features_segmented[2]).astype(np.float32)}
        with open(out_dir, "wb") as fout:
            pickle.dump(audio_features, fout)
        fout.close()


def normalize_train_spec_data(features_path):
    gc.enable()
    feat_list = os.listdir(features_path)
    spec_data_list = []
    print('start load features without normalization...')
    for feat in tqdm(feat_list):
        f_read = open(os.path.join(features_path, feat), 'rb')
        feat = pickle.load(f_read)
        spec_feat = feat["seg_spec"]
        spec_data_list.append(spec_feat)
    spec_data = np.vstack(spec_data_list).astype(np.float32)
    print(spec_data.shape)
    del spec_data_list
    nch = spec_data.shape[1]
    nfreq = spec_data.shape[2]
    ntime = spec_data.shape[3]
    rearrange = lambda x: x.transpose(1, 0, 3, 2).reshape(nch, -1, nfreq)
    spec_data = rearrange(spec_data)

    # scaler type
    scaler = eval('MinMaxScaler(feature_range=(0,1))')
    print('normalization start!')
    for ch in range(nch):
        # get scaling values from training data
        scale_values = scaler.fit(spec_data[ch])
        # apply to all
        spec_data[ch] = scaler.transform(spec_data[ch])

    # Shape the data back to (N,C,F,T)
    rearrange = lambda x: x.reshape(nch, -1, ntime, nfreq).transpose(1, 0, 3, 2)
    spec_data = rearrange(spec_data)
    print('normalization finished!')
    # AlexNet preprocessing
    alexnet_preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert format to uint8, flip the frequency axis to orientate image upward,
    #   duplicate into 3 channels
    spec_data = np.clip(spec_data, 0.0, 1.0)
    spec_data = (spec_data * 255.0).astype(np.uint8)
    spec_data = np.flip(spec_data, axis=2)
    spec_data = np.moveaxis(spec_data, 1, -1)
    spec_data = np.repeat(spec_data, 3, axis=-1)
    print('saving new features...')
    for i, seg in tqdm(enumerate(spec_data)):
        img = Image.fromarray(seg, mode='RGB')
        seg_spec = alexnet_preprocess(img)
        f_read = open(os.path.join(features_path, feat_list[i]), 'rb')
        audio_features = pickle.load(f_read)
        x_inp = audio_features["X_inp"]
        label = audio_features["label"]
        seg_mfcc = audio_features["seg_mfcc"].astype(np.float32)
        seg_audio = np.squeeze(audio_features["seg_audio"].astype(np.float32))
        new_audio_features = {"X_inp": x_inp, "label": label, "seg_spec": seg_spec, "seg_mfcc": seg_mfcc,
                              "seg_audio": seg_audio}
        with open(os.path.join(features_path, feat_list[i]), "wb") as fout:
            pickle.dump(new_audio_features, fout)
        fout.close()
    print("Finished!!")


def normalize_eval_spec_data(features_path):
    gc.enable()
    feat_list = os.listdir(features_path)
    spec_data_list = []
    print('start load features without normalization...')
    for feat in tqdm(feat_list):
        f_read = open(os.path.join(features_path, feat), 'rb')
        feat = pickle.load(f_read)
        f_read.close()
        spec_feat = feat["seg_spec"]
        spec_data_list.append(spec_feat)
    spec_data = np.vstack(spec_data_list).astype(np.float32)
    print(spec_data.shape)
    del spec_data_list
    nch = spec_data.shape[1]
    nfreq = spec_data.shape[2]
    ntime = spec_data.shape[3]
    rearrange = lambda x: x.transpose(1, 0, 3, 2).reshape(nch, -1, nfreq)
    spec_data = rearrange(spec_data)

    # scaler type
    scaler = eval('MinMaxScaler(feature_range=(0,1))')
    print('normalization start!')
    for ch in range(nch):
        # get scaling values from training data
        scale_values = scaler.fit(spec_data[ch])
        # apply to all
        spec_data[ch] = scaler.transform(spec_data[ch])

    # Shape the data back to (N,C,F,T)
    rearrange = lambda x: x.reshape(nch, -1, ntime, nfreq).transpose(1, 0, 3, 2)
    spec_data = rearrange(spec_data)
    print('normalization finished!')
    # AlexNet preprocessing
    alexnet_preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert format to uint8, flip the frequency axis to orientate image upward,
    #   duplicate into 3 channels
    spec_data = np.clip(spec_data, 0.0, 1.0)
    spec_data = (spec_data * 255.0).astype(np.uint8)
    spec_data = np.flip(spec_data, axis=2)
    spec_data = np.moveaxis(spec_data, 1, -1)
    spec_data = np.repeat(spec_data, 3, axis=-1)
    print('saving new features...')
    for i, seg in tqdm(enumerate(spec_data)):
        img = Image.fromarray(seg, mode='RGB')
        seg_spec = alexnet_preprocess(img)
        f_read = open(os.path.join(features_path, feat_list[i]), 'rb')
        audio_features = pickle.load(f_read)
        x_inp = audio_features["X_inp"]
        seg_mfcc = audio_features["seg_mfcc"].astype(np.float32)
        seg_audio = np.squeeze(audio_features["seg_audio"].astype(np.float32))
        new_audio_features = {"X_inp": x_inp, "seg_spec": seg_spec, "seg_mfcc": seg_mfcc, "seg_audio": seg_audio}
        with open(os.path.join(features_path, feat_list[i]), "wb") as fout:
            pickle.dump(new_audio_features, fout)
        fout.close()
    print("Finished!!")


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, config):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.feat_list = os.listdir(config['train_feat_path'])
        self.config = config

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, index):
        f_read = open(os.path.join(self.config['train_feat_path'], self.feat_list[index]), 'rb')
        wav_feat = pickle.load(f_read)
        x_inp = wav_feat["X_inp"]
        audio_spec = wav_feat["seg_spec"]
        audio_mfcc = wav_feat["seg_mfcc"]
        audio_wav = wav_feat['seg_audio']
        y = wav_feat["label"]

        return x_inp, y, audio_spec, audio_mfcc, audio_wav


class Dataset_ASVspoof2019_dev(Dataset):
    def __init__(self, config):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.feat_list = os.listdir(config['dev_feat_path'])
        self.config = config

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, index):
        key = self.feat_list[index].replace('.pkl', '')
        f_read = open(os.path.join(self.config['dev_feat_path'], self.feat_list[index]), 'rb')
        wav_feat = pickle.load(f_read)
        x_inp = wav_feat["X_inp"]
        audio_spec = wav_feat["seg_spec"]
        audio_mfcc = wav_feat["seg_mfcc"]
        audio_wav = wav_feat['seg_audio']

        return x_inp, key, audio_spec, audio_mfcc, audio_wav


class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, config, args):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.feat_list = os.listdir(config['eval_feat_path'])
        self.pick_list = []
        for wav in self.feat_list:
            if args.pick == '0':
                self.pick_list.append(wav)
            elif args.pick == '1':  # 分情感测试
                if wav[0] == 'S':
                    emo = wav.split('_')[1].split('2')[1]
                else:
                    emo = wav.split('_')[1]
                if emo == args.emotion:
                    self.pick_list.append(wav)
            elif args.pick == '2':  # 分系统测试
                if wav[0] == 'S':
                    sys = wav.split('_')[0]
                    if sys == args.system:
                        self.pick_list.append(wav)
                else:
                    self.pick_list.append(wav)
        self.config = config
        self.args = args
        if self.args.pick == '0':
            print("正在测试！")
        elif args.pick == '1':
            print("正在分情感测试！情感为：", args.emotion)
        elif args.pick == '2':
            print("正在分系统测试！系统为：", args.system)
        print("eval_dataset files number: ", len(self.pick_list))

    def __len__(self):
        return len(self.pick_list)

    def read_feature_file(self, index):
        f_read = open(os.path.join(self.config['eval_feat_path'], self.pick_list[index]), 'rb')
        wav_feat = pickle.load(f_read)
        x_inp = wav_feat["X_inp"]
        audio_spec = wav_feat["seg_spec"]
        audio_mfcc = wav_feat["seg_mfcc"]
        audio_wav = wav_feat['seg_audio']
        return x_inp, audio_spec, audio_mfcc, audio_wav

    def __getitem__(self, index):
        key = self.pick_list[index].replace('.pkl', '')
        x_inp, audio_spec, audio_mfcc, audio_wav = self.read_feature_file(index)
        return x_inp, key, audio_spec, audio_mfcc, audio_wav
