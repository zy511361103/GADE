import os
os.environ["OMP_NUM_THREADS"] = "1"
from data_utils import genSpoof_list, extract_eval_features, \
    extract_train_features, normalize_eval_spec_data, normalize_train_spec_data

wav2vec_pretrained_path = "/data6/zhaoyan/code/CA-MSER-main/features_extraction/pretrained_model/wav2vec2-base-960h"
ext = '.flac'
cut = 64600  # take ~4 sec audio (64600 samples)


def gen_train_data(db_path, list_path, db_name, save_path):
    # train data
    train_labels, train_list_IDs = genSpoof_list(dir_meta=list_path,
                                                 db_name=db_name,
                                                 is_train=True,
                                                 is_eval=False)
    print("no. training files:", len(train_list_IDs))
    extract_train_features(db_path, train_list_IDs, train_labels, cut, wav2vec_pretrained_path, ext, save_path)
    normalize_train_spec_data(features_path=save_path)


def gen_dev_data(db_path, list_path, db_name, save_path):
    # dev data
    dev_labels, dev_list_IDs = genSpoof_list(dir_meta=list_path,
                                             db_name=db_name,
                                             is_train=False,
                                             is_eval=False)
    print("no. validation files:", len(dev_list_IDs))
    extract_eval_features(db_path, dev_list_IDs, cut, wav2vec_pretrained_path, ext, save_path)
    normalize_eval_spec_data(features_path=save_path)


def gen_eval_data(db_path, list_path, db_name, save_path):
    # eval data
    eval_list_IDs = genSpoof_list(dir_meta=list_path,
                                  db_name=db_name,
                                  is_train=False,
                                  is_eval=True)
    print("no. evaluation files:", len(eval_list_IDs))
    extract_eval_features(db_path, eval_list_IDs, cut, wav2vec_pretrained_path, ext, save_path)
    normalize_eval_spec_data(features_path=save_path)

#Train:
train_wav_path = '/data6/zhaoyan/data/ADD2023/Track1.2/train/wav/'
train_label_path = '/data6/zhaoyan/data/ADD2023/Track1.2/train/label.txt'
train_feat_save_path = '/data6/zhaoyan/code/aasist-main/data/ADD2023/train/'

#Dev:
dev_wav_path = '/data6/zhaoyan/data/ADD2023/Track1.2/dev/wav/'
dev_label_path = '/data6/zhaoyan/data/ADD2023/Track1.2/dev/label.txt'
dev_feat_save_path = '/data6/zhaoyan/code/aasist-main/data/ADD2023/dev/'
train_and_dev_database_name = "ADD"

#Eval:
eval_wav_path = '/data1/zhaoyan/data/ASV2019_LA/eval/flac/'
eval_label_path = '/data1/zhaoyan/data/ASV2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
eval_feat_save_path = "/data6/zhaoyan/code/aasist-main/data/ASV/2019test/"
eval_database_name = "ASV2019"
# gen_train_data(train_wav_path, train_label_path, train_and_dev_database_name, train_feat_save_path)
# gen_dev_data(dev_wav_path, dev_label_path, train_and_dev_database_name, dev_feat_save_path)
gen_eval_data(eval_wav_path, eval_label_path, eval_database_name, eval_feat_save_path)
