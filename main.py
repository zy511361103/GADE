"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import shutil
from tqdm import tqdm
from data_utils import Dataset_ASVspoof2019_train, Dataset_ASVspoof2019_dev, Dataset_ASVspoof2019_eval
from ser.ser_model import Ser_Model
import sys
import warnings
from importlib import import_module
from shutil import copy
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from evaluate import *
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    dev_trial_path = config["dev_list_path"]

    # define model related paths
    model_tag = 'exp/' + args.exp_name
    # make directory for metric logging
    metric_path = model_tag + "/scores"
    os.makedirs(metric_path, exist_ok=True)
    model_save_path = os.path.join(model_tag, 'models')
    os.makedirs(model_save_path, exist_ok=True)
    log_path = os.path.join(model_tag, 'log')
    copy(args.config, model_tag + "/config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    ser_model = Ser_Model(model_config).to(device)
    ser_model.load_state_dict(torch.load(model_config['ser_config']['ser_model_path'], map_location=device))
    ser_model.eval()

    # evaluates pretrained model and exit script
    if args.eval:
        eval_trial_path = config["eval_list_path"]
        eval_database_name = config["eval_database_name"]
        eval_loader = get_loader('eval', args, config)
        print("List file of evaluation dataset:", eval_trial_path)
        print("Start evaluation...")
        eer_total = 0.0
        for trained_model in args.trained_models:
            eval_score_name = config["eval_database_name"] + '_epoch_' + trained_model.split(os.sep)[-1].split('_')[
                -2] + '.txt'
            eval_score_path = os.path.join(metric_path, eval_score_name)
            model.load_state_dict(torch.load(trained_model, map_location=device))
            print("Model loaded : {}".format(trained_model))
            produce_evaluation_file(eval_loader, model, ser_model, device, eval_score_path)
            eer = eval_to_score_file(eval_score_path, eval_trial_path, eval_database_name)
            eer_total = eer_total + eer
        final_eer = eer_total / len(args.trained_models)
        print("平均之后的eer(百分之):%.2f" % final_eer)
        print("Evaluation DONE.")
        sys.exit(0)

    # define dataloaders
    train_and_dev_database_name = config["train_and_dev_database_name"]
    trn_loader = get_loader('train', args, config)
    dev_loader = get_loader('dev', args, config)
    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_name = ''
    n_swa_update = 0  # number of snapshots of model to use in SWA
    min_dev_epoch = 0

    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch: {:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, ser_model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file(dev_loader, model, ser_model, device, os.path.join(metric_path, "dev_score.txt"))
        dev_eer = eval_to_score_file(os.path.join(metric_path, "dev_score.txt"), dev_trial_path,
                                     train_and_dev_database_name)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
            running_loss, dev_eer))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        torch.save(model.state_dict(),
                   os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)))
        if best_dev_eer >= dev_eer:
            if best_dev_eer > dev_eer:
                min_dev_epoch = epoch
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            if not best_name == '':
                os.remove(os.path.join(model_save_path, best_name))
            best_name = "best_epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)
            torch.save(model.state_dict(), os.path.join(model_save_path, best_name))
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        if epoch - min_dev_epoch >= config["early_stop"]:
            print("Early stop!best dev eer firstly found at epoch:", min_dev_epoch)
            sys.exit(0)

    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)

    torch.save(model.state_dict(), os.path.join(model_save_path, "swa.pth"))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(db_type, args, config):
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    if db_type == 'train':
        train_set = Dataset_ASVspoof2019_train(config)
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)
        return trn_loader
    elif db_type == 'dev':
        dev_set = Dataset_ASVspoof2019_dev(config)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dev_loader
    elif db_type == 'eval':
        eval_set = Dataset_ASVspoof2019_eval(config, args)
        eval_loader = DataLoader(eval_set,
                                 batch_size=config["batch_size"],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
        return eval_loader


def produce_evaluation_file(
        data_loader: DataLoader,
        model, ser_model,
        device: torch.device,
        save_path: str,
) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    for batch_x, utt_id, data_spec, data_mfcc, data_audio in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_spec = data_spec.to(device)
        batch_mfcc = data_mfcc.to(device)
        batch_audio = data_audio.to(device)
        emo_deep_feat = ser_model(batch_spec, batch_mfcc, batch_audio)
        with torch.no_grad():
            _, batch_out = model(batch_x, emo_deep_feat.detach())
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            fh.write("{} {}\n".format(fn, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
        trn_loader: DataLoader,
        model, ser_model,
        optim: Union[torch.optim.SGD, torch.optim.Adam],
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y, data_spec, data_mfcc, data_audio in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_spec = data_spec.to(device)
        batch_mfcc = data_mfcc.to(device)
        batch_audio = data_audio.to(device)
        emo_deep_feat = ser_model(batch_spec, batch_mfcc, batch_audio)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, emo_deep_feat.detach(), Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config", dest="config", type=str,
                        default='/data6/zhaoyan/code/aasist-main/config/AASIST.conf')
    parser.add_argument("--seed", type=int, default=3742, help="random seed (default: 1234)")
    parser.add_argument("--eval", action="store_true",
                        default=True, help="when this flag is given, evaluates given model and exit")
    parser.add_argument('--pick', type=str, default='0')  # 0为普通测试，1为分情感测试，2为分系统测试
    parser.add_argument('--system', type=str, default='S7')  # pick=2时有效
    parser.add_argument('--emotion', type=str, default='Sad')  # pick=1时有效
    parser.add_argument("--exp_name", type=str, default='GADE', help="comment to describe the saved model")
    parser.add_argument("--trained_models", type=list,
                        default=["data/exp/GADE/models/epoch_10_0.000.pth",
                                 "data/exp/GADE/models/epoch_15_0.000.pth",
                                 "data/exp/GADE/models/epoch_20_0.000.pth"],
                        help="训练好的模型列表，直接取三次平均")
    main(parser.parse_args())
