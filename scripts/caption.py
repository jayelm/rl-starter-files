"""
Caption demos
"""


import pickle
import blosc
from teacher import Teacher
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from torch import nn, optim
import util
from contextlib import nullcontext
import pandas as pd
import os
import bleu


def to_text(tokens, vocab):
    if isinstance(tokens, torch.Tensor):
        # Single example, tensor
        if tokens.ndim == 1:
            tokens = [tokens]

    texts = []
    for caption in tokens:
        this_text = []
        for tok in caption:
            t = vocab[tok.item()]
            if t == "<SOS>":
                continue
            if t in {"<EOS>", "<PAD>"}:
                break
            this_text.append(t)
        texts.append(" ".join(this_text))
    return texts


def demo_collate(batch):
    obs, dirs, acts, obs_lens, langs, lang_lens = zip(*batch)
    obs_lens = torch.tensor(obs_lens)
    lang_lens = torch.tensor(lang_lens)
    obs_pad = pad_sequence(obs, batch_first=True, padding_value=0)
    dirs_pad = pad_sequence(dirs, batch_first=True, padding_value=0)
    acts_pad = pad_sequence(acts, batch_first=True, padding_value=0)
    langs_pad = pad_sequence(langs, batch_first=True, padding_value=0)
    return (obs_pad, dirs_pad, acts_pad, obs_lens, langs_pad, lang_lens)


def make_vocab(items, str_format=False):
    w2i = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
    }
    for item in items:
        if str_format:
            item = item.split(" ")
        for tok in item:
            tok = str(tok)
            if tok not in w2i:
                w2i[tok] = len(w2i)
    return w2i


class Demos:
    def __init__(self, demos):
        self.obs = []
        self.obs_lens = []
        self.langs = []
        self.lang_lens = []
        self.dirs = []
        self.acts = []

        missions, packed_obs, directions, actions = zip(*demos)

        self.lang_w2i = make_vocab(missions, str_format=True)
        self.dirs_w2i = make_vocab(directions)
        self.acts_w2i = make_vocab(actions)

        for mission, packed_obs, dirs, acts in demos:
            missions_i = [1, *(self.lang_w2i[t] for t in mission.split(" ")), 2]
            # src doesn't need start of sentence token.
            dirs_i = [self.dirs_w2i[str(t)] for t in dirs]
            acts_i = [self.acts_w2i[str(t)] for t in acts]

            missions_i = np.array(missions_i, dtype=np.int64)
            dirs_i = np.array(dirs_i, dtype=np.int64)
            acts_i = np.array(acts_i, dtype=np.int64)

            obs = blosc.unpack_array(packed_obs)
            # Transpose - channels first
            obs = np.transpose(obs, (0, 3, 1, 2))

            self.obs.append(obs)
            self.obs_lens.append(obs.shape[0])
            self.langs.append(missions_i)
            self.lang_lens.append(len(missions_i))
            self.dirs.append(dirs_i)
            self.acts.append(acts_i)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.obs[i]),
            torch.from_numpy(self.dirs[i]),
            torch.from_numpy(self.acts[i]),
            self.obs_lens[i],
            torch.from_numpy(self.langs[i]),
            self.lang_lens[i],
        )

    def __len__(self):
        return len(self.langs)


def load_demos(*demos_files, n_per_file=100000):
    # TODO - tag the demos
    all_demos = []
    for df in demos_files:
        with open(df, "rb") as f:
            demos = pickle.load(f)
            demos = demos[:n_per_file]
            all_demos.extend(demos)

    demos_dset = Demos(all_demos)
    return demos_dset


def run(split, epoch, model, criterion, optimizer, dataloader):
    training = split == "train"

    if training:
        ctx = nullcontext
        model.train()
        to_measure = ["top1", "top5", "loss"]
    else:
        ctx = torch.no_grad
        model.eval()
        to_measure = ["top1", "top5", "loss", "bleu4"]

    meters = {m: util.AverageMeter() for m in to_measure}

    ranger = tqdm(dataloader, desc=str(epoch))
    for batch_i, batch in enumerate(ranger):
        obs, dirs, acts, obs_lens, langs, lang_lens = batch
        batch_size = obs.shape[0]
        obs = obs.float()

        if args.cuda:
            obs = obs.cuda()
            dirs = dirs.cuda()
            acts = acts.cuda()
            obs_lens = obs_lens.cuda()
            langs = langs.cuda()
            lang_lens = lang_lens.cuda()

        with ctx():
            logits = model(obs, dirs, acts, obs_lens, langs, lang_lens)

        targets = langs[:, 1:]
        target_lens = lang_lens - 1
        n_targets = target_lens.sum().item()
        targets_packed = pack_padded_sequence(
            targets, target_lens, batch_first=True, enforce_sorted=False
        )
        targets = targets_packed.data

        this_loss = criterion(logits, targets)

        if training:
            optimizer.zero_grad()
            this_loss.backward()
            optimizer.step()
        else:
            # Sample
            sampled_lang, sampled_lengths = model.sample(obs, dirs, acts, obs_lens)
            sampled_text = to_text(sampled_lang, model.rev_vocab)
            true_text = to_text(langs, model.rev_vocab)
            true_text = [[t] for t in true_text]

            b4 = bleu.compute_bleu(true_text, sampled_text)[0] * 100
            meters["bleu4"].update(b4, batch_size)

        top5 = util.accuracy(logits, targets, 5)
        meters["top5"].update(top5, n_targets)
        top1 = util.accuracy(logits, targets, 1)
        meters["top1"].update(top1, n_targets)

        meters["loss"].update(this_loss.item(), n_targets)

    metrics = util.compute_average_metrics(meters)
    return metrics


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Caption demos", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--demos",
        default=["./demos/BabyAI-GoTo-v0.pkl",],
        nargs="+",
        help="Demos pickle files",
    )
    parser.add_argument(
        "--n_per_file",
        default=100000,
        type=int,
        help="How many demos to load per pickle file",
    )
    parser.add_argument(
        "--exp_dir", default="exp/debug", help="Path to exp dir",
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    util.save_args(args, args.exp_dir)

    demos = load_demos(*args.demos, n_per_file=args.n_per_file)
    val_size = int(len(demos) * 0.1)
    test_size = int(len(demos) * 0.1)
    dsets = torch.utils.data.random_split(
        demos, [len(demos) - val_size - test_size, val_size, test_size]
    )

    def to_dl(d):
        return torch.utils.data.DataLoader(
            d, batch_size=100, pin_memory=True, num_workers=4, collate_fn=demo_collate
        )

    dataloaders = {
        "train": to_dl(dsets[0]),
        "val": to_dl(dsets[1]),
        "test": to_dl(dsets[2]),
    }

    # Hardcode obs space for nwo
    # ==== MODEL ====
    #  obs_space = {'image': (7, 7)}
    model = Teacher(len(demos.dirs_w2i), len(demos.acts_w2i), demos.lang_w2i)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    if args.cuda:
        model = model.cuda()

    losses = util.AverageMeter()
    top5 = util.AverageMeter()
    top1 = util.AverageMeter()

    records = []
    for epoch in range(args.epochs):
        train_metrics = run(
            "train", epoch, model, criterion, optimizer, dataloaders["train"]
        )
        tqdm.write(
            f"TRAIN {epoch} loss {train_metrics['loss']:.3f} top1 {train_metrics['top1']:.3f} top5 {train_metrics['top5']:.3f}"
        )

        val_metrics = run("val", epoch, model, criterion, optimizer, dataloaders["val"])
        tqdm.write(
            f"VAL {epoch} loss {val_metrics['loss']:.3f} top1 {val_metrics['top1']:.3f} top5 {val_metrics['top5']:.3f} bleu4 {val_metrics['bleu4']:.3f}"
        )

        records.append(
            {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in train_metrics.items()},
                "epoch": epoch,
            }
        )

        pd.DataFrame(records).to_csv(
            os.path.join(args.exp_dir, "metrics.csv"), index=False
        )
