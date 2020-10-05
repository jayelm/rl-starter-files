"""
utils
"""


import os
import subprocess
import json
import logs


logger = logs.get_logger(__name__)


def current_git_hash():
    """
    Get the hash of the latest commit in this repository. Does not account for unstaged changes.
    Returns
    -------
    git_hash : ``str``, optional
        The string corresponding to the current git hash if known, else ``None`` if something failed.
    """
    unstaged_changes = False
    try:
        subprocess.check_output(["git", "diff-index", "--quiet", "HEAD", "--"])
    except subprocess.CalledProcessError as grepexc:
        if grepexc.returncode == 1:
            logger.warn("Running experiments with unstaged changes.")
            unstaged_changes = True
    except FileNotFoundError:
        logger.warn("Git not found")
    try:
        git_hash = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )
        return git_hash, unstaged_changes
    except subprocess.CalledProcessError:
        return None, None


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, running_avg=False):
        self.reset()
        self.compute_running_avg = running_avg
        if self.compute_running_avg:
            self.reset_running_avg()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_running_avg(self):
        self.running_val = 0
        self.running_avg = 0
        self.running_sum = 0
        self.running_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.compute_running_avg:
            self.update_running_avg(val, n)

    def update_running_avg(self, val, n):
        self.running_val = val
        self.running_sum += val * n
        self.running_count += n
        self.running_avg = self.running_sum / self.running_count

    def __str__(self):
        return f"AverageMeter(mean={self.avg:f}, count={self.count:d})"

    def __repr__(self):
        return str(self)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    scores and labels are 2d predictions, and we don't care about seqs of
    predictions. So if this is a seq prediction task, we give partial credit

    :param scores: torch.Tensor scores from the model
    :param targets: torch.Tensor true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def compute_average_metrics(meters):
    """
    Compute averages from meters. Handle tensors vs floats (always return a
    float)

    Parameters
    ----------
    meters : Dict[str, util.AverageMeter]
        Dict of average meters, whose averages may be of type ``float`` or
        ``torch.Tensor``

    Returns
    -------
    metrics : Dict[str, float]
        Average value of each metric
    """
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {
        m: float(v) if isinstance(v, float) or isinstance(v, int) else v.item()
        for m, v in metrics.items()
    }
    return metrics


def save_args(args, exp_dir, filename="args.json"):
    """
    Save arguments in the experiment directory. This is REALLY IMPORTANT for
    reproducibility, so you know exactly what configuration of arguments
    resulted in what experiment result! As a bonus, this function also saves
    the current git hash so you know exactly which version of the code produced
    your result (that is, as long as you don't run with unstaged changes).
    Parameters
    ----------
    args : ``argparse.Namespace``
        Arguments to save
    exp_dir : ``str``
        Folder to save args to
    filename : ``str``, optional (default: 'args.json')
        Name of argument file
    """
    args_dict = vars(args)
    args_dict["git_hash"], args_dict["git_unstaged_changes"] = current_git_hash()
    with open(os.path.join(exp_dir, filename), "w") as f:
        json.dump(args_dict, f, indent=4, separators=(",", ": "), sort_keys=True)
