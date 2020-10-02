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
