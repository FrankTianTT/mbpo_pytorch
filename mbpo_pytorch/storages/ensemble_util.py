import numpy as np
import torch

class EnsembleBatchSampler():
    def __init__(self, ranges, batch_size: int, drop_last: bool, ensemble_size=7) -> None:
        self.samplers = np.array([np.random.permutation(ranges) for _ in range(ensemble_size)])
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ensemble_size = ensemble_size

        self.now_steps = 0

        if self.drop_last:
            self.length = len(ranges) // self.batch_size
        else:
            self.length = (len(ranges) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        while self.now_steps < self.length:
            idxes = self.samplers[:, self.now_steps * self.batch_size: (self.now_steps + 1) * self.batch_size]
            self.now_steps += 1
            yield torch.from_numpy(idxes)

    def __len__(self):
        return self.length


def test_ensemble_batch_sampler():
    v = torch.randn(20, 5)
    # v = np.random.randn(20, 5)

    sampler = EnsembleBatchSampler(range(17), 4, drop_last=False, ensemble_size=7)
    for s in sampler:
        s = torch.from_numpy(s)
        a = v[s]
        print(a.shape)


if __name__ == "__main__":
    test_ensemble_batch_sampler()
    #
    # v = torch.randn(20, 5)
    # idx = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    # print(idx.shape)
    # print(v[idx].shape)