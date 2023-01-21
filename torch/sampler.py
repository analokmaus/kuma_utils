from typing import Any, Callable, Iterator, List, Optional, Union
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler


class DistributedProxySampler(DistributedSampler):
    """Distributed sampler proxy to adapt user's sampler for distributed data parallelism configuration.
    Code is borrowed from: https://pytorch.org/ignite/_modules/ignite/distributed/auto.html#DistributedProxySampler
    Code is based on https://github.com/pytorch/pytorch/issues/23430#issuecomment-562350407

    Args:
        sampler: Input torch data sampler.
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process within ``num_replicas``.

    .. note::
        Input sampler is assumed to have a constant size.
    """

    def __init__(self, sampler: Sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None) -> None:

        if not isinstance(sampler, Sampler):
            raise TypeError(f"Argument sampler should be instance of torch Sampler, but given: {type(sampler)}")

        if isinstance(sampler, DistributedSampler):
            raise TypeError("Argument sampler must not be a distributed sampler already")

        if not hasattr(sampler, "__len__"):
            raise TypeError("Argument sampler should have length")

        super(DistributedProxySampler, self).__init__(
            sampler, num_replicas=num_replicas, rank=rank, shuffle=False  # type: ignore[arg-type]
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)

        indices = []  # type: List
        while len(indices) < self.total_size:
            indices += list(self.sampler)

        if len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"{len(indices)} vs {self.num_samples}")

        return iter(indices)
    