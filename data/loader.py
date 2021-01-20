from data.MSCOCO import MSCOCO
import torch

def build_dataloader(args, train=True):
    dataset = MSCOCO(args, args.data_dir, train)

    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=trivial,
        num_workers=args.num_workers
    )

    return dataloader

def trivial(any_input):
    return any_input
