from data.MSCOCO import MSCOCO
import torch

def build_dataloader(args, train=True):
    dataset = MSCOCO(args, args.data_dir)

    sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=trivial,
        num_workers=args.num_workers,
        drop_last=True
    )

    return dataloader

def trivial(any_input):
    return any_input
