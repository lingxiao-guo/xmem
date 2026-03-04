"""
Headless propagation for XMem interactive workspaces.
"""

from argparse import ArgumentParser
from contextlib import nullcontext
from os import path

import torch

from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask
from inference.interact.resource_manager import ResourceManager

torch.set_grad_enabled(False)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/XMem.pth')
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace', help='Workspace with images/masks.', default=None)
    parser.add_argument('--buffer_size', type=int, default=100)
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument('--size', default=480, type=int,
                        help='Resize shorter side to this size. -1 keeps original resolution.')

    parser.add_argument('--max_mid_term_frames', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', type=int, default=5)
    parser.add_argument('--max_long_term_elements', type=int, default=10000)
    parser.add_argument('--num_prototypes', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=10)
    parser.add_argument('--deep_update_every', type=int, default=-1)
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--no_amp', action='store_true')

    parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both')
    parser.add_argument('--start_frame', type=int, default=None,
                        help='Seed frame index. Defaults to the first frame that already has a mask.')
    parser.add_argument('--overwrite_seed', action='store_true',
                        help='Overwrite the seed frame mask using model output after seeding.')
    return parser.parse_args()


def find_seed_frame(res_man: ResourceManager, requested_seed):
    if requested_seed is not None:
        if requested_seed < 0 or requested_seed >= len(res_man):
            raise ValueError(f'start_frame out of range: {requested_seed}')
        seed_mask = res_man.get_mask(requested_seed)
        if seed_mask is None:
            raise ValueError(f'No seed mask found at frame index {requested_seed}')
        return requested_seed, seed_mask

    for ti in range(len(res_man)):
        mask = res_man.get_mask(ti)
        if mask is not None:
            return ti, mask
    raise ValueError('No seed mask found in workspace/masks. Import or create at least one mask first.')


def propagate_direction(res_man, network, config, seed_ti, seed_mask, indices, overwrite_seed=False):
    processor = InferenceCore(network, config=config)
    processor.set_all_labels(list(range(1, config['num_objects'] + 1)))

    seed_image = res_man.get_image(seed_ti)
    seed_image_torch, _ = image_to_torch(seed_image, device)
    seed_mask_one_hot = index_numpy_to_one_hot_torch(seed_mask, config['num_objects'] + 1).to(device)
    seed_prob = processor.step(seed_image_torch, seed_mask_one_hot[1:])

    if overwrite_seed:
        res_man.save_mask(seed_ti, torch_prob_to_numpy_mask(seed_prob), async_write=False)

    processed = 0
    for ti in indices:
        image = res_man.get_image(ti)
        image_torch, _ = image_to_torch(image, device)
        prob = processor.step(image_torch, end=(ti == indices[-1]))
        res_man.save_mask(ti, torch_prob_to_numpy_mask(prob), async_write=False)

        processed += 1
        if processed % 100 == 0:
            print(f'Processed {processed} frames in current pass...')


def main():
    args = parse_args()
    config = vars(args)
    config['enable_long_term'] = not args.disable_long_term
    config['enable_long_term_count_usage'] = not args.disable_long_term
    config['async_writes'] = False

    if config['workspace'] is None:
        if config['images'] is not None:
            basename = path.basename(config['images'])
        elif config['video'] is not None:
            basename = path.basename(config['video'])[:-4]
        else:
            raise NotImplementedError('Either images, video, or workspace has to be specified')
        config['workspace'] = path.join('./workspace', basename)

    print(f'Running headless propagation on {device.type}.')
    if args.disable_long_term:
        print('Long-term memory disabled: faster but less robust to long occlusions.')

    res_man = ResourceManager(config)
    seed_ti, seed_mask = find_seed_frame(res_man, args.start_frame)
    print(f'Using seed frame: {seed_ti}')

    autocast_ctx = torch.cuda.amp.autocast(enabled=not args.no_amp) if device.type == 'cuda' else nullcontext()
    with autocast_ctx:
        network = XMem(config, args.model, map_location=device).to(device).eval()

        if args.direction in ('backward', 'both'):
            backward_indices = list(range(seed_ti - 1, -1, -1))
            if backward_indices:
                print(f'Backward propagation: {len(backward_indices)} frames')
                propagate_direction(
                    res_man, network, config, seed_ti, seed_mask, backward_indices,
                    overwrite_seed=args.overwrite_seed
                )

        if args.direction in ('forward', 'both'):
            forward_indices = list(range(seed_ti + 1, len(res_man)))
            if forward_indices:
                print(f'Forward propagation: {len(forward_indices)} frames')
                propagate_direction(
                    res_man, network, config, seed_ti, seed_mask, forward_indices,
                    overwrite_seed=args.overwrite_seed
                )

    print('Headless propagation finished.')


if __name__ == '__main__':
    main()
