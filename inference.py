"""AINIME @ WACV 2027 challenge, submission example.

    python inference.py

Reads every clip (video file or folder of frames) in ./val and writes
./val_output/<clip_id>/000000.png, 000001.png, ... (one lossless PNG per input frame).
Output frames must have the same resolution as the input frames.
Uses every visible GPU (one process per GPU, clips split between them), else the CPU.

Example model: pretrained FBCNN (JPEG artifact removal). To use your own model,
replace load_model() and restore(); keep the rest.
"""
import argparse
import os
import re
import time

import cv2
import torch
import torch.multiprocessing as mp

from fbcnn_arch import FBCNN

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
VID_EXT = ('.mp4', '.mkv', '.avi', '.mov', '.webm')


# ----------------------------------------------------------------------------------------
# Model-specific part: replace these two functions with your own model.
# ----------------------------------------------------------------------------------------
def load_model(checkpoint, device):
    model = FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.eval().to(device)


@torch.no_grad()
def restore(model, img, device):
    """RGB uint8 (H, W, 3) -> RGB uint8 (H, W, 3), same size."""
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    out = model(x)[0]  # FBCNN returns (image, predicted quality factor)
    return out.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------------------------
# I/O and multi-GPU: no need to change.
# ----------------------------------------------------------------------------------------
def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def list_clips(root):
    """[(clip_id, path)] for every sub-folder or video file in root."""
    clips = []
    for name in sorted(os.listdir(root), key=natural_key):
        path = os.path.join(root, name)
        stem, ext = os.path.splitext(name)
        if os.path.isdir(path):
            clips.append((name, path))
        elif ext.lower() in VID_EXT:
            clips.append((stem, path))
    return clips


def read_frames(path):
    """Yield RGB uint8 frames of a video file or of a folder of images."""
    if os.path.isdir(path):
        for name in sorted((f for f in os.listdir(path) if f.lower().endswith(IMG_EXT)), key=natural_key):
            yield cv2.cvtColor(cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f'cannot open video {path}')
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cap.release()


def worker(rank, devices, clips, checkpoint, output):
    device = devices[rank]
    model = load_model(checkpoint, device)
    for cid, path in clips[rank::len(devices)]:
        t0, count = time.time(), 0
        try:
            os.makedirs(os.path.join(output, cid), exist_ok=True)
            for i, frame in enumerate(read_frames(path)):
                out = restore(model, frame, device)
                if out.shape != frame.shape:
                    raise ValueError(f'output {out.shape[1]}x{out.shape[0]} != input {frame.shape[1]}x{frame.shape[0]}')
                cv2.imwrite(os.path.join(output, cid, f'{i:06d}.png'), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                count += 1
        except Exception as e:  # keep going with the other clips
            print(f'[{device}] {cid}: FAILED ({e})', flush=True)
            continue
        dt = time.time() - t0
        print(f'[{device}] {cid}: {count} frames, {dt:.1f}s ({dt / max(count, 1):.2f} s/frame)', flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.join(here, 'model_zoo', 'fbcnn_color.pth'))
    ap.add_argument('--input', default=os.path.join(here, 'val'))
    ap.add_argument('--output', default=os.path.join(here, 'val_output'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f'checkpoint not found: {args.checkpoint}')
    if not os.path.isdir(args.input):
        raise SystemExit(f'input folder not found: {args.input}')
    clips = list_clips(args.input)
    if not clips:
        raise SystemExit(f'no clips found in {args.input}')

    n_gpu = 0 if args.cpu else torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(n_gpu)] or [torch.device('cpu')]
    devices = devices[:len(clips)]
    print(f'{len(clips)} clips on {", ".join(map(str, devices))}')

    if len(devices) == 1:
        worker(0, devices, clips, args.checkpoint, args.output)
    else:
        mp.spawn(worker, args=(devices, clips, args.checkpoint, args.output), nprocs=len(devices))


if __name__ == '__main__':
    main()
