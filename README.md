# AINIME @ WACV 2027 Challenge: submission example

This folder shows the structure of a submission for the Anime Video Enhancement challenge
(Re-Anime600, https://mv-lab.github.io/ainime/). The example model is a pretrained FBCNN.

```
inference.py        entry point
fbcnn_arch.py       example model code
model_zoo/          example checkpoint (fbcnn_color.pth)
requirements.txt    pinned dependencies
val/                input clips (.mp4); here only one example clip
val_output/         results (enhanced clips, written by inference.py)
```

## What to submit

A zip with your `inference.py`, your model code and checkpoint, and `requirements.txt`.
Do not include `val/` or `val_output/`: the organizers replace `val/` with the folder containing all the hidden clips, place it next to `inference.py`
and run `python inference.py` (no arguments). The script must not install packages or download files.

## Rules for the script

1. Reads every clip (`.mp4`) in the `val/` folder next to `inference.py` (during evaluation: the hidden validation/test clips; for local testing: the example clip, or e.g. the public train clips copied into `val/`).
2. Writes one enhanced video per input clip into `val_output/` (created if missing), with the same file name as the input (e.g. `val/238930.mp4` → `val_output/238930.mp4`).
3. Output videos keep the same resolution, number of frames and frame rate (23.98 fps) as the input.
4. Output videos must be encoded losslessly (e.g. H.264 with `-crf 0` or FFV1) to avoid compression artifacts affecting the evaluation.
5. Uses every visible GPU when there are several, else the CPU.

To plug in your model, replace `load_model()` and `restore()` in `inference.py`.

## Run the example

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python inference.py
```

The enhanced example clip appears in `val_output/`.

`requirements.txt` was tested on 2 x Quadro P5000 (Pascal, 16 GB each). Pascal GPUs need the
CUDA 12.6 build of PyTorch, already selected in the file. For other GPUs, install the matching build from pytorch.org.

## Credits

FBCNN: Jiang et al., Towards Flexible Blind JPEG Artifacts Removal, ICCV 2021 (Apache-2.0).
https://github.com/jiaxi-jiang/FBCNN

Fargetta, G., et al. Re-Anime600: A Curated Benchmark Dataset for Anime
Video Quality Assessment. CVPR Workshop, 2026.

Pan, Z., Zhu, Y., Mu, Y. Sakuga-42M dataset: Scaling up cartoon research.
arXiv preprint arXiv:2405.07425, 2024.
