# AINIME @ WACV 2027 Challenge: submission example

This folder shows the structure of a submission for the Anime Video Enhancement challenge
(Re-Anime600, https://mv-lab.github.io/ainime/). The example model is a pretrained FBCNN.

```
inference.py        entry point
fbcnn_arch.py       example model code
model_zoo/          example checkpoint (fbcnn_color.pth)
requirements.txt    pinned dependencies
val/                input clips (video files or folders of frames)
val_output/         results
```

## What to submit

A zip with your `inference.py`, your model code and checkpoint, and `requirements.txt`.
Do not include `val/` or `val_output/`: the organizers place the clips folder next to `inference.py`
and run `python inference.py`. The script must not install packages or download files.

## Rules for the script

1. Reads every clip in `val/` (video file or folder of frames).
2. Writes `val_output/<clip_id>/000000.png, 000001.png, ...`: one lossless PNG per input frame.
3. Each output frame has the same resolution as the input frame.
4. Uses every visible GPU when there are several, else the CPU.

To plug in your model, replace `load_model()` and `restore()` in `inference.py`.

## Run the example

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python inference.py
```

`requirements.txt` was tested on 2 x Quadro P5000 (Pascal, 16 GB each). Pascal GPUs need the
CUDA 12.6 build of PyTorch, already selected in the file. For other GPUs, install the matching build from pytorch.org.

## Credits

FBCNN: Jiang et al., Towards Flexible Blind JPEG Artifacts Removal, ICCV 2021 (Apache-2.0).
https://github.com/jiaxi-jiang/FBCNN
