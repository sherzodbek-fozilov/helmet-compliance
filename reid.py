# person re-identification embeddings for the repeat violator journal.
# osnet_x0_25 trained on msmt17 by the torchreid author (mit), ~41 crops/s on one cpu core.
# on market-1501 (unseen people and cameras) it finds repeat violators ~17x more often than the
# imagenet mobilenetv3 used before, at the same false merge rate: see tools/eval_reid.py
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from osnet import osnet_x0_25

WEIGHTS_URL = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
               "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "osnet_x0_25_msmt17.pth"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_H, _W = 256, 128  # training input size


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Скачиваю веса ReID (9 МБ): {path.name}")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.replace(path)


def load_osnet(weights: Path = WEIGHTS_PATH) -> torch.nn.Module:
    if not weights.exists():
        _download(weights)
    model = osnet_x0_25(num_classes=1000, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    # checkpoints saved from DataParallel have "module." prefixes; the id classifier is not needed
    state = {k.removeprefix("module."): v for k, v in state.items()
             if not k.removeprefix("module.").startswith("classifier")}
    model.load_state_dict(state, strict=False)
    return model.eval()


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(crop_bgr, (_W, _H), interpolation=cv2.INTER_LINEAR)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _MEAN) / _STD
    return img.transpose(2, 0, 1)


class ReIDExtractor:
    def __init__(self, weights: Path = WEIGHTS_PATH) -> None:
        self._model = load_osnet(weights)

    @torch.no_grad()
    def extract(self, person_bgr: np.ndarray) -> np.ndarray | None:
        """l2-normalised 512-d vector of a full person crop, None if the crop is too small"""
        h, w = person_bgr.shape[:2]
        if h < 40 or w < 15:
            return None
        vec = self._model(torch.from_numpy(preprocess(person_bgr)).unsqueeze(0))
        return torch.nn.functional.normalize(vec, dim=1).squeeze(0).numpy().astype(np.float32)
