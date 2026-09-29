"""IP-Adapter FaceID for create_image (added 2026-09-28 at the user's request, with RealVisXL V5.0): a NEW image of
the person in an attached photo ("make me an astronaut", "this person on a beach at sunset").

How it works:
- The face in the photo becomes a 512-number identity (ArcFace w600k_r50, the recognition model of InsightFace's
  buffalo_l, run with onnxruntime on the CPU). The face is found with OpenCV's YuNet, which also gives the 5
  landmarks (eyes, nose, mouth corners) used to align it to ArcFace's 112x112 template, as InsightFace does.
  The insightface package itself is not used: it has no wheels for this Python (3.14) and would need compiling.
- h94/IP-Adapter-FaceID's SDXL adapter (ip-adapter-faceid_sdxl.bin) turns that identity into extra conditioning for
  the SDXL UNet, through diffusers' own load_ip_adapter. It works with every current image style (all SDXL).
- The adapter is loaded into the pipeline just before a face generation and unloaded right after, so ordinary
  images and edit_image are never affected by it, and its ~1GB leaves memory between face images.

Licences: InsightFace's models and the FaceID weights are for non-commercial research use (see their model cards).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.tools.face_id")

# ArcFace's 112x112 alignment template (InsightFace's arcface_dst): left eye, right eye (as seen in the image),
# nose tip, left and right mouth corners. YuNet returns its 5 landmarks in the same order.
_ARCFACE_TEMPLATE = (
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041),
)


class FaceNotFound(ValueError):
    """The photo has no face clear enough to use."""


@dataclass(frozen=True)
class FaceIDConfig:
    adapter_repo: str = "h94/IP-Adapter-FaceID"
    adapter_file: str = "ip-adapter-faceid_sdxl.bin"
    recognition_repo: str = "immich-app/buffalo_l"  # InsightFace buffalo_l, repackaged; recognition = w600k_r50
    recognition_file: str = "recognition/model.onnx"
    detector_path: str = "models/faceid/face_detection_yunet_2023mar.onnx"
    scale: float = 0.8  # how strongly the face steers the image (diffusers' set_ip_adapter_scale)


class FaceEmbedder:
    """Photo -> the largest face's normalized ArcFace embedding (512 floats). CPU only; models load on first use."""

    def __init__(self, config: FaceIDConfig, root: Path) -> None:
        self._config = config
        self._detector_path = (root / config.detector_path) if not Path(config.detector_path).is_absolute() else Path(config.detector_path)
        self._session = None
        self._lock = threading.Lock()

    def _recognizer(self):
        if self._session is None:
            import onnxruntime
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self._config.recognition_repo, self._config.recognition_file)
            self._session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        return self._session

    def _landmarks(self, bgr) -> tuple[Any, int]:
        """The 5 landmarks of the largest face, as a (5, 2) float32 array, and how many faces the photo has."""
        import cv2
        import numpy as np

        height, width = bgr.shape[:2]
        scale = min(1.0, 1280 / max(height, width))  # YuNet is fast on a smaller copy; landmarks are scaled back
        small = cv2.resize(bgr, (round(width * scale), round(height * scale))) if scale < 1 else bgr
        detector = cv2.FaceDetectorYN.create(str(self._detector_path), "", (small.shape[1], small.shape[0]), 0.7)
        _, faces = detector.detect(small)
        if faces is None or len(faces) == 0:
            raise FaceNotFound("No face was found in that photo. Use a clear photo where the face is visible.")
        # The largest clear face: a borderline detection (a blurred face in the background, a partial one) can
        # be bigger than the person meant, so faces well below the best score only count when nothing is clearer.
        best = max(float(f[14]) for f in faces)
        face = max((f for f in faces if float(f[14]) >= best - 0.1), key=lambda f: f[2] * f[3])
        return (np.asarray(face[4:14], dtype=np.float32).reshape(5, 2) / scale).astype(np.float32), len(faces)

    def embed(self, photo_path: Path) -> tuple[Any, int]:
        """(embedding, number of faces in the photo)."""
        import cv2
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(photo_path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            points, count = self._landmarks(bgr)
            matrix, _ = cv2.estimateAffinePartial2D(points, np.asarray(_ARCFACE_TEMPLATE, dtype=np.float32), method=cv2.LMEDS)
            aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)
            # InsightFace's ArcFaceONNX input: RGB, (x - 127.5) / 127.5, NCHW
            blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            session = self._recognizer()
            embedding = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
        return embedding / np.linalg.norm(embedding), count  # InsightFace's normed_embedding


class FaceID:
    """Face-conditioned generation for an SDXL diffusers pipeline."""

    def __init__(self, config: FaceIDConfig, root: Path, embedder: FaceEmbedder | None = None) -> None:
        self.config = config
        self._embedder = embedder or FaceEmbedder(config, root)

    def embed(self, photo_path: Path) -> tuple[Any, int]:
        return self._embedder.embed(photo_path)

    def generation_kwargs(self, embedding, guidance: float, device: str = "mps") -> dict[str, Any]:
        """ip_adapter_image_embeds for the pipeline call: [negative, positive] with CFG (guidance > 1), as diffusers
        expects for precomputed embeds; only the positive without it."""
        import torch

        positive = torch.from_numpy(embedding).reshape(1, 1, 1, -1).to(torch.float16)
        embeds = torch.cat([torch.zeros_like(positive), positive]) if guidance > 1 else positive
        return {"ip_adapter_image_embeds": [embeds.to(device)]}

    def load(self, pipe) -> None:
        pipe.load_ip_adapter(self.config.adapter_repo, subfolder=None, weight_name=self.config.adapter_file,
                             image_encoder_folder=None)
        pipe.set_ip_adapter_scale(self.config.scale)
        logger.info("FaceID adapter loaded (%s, scale %.2f)", self.config.adapter_file, self.config.scale)

    def unload(self, pipe) -> None:
        pipe.unload_ip_adapter()
        logger.info("FaceID adapter unloaded")
