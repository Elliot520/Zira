# Kokoro TTS

- Model: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0. Voices are the model's own
  `voices/*.pt` (54 of them, e.g. `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` for Hindi, `af_heart` for American English).
- Code: the `kokoro` and `misaki` packages (Apache-2.0) from PyPI, run from `.venv-kokoro` by `worker.py`.
- Hindi G2P uses espeak-ng (GPL-3.0) through `espeakng-loader`, which bundles the library; nothing is installed
  system-wide.
- Everything runs locally. No text is sent to any service; the only network use is the one-time model download.
