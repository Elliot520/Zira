# third_party/indicf5

Inference code for the optional IndicF5-Hinglish voice (see README "Voice"). Runs only inside the separate
`.venv-tts` environment, in its own process (`worker.py`); Zira's main environment never imports it.

- `f5_tts/` - F5-TTS model and inference code as shipped in https://github.com/saravananravi08/indicf5-finetune
  (MIT licence per its pyproject.toml; F5-TTS itself is MIT). The Gradio UI files were left out.
- `voices/` - reference voice clips and their transcripts from the same repository.
- Model weights are not stored here: `Saravananravi/indicf5-hinglish` (Apache-2.0), plus the vocabulary from the
  gated `ai4bharat/IndicF5` (MIT, access must be requested on Hugging Face) and the `charactr/vocos-mel-24khz`
  vocoder, all loaded from the local Hugging Face cache.
