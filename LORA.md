# LoRA training (Zira LoRA Studio)

Train a small SDXL LoRA (style/person/object) on your images, for Zira's REALISTIC image model.

## Where things are
| What | Where |
|---|---|
| Studio app (separate FastAPI) | `/Users/rehanali/worksapce/AI/zira_lora_studio_fixed` (own `.venv`) |
| Studio code | `app.py` (API + trainer launch), `image_downloader.py`, `static/index.html` |
| Studio settings | `studio_settings.json` (last project) |
| Zira launcher/proxy | `jarvis-ai/app/api/lora_studio.py` (`POST /api/lora-studio/start`, proxied at `/lora-studio/`) |
| Studio log | `jarvis-ai/logs/lora_studio.log` |
| Downloaded images | `jarvis-ai/data/image_downloads/<slug>/` |
| Training workspace | `jarvis-ai/data/lora_training/<slug>/` (images + `config.json`) |
| Trained LoRA weights | `jarvis-ai/models/loras/<slug>/pytorch_lora_weights.safetensors` |
| Base model | Perfection Realistic ILXL HF snapshot folder (the one with `model_index.json`); auto-detected from `IMAGE_STYLE_REALISTIC_MODEL` |

## How to use
1. In Zira press **LoRA** (desktop header; phone: Settings → LoRA training). Zira starts the Studio (127.0.0.1:8765, detached, survives Zira restarts) and opens it via `/lora-studio/`.
2. Either:
   - **CREATE LoRA (one click)**: type a name → downloads ~20 images (ddgs search, 500px, duplicates skipped, optional single-face filter YuNet) → trains. Name = trigger = slug. Resolution = image size rounded up to 64 (500 → 512).
   - **Manual**: pick an image folder, set name, trigger, base model, steps (1200), rank (16), learning rate (1e-4), resolution, then Train.
3. Watch progress in the page (`GET /api/train/status`). Press **STOP LoRA STUDIO** when done (kills trainer + Studio; `POST /api/shutdown`).

## How training works
- `prepare_dataset()` copies images to `data/lora_training/<slug>/images`, writes `config.json`.
- `run_train()` downloads diffusers' `train_dreambooth_lora_sdxl.py` (matching the installed diffusers version) and runs `accelerate launch` with: `--instance_prompt "a photo of <trigger>"`, batch 1, constant LR, `--max_train_steps`, `--rank`. LoRA alpha = rank (the trainer has no alpha flag).
- Trainer runs in its own process group; Stop kills it.

## Tips / known issues
- 16GB Mac: **free Zira's image/video models first** (⟲ Reset button). An earlier training produced no weights, probably out of memory.
- Use 15–30 varied, clear images; 1000–1500 steps; rank 16.
- Only train on images you have rights to use; person LoRAs only of consenting people (yourself).

## Pending (not built yet)
- **Zira does not load trained LoRAs yet.** Needed: in `app/tools/image.py`, after `get_txt2img()`, call `pipe.load_lora_weights(models/loras/<slug>, adapter_name=...)` + `pipe.set_adapters([...], adapter_weights=[0.8])` when the prompt contains a known trigger; unload with `pipe.unload_lora_weights()`. The Studio's "integration/apply" button only writes a helper `app/tools/lora_identity.py` (`load_identity_lora(pipe, path, scale)`), it does not wire it in.
- Optional: a LoRA picker in the image settings drawer; list `models/loras/*`.
