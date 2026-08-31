# Omniscribe

Portable, fully local captioner for **video, GIF, image and audio** — plus a prompt writer that turns a pile of reference media into a single MiniMax H3 video prompt.

It runs on one model, [ASID-Captioner-7B](https://huggingface.co/AudioVisual-Caption/ASID-Captioner-7B), a Qwen2.5-Omni-7B fine-tune loaded in 4-bit. Nothing is uploaded anywhere. No API keys, no accounts, no telemetry.

The thing that separates it from most captioners: **it hears the audio track.** Speech, music, and sound effects all land in the caption, and dialogue is transcribed verbatim.

---

## What it does

Three tabs, one shared settings rail.

### Batch

Point it at a folder, pick which file types to include, and it writes a `.txt` caption beside every file. Recurses into subfolders, skips files that already have a caption, and survives failures — one bad file doesn't stop the run.

Built for building training-caption sets.

### Single file

Drop one file, get one caption on screen. Nothing is written to disk. Use it to dial in a prompt before committing to a batch of a few thousand.

### Prompt maker

Feed it a mixed pile of references — up to **9 images, 3 videos and 3 audio clips** — say what you want in plain language, and it writes one MiniMax H3 prompt.

Each file is labelled the way H3 expects (`<Picture 1>`, `<Video 1>`, `<Audio 1>`) and every label is given an explicit job in the output, which is the single most important thing about prompting H3 well.

It runs in stages rather than one shot: describe each reference, then draft the shot list, then assemble the fields. A 7B model can't hold that whole task in one pass, and asking it to try produces mush.

Two output formats:

- **Six fields** — `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. This is H3's documented reference-to-video structure.
- **Simple prose** — one flowing paragraph, for text-to-video.

The instruction block is a normal editable textbox. The default is a reasonable starting point, not scripture.

---

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 |
| GPU | NVIDIA, **8 GB VRAM minimum**, 12 GB+ comfortable |
| Disk | ~25 GB (18 GB model, ~7 GB environment) |
| Driver | CUDA 13.0 capable |

CPU-only technically runs and is unusably slow. If `nvidia-smi` reports CUDA below 13.0, open `install.bat` and change `set "CUDA=cu130"` to `cu126`.

ffmpeg is **not** a separate install — it comes down with the conda environment.

---

## Install

**1. Build the environment**

```bat
install.bat
```

Downloads a private Miniconda into `installer_files\`, creates a Python 3.12 environment with ffmpeg, then installs torch and the rest. Takes a while. Safe to re-run — it resumes rather than starting over.

**2. Get the model**

```bat
model.bat
```

About 18 GB. Interrupted downloads resume — just run it again.

The weights must sit **flat** in `models\` — `models\model-00001-of-00004.safetensors`, not `models\ASID-Captioner-7B\...`. `model.bat` checks this for you and says so if they land nested.

**3. Run**

```bat
start.bat
```

Opens `http://127.0.0.1:7870`. First caption of each session loads the model into VRAM and takes an extra minute; after that it stays resident.

---

## Settings

| Setting | Does what |
|---|---|
| **Preset** | Which instruction goes to the model. `Auto` picks per file type — video, image or audio. Any other preset unlocks the text box for editing. |
| **Max caption tokens** | Length ceiling. 512 is fine for general captions; raise for long structured formats. |
| **Max frame resolution** | Per-frame pixel cap. Higher catches fine detail — faces, small text — at real VRAM cost. |
| **Use the audio track** | Off makes video captioning faster and blind to sound. |
| **CPU offload** | Only if you're hitting OOM. Considerably slower. |
| **Skip files that already have a caption** | Makes an interrupted batch resumable. |
| **Caption extension** | `txt` by default. |

Prompt maker adds its own three: token budget, reference resolution, and **video sample fps** — how densely reference clips are sampled. The default of 1.0 sees roughly one frame per second, which is enough for scene and mood but *not* enough to describe choreography or fast action. Raise it to 3–4 for anything motion-heavy.

---

## Known limitations

**The `MiniMax H3 T2VA` preset is unreliable.** It asks a 7B model to follow a long multi-field format in a single pass, and it usually just returns an ordinary caption instead. It's kept because it occasionally works on simple clips. **Use the Prompt maker tab instead** — same goal, staged approach, far better results.

**Reference-to-video is not motion transfer.** H3 takes the character of a movement — pacing, energy, style — not exact pose sequences. If you need a dance copied move-for-move, that's a pose-control job (DWPose plus a pose-conditioned model), not this.

**Fine visual detail depends on resolution.** At the default 448px the model reliably gets clothing, setting and broad appearance, and less reliably gets things like heterochromia, small props, or logo text. Raise the resolution slider when identity precision matters.

**GIFs are transcoded.** An `.mp4` is written next to the source and captioned as video. Both files get the caption.

---

## Troubleshooting

**`No model found in models\`** — weights are missing or nested one folder too deep. See step 2.

**CUDA out of memory** — lower Max frame resolution first, then enable CPU offload. On the Prompt maker, also lower video sample fps; each reference clip costs frames.

**`ffmpeg not found`** — the environment is incomplete. Re-run `install.bat`, or drop `ffmpeg.exe` into a `bin\` folder in the project root.

**Audio files fail** — needs ffmpeg for decoding. Same fix.

**Moving the folder** is fine — every script resolves paths relative to itself. The `Scripts\*.exe` shims (`pip.exe`, `hf.exe`) keep the old absolute path baked in, which is normal pip behaviour; call them as `python.exe -m pip` instead, or re-run `install.bat` to regenerate them.

---

## Layout

```
app.py             Gradio UI - layout, CSS, wiring
captioner.py       Everything else - model loading, media, prompts, the
                   staged prompt-maker pipeline
install.bat        Builds installer_files\ from nothing
model.bat          Downloads the weights into models\
start.bat          Launches the UI
requirements.txt   Python deps (torch deliberately excluded - install.bat
                   pulls it from the CUDA index so pip can't substitute
                   the CPU build)
```

`installer_files\` and `models\` are both gitignored — together they're ~25 GB and both are rebuilt by the scripts above.

---

## Credits

- **[ASID-Captioner-7B](https://huggingface.co/AudioVisual-Caption/ASID-Captioner-7B)** — the captioning model
- **[Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)** — the base model and `qwen-omni-utils`
- **[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)** — the prompt format the Prompt maker targets
- **[Gradio](https://www.gradio.app/)** — the UI

The model carries its own license, separate from this code.
