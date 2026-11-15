# ❓ Troubleshooting & FAQ

Common issues, their solutions, and frequently asked questions.

---

## Common Errors

### 🔴 `GOOGLE_API_KEY` not set

**Error:** Gemini AI analysis fails with authentication error.

**Fix:** Ensure your `.env` file contains a valid Gemini API key:
```
GOOGLE_API_KEY=your-key-here
```
Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

---

### 🔴 FFmpeg not found

**Error:** `FileNotFoundError: ffmpeg not found` or `ffmpeg is not recognized`

**Fix:** Install FFmpeg and ensure it's in your system PATH:
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Colab/Kaggle:** FFmpeg is pre-installed

---

### 🔴 CUDA / GPU Memory Error

**Error:** `torch.cuda.OutOfMemoryError` or `RuntimeError: CUDA out of memory`

**Fix:** Try these solutions in order:
1. Use a smaller Whisper model: `--whisper-model medium`
2. Use int8 compute type: `--whisper-compute-type int8`
3. Force CPU: `--whisper-device cpu`
4. Use `--use-dlp-subs` to skip Whisper entirely (YouTube sources only)

---

### 🔴 `float16` not supported (Kaggle)

**Error:** Whisper crashes with compute type errors on Kaggle T4.

**Fix:** Use float32:
```bash
python main.py --url "VIDEO_URL" --whisper-compute-type float32
```

---

### 🔴 `DiarizeOutput` has no attribute `itertracks`

**Error:** Split-screen or camera-switch fails with Pyannote error.

**Fix:** This is a Pyannote version compatibility issue. Ensure you have the latest version:
```bash
pip install --upgrade pyannote.audio
```

---

### 🔴 `torchaudio` has no attribute `AudioMetaData`

**Error:** PyAnnote stops while importing with `AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`.

**Cause:** Older `pyannote.audio` releases use audio metadata APIs removed from newer `torchaudio` versions.

**Fix:** From the repository root, update the project environment to the supported PyAnnote version, then restart the backend:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 🔴 HuggingFace Token Error (Podcast Modes)

**Error:** PyAnnote cannot download `pyannote/speaker-diarization-3.1` or `pyannote/segmentation-3.0`, or returns `401 Unauthorized` / `NoneType` has no attribute `eval`.

**Cause:** Both models are gated. A token being present is not enough; the Hugging Face account must have accepted the terms for both repositories, and the token must belong to that account and be authorized to read them.

**Fix:**
1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the terms for [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. In the dashboard's **Settings**, enter the token in **HuggingFace Token** and save it. Alternatively, set `HF_TOKEN=hf_your-token` in the repository's git-ignored `.env`.
4. Start a new Auto Story job. The failed job cannot resume from the diarization step.

**Alternative:** Use `--split-trigger face` which doesn't require a token:
```bash
python main.py --url "VIDEO_URL" --split-screen --dynamic-split --split-trigger face
```

---

### 🔴 Google Drive Download Failed (429)

**Error:** `HTTP Error 429: Too Many Requests` when downloading from Google Drive.

**Fix:** The system uses `gdown` for Google Drive downloads. If rate-limited:
1. Wait a few minutes and retry
2. Use a direct download link
3. Download the file manually and use `--source local`

---

### 🔴 AV1 Codec Crash

**Error:** Rendering crashes when source video uses AV1 codec.

**Fix:** The system automatically excludes AV1 (`av01`) codecs. If issues persist, try:
```bash
python main.py --url "VIDEO_URL" --source-height 1080
```

---

### 🔴 TikTok HEVC Crash

**Error:** `IndexError: tuple index out of range` during Whisper transcription for TikTok videos.

**Fix:** The system automatically prefers H.264 for non-YouTube sources. If issues persist, the video may need to be downloaded manually.

---

### 🔴 Glitch Transition Disappears (High Resolution)

**Error:** Hook glitch teaser is invisible or glitchy at 2K/4K resolution.

**Fix:** This was fixed in v0.9.11. Ensure you're using the latest version. The glitch video now dynamically scales to match output dimensions.

---

## Frequently Asked Questions

### Q: What GPUs are supported?

Any **NVIDIA CUDA-compatible GPU** works. Tested on:
- NVIDIA T4 (Google Colab / Kaggle)
- NVIDIA RTX 3060 / 3070 / 3080 / 3090
- NVIDIA RTX 4060 / 4070 / 4080 / 4090

CPU-only mode is also available (slower but functional).

---

### Q: How long does processing take?

Processing time depends on:
- Video length
- Number of clips
- GPU vs CPU
- Enabled features

Typical benchmarks for a 30-minute source video, 7 clips:
| GPU | Time |
|---|---|
| RTX 3080 | ~15-20 minutes |
| T4 (Colab) | ~25-35 minutes |
| CPU only | ~60-90 minutes |

---

### Q: Can I use this commercially?

Yes! The project is open source. However, ensure your BGM music is royalty-free and your source video content is licensed for commercial use.

---

### Q: How do I skip Whisper transcription?

Use YouTube's built-in subtitles:
```bash
python main.py --url "VIDEO_URL" --use-dlp-subs
```

---

### Q: Can I process non-YouTube videos?

Yes! Supported platforms:
- TikTok: `--source tiktok`
- Instagram: `--source instagram`
- Google Drive: `--source gdrive`

---

### Q: How do I reproduce/debug a specific clip?

1. Find the `gemini_response.json` in your output directory
2. Re-run with `--load-gemini-json` to skip the AI analysis step:
   ```bash
   python main.py --url "VIDEO_URL" --load-gemini-json
   ```

---

### Q: What if AI generates bad clips?

Try these approaches:
1. **Change the Gemini model**: `--gemini-model gemini-2.5-flash`
2. **Increase clip count**: `--clips 10` (more choices = better chance of good clips)
3. **Try NVIDIA NIM**: `--ai-provider nvidia`
4. **Manual curation**: Use `--load-gemini-json` to edit the JSON and re-render

---

## Disk Cleanup

The pipeline creates intermediate files that can take up significant space. Clean up with:

```bash
bash cleanup.sh
```

This removes temporary files while preserving final clips and job history.

---

## See Also

- [Getting Started](Getting-Started) — Installation guide
- [Google Colab Guide](Google-Colab-Guide) — Cloud GPU setup
- [CLI Reference](CLI-Reference) — All available options
