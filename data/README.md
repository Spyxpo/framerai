# Training data

FramerAI trains from scratch on your own local data - there are no external
teacher models. Put your files anywhere under this directory (subfolders are
scanned recursively) and point training at it:

```bash
python build.py --mode all --size tiny --data-dir data
```

The bulk of this directory is git-ignored; only this README and the `examples/`
folder are tracked. Ready-to-use example datasets with real text, image, and
audio media are located in `data/examples/`.

## Supported formats

### Text

- `*.txt` - plain text. Documents are split on blank lines into training samples.
- `*.jsonl` - one JSON object per line, using any of:
  - `{"text": "..."}`
  - `{"prompt": "...", "response": "..."}` - formatted as a `<user>`/`<assistant>` turn.
  - `{"instruction": "...", "input": "...", "output": "..."}` - `input` is optional.

### Image captions (for image generation, with `--train-modalities`)

`*.jsonl` records with an image path and a caption:

```json
{"image": "media/sunset.png", "caption": "a sunset over the ocean with orange clouds"}
```

- **Metadata format**: JSON object with `"image"` (path to image file) and `"caption"` fields.
- **Path format**: Relative paths (e.g. `media/sunset.png`) are resolved relative to the directory containing the `.jsonl` file. Absolute paths are also supported.
- **Image formats**: PNG, JPEG, WEBP, BMP (opened via Pillow and resized to training resolution).

### Audio captions (for audio generation, with `--train-modalities`)

`*.jsonl` records with an audio path and its text transcript:

```json
{"audio": "media/greeting.wav", "text": "hello and welcome to framerai"}
```

- **Metadata format**: JSON object with `"audio"` (path to audio file) and `"text"` fields.
- **Path format**: Relative paths (e.g. `media/greeting.wav`) are resolved relative to the directory containing the `.jsonl` file. Absolute paths are also supported.
- **Audio formats**: WAV, FLAC, OGG (read via `soundfile` or `torchaudio` and resampled automatically to the model sample rate).

## Example datasets

Sample datasets with real media files are provided under `data/examples/`:

- `data/examples/text.txt` & `data/examples/corpus.jsonl` - Text samples.
- `data/examples/image_captions.jsonl` - Image caption pairs referencing images in `data/examples/media/` (`sunset.png`, `mountains.png`).
- `data/examples/audio_captions.jsonl` - Audio caption pairs referencing audio clips in `data/examples/media/` (`greeting.wav`, `count.wav`).

## Enabling modality training

Text trains by default. To train the text backbone as well as the image and audio generators on the example datasets:

```bash
python build.py --mode all --size tiny --data-dir data/examples --train-modalities
```
