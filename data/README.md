# Training data

FramerAI trains from scratch on your own local data - there are no external
teacher models. Put your files anywhere under this directory (subfolders are
scanned recursively) and point training at it:

```bash
python build.py --mode all --size tiny --data-dir data
```

The bulk of this directory is git-ignored; only this README and the `examples/`
folder are tracked.

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
{"image": "images/cat.png", "caption": "a cat sitting on a windowsill"}
```

Relative paths are resolved against the `.jsonl` file's own folder.

### Audio captions (for audio generation, with `--train-modalities`)

`*.jsonl` records with an audio path and its text:

```json
{"audio": "clips/hello.wav", "text": "hello and welcome"}
```

Audio is read with `soundfile` or `torchaudio` and resampled to the model's
sample rate.

## Enabling modality training

Text trains by default. To also train the image and audio generators on caption
pairs:

```bash
python build.py --mode all --size tiny --data-dir data --train-modalities
```

See [examples/](examples/) for ready-to-read samples.
