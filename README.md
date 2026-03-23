# PP-DocLayout fine-tuning

## Doc
* https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html

## Install
```bash
install.sh
```

## Label Studio to PaddleDetection conversion

Use `dataset.py` to convert Label Studio layout annotations to COCO JSON
compatible with PaddleDetection.

### 1) Convert to one COCO annotation file

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py \
	--input .\dataset\annuaires-test\ev9iv0_label_studio.json \
	--output .\dataset\annuaires-test\ev9iv0_coco.json
```

### 2) Convert and create train/val split files

This mode writes 3 files:
- full annotations (`--output`)
- train split (`*_train.json`)
- val split (`*_val.json`)

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py \
	--input .\dataset\annuaires-test\ev9iv0_label_studio.json \
	--output .\dataset\annuaires-test\ev9iv0_coco.json \
	--split-train-val \
	--val-ratio 0.2 \
	--seed 42
```

### 3) Optional flags

- `--image-prefix JPEGImages`: prefix image file names in COCO (`JPEGImages/f22.jpg`, etc.)
- `--include-transcription`: add Label Studio textarea content to each annotation under `transcription`
- `--no-background`: remove category id `0` (`_background_`)
- `--train-output <path>` and `--val-output <path>`: custom split output paths

Example with explicit split outputs:

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py \
	--input .\dataset\annuaires-test\ev9iv0_label_studio.json \
	--output .\dataset\annuaires-test\ev9iv0_coco.json \
	--split-train-val \
	--train-output .\dataset\annuaires-test\train.json \
	--val-output .\dataset\annuaires-test\val.json
```

## To do
- [X] Soduco v4 -> Label Studio (test with one directory only)
- [ ] Download images from Gallica... (en cours............)
- [X] Label Studio -> PPDocLayout
- [ ] Check dataset files for fine-tuning
- [ ] Fine-tuning