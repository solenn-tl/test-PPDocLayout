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

### 2b) Restrict full/train/val JSON annotations to images currently in `imgs`

Use `--images-dir` to keep only images present in the folder (matching by file name)
before writing the full/train/val JSON files.

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py \
	--input .\dataset\annuaires-test\ev9iv0_label_studio.json \
	--output .\dataset\annuaires-test\ev9iv0_coco.json \
	--split-train-val \
	--images-dir .\dataset\annuaires-test\dataset\imgs \
	--val-ratio 0.2 \
	--seed 42 \
	--train-output .\dataset\annuaires-test\dataset\annotations\train.json \
	--val-output .\dataset\annuaires-test\dataset\annotations\val.json
```

### 3) Optional flags

- `--image-prefix JPEGImages`: prefix image file names in COCO (`JPEGImages/f22.jpg`, etc.)
- `--include-transcription`: add Label Studio textarea content to each annotation under `transcription`
- `--no-background`: remove category id `0` (`_background_`)
- `--train-output <path>` and `--val-output <path>`: custom split output paths
- `--images-dir <path>`: keep only images (and their annotations) that exist in this folder before writing outputs/splits
- `--include-unannotated-images`: include source-folder images that are not in Label Studio as images with zero annotations (requires `--images-dir`)
- `--dataset-dir <path>`: create `annotations/train.json`, `annotations/val.json`, and copy split images into `imgs/` with renamed file names (`train_000001.jpg`, `val_000001.jpg`, etc.)

Example with explicit split outputs:

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py \
	--input .\dataset\annuaires-test\ev9iv0_label_studio.json \
	--output .\dataset\annuaires-test\ev9iv0_coco.json \
	--split-train-val \
	--images-dir .\dataset\annuaires-test\dataset\imgs \
	--train-output .\dataset\annuaires-test\dataset\annotations\train.json \
	--val-output .\dataset\annuaires-test\dataset\annotations\val.json
```

### 4) Build PaddleDetection dataset folders and copy images

This mode creates:
- `dataset_out/annotations/train.json`
- `dataset_out/annotations/val.json`
- `dataset_out/imgs/*` copied from source images and renamed by split

Notes:
- `--dataset-dir` always produces train/val files (it does not require `--split-train-val`).
- `--images-dir` is required with `--dataset-dir`.
- `--include-unannotated-images` requires Pillow (`pip install Pillow`) to read image sizes.

```bash
d:/Pro/codes/layout-ocr/labelstudio_env/Scripts/python.exe .\dataset.py --input .\dataset\annuaires\ev9iv0_label_studio.json --images-dir .\dataset\annuaires\imgs --dataset-dir .\dataset\annuaires\dataset --val-ratio 0.2 --seed 42 
```

## To do
- [X] Soduco v4 -> Label Studio (test with one directory only)
- [ ] Download images from Gallica... (en cours............)
- [X] Label Studio -> PPDocLayout
- [ ] Check dataset files for fine-tuning
- [ ] Fine-tuning