# ------
# Essai d'entraînement de PP-StructureV3
# https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#41-data-preparation
# ------

# Préparation des données
mkdir -p dataset
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_layout_examples.tar -P ./dataset
tar -xf ./dataset/det_layout_examples.tar -C ./dataset/

# Vérification du dataset
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples

# Lancement de l'entraînement
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_layout_examples \
    -o Global.device=gpu:0