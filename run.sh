# ------
# Essai d'entraînement de PP-StructureV3
# https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#41-data-preparation
# ------

# Préparation des données (si téléchargement du dataset d'exemple)
#mkdir -p dataset
#wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_layout_examples.tar -P ./dataset
#tar -xf ./dataset/det_layout_examples.tar -C ./dataset/

# Changement de répertoire vers le projet PaddleX (où se trouve main.py)
cd paddlex-xp/PaddleX

# Vérification du dataset
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./../../dataset/annuaires-test

# Lancement de l'entraînement
## Test sur 1 epoch
## Vérifier/Modifier le numéro de GPU selon la machine
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./../../dataset/annuaires-test \
    -o Global.device=gpu:1 \
    -o Global.output=./../../output \
    -o Train.epochs_iters=1