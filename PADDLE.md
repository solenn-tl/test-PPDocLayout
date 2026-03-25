# Fine-tune Paddle Detection

## Env
* Python entre 3.8 et 3.12
```bash
source /mnt/d/Pro/codes/layout-ocr/paddlex-xp/.venv_paddelx/bin/activate
python -V
which python
```

## Installation
```bash
chmod ./install.sh
./install.sh
```
## Fine-tuning
### Quelques remarques
* Il faut être dans le dossier ```paddlex-xp/PaddleX``` où se trouve le fichier ```main.py```.
* Les dossiers du dataset doivent obligatoirement s'appeler ```annotations``` et ```images```.
* Les fichiers avec les annotations doivent obligatoirement s'appeler ```instance_train.json``` et ```instance_val.json```.
* Il ne doit pas y avoir d'autres fichiers ou répertoires dans le répertoire du dataset.

### Exécuter
```bash
./run.sh
```
* Test la structure du dataset
* Réalise le fine-tuning

## Evaluation
```bash
./evaluate.sh
```
* Evalue le résultat sur le val set
## Inference
```bash
./inference.sh
```
* Prédictions sur une image