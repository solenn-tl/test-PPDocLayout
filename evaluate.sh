# Changement de répertoire vers le projet PaddleX (où se trouve main.py)
cd paddlex-xp/PaddleX

#Evaluation 
## Mettre à jour le chemin vers le modèle à tester si besoin
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./../../dataset/annuaires-test/dataset \
    -o Evaluate.weight_path=./output/best_model/best_model/model.pdparams