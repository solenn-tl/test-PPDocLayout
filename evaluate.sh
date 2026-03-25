#Evaluation 
## Mettre à jour le chemin vers le modèle à tester si besoin
## Pour l'instant, ne fonctionne pas car le modèle  entraîné sur une epoch 
## prédit plus de classes que celles attendues dans le dataset (ce qui créé une erreur)
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./../../dataset_test \
    -o Evaluate.weight_path=./output/best_model/best_model/model.pdparams \
    -o Global.num_classes=6 