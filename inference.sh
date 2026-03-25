# Inference
## Mettre à jour le chemin vers le modèle à tester si besoin
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="./../../dataset_test/images/val_000001.jpg"