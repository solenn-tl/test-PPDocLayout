# Import pre-annotated data to label studios

## Env
```bash
py -3.11 -m venv labelstudio_env
source labelstudio_env/bin/activate
python -m pip install label-studio

label-studio
```

## Load pre-annotated data

1. Retrieve a BNF manifest and an annotations json from annuaires-viewer.
2. Execute ```convert.py```
3. Load the json whose name include "label_studio" in a Label Studio annotation project for the OCR task. 
4. Setup the tabels of your project (here : SectionHeader and Text)