# ------
# Initialisation du projet
# Testé avec Python 3.11
# ------

mkdir paddlex-xp
cd paddlex-xp

# Création de l'environnement virtuel avec Python 3.11
python3 -m venv .venv

# Activation de l'environnement
source .venv/bin/activate

# Mise à jour de pip
pip install --upgrade pip

# ------
# Installation de PaddlePaddle
# https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html#installing-paddlepaddle-via-docker
# ------

# Installation de la version GPU (CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# ------
# Installation de PaddleX
# https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html#12-plugin-installation-mode
# ------

git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX

# Installation de PaddleX en mode éditable avec ses dépendances de base
pip install -e ".[base]"

# Installation des modules de détection via la CLI PaddleX
paddlex --install PaddleDetection

# Post-bidouillage pas dans la doc mais nécessaire car PaddleDetection attends d'anciennes vieilles versions de pip et OpenCV
pip install "setuptools<82"	      
pip install "opencv-python==4.5.5.64"