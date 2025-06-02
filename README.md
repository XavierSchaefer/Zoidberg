

Crée environnement 
>python -m venv NOM_DE_LENVIRONNEMENT
>py -3.10 -m venv monenv

#### Lancer l'environnement : 
.\monenv\Scripts\activate
Linux: source monenv/bin/activate

### Gérer un .requirement.txt
    pip freeze > requirements.txt

### Installer les dépendances du fichier 
    pip install -r requirements.txt
    

### Lancer tensorboard : Lancer un entrainement puis 
    tensorboard --logdir=./logs

Lancer : python main.py