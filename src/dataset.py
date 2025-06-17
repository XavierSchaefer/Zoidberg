# from torchvision import datasets, transforms
# from sklearn.utils import resample

# import numpy as np
# from PIL import Image # Assurez-vous que Pillow est installé (pip install Pillow)
# import sys # Pour afficher la progression sur la même ligne
# class Dataset:
#   def __init__(self,dataPath,type="img",transform=None):
#     self.dataPath = dataPath
#     self.type = type
#     self.dataset = None
#     self.transform = transform
#     self.loadDataset()
#   def __len__(self):
#     return len(self.dataset) if self.dataset else 0
#   def loadDataset(self):
#     if self.type == "img":
#       self.dataset = datasets.ImageFolder(root=self.dataPath, transform=self.transform)
#     else:
#       raise ValueError(f"Type de dataset non pris en charge : {self.type}.")
#     return self.dataset
  
#   def setTransform(self,transform):
#     self.transform = transform
#     if self.dataset is not None:
#       self.loadDataset()
#     return self.transform
#   def getTransform(self):
#     return self.transform
#   def getDataset(self):
#     return self.dataset
#   def dataset_to_numpy(dataset, target_height, target_width): # <-- Ajout des arguments
#     """
#     Convertit un dataset itérable (retournant des (Image PIL, label))
#     en tableaux NumPy X et y, en redimensionnant les images et
#     en les convertissant en niveaux de gris.
#     """
#     X = []
#     y = []
#     print(f"⏳ Conversion du dataset en NumPy ({target_height}x{target_width}, Grayscale)...")
#     count = 0
#     total = 0
#     # Essayons d'obtenir la taille totale si possible (peut échouer pour certains itérables)
#     try:
#         total = len(dataset)
#         if total == 0 : total = -1 # Indique qu'on ne connait pas la taille
#     except TypeError:
#         total = -1 # On ne connait pas la taille
#         print("   (Taille totale du dataset inconnue, affichage simple du compteur)")

#     for img, label in dataset:
#         count += 1
#         try:
#             # --- CHANGEMENT ICI ---
#             # 1. Redimensionner l'image PIL
#             #    Utiliser Image.Resampling.LANCZOS pour une bonne qualité de redimensionnement
#             img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

#             # 2. Convertir en niveaux de gris ('L')
#             img_gray = img_resized.convert('L')

#             # 3. Convertir l'image traitée en tableau NumPy
#             img_array = np.array(img_gray)
#             # --- FIN CHANGEMENT ---

#             # Vérification de la forme (devrait maintenant être constante)
#             if img_array.shape != (target_height, target_width):
#                  print(f"\n⚠️ Image {count} a une forme inattendue {img_array.shape} après traitement. Ignorée.", file=sys.stderr)
#                  continue

#             X.append(img_array)
#             y.append(label)

#             # Afficher la progression
#             progress_char = "."
#             if total > 0:
#                 percent = int(100 * count / total)
#                 bar_len = 20
#                 filled_len = int(bar_len * count // total)
#                 bar = '█' * filled_len + '-' * (bar_len - filled_len)
#                 sys.stdout.write(f'\r   Progression: |{bar}| {percent}% ({count}/{total})')
#                 sys.stdout.flush()
#             elif count % 100 == 0: # Simple compteur si taille inconnue
#                  sys.stdout.write(f'\r   Images traitées: {count}{progress_char * (count//100 % 4)}')
#                  sys.stdout.flush()


#         except Exception as e:
#             # Affiche l'erreur mais continue si possible
#             print(f"\n❌ Erreur lors du traitement de l'image {count}: {e}. Image ignorée.", file=sys.stderr)
#             continue # Passe à l'image suivante

#     # Aller à la ligne après la barre de progression
#     sys.stdout.write('\n')
#     print(f"✅ Conversion terminée. {len(X)} images chargées avec succès.")

#     # Vérification finale avant de retourner
#     if not X:
#          print("❌ Aucune image n'a pu être chargée. Vérifiez le dataset et les erreurs ci-dessus.", file=sys.stderr)
#          # Retourner des tableaux vides pour éviter une erreur plus tard
#          return np.array([]), np.array([])

#     # np.array(X) devrait maintenant fonctionner car toutes les img_array ont la même shape
#     try:
#         X_np = np.array(X)
#         y_np = np.array(y)
#         print(f"   Forme de X résultant: {X_np.shape}") # Pour vérifier
#         return X_np, y_np
#     except Exception as e:
#         print(f"\n❌ Erreur finale lors de la création du tableau NumPy global : {e}", file=sys.stderr)
#         print(f"   Cela peut arriver si certaines images ont échappé aux vérifications de forme.", file=sys.stderr)
#         # Tenter de trouver la ou les formes problématiques (peut être lent)
#         shapes = {arr.shape for arr in X}
#         print(f"   Formes détectées dans la liste X avant l'erreur: {shapes}", file=sys.stderr)
#         # Retourner des tableaux vides pour éviter un crash complet
#         return np.array([]), np.array([])
#   from sklearn.utils import resample

#   def balance_dataset(X, y):
#     # Sépare les classes
#     X_0 = X[y == 0]
#     X_1 = X[y == 1]

#     n_samples = min(len(X_0), len(X_1))  # Prend la plus petite classe

#     X_0_bal = resample(X_0, replace=False, n_samples=n_samples, random_state=42)
#     X_1_bal = resample(X_1, replace=False, n_samples=n_samples, random_state=42)

#     y_0_bal = np.zeros(n_samples, dtype=int)
#     y_1_bal = np.ones(n_samples, dtype=int)

#     # Fusion et shuffle
#     X_balanced = np.vstack((X_0_bal, X_1_bal))
#     y_balanced = np.concatenate((y_0_bal, y_1_bal))

#     # Mélange les données
#     indices = np.random.permutation(len(X_balanced))
#     return X_balanced[indices], y_balanced[indices]
