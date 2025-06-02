import tensorflow as tf
from collections import OrderedDict
import numpy as np
import os
from src.analyse.CNNAnalyser import analyseModel
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
from src.analyse.CNNAnalyser import get_accuracy
import json
mixed_precision.set_global_policy('mixed_float16')

TRAIN_DATASET_PATH = 'dataset/initial/train'
TEST_DATASET_PATH = 'dataset/initial/test'
VALIDATION_DATASET_PATH = 'dataset/initial/val'

WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 16
OUTPUT_SIZE = 1
EPOCHS = 50
SEED = 42
PATIENCE = 8
LEARNING_RATE = 0.001
AUTOTUNE = tf.data.AUTOTUNE

# CLASS=["Sain","Malade"]
def count_classes(ds: tf.data.Dataset,
                  class_names: list[str],
                  dataset_name: str = "") -> dict[str, int]:
    """
    Compte le nombre d'échantillons par classe dans un tf.data.Dataset batché.

    Args:
      ds            : tf.data.Dataset retournant (x_batch, y_batch).
      class_names   : liste des noms de classes, dans l'ordre des labels (0, 1, …).
      dataset_name  : étiquette affichée dans le titre du tableau.

    Returns:
      Un dict {nom_de_classe: count}.
    """
    # Initialisation du compteur
    num_classes = len(class_names)
    total_counts = np.zeros(num_classes, dtype=int)

    # Parcours par batch
    for _, y_batch in ds:
        # y_batch est un tenseur shape=(batch_size, ) ou (batch_size, 1)
        labels = y_batch.numpy().flatten().astype(int)
        # Comptage vectorisé pour ce batch
        batch_counts = np.bincount(labels, minlength=num_classes)
        total_counts += batch_counts

    # Calcul du total global
    total_samples = total_counts.sum()

    # Préparation du dict résultat
    counts_dict = OrderedDict(
        (class_names[i], int(total_counts[i]))
        for i in range(num_classes)
    )

    # Affichage formaté
    header = f"Répartition pour « {dataset_name} »"
    print(f"\n{header}")
    print("-" * len(header))
    print(f"{'Classe':<20}{'Nombre':>10}{'Pourcentage':>15}")
    print("-" * 45)
    for cls, cnt in counts_dict.items():
        pct = (cnt / total_samples * 100) if total_samples else 0
        print(f"{cls:<20}{cnt:>10}{pct:>14.2f}%")
    # Ligne Total
    print("-" * 50)
    print(f"{'Total':<20}{total_samples:>10}{'100.00%':>14}")
    return counts_dict

def ConvolutionalNeuralNetworkV3():
  augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.1),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomFlip(mode='horizontal'),
  ])
  # Importation des jeux de données
  print("1. Loading datasets")

  print("1.1 Loading TRAIN dataset")
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DATASET_PATH,
    image_size=(HEIGHT, WIDTH),
    batch_size=None,        # on gère le batch après l'unbatch
    color_mode='grayscale',
    shuffle=True,           # important !
    seed=SEED
  ) 
  class_names = train_ds.class_names

  train_ds = (
      train_ds
      .cache()                                        # garde tout en RAM après premier epoch
      .shuffle(buffer_size=2000, seed=SEED)           # shuffle large
      .map(lambda x, y: (augment(x), y),
          num_parallel_calls=AUTOTUNE)               # augmentation à la volée
      .map(lambda x, y: (x/255.0, y), num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE, drop_remainder=True)
      .prefetch(AUTOTUNE)
  )
  print("1.2 Loading VALIDATION dataset")
  # 2) Validation et test → pas d'augmentation, seulement normalisation
  validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
      VALIDATION_DATASET_PATH,
      image_size=(HEIGHT, WIDTH),
      batch_size=BATCH_SIZE,
      color_mode='grayscale',
      shuffle=False,
      seed=SEED
  )
  validation_ds = validation_ds.map(lambda x, y: (x/255.0, y),
                                    num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
  print("1.3 Loading TEST dataset")
  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=False,
    seed=SEED
  )
  test_ds = test_ds.map(lambda x, y: (x/255.0, y),
                        num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    
	# 2. Dataset classification and repartition
  print("2. Dataset classification and repartition (train, validation, test)")

  # Nom des classes
  class_namesTrain = class_names
  class_namesValidation = class_names
  class_namesTest = class_names
  # Répartition des classes
  train_counts = count_classes(train_ds,     class_namesTrain, dataset_name="train")
  val_counts   = count_classes(validation_ds, class_namesValidation, dataset_name="validation")
  test_counts  = count_classes(test_ds,       class_namesTest, dataset_name="test")

  # 5. Model creation
  print("5. Model creation...")
  
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(228, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(2048, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid'),
  ])
  # 6. Model compilation  
  print("6. Model compilation...")
  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  # 7. Model training
  print("7. Model training...")
  y_train = np.concatenate([y.numpy().flatten() for _, y in train_ds], axis=0)
  weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
  )
  class_weight = dict(enumerate(weights))
  print("Class weights:")
  print(class_weight)
  model.fit(train_ds,class_weight=class_weight, validation_data=validation_ds, epochs=EPOCHS,callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
  
  # 8. Model evaluation
  print("8. Model evaluation...")
  model.evaluate(test_ds)

  # 9. Model prediction
  print("9. Model prediction...")
  y_pred = model.predict(test_ds)
  y_true = np.concatenate([y.numpy().flatten() for _, y in test_ds], axis=0)

  accuracy = get_accuracy(y_true, y_pred)

 
  # 10. Model save with sample name
  print("10. Model save...")
  MODEL_FILENAME = f"{accuracy:.4f}_CNNV3_{EPOCHS}epochs_{LEARNING_RATE}lr_{PATIENCE}patience_{SEED}seed.keras"
  MODEL_NAME = "CNNV3"
  ROOT = os.getcwd() 
  MODEL_PATH = os.path.join(ROOT,"src", "models", MODEL_NAME, MODEL_FILENAME)
  model.save(MODEL_PATH)
  print(f"Model saved at: {MODEL_PATH}")
  # Save history
  history = model.history.history
  history_path = os.path.splitext(MODEL_PATH)[0] + "_history.json" 
  # Enregistrer l’historique dans un fichier JSON
  with open(history_path, "w", encoding="utf-8") as f:
      json.dump(history, f, indent=4)

  # 11. Analyse du modèle
  print("11. Model analysis...")
  test_path  = os.path.join(ROOT, "dataset", "initial", "test")
  analyseModel(MODEL_PATH, test_path)
