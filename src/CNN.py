# # src/CNNModel.py
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dropout, Dense, InputLayer, BatchNormalization, ReLU, Add ,Resizing, RandomRotation, RandomTranslation, RandomZoom, Rescaling
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.metrics import Precision, Recall
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from sklearn.metrics import recall_score
# from imblearn.over_sampling import RandomOverSampler
# from src.dataset import Dataset
# from src.Model_Analyser import ModelAnalyser

# # Configurez TensorBoard
# tensorboard_callback = TensorBoard(
#     log_dir="./logs",         # répertoire pour stocker les logs
#     histogram_freq=1,         # calculez les histogrammes des poids (et activations) toutes les 1 epoch
#     write_graph=True,         # enregistre le graph du modèle
#     write_images=True,        # enregistre des images des poids si possible
#     update_freq='epoch'       # mise à jour à la fin de chaque epoch
# )

# def residual_block(x, filters):
#     """
#     Petit bloc résiduel :
#       - Conv -> BN -> ReLU
#       - Conv -> BN
#       - skip connection + ReLU final
#     """
#     x_skip = x  # sauvegarde l'entrée pour la skip-connection
    
#     # 1ère conv
#     x = Conv2D(filters, (3,3), padding='same', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
    
#     # 2ème conv
#     x = Conv2D(filters, (3,3), padding='same', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)

#     # skip connection
#     x = Add()([x, x_skip]) 
#     x = ReLU()(x)
#     return x

# def build_resnet_model(height=128, width=128, outputSize=1):
#     inp = Input(shape=(height, width, 1))
    
#     x = Rescaling(1./255)(inp) # 'x' prend la sortie de Rescaling(inp)

#     # Bloc 1 : Conv + BN + Pool
#     # La première couche Conv2D utilise 'x' (qui est la sortie de Rescaling)
#     x = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(0.001))(x) 
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D((2,2))(x)
    
#     # Bloc résiduel 1
#     x = residual_block(x, 32)
#     x = MaxPooling2D((2,2))(x)

#     # Bloc 2 : conv normal
#     x = Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D((2,2))(x)
    
#     # Bloc résiduel 2
#     x = residual_block(x, 64)
#     x = MaxPooling2D((2,2))(x)

#     # Bloc 3 : conv normal
#     x = Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D((2,2))(x)

#     # Bloc résiduel 3
#     x = residual_block(x, 128)
#     x = MaxPooling2D((2,2))(x)

#     # On peut continuer, 
#     # ou s'arrêter là si la taille spatiale devient trop petite

#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.4)(x)
#     out = Dense(outputSize, activation='sigmoid')(x)
    
#     model = Model(inputs=inp, outputs=out)
#     return model
# def ConvolutionalNeuralNetworkModel():
#     # 📦 Paramètres globaux
#     height = 128
#     width = 128
#     outputSize = 1
#     epoch = 50
#     batchSize = 32
#     random_seed = 42
#     patience = 5
#     learningRate = 0.001

#     # 🔄 Chargement et transformation des données

#     data_augmentation = Sequential(
#         [
#             # InputLayer est utile ici si on applique ce modèle séparément
#             # InputLayer(input_shape=(height, width, 1)), # Pas nécessaire si intégré au pipeline tf.data .map
#             # Note: Resizing et Grayscale sont gérés lors du chargement initial ou avant oversampling ici.
#             # Les augmentations aléatoires sont appliquées uniquement au training set.
#             RandomRotation(factor=0.1), # Rotation de +/- 10% * 2*pi
#             RandomTranslation(height_factor=0.1, width_factor=0.1),
#             RandomZoom(height_factor=0.1, width_factor=0.1),
#             # RandomFlip("horizontal") # Si vous voulez le flip horizontal
#             # Rescaling(1./255) # Normalisation des pixels si les images sont chargées en 0-255
#                                 # Si ToTensor() normalisait déjà en 0-1, ceci n'est pas nécessaire.
#                                 # Si les images sont déjà en 0-1 après Dataset, pas besoin.
#         ],
#         name="data_augmentation",
#     )

#     # 🔄 Chargement et transformation des données
#     print("🔄 Chargement des données initiales...")
#     train_data = Dataset("dataset/initial/train", "img", None) # Utiliser pour entrainer le modèle
#     x_train_initial, y_train_initial = Dataset.dataset_to_numpy(train_data.getDataset(),height,width)

#     test_data = Dataset("dataset/initial/test", "img", None) # Utiliser pour l'évaluation finale
#     x_test, y_test = Dataset.dataset_to_numpy(test_data.getDataset(),height,width) 

#     # Vérifier si les données ont été chargées correctement
#     if x_train_initial.size == 0 or x_test.size == 0: # Utilise x_test ici
#          print("❌ ERREUR: Échec du chargement des données initiales. Arrêt.")
#          # Peut-être retourner None ou lever une exception pour arrêter proprement
#          return None
#     print(f"✅ {len(x_train_initial)} images entraînement, {len(x_test)} images test")

#     # Séparation entrainement en 2 ensembles: train et validation ( permet de surveiller l'entrainement )
#     x_train_split, x_val, y_train_split, y_val = train_test_split(
#         x_train_initial,
#         y_train_initial,
#         test_size=0.2,
#         random_state=random_seed,
#         stratify=y_train_initial
#     )
#     print(f" Entrainement séparé en 2 ensembles: Entrainement: {len(x_train_split)}, Validation: {len(x_val)}")


#     print("⚖️ Application de l'Oversampling sur le jeu d'entraînement...")
#     n_samples_train,h,w = x_train_split.shape
#     x_train_split_flat = x_train_split.reshape((n_samples_train, -1))

#     # 🔁 Oversampling pour équilibrer
#     ros = RandomOverSampler(random_state=random_seed)
#     x_train_resampled_flat, y_train_resampled = ros.fit_resample(x_train_split_flat, y_train_split)

#     # Remettre en forme d'image pour le CNN
#     x_train_resampled = x_train_resampled_flat.reshape((-1, h, w))
#     print(f"Train après Oversampling: {len(x_train_resampled)}")

#     # 💡 Vérification: Assurez-vous que les images sont bien (N, H, W) ou (N, H, W, 1)
#     # Et que les pixels sont normalisés (ex: 0-1 ou -1 à 1)
#     # Ajout de la dimension 'channel' si nécessaire (pour grayscale, c'est 1)
#     print("⚙️ Préparation finale des tenseurs...")
#     if x_train_resampled.ndim == 3:
#         x_train_resampled = np.expand_dims(x_train_resampled, axis=-1)
#     if x_val.ndim == 3:
#         x_val = np.expand_dims(x_val, axis=-1)
#     if x_test.ndim == 3:
#         x_test = np.expand_dims(x_test, axis=-1)

#     print(f"   Shape après ajout canal: x_train={x_train_resampled.shape}, x_val={x_val.shape}, x_test={x_test.shape}") # Log ajouté

#     # S'assurer que les types sont corrects (float32 pour les images, int ou float pour labels selon la loss)
#     x_train_resampled = x_train_resampled.astype(np.float32)
#     y_train_resampled = y_train_resampled.astype(np.int32) # Ou float32 si besoin
#     x_val = x_val.astype(np.float32)
#     y_val = y_val.astype(np.int32)
#     x_test = x_test.astype(np.float32)
#     y_test = y_test.astype(np.int32)

#     print("🚀 Création des pipelines tf.data...")
#     AUTOTUNE = tf.data.AUTOTUNE # Permet à tf.data de gérer le parallélisme

#     # Fonction pour appliquer l'augmentation (uniquement au train set)
#     def augment_data(image, label):
#         # L'augmentation est appliquée à chaque image individuellement
#         image = data_augmentation(image, training=True)
#         return image, label
    
#     # Création du dataset d'entraînement
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train_resampled, y_train_resampled))
#     train_dataset = (
#         train_dataset
#         .shuffle(len(x_train_resampled), seed=random_seed) # Mélanger les données
#         .batch(batchSize)             # Créer des batches
#         .map(augment_data, num_parallel_calls=AUTOTUNE) # Appliquer l'augmentation aux batches
#         .prefetch(buffer_size=AUTOTUNE) # Précharger les prochains batches
#     )

#     # Création du dataset de validation (pas d'augmentation, pas de shuffle)
#     val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#     val_dataset = (
#         val_dataset
#         .batch(batchSize)
#         .prefetch(buffer_size=AUTOTUNE)
#     )

#     # Création du dataset de test (pas d'augmentation, pas de shuffle)
#     test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#     test_dataset = (
#         test_dataset
#         .batch(batchSize)
#         .prefetch(buffer_size=AUTOTUNE)
#     )


#     # 🧬 Création du modèle
#     model = build_resnet_model(height=height, width=width, outputSize=outputSize)

#     # Compilation du modèle
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

#     early_stop = EarlyStopping(
#     monitor='val_loss',   # on surveille la validation loss
#     patience=patience,           # nombre d'epochs sans amélioration qu'on tolère
#     restore_best_weights=True  # pour récupérer le meilleur modèle final
#     )

#     print("🚀 Entraînement...")
#     history = model.fit(train_dataset, epochs=epoch, callbacks=[early_stop,tensorboard_callback], validation_data=val_dataset)

#     results = model.evaluate(test_dataset, verbose=0)
#     test_loss = results[0]
#     test_acc = results[1]
#     test_precision = results[2]
#     test_recall = results[3] # L'index dépend de l'ordre dans model.compile


#     print(f"📉 Test Loss: {test_loss:.4f}")
#     print(f"✅ Test Accuracy: {test_acc:.4f}")
#     print(f"🎯 Test Precision: {test_precision:.4f}")
#     print(f"📌 Test Recall: {test_recall:.4f}")
#     param_count = model.count_params()
#     param_bytes = param_count * 4
#     param_megabytes = param_bytes / 1024**2
#     print(f"Nombre de paramètres : {param_count}")
#     print(f"Taille ~ {param_megabytes:.2f} Mo (float32)")

#     # 🔍 Prédictions
#     print("🔍 Génération des prédictions sur le jeu de test...")
#     y_pred_probs = model.predict(x_test)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()
#     y_pred_probs_flat = y_pred_probs.flatten()
#     # Assurez-vous que y_test est aussi un array 1D pour la comparaison
#     if y_test.ndim > 1:
#       y_test_flat = y_test.flatten()
#     else:
#       y_test_flat = y_test


#     recall_sklearn = recall_score(y_test_flat, y_pred)
#     print(f"📌 Test Recall (scikit-learn): {recall_sklearn:.4f}")


#     model_config = {
#         "height": height,
#         "width": width,
#         "hiddenUnitsSize": 128, # Taille de l'avant-dernière couche Dense
#         "hiddenUnitStacks": 3, # Nombre de blocs résiduels (approx)
#         "generationSize": history.epoch[-1] + 1, # Nombre réel d'époques effectuées
#         "batchSize": batchSize,
#         "test_acc": test_acc,
#         "test_loss": test_loss,
#         "test_precision": test_precision,
#         "test_recall": test_recall,
#         # 'test_auc': roc_auc <-- L'AUC sera ajouté par ModelAnalyser lui-même
#     }

#     analyser = ModelAnalyser(
#         history=history,              # Objet History Keras
#         y_true=y_test_flat,           # Vraies étiquettes (1D)
#         y_pred=y_pred,                # Prédictions binaires (1D)
#         y_pred_probs=y_pred_probs_flat, # Probabilités prédites (1D) <-- NOUVEAU
#         model_config=model_config,    # Configuration
#         class_names=['NORMAL', 'PNEUMONIA'], # Noms des classes
#         save_dir="resultsCNNNew"      # Répertoire de sauvegarde
#     )
#     analyser.plot_and_analyse()

#     return model