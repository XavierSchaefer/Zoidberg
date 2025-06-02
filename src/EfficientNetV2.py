# src/EfficientNetV2.py

import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2S,  # variante Small
    preprocess_input  # preprocessing spécifique à V2
)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

# ─── Paramètres modifiables ────────────────────────────────────────────────────
EFN_VARIANT   = 'S'         # 'S' (Small), ou 'M','L' pour Medium/Large
INPUT_SHAPE   = (224,224,3) # EfficientNet-V2 attend du RGB
DROPOUT_RATE  = 0.4
OUTPUT_SIZE   = 1           # binaire
BATCH_SIZE    = 16
SEED          = 42
AUTOTUNE      = tf.data.AUTOTUNE
# Head
EPOCHS        = 50
LEARNING_RATE = 0.001
PATIENCE      = int(EPOCHS * 0.2)
# Backbone
BACKBONE_EPOCHS   = int(EPOCHS * 1.3)
BACKBONE_LR       = LEARNING_RATE * 0.1 # Réduction significative du LR pour le fine-tuning
BACKBONE_PATIENCE = int(PATIENCE)
# ──────────────────────────────────────────────────────────────────────────────

def count_classes(ds, dataset_name=""):
    """
    Compte le nombre d'échantillons pour chaque classe dans un tf.data.Dataset.
    """
    counts = {0: 0, 1: 0} # Initialise pour 2 classes
    # Itère sur le dataset pour compter les labels
    # Note: ds.unbatch() peut être très lent sur de gros datasets.
    # S'il y a des problèmes de performance, envisager d'itérer sur les batchs.
    for _, y_batch in ds:
        for y_tensor in y_batch:
            y_numpy = y_tensor.numpy()
            # Gérer le cas où y_numpy est un scalaire ou un tableau avec un seul élément
            label = int(y_numpy) if np.isscalar(y_numpy) or y_numpy.size == 1 else int(y_numpy[0])
            if label in counts:
                counts[label] += 1
            else:
                # Si une classe inattendue apparaît, l'ajouter au comptage
                counts[label] = 1
    print(f"Distribution des classes pour {dataset_name}:")
    total_samples = sum(counts.values())
    for cls, count in counts.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  Classe {cls}: {count} échantillons ({percentage:.2f}%)")
    return counts

def to_rgb(x, y):
    """
    Duplique le canal grayscale en 3 canaux RGB, puis cast en float32.
    """
    x = tf.image.grayscale_to_rgb(x)        # (B,H,W,1) → (B,H,W,3)
    x = tf.cast(x, tf.float32)
    return x, y

def build_efficientnet_v2_model(
    variant: str,
    input_shape=INPUT_SHAPE,
    dropout_rate=DROPOUT_RATE,
    output_size=OUTPUT_SIZE
) -> Model:
    """
    Charge la base EfficientNet-V2 (Small/Medium/Large) pré-entraînée sur ImageNet,
    gèle ses poids, puis lui ajoute :
      - un GlobalAveragePooling,
      - un Dropout pour la robustesse,
      - une couche Dense sigmoïde pour la classification binaire.
    """
    # 1) Récupère la classe correspondant à la variante choisie
    Base = getattr(tf.keras.applications, f'EfficientNetV2{variant}')
    base_model = Base(
        include_top=False,      # on enlève la tête ImageNet
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )
    # print model name
    print(f"Used model: {base_model.name}")
    base_model.trainable = False  # on gèle le backbone au début

    # 2) Assemble le modèle final
    inputs = Input(shape=input_shape)
    x = preprocess_input(inputs)              # preprocessing spécifique V2
    x = base_model(x, training=False)         # passe dans EfficientNetV2
    x = GlobalAveragePooling2D()(x)           # moyenne spatiale globale
    x = Dropout(dropout_rate)(x)              # robustesse (simule pannes)
    outputs = Dense(output_size, activation='sigmoid')(x)
    return Model(inputs, outputs)

def plot_confusion_pr(model, test_ds, best_thr):
    """
    Trace la matrice de confusion et la distribution des scores sur test_ds
    en utilisant le seuil best_thr.
    """
    # Predictions + labels
    p = model.predict(test_ds).ravel()
    y = np.concatenate([y for _, y in test_ds]).ravel()
    y_pred = (p >= best_thr).astype(int)

    # Matrice de confusion
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred),
        display_labels=['Normal','Pneumonia']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Test set – seuil {best_thr:.2f}")
    # plt.show( )
    plt.savefig(f"confusion_matrix_{best_thr:.2f}.png")

    # Distribution des scores (pour affiner le seuil)
    prec, rec, thr = precision_recall_curve(y, p)
    plt.figure()
    plt.plot(thr, prec[1:], label='Precision')
    plt.plot(thr, rec[1:],  label='Recall')
    plt.xlabel("Seuil")
    plt.legend()
    plt.title("Precision / Recall vs Threshold")
    plt.savefig(f"precision_recall_curve_{best_thr:.2f}.png")

def EfficientNetV2():
    """
    Pipeline complète :
     1. Chargement / split des données
     2. Conversion grayscale→RGB + preprocess_input
     3. Data augmentation (optionnel)
     4. Construction et compilation du modèle EfficientNetV2
     5. Entraînement head + fine-tuning
     6. Évaluation et affichages
    """
    # 1. Chargement des datasets (training / validation / test)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/initial/train",
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=INPUT_SHAPE[:2],
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/initial/train",
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=INPUT_SHAPE[:2],
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/initial/test",
        image_size=INPUT_SHAPE[:2],
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 2. Conversion en RGB (+ casting) et preprocessing EfficientNetV2
    train_ds = train_ds.map(to_rgb, num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(to_rgb,   num_parallel_calls=AUTOTUNE)
    test_ds  = test_ds.map(to_rgb,  num_parallel_calls=AUTOTUNE)

    # 3. (Optionnel) Data augmentation sur train_ds
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(
        lambda x,y: (augment(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # 4. Prefetch pour la performance
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    # Optionnel : Visualiser un batch d'images du train_ds pour vérification
    # Décommentez pour utiliser
    # print("Visualisation d'un batch de train_ds avant l'entrée dans le modèle...")
    # for images, labels in train_ds.take(1):
    #     plt.figure(figsize=(12, 12))
    #     for i in range(min(9, BATCH_SIZE)):
    #         ax = plt.subplot(3, 3, i + 1)
    #         # Les images ici sont après to_rgb et augment, donc RGB, float32, [0,255]
    #         # preprocess_input est appliqué DANS le modèle.
    #         img_display = images[i].numpy().astype("uint8")
    #         plt.imshow(img_display)
    #         plt.title(f"Label: {labels[i].numpy()}")
    #         plt.axis("off")
    #     plt.savefig("sample_batch_train_ds_before_preprocess.png")
    #     plt.close()
    #     print("Batch d'échantillons sauvegardé dans sample_batch_train_ds_before_preprocess.png")
    #     break

    # Afficher la distribution des classes
    print("\n--- Vérification de la distribution des classes ---")
    train_counts = count_classes(train_ds, "train_ds")
    val_counts = count_classes(val_ds, "val_ds")
    _ = count_classes(test_ds, "test_ds")

    # Calcul des class_weight (exemple simple, à ajuster si nécessaire)
    # class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}
    # Pour l'instant, une version basique si déséquilibre
    class_weights = None
    if train_counts[0] > 0 and train_counts[1] > 0: # S'assurer qu'il y a au moins une instance de chaque classe
        total_train_samples = train_counts[0] + train_counts[1]
        weight_for_0 = (1 / train_counts[0]) * (total_train_samples / 2.0)
        weight_for_1 = (1 / train_counts[1]) * (total_train_samples / 2.0)
        class_weights = {0: weight_for_0, 1: weight_for_1}
        print(f"Poids de classe calculés : {class_weights}")
    else:
        print("Avertissement : Le dataset d'entraînement ne contient pas les deux classes, class_weight non appliqué.")
    print("--- Fin de la vérification ---\n")

    # 5. Création du modèle
    model = build_efficientnet_v2_model(
        variant=EFN_VARIANT,
        input_shape=INPUT_SHAPE,
        dropout_rate=DROPOUT_RATE,
        output_size=OUTPUT_SIZE
    )
    # Optionnel: Vérifier la structure du modèle et l'index du base_model
    # print("Résumé du modèle AVANT de rendre le backbone entraînable:")
    # model.summary() 

    # 6. Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        #  Add validation loss
        metrics=['loss','accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    # 7. Entraînement de la tête uniquement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=PATIENCE, restore_best_weights=True),
            TensorBoard(log_dir="./logsEfficientNetV2", histogram_freq=1)
        ],
        class_weight=class_weights
    )

    # 8. Fine-tuning : dégeler le backbone et ré-entraîner légèrement
    # print("Résumé du modèle APRÈS avoir rendu le backbone entraînable (pour vérifier les couches):")
    # model.summary() # Décommentez pour vérifier l'index de la couche base_model
    if len(model.layers) > 1 and model.layers[1].name == EFN_VARIANT.lower() + "_base": # Heuristique pour trouver le nom, ex: 's_base'
        print(f"Activation de l'entraînement pour la couche: {model.layers[1].name}")
        model.layers[1].trainable = True
    elif len(model.layers) > 1 and "efficientnetv2" in model.layers[1].name:
        print(f"Activation de l'entraînement pour la couche: {model.layers[1].name} (correspondance partielle)")
        model.layers[1].trainable = True
    else:
        print(f"AVERTISSEMENT: Impossible de confirmer automatiquement l'index du base_model (attendu à model.layers[1] et nom contenant efficientnetv2). Veuillez vérifier model.summary().")
        # Par défaut, on suppose que c'est la couche 1 si on ne trouve pas par nom
        if len(model.layers) > 1: model.layers[1].trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BACKBONE_LR),
        loss='binary_crossentropy',
        metrics=['loss','accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=BACKBONE_EPOCHS,
        callbacks=[EarlyStopping(patience=BACKBONE_PATIENCE, restore_best_weights=True)]
    )

    # 9. ÉVALUATION SUR L'ENSEMBLE DE VALIDATION (pour le choix du seuil)
    print("\n--- 9. Évaluation sur l'ensemble de VALIDATION pour le choix du seuil ---")
    p_val = model.predict(val_ds, verbose=1).ravel()
    y_val = np.concatenate([y for _, y in val_ds]).ravel()

    precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, p_val)
    
    # Les F1-scores doivent correspondre aux seuils. 
    # precisions_val et recalls_val ont une longueur de len(thresholds_val) + 1.
    # Nous utilisons donc [:-1] pour les aligner avec thresholds_val.
    pr_val_slice = precisions_val[:-1]
    re_val_slice = recalls_val[:-1]

    # Calcul F1 scores, en évitant la division par zéro et les NaNs
    denominator = pr_val_slice + re_val_slice
    f1_scores_val = np.zeros_like(denominator, dtype=float)
    # Calculer F1 seulement où le dénominateur est significatif
    valid_indices = denominator > 1e-9
    f1_scores_val[valid_indices] = (2 * pr_val_slice[valid_indices] * re_val_slice[valid_indices]) / denominator[valid_indices]

    optimal_threshold = 0.5 # Fallback
    best_f1_score_val = 0.0 # Fallback
    best_f1_idx_val = -1    # Fallback

    if len(f1_scores_val) > 0:
        best_f1_idx_val = np.nanargmax(f1_scores_val)
        # S'assurer que thresholds_val n'est pas vide et que l'index est valide
        if len(thresholds_val) > 0 and best_f1_idx_val < len(thresholds_val):
            optimal_threshold = thresholds_val[best_f1_idx_val]
            best_f1_score_val = f1_scores_val[best_f1_idx_val]
        elif len(thresholds_val) > 0: # Si l'index est hors limites mais thresholds existent
            print(f"Attention: L'index F1 optimal ({best_f1_idx_val}) est hors limites pour thresholds_val (longueur {len(thresholds_val)}). Vérifiez les prédictions p_val.")
            optimal_threshold = thresholds_val[-1] # Ou une autre stratégie de fallback
            best_f1_score_val = f1_scores_val[min(best_f1_idx_val, len(f1_scores_val)-1)]
        else: # thresholds_val est vide
             print("Attention: thresholds_val est vide. Impossible de déterminer un seuil optimal à partir de F1.")
    else:
        print("Attention: Aucun score F1 n'a pu être calculé (f1_scores_val est vide). Vérifiez les prédictions p_val.")

    print(f"Seuil optimal (maximisant F1-score pour classe 1 sur val_ds) = {optimal_threshold:.4f} (F1 = {best_f1_score_val:.4f})")

    # Sauvegarde de la courbe Précision-Rappel vs Seuil (VALIDATION)
    plt.figure(figsize=(8, 6))
    # thresholds_val peut avoir une longueur différente de pr_val_slice si p_val est constant ou problématique
    # Afficher uniquement s'ils ont des longueurs compatibles
    if len(thresholds_val) == len(pr_val_slice):
        plt.plot(thresholds_val, pr_val_slice, label='Précision (Validation)', color='blue')
        plt.plot(thresholds_val, re_val_slice, label='Rappel (Validation)', color='orange')
        if best_f1_idx_val != -1 and best_f1_idx_val < len(pr_val_slice) : # S'assurer que l'index est valide pour pr/re slices
            plt.scatter(optimal_threshold, pr_val_slice[best_f1_idx_val], marker='o', color='blue', s=100, label=f'Précision au seuil optimal ({pr_val_slice[best_f1_idx_val]:.2f})')
            plt.scatter(optimal_threshold, re_val_slice[best_f1_idx_val], marker='o', color='orange', s=100, label=f'Rappel au seuil optimal ({re_val_slice[best_f1_idx_val]:.2f})')
    else:
        print("Impossible de tracer P-R vs Seuils car les longueurs de tableaux ne correspondent pas.")
    plt.title("Courbe Précision-Rappel vs Seuil (Ensemble de Validation)")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"precision_recall_vs_threshold_validation_set.png")
    plt.close()
    print("Graphique 'precision_recall_vs_threshold_validation_set.png' sauvegardé.")

    # 10. ÉVALUATION FINALE SUR L'ENSEMBLE DE TEST
    print("\n--- 10. Évaluation finale sur l'ensemble de TEST ---")
    
    # Métriques de base avec model.evaluate() (utilise un seuil par défaut de 0.5 pour acc, prec, rec)
    print("\nMétriques de base sur test_ds (évaluées par Keras avec seuil ~0.5 pour acc/prec/rec):")
    results_test = model.evaluate(test_ds, verbose=1)
    # test_metrics = dict(zip(model.metrics_names, results_test))
    print("  Résultats bruts de model.evaluate():")
    for name, value in zip(model.metrics_names, results_test):
        if isinstance(value, float):
            print(f"    {name}: {value:.4f}")
        else:
            print(f"    {name}: {value}")

    # Prédictions sur l'ensemble de test
    p_test = model.predict(test_ds, verbose=1).ravel()
    y_test = np.concatenate([y for _, y in test_ds]).ravel()
    
    # Classification avec le seuil optimal trouvé sur val_ds
    y_pred_test_optimal_threshold = (p_test >= optimal_threshold).astype(int)

    print(f"\nÉvaluation sur test_ds avec le seuil optimal = {optimal_threshold:.4f} (déterminé sur val_ds):")

    # Matrice de confusion pour test_ds avec le seuil optimal
    cm_test_optimal = confusion_matrix(y_test, y_pred_test_optimal_threshold, labels=[0, 1])
    tn_opt, fp_opt, fn_opt, tp_opt = cm_test_optimal.ravel()

    disp_optimal = ConfusionMatrixDisplay(confusion_matrix=cm_test_optimal, display_labels=['Sain (0)', 'Malade (1)'])
    disp_optimal.plot(cmap='Blues', values_format='d')
    plt.title(f"Matrice de Confusion (Test Set) - Seuil Optimal = {optimal_threshold:.4f}")
    plt.savefig(f"confusion_matrix_test_set_optimal_threshold.png")
    plt.close()
    print("Graphique 'confusion_matrix_test_set_optimal_threshold.png' sauvegardé.")
    print(f"  Vrais Négatifs (Sain pred. Sain): {tn_opt}")
    print(f"  Faux Positifs (Sain pred. Malade): {fp_opt}")
    print(f"  Faux Négatifs (Malade pred. Sain): {fn_opt}")
    print(f"  Vrais Positifs (Malade pred. Malade): {tp_opt}")
    
    # Métriques détaillées par classe pour test_ds avec le seuil optimal
    accuracy_test_optimal = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt + 1e-9)
    
    precision_class1_optimal = tp_opt / (tp_opt + fp_opt + 1e-9)
    recall_class1_optimal = tp_opt / (tp_opt + fn_opt + 1e-9)
    f1_class1_optimal = 2 * (precision_class1_optimal * recall_class1_optimal) / (precision_class1_optimal + recall_class1_optimal + 1e-9)
    
    precision_class0_optimal = tn_opt / (tn_opt + fn_opt + 1e-9) # Précision pour "Sain"
    recall_class0_optimal = tn_opt / (tn_opt + fp_opt + 1e-9)    # Rappel pour "Sain" (Spécificité)
    f1_class0_optimal = 2 * (precision_class0_optimal * recall_class0_optimal) / (precision_class0_optimal + recall_class0_optimal + 1e-9)

    print(f"  Accuracy globale (seuil optimal): {accuracy_test_optimal:.4f}")
    print(f"  Classe 'Malade (1)' (seuil optimal):")
    print(f"    Précision: {precision_class1_optimal:.4f}")
    print(f"    Rappel:    {recall_class1_optimal:.4f}")
    print(f"    F1-score:  {f1_class1_optimal:.4f}")
    print(f"  Classe 'Sain (0)' (seuil optimal):")
    print(f"    Précision: {precision_class0_optimal:.4f}")
    print(f"    Rappel (Spécificité): {recall_class0_optimal:.4f}")
    print(f"    F1-score:  {f1_class0_optimal:.4f}")

    # Optionnel: Évaluation avec un seuil de 0.5 pour comparaison
    # y_pred_test_05_threshold = (p_test >= 0.5).astype(int)
    # print(f"\nÉvaluation sur test_ds avec le seuil fixe = 0.5 :")
    # cm_test_05 = confusion_matrix(y_test, y_pred_test_05_threshold, labels=[0, 1])
    # tn_05, fp_05, fn_05, tp_05 = cm_test_05.ravel()
    # ... (calculer et afficher métriques pour seuil 0.5 de la même manière si besoin) ...


    print("\n--- Résumé du Modèle ---")
    model.summary()

    return model
