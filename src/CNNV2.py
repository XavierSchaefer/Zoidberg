# src/CNNModel.py
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, ReLU, Add, RandomRotation, RandomTranslation, RandomZoom, Rescaling, GlobalAveragePooling2D, RandomFlip
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, precision_recall_curve,ConfusionMatrixDisplay
mixed_precision.set_global_policy('mixed_float16')


# 📦 Paramètres globaux
# NETWORK_STRUCTURE= (2,2,2,2) # ResNet-18
# NETWORK_STRUCTURE = (3,4,6,3) # ResNet-50 
NETWORK_STRUCTURE = (3,4,23,3) # ResNet-101
# NETWORK_STRUCTURE = (3,8,36,3) # ResNet-152
INITIAL_FILTERS = 16 # Nombre de filtres pour chaque bloc résiduel
HEIGHT,WIDTH = 64,64 # Taille de l'image
OUTPUTSIZE = 1
EPOCH = 10
BATCH = 16
SEED = 42
PATIENCE = 3
LEARNINGRATE = 0.001
AUTOTUNE = tf.data.AUTOTUNE
def count_classes(ds):
    counts = [0, 0]
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1
    return counts

# # Configurez TensorBoard
# tensorboard_callback = TensorBoard(
#     log_dir="./logsV2",         # répertoire pour stocker les logs
#     histogram_freq=1,         # calculez les histogrammes des poids (et activations) toutes les 1 epoch
#     write_graph=True,         # enregistre le graph du modèle
#     write_images=True,        # enregistre des images des poids si possible
#     update_freq='epoch'       # mise à jour à la fin de chaque epoch
# )
def gradcam(model, img, last_conv_layer_name):
    """
    Calcule la heatmap Grad-CAM pour une image et un modèle donnés.
    - model : votre modèle Keras.
    - img   : tenseur (1, H, W, C) pré-normalisé (ici 0..1).
    - last_conv_layer_name : nom exact de la dernière couche Conv2D.
    """
    # 1. Créer un sous-modèle qui renvoie les activations de la couche conv et la sortie
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Calculer les gradients de la classe prédite par rapport aux activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        # si binaire, predictions shape=(1,1) : on prend [:,0]
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)           # shape=(1,h,w,filters)
    # 3. Moyenne spatiale des gradients ➔ poids par canal
    pooled_grads = tf.reduce_mean(grads, axis=(1,2))    # shape=(1,filters)

    # 4. Pondération des activations par ces poids, puis sommation
    conv_outputs = conv_outputs[0]                      # shape=(h,w,filters)
    pooled_grads = pooled_grads[0]                      # shape=(filters,)
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # 5. Post-traitement : ReLU + normalisation
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4, cmap='jet'):
    """
    Superpose la heatmap sur l'image originale.
    - img     : (H,W) numpy array.
    - heatmap : même (H,W), valeurs [0..1].
    """
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.show()
    
def residual_block(x, filters,stride=1):
    """
    Petit bloc résiduel :
      - Conv -> BN -> ReLU
      - Conv -> BN
      - skip connection + ReLU final
    """
    x_skip = x  # sauvegarde l'entrée pour la skip-connection
    
    # 1ère conv
    x = Conv2D(filters, (3,3), padding='same', strides=stride)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # 2ème conv
    x = Conv2D(filters, (3,3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)

    # ajustement du skip si on change le nombre de filtres ou la résolution
    if stride != 1 or x_skip.shape[-1] != filters:
        x_skip = Conv2D(filters, (1,1), strides=stride, padding='same')(x_skip)
        x_skip = BatchNormalization()(x_skip)


    # skip connection
    x = Add()([x, x_skip]) 
    x = ReLU()(x)
    return x
def bottleneck_block(x, filters, stride=1, expansion=4):
    """
    Bottleneck résiduel :
      - Conv1 (1×1) → BN → ReLU
      - Conv2 (3×3) → BN → ReLU
      - Conv3 (1×1) → BN
      - Skip + ReLU
    """
    shortcut = x
    reduced_filters = filters // expansion

    # 1) 1×1 réduction
    x = Conv2D(reduced_filters, (1,1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 2) 3×3 convolution
    x = Conv2D(reduced_filters, (3,3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 3) 1×1 restauration
    x = Conv2D(filters, (1,1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # 4) ajustement du skip si dimension ou résolution changent
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # 5) skip connection + activation finale
    x = Add()([x, shortcut])
    return ReLU()(x)
def build_resnet_model(height=128, width=128, outputSize=1, blocks_per_stage=(2,2,2),initial_filters=64):
    inp = Input(shape=(height, width, 1))
    x = Rescaling(1./255)(inp) # 'x' prend la sortie de Rescaling(inp)

    # Bloc 1 : Conv + BN + Pool
    # La première couche Conv2D utilise 'x' (qui est la sortie de Rescaling)
    x = Conv2D(initial_filters, (7,7), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    
    # Stages
    filters = initial_filters
    for i, n_blocks in enumerate(blocks_per_stage):
        # 1er bloc du stage : downsample sauf pour le 1er stage
        stride = 1 if i == 0 else 2
        x = residual_block(x, filters, stride=stride)
        # blocs suivants en stride=1
        for _ in range(n_blocks - 1):
            x = residual_block(x, filters, stride=1)
        filters *= 2  # double les filtres au stage suivant

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    out = Dense(outputSize, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=out)
def build_resnet_bottleneck(
    height=224, width=224, output_size=1,
    blocks_per_stage=(3, 4, 6, 3),  # configuration ResNet-50
    initial_filters=64,
    expansion=4
):
    inp = Input((height, width, 1))
    x   = Rescaling(1./255)(inp)

    # Stem
    x = Conv2D(initial_filters, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    # Stages
    filters = initial_filters
    for i, n_blocks in enumerate(blocks_per_stage):
        # premier bloc du stage (downsample si i>0)
        stride = 1 if i == 0 else 2
        x = bottleneck_block(x, filters, stride=stride, expansion=expansion)
        # blocs restants sans downsample
        for _ in range(n_blocks - 1):
            x = bottleneck_block(x, filters, stride=1, expansion=expansion)
        filters *= 2

    # Head de classification
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    out = Dense(output_size, activation='sigmoid')(x)
    return Model(inp, out)




def ConvolutionalNeuralNetworkModelV2():

    # 1. Chargement et transformation des données
    train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/initial/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(HEIGHT,WIDTH),
    color_mode="grayscale",   # 1 canal
    batch_size=BATCH,
    shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/initial/train",
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(HEIGHT,WIDTH),
        color_mode="grayscale",
        batch_size=BATCH,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/initial/test",
        image_size=(HEIGHT,WIDTH),
        color_mode="grayscale",
        batch_size=BATCH,
        shuffle=False,
    )

    # 2. Oversampling pour équilibrer
    augment = tf.keras.Sequential([
        RandomRotation(factor=0.1), # Rotation de +/- 10% * 2*pi
        RandomTranslation(height_factor=0.1, width_factor=0.1),
        RandomZoom(height_factor=0.1, width_factor=0.1),
        RandomFlip(mode='horizontal'),
    ])

    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)
    # 2) Mixed precision & XLA
    # mixed_precision.set_global_policy('mixed_float16')
    # tf.config.optimizer.set_jit(True) # Temporarily commented out XLA JIT
    tb_cb = TensorBoard(log_dir="logs/", histogram_freq=10)


    # print("Train  :", count_classes(train_ds), "   [Normal, Pneumonia]")
    # print("Val    :", count_classes(val_ds))
    # print("Test   :", count_classes(test_ds))
    # n0, n1 = count_classes(train_ds)
    # total   = n0 + n1
    # class_weight = {0: total/(2*n0), 1: total/(2*n1)}
    # class_weight = {1:1.91, 0:0.68}
    class_weight = {0:20, 1:1}
    print("[INFO] class_weight =", class_weight)
    # Augmentation + prefetch A VALIDER SI PERTINENT
    train_ds = (
    train_ds
      .map(lambda x, y: (augment(x, training=True), y),
           num_parallel_calls=AUTOTUNE)
      .prefetch(AUTOTUNE)
    )

    # 3. Création du modèle
    # model = build_resnet_model(height=HEIGHT, width=WIDTH, outputSize=OUTPUTSIZE, blocks_per_stage=(2,2,2),initial_filters=64)
    model = build_resnet_bottleneck(height=HEIGHT, width=WIDTH, output_size=OUTPUTSIZE, blocks_per_stage=NETWORK_STRUCTURE, initial_filters=INITIAL_FILTERS, expansion=4)

    # 4. Compilation du modèle
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNINGRATE,weight_decay=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'),tf.keras.metrics.AUC(name='auc_roc'), tf.keras.metrics.AUC(curve='PR', name='auc_pr')])

    # 5. Entrainement du modèle
    history = model.fit(train_ds,class_weight=class_weight, validation_data=val_ds, epochs=EPOCH,callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True),tb_cb])


    result = model.evaluate(test_ds)
    print(result)   
    # 6. Analyse des résultats | Matrice de confusion au meilleur seuil
    p_val  = model.predict(val_ds,  verbose=0).ravel()
    y_val  = np.concatenate([y for _, y in val_ds]).ravel()

    prec, rec, thr = precision_recall_curve(y_val, p_val)

    # on n'utilise que les points associés à thr
    prec2 = prec[1:]   
    rec2  = rec[1:]
    thr2  = thr      # len(thr2) = len(prec2) = len(rec2)

    f1_scores = 2*prec2*rec2/(prec2 + rec2 + 1e-9)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = thr2[best_idx]
    # best_thr = 0.5
    print("Seuil F1 corrigé =", best_thr)




    p_test = model.predict(test_ds, verbose=0).ravel()
    y_test = np.concatenate([y for _, y in test_ds]).ravel()

    y_pred = (p_test >= best_thr).astype(int)
    ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred, labels=[0,1]),
            display_labels=['Normal','Pneumonia']
    ).plot(cmap='Blues', values_format='d')
    plt.title(f"Test set – seuil {best_thr:.2f}")
    # plt.show()
    plt.savefig("Matrice.png")
    plt.close();

    plt.hist(p_val[y_val==0], bins=50, alpha=0.5, label='Normal')
    plt.hist(p_val[y_val==1], bins=50, alpha=0.5, label='Pneumonia')
    plt.legend()
    plt.xlabel("Score (p_val)")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des scores sur validation")
    # plt.show()
    plt.savefig("Distribution.png")
    plt.close();


    # Show result
    model.summary()
