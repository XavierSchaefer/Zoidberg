# src/analyse/CNNAnalyser.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score


def get_recall_precision_index(y_true, y_pred, class_names):
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    index = 0
    for i, r in enumerate(recall):
        index += r * precision[i]
    index /= len(recall)
    return index

def get_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def load_test_dataset(path, image_size=(224,224), batch_size=32, seed=42):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=image_size, batch_size=batch_size,
        color_mode='grayscale', shuffle=False, seed=seed
    )
    class_names = ds.class_names

    return ds.map(lambda x, y: (x/255.0, y)), class_names

def analyseModel(model_path: str, test_dataset_path: str):
    model_path = os.path.normpath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    # 1) Charger le modèle
    model = tf.keras.models.load_model(model_path)
    print("\n=== Model Summary ===")
    model.summary()

    # 2) Charger l'historique
    hist_path = os.path.splitext(model_path)[0] + "_history.json"
    history = {}
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            history = json.load(f)
        print("\n=== Training History Keys ===")
        print(list(history.keys()))
    else:
        print(f"\n⚠️ History file not found at {hist_path}")

    # 3) Charger et préparer test_ds
    test_ds, class_names = load_test_dataset(test_dataset_path,
                                image_size=model.input_shape[1:3],
                                batch_size=16)

    # 4) Évaluation
    print("\n=== Evaluate on test set ===")
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"{k:>12} : {v:.4f}")

    # 5) Prédictions et métriques
    y_true = np.concatenate([y.numpy().flatten() for _, y in test_ds])
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # print("\n=== Classification Report ===")
    # print(classification_report(y_true, y_pred,
    #                             target_names=class_names,
    #                             zero_division=0))
  
    # Get recall for each class
    recall = recall_score(y_true, y_pred, average=None)
    # print("\n=== Recall for each class ===")
    # for i, r in enumerate(recall):
    #     print(f"{class_names[i]}: {r:.4f}")

    # Get precision for each class
    precision = precision_score(y_true, y_pred, average=None)
    # print("\n=== Precision for each class ===")
    # for i, p in enumerate(precision):
    #     print(f"{class_names[i]}: {p:.4f}")

    # Multiply recall and precision for each class and do the mean
    index = get_recall_precision_index(y_true, y_pred, class_names)
    print(f"Index: {index:.4f}")

    # Get % of correct predictions
    accuracy = get_accuracy(y_true, y_pred)
    accuracy_percentage = accuracy * 100
    print(f"Accuracy % : {accuracy_percentage:.4f}") 

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))
    # 6) Retour d’une structure de résultats
    return {
        "model": model,
        "history": history,
        "evaluation": results,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
