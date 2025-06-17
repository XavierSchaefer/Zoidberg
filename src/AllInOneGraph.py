import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def generate_combined_diagnostic_plot(history, y_test, y_pred, model_config, report_text, save_path="results/combined_diagnostics.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sain", "Malade"])

    fig = plt.figure(figsize=(18, 7))
    plt.suptitle("🧠 Diagnostic complet du modèle", fontsize=16, fontweight="bold")

    # Courbe de loss
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.plot(history.history["loss"], label="Loss (train)", marker='o')
    ax1.plot(history.history["val_loss"], label="Loss (val)", marker='o')
    ax1.set_title("📉 Courbe de perte")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte")
    ax1.grid(True)
    ax1.legend()

    # Courbe d'accuracy
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax2.plot(history.history["accuracy"], label="Accuracy (train)", marker='o')
    ax2.plot(history.history["val_accuracy"], label="Accuracy (val)", marker='o')
    ax2.set_title("🎯 Courbe de précision")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Précision")
    ax2.grid(True)
    ax2.legend()

    # Matrice de confusion
    ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    disp.plot(ax=ax3, cmap="Blues", values_format="d", colorbar=False)
    ax3.set_title("🧩 Matrice de confusion")

    # Rapport classification
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax4.axis('off')
    ax4.text(0, 0.8, "📊 Rapport de classification", fontsize=12, fontweight="bold")
    ax4.text(0, 0.6, report_text, fontsize=9, family="monospace")

    # Description du modèle
    description = (
        f"Image size: {model_config['height']}x{model_config['width']}, "
        f"Hidden units: {model_config['hiddenUnitsSize']} x{model_config['hiddenUnitStacks']}, "
        f"Epochs: {model_config['generationSize']}, "
        f"Batch size: {model_config['batchSize']}, "
        f"Test Accuracy: {model_config['test_acc']:.2f}, "
        f"Test Loss: {model_config['test_loss']:.2f}"
    )
    ax4.text(0, 0.05, description, fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return save_path
