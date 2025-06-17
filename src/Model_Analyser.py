# src/Model_Analyser.py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, # <-- Added for ROC Curve
    precision_recall_curve # <-- Added for Precision-Recall Curve
)
import json # For saving config and report

class ModelAnalyser:
    """
    Analyzes a Keras model's training history and test set predictions.

    Generates plots including:
    - Training/Validation Loss, Accuracy, Precision, Recall vs. Epochs
    - Confusion Matrix
    - ROC Curve with AUC
    - Precision-Recall Curve
    - Provides a text-based decision aid.
    Saves the plot and a JSON file with metrics and configuration.
    """
    def __init__(self, history, y_true, y_pred, y_pred_probs, model_config, class_names=['NORMAL', 'PNEUMONIA'], save_dir="results"):
        """
        Initializes the analyser.

        Args:
            history (tf.keras.callbacks.History): Keras training history object.
            y_true (np.ndarray): True labels for the test set (1D array).
            y_pred (np.ndarray): Predicted labels for the test set (1D array, 0s and 1s).
            y_pred_probs (np.ndarray): Predicted probabilities for the positive class (1D array).
            model_config (dict): Dictionary containing model hyperparameters and final test metrics.
            class_names (list): List of class names for display (e.g., ['Negative', 'Positive']).
            save_dir (str): Directory to save the output plot and JSON file.
        """
        self.history = history.history # Access the dictionary directly
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_probs = y_pred_probs # <-- Store probabilities
        self.model_config = model_config
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True) # Create save directory

    def _plot_history(self, ax, metric_name, val_metric_name, title, ylabel):
        """Helper function to plot training history metrics."""
        train_metric = self.history.get(metric_name)
        val_metric = self.history.get(val_metric_name)

        if train_metric:
            ax.plot(train_metric, label=f'Train {ylabel}', color='blue', alpha=0.8)
        if val_metric:
            ax.plot(val_metric, label=f'Validation {ylabel}', color='orange', linestyle='--')

        if train_metric or val_metric:
            ax.set_xlabel('Époques')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, f'{ylabel} non trouvée\ndans l\'historique', ha='center', va='center')
            ax.set_title(title, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    def plot_and_analyse(self):
        """Generates and saves the analysis plots and data."""
        print("\n--- Analyse du Modèle ---")

        # --- Calculate Metrics ---
        report_dict = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0 # Avoid warnings if a class has no predictions
        )
        report_text = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        print("Classification Report:\n", report_text)

        # Calculate ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_pred_probs)
        roc_auc = auc(fpr, tpr)
        print(f"Aire sous la courbe ROC (AUC): {roc_auc:.4f}")
        self.model_config['test_auc'] = roc_auc # Store AUC in config

        # Calculate Precision-Recall curve
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(self.y_true, self.y_pred_probs)

        # --- Store per-class metrics in model_config and create filename suffix ---
        filename_metric_suffix = f"_AUC_{roc_auc:.2f}" # Start with AUC
        for class_name in self.class_names:
            if class_name in report_dict:
                precision = report_dict[class_name].get('precision', 0)
                recall = report_dict[class_name].get('recall', 0)
                # Store in model_config for title display
                self.model_config[f'{class_name}_precision'] = precision
                self.model_config[f'{class_name}_recall'] = recall
                # Add to filename suffix, sanitize class_name for filename
                sanitized_class_name = class_name.replace(" ", "_").upper()
                filename_metric_suffix += f"_{sanitized_class_name}P{precision:.2f}_R{recall:.2f}"
            else:
                # Handle cases where a class might not be in the report (e.g., due to zero_division or other issues)
                self.model_config[f'{class_name}_precision'] = 0
                self.model_config[f'{class_name}_recall'] = 0
                sanitized_class_name = class_name.replace(" ", "_").upper()
                filename_metric_suffix += f"_{sanitized_class_name}P0.00_R0.00"


        # --- Create Plots ---
        fig, axes = plt.subplots(3, 2, figsize=(16, 18)) # 3 rows, 2 columns
        fig.patch.set_facecolor('white') # Ensure white background

        # Plot History Metrics
        self._plot_history(axes[0, 0], 'loss', 'val_loss', 'Évolution de la Perte (Loss)', 'Perte')
        self._plot_history(axes[0, 1], 'accuracy', 'val_accuracy', 'Évolution de la Précision (Accuracy)', 'Précision')
        # Check for precision/recall keys (might have different names like 'precision_1', 'recall_1')
        prec_key = next((k for k in self.history if 'precision' in k and 'val' not in k), None)
        val_prec_key = next((k for k in self.history if 'precision' in k and 'val' in k), None)
        rec_key = next((k for k in self.history if 'recall' in k and 'val' not in k), None)
        val_rec_key = next((k for k in self.history if 'recall' in k and 'val' in k), None)
        self._plot_history(axes[1, 0], prec_key, val_prec_key, 'Évolution de la Précision (Precision)', 'Precision')
        self._plot_history(axes[1, 1], rec_key, val_rec_key, 'Évolution du Rappel (Recall)', 'Rappel')


        # Plot Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
        disp.plot(cmap='Blues', ax=axes[2, 0], colorbar=False)
        axes[2, 0].set_title('Matrice de Confusion (Test Set)', fontsize=12)
        axes[2, 0].grid(False)

        # Plot ROC Curve
        axes[2, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
        axes[2, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (AUC = 0.50)')
        axes[2, 1].set_xlim([0.0, 1.0])
        axes[2, 1].set_ylim([0.0, 1.05])
        axes[2, 1].set_xlabel('Taux de Faux Positifs (1 - Spécificité)')
        axes[2, 1].set_ylabel('Taux de Vrais Positifs (Rappel / Sensibilité)')
        axes[2, 1].set_title('Courbe ROC (Receiver Operating Characteristic)', fontsize=12)
        axes[2, 1].legend(loc="lower right")
        axes[2, 1].grid(True, linestyle='--', alpha=0.6)

        # (Optional: Add Precision-Recall Curve - uncomment if needed, might replace ROC or need a bigger grid)
        # axes[1, 1].plot(recall_vals, precision_vals, color='blue', lw=2, label='Courbe PR')
        # axes[1, 1].set_xlabel('Rappel (Recall)')
        # axes[1, 1].set_ylabel('Précision (Precision)')
        # axes[1, 1].set_title('Courbe Précision-Rappel', fontsize=12)
        # axes[1, 1].set_ylim([0.0, 1.05])
        # axes[1, 1].grid(True, linestyle='--', alpha=0.6)

        # --- Add Title and Decision Aid ---
        fig.suptitle(self._format_model_config(), fontsize=16, fontweight='bold', y=0.99)

        # --- Create custom metrics text (P*R per class and Global Accuracy) ---
        custom_metrics_lines = ["--- Performance Clés (Test Set) ---"]
        for class_name in self.class_names:
            if class_name in report_dict:
                precision = report_dict[class_name].get('precision', 0)
                recall = report_dict[class_name].get('recall', 0)
                custom_metrics_lines.append(f"{class_name}: Précision * Rappel = {precision * recall:.2f}")
            else:
                # Should not happen if class_names is derived from y_true/y_pred context
                custom_metrics_lines.append(f"{class_name}: Précision * Rappel = N/A")

        accuracy = report_dict.get('accuracy', 0) # Overall accuracy
        custom_metrics_lines.append(f"\nTaux de succès global (Accuracy): {accuracy * 100:.2f}%")
        custom_metrics_display_text = "\n".join(custom_metrics_lines)

        # --- Add Custom Metrics text to figure ---
        plt.figtext(
            0.5, # x-coordinate (center of the figure)
            0.11, # y-coordinate: top of this text box will be at 11% from bottom
            custom_metrics_display_text, # The custom metrics string
            ha="center", # Horizontal alignment
            va="top", # Vertical alignment (anchor top of text box at y)
            fontsize=8, 
            family='sans-serif', # Default sans-serif font
            bbox=dict(boxstyle="round,pad=0.4", fc='lightblue', ec='gray', alpha=0.85)
        )

        # --- Add Decision Aid text to figure ---
        decision_text = self._decision_aid(report_dict, roc_auc)
        plt.figtext(
            0.5, # x-coordinate (center)
            0.01, # y-coordinate: bottom of this text box will be at 1% from bottom
            decision_text, 
            fontsize=9, 
            ha='center', 
            va='bottom', # Anchor bottom of text box at y, allowing it to grow upwards
            wrap=True,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='gray', alpha=0.8)
        )

        # --- Save and Show ---
        # Adjust bottom margin in rect to make space for the two text boxes below plots
        plt.tight_layout(rect=[0, 0.14, 1, 0.95]) # Increased bottom margin from 0.08 to 0.14
        save_path_plot = os.path.join(self.save_dir, f"model_diagnostic{filename_metric_suffix}.png")
        plt.savefig(save_path_plot)
        print(f"\n📈 Plot sauvegardé : {save_path_plot}")

        # Save config and report to JSON
        output_data = {
            'model_config': self.model_config,
            'classification_report': report_dict,
            'history_summary': {k: v[-1] for k, v in self.history.items() if v} # Last value of each history metric
        }
        save_path_json = os.path.join(self.save_dir, f"model_metrics{filename_metric_suffix}.json")
        try:
            with open(save_path_json, 'w') as f:
                # Use custom encoder for numpy types if needed, but basic types should be fine
                json.dump(output_data, f, indent=4)
            print(f"📝 Métriques sauvegardées : {save_path_json}")
        except TypeError as e:
             print(f"⚠️ Erreur lors de la sauvegarde JSON (peut contenir des types NumPy non sérialisables): {e}")
             # Attempt basic string conversion as fallback
             try:
                 output_data_str = {k: str(v) for k, v in output_data.items()}
                 with open(save_path_json, 'w') as f:
                    json.dump(output_data_str, f, indent=4)
                 print(f"📝 Métriques sauvegardées (converties en str) : {save_path_json}")
             except Exception as e_inner:
                  print(f"❌ Échec de la sauvegarde JSON même après conversion: {e_inner}")


        plt.show()
        print("--- Fin de l'analyse ---")


    def _decision_aid(self, report_dict, roc_auc):
        """Provides automated textual feedback based on metrics."""
        # Extract overall metrics (macro avg is often good for imbalance)
        acc = report_dict['accuracy']
        macro_avg = report_dict.get('macro avg', {})
        recall = macro_avg.get('recall', 0)
        precision = macro_avg.get('precision', 0)
        f1 = macro_avg.get('f1-score', 0)

        # Extract metrics for the positive class (assuming it's the second one)
        positive_class = self.class_names[1]
        pos_recall = report_dict.get(positive_class, {}).get('recall', 0)
        pos_precision = report_dict.get(positive_class, {}).get('precision', 0)

        loss = self.model_config.get('test_loss', float('inf'))

        decisions = ["--- Aide à la Décision (Basée sur le jeu de test) ---"]

        # Analyse AUC
        if roc_auc > 0.9: decisions.append(f"🟢 AUC ({roc_auc:.2f}) excellent : Très bonne capacité de discrimination.")
        elif roc_auc > 0.8: decisions.append(f"🟡 AUC ({roc_auc:.2f}) bon : Bonne capacité de discrimination.")
        elif roc_auc > 0.7: decisions.append(f"🟠 AUC ({roc_auc:.2f}) correct : Capacité de discrimination modérée.")
        elif roc_auc > 0.5: decisions.append(f"🔴 AUC ({roc_auc:.2f}) faible : Discrimination à peine meilleure que l'aléatoire.")
        else: decisions.append(f"⚫ AUC ({roc_auc:.2f}) mauvais : Problème probable (vérifier labels/prédictions).")

        # Analyse Recall (Macro et/ou Positif - crucial pour médical)
        if pos_recall > 0.9: decisions.append(f"🟢 Rappel {positive_class} ({pos_recall:.2f}) excellent : Peu de faux négatifs manqués.")
        elif pos_recall > 0.75: decisions.append(f"🟡 Rappel {positive_class} ({pos_recall:.2f}) acceptable : Risque modéré de faux négatifs.")
        else: decisions.append(f"🔴 Rappel {positive_class} ({pos_recall:.2f}) faible : Risque élevé de manquer des cas positifs. (Priorité!)")
        # Optionnel: commenter le rappel macro avg
        # if recall > 0.8: decisions.append(f"Rappel Macro ({recall:.2f}) bon.")
        # else: decisions.append(f"Rappel Macro ({recall:.2f}) moyen/faible.")

        # Analyse F1-score (équilibre Precision/Recall)
        if f1 > 0.8: decisions.append(f"🟢 F1 Macro ({f1:.2f}) bon : Bon équilibre Precision/Recall global.")
        elif f1 > 0.65: decisions.append(f"🟡 F1 Macro ({f1:.2f}) moyen : Déséquilibre possible.")
        else: decisions.append(f"🔴 F1 Macro ({f1:.2f}) faible : Mauvais équilibre ou faible performance.")

        # Analyse Précision (Macro et/ou Positif - important si coût des faux positifs élevé)
        # if pos_precision > 0.8: decisions.append(f"Précision {positive_class} ({pos_precision:.2f}) bonne.")
        # else: decisions.append(f"Précision {positive_class} ({pos_precision:.2f}) améliorable (risque de faux positifs).")

        # Analyse Loss
        if loss < 0.2: decisions.append(f"🟢 Perte ({loss:.2f}) basse : Bonne convergence.")
        elif loss < 0.5: decisions.append(f"🟡 Perte ({loss:.2f}) acceptable.")
        else: decisions.append(f"🟠 Perte ({loss:.2f}) élevée : Convergence moyenne ou sur-apprentissage?")

        # Analyse Overfitting (basé sur la fin de l'historique)
        train_acc_last = self.history.get('accuracy', [0])[-1]
        val_acc_last = self.history.get('val_accuracy', [0])[-1]
        if val_acc_last > 0 and abs(train_acc_last - val_acc_last) > 0.15:
             decisions.append(f"⚠️ Gap Acc Train/Val ({train_acc_last:.2f} vs {val_acc_last:.2f}) : Signe de sur-apprentissage possible (vérifier courbes). Régularisation?")

        train_loss_last = self.history.get('loss', [float('inf')])[-1]
        val_loss_last = self.history.get('val_loss', [float('inf')])[-1]
        if val_loss_last < float('inf') and val_loss_last > train_loss_last * 1.5 and val_loss_last > 0.5 :
             decisions.append(f"⚠️ Gap Loss Train/Val ({train_loss_last:.2f} vs {val_loss_last:.2f}) : Signe de sur-apprentissage possible (vérifier courbes). Régularisation?")

        # Suggestions générales (à adapter !)
        if pos_recall < 0.75 and roc_auc < 0.8 : decisions.append("💡 Suggestion: Revoir équilibrage (oversampling, weights), augmenter données positives, ou architecture modèle.")
        if abs(train_acc_last - val_acc_last) > 0.15 : decisions.append("💡 Suggestion: Augmenter régularisation (Dropout, L2), data augmentation, ou réduire complexité modèle.")

        return "\n".join(decisions)


    def _format_model_config(self):
        """Formats the model configuration for the main title."""
        cfg = self.model_config
        # Format common keys nicely, include others as key=value
        title_parts = [f"Diagnostic Modèle CNN (Test Set)"]
        core_params = { # Prioritize these
            'height': 'Image H', 'width': 'Image W', 'batchSize': 'Batch',
            'test_acc': 'Acc (Keras)', 'test_loss': 'Loss (Keras)', 'test_precision': 'Prec (Keras)',
            'test_recall': 'Recall (Keras)', 'test_auc': 'AUC (Global)'
         }
        formatted_parts = []
        other_parts = []

        for key, val in cfg.items():
            label = core_params.get(key)
            if label:
                if isinstance(val, float):
                    formatted_parts.append(f"{label}: {val:.2f}")
                else:
                    formatted_parts.append(f"{label}: {val}")
            # elif key not in ['hiddenUnitsSize','hiddenUnitStacks', 'generationSize']: # Avoid cluttering title too much
            #     other_parts.append(f"{key}: {val}")

        title_parts.extend(formatted_parts)

        # Add per-class precision and recall to title
        class_metrics_parts = []
        for class_name in self.class_names:
            precision = cfg.get(f'{class_name}_precision')
            recall = cfg.get(f'{class_name}_recall')
            if precision is not None and recall is not None:
                 class_metrics_parts.append(f"{class_name} - P: {precision:.2f}, R: {recall:.2f}")
        
        if class_metrics_parts:
            title_parts.append("--- Précision/Rappel par Classe (du Rapport) ---")
            title_parts.extend(class_metrics_parts)
        
        # title_parts.extend(other_parts) # Uncomment to add all other params

        return "\n".join(title_parts)