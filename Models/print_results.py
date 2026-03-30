import pandas as pd

binary_metrics = {
    "Modèle": [
        "Decision Tree",
        "Gradient Boosting",
        "KNN",
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Isolation Forest (Non-Supervisé)"
    ],
    "Accuracy": [0.9932, 0.9968, 0.9938, 0.9846, 0.9968, 0.9952, None],
    "Precision": [0.8198, 0.9947, 0.9545, 0.7178, 1.0000, 0.9694, 0.4135],
    "Recall":    [0.8636, 0.8511, 0.7386, 0.4307, 0.8466, 0.7932, 0.8837],
    "F1-Score":  [0.8412, 0.9173, 0.8328, 0.5384, 0.9169, 0.8725, 0.6633],
    "ROC-AUC":   [0.9298, 0.9476, 0.9027, 0.9348, 0.9462, 0.9143, 0.9632],
    "Type":      ["Supervisé", "Supervisé", "Supervisé", "Supervisé", "Supervisé", "Supervisé", "Non-Supervisé"]
}
df_binary = pd.DataFrame(binary_metrics)
df_binary = df_binary.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

multiclass_metrics = {
    "Modèle": [
        "Decision Tree",
        "Gradient Boosting",
        "KNN",
        "Logistic Regression",
        "Random Forest",
        "SVM Optimized"
    ],
    "Accuracy (Multi-classes)": [0.9963, 0.9834, 0.9933, 0.9890, 0.9963, 0.0834],
    "Type": ["Supervisé"] * 6
}
df_multi = pd.DataFrame(multiclass_metrics)
df_multi = df_multi.sort_values(by="Accuracy (Multi-classes)", ascending=False).reset_index(drop=True)

print("==========================================================================")
print("               RÉSULTATS DE LA CLASSIFICATION BINAIRE                     ")
print("==========================================================================\n")
print(df_binary.to_string(index=True))

print("\n\n==========================================================================")
print("               RÉSULTATS DE LA CLASSIFICATION MULTI-CLASSES               ")
print("==========================================================================\n")
print(df_multi.to_string(index=True))
