"""
=============================================================
PORTFOLIO: ML-Based Tumor Subtype Classification
Dataset: TCGA Breast Cancer Gene Expression (simulated)
Models: Random Forest, Logistic Regression, SVM
Author: Portfolio de Bioinformática Médica
=============================================================

Objective:
Apply machine learning to classify breast cancer molecular 
subtypes (Luminal A, Luminal B, HER2-enriched, Triple Negative)
from gene expression profiles — demonstrating how AI can 
automate tumor classification for precision oncology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# CONFIGURACIÓN VISUAL
# ─────────────────────────────────────────────
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

COLORES_SUBTIPOS = {
    "Luminal_A":       "#3498db",
    "Luminal_B":       "#2ecc71",
    "HER2_enriched":   "#e74c3c",
    "Triple_Negative": "#9b59b6"
}

# ─────────────────────────────────────────────
# MÓDULO 1: CARGA Y EXPLORACIÓN
# ─────────────────────────────────────────────
print("=" * 60)
print("MÓDULO 1: CARGA Y EXPLORACIÓN DE DATOS")
print("=" * 60)

df = pd.read_csv("tcga_breast_expression.csv")
genes = [c for c in df.columns if c != "subtype"]

print(f"\n📊 Dataset: {len(df)} tumor samples, {len(genes)} genes")
print(f"\n📈 Subtype distribution:")
for subtype, n in df["subtype"].value_counts().items():
    barra = "█" * n
    print(f"   {subtype:<20} {barra} ({n})")

print(f"\n🔬 Gene expression summary (first 5 genes):")
print(df[genes[:5]].describe().round(2).to_string())

# ─────────────────────────────────────────────
# MÓDULO 2: PREPROCESAMIENTO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MÓDULO 2: PREPROCESAMIENTO")
print("=" * 60)

# Features y labels
X = df[genes].values
y = df["subtype"].values

# Encoding de labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✅ Train set: {len(X_train)} samples")
print(f"✅ Test set:  {len(X_test)} samples")
print(f"✅ Features normalized with StandardScaler")
print(f"✅ Stratified split — balanced subtypes in both sets")

# ─────────────────────────────────────────────
# MÓDULO 3: ENTRENAMIENTO DE MODELOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MÓDULO 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("=" * 60)

modelos = {
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":               SVC(kernel="rbf", probability=True, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():
    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(modelo, X_scaled, y_encoded, cv=5)

    resultados[nombre] = {
        "modelo": modelo,
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred
    }

    print(f"\n🤖 {nombre}")
    print(f"   Test Accuracy:     {acc:.4f} ({acc*100:.1f}%)")
    print(f"   Cross-val (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Mejor modelo
mejor_nombre = max(resultados, key=lambda x: resultados[x]["accuracy"])
mejor_modelo = resultados[mejor_nombre]["modelo"]
mejor_pred = resultados[mejor_nombre]["y_pred"]

print(f"\n🏆 Best model: {mejor_nombre}")
print(f"   Accuracy: {resultados[mejor_nombre]['accuracy']*100:.1f}%")
print(f"\n📋 Detailed classification report ({mejor_nombre}):")
print(classification_report(y_test, mejor_pred,
                           target_names=le.classes_))

# ─────────────────────────────────────────────
# MÓDULO 4: VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MÓDULO 4: VISUALIZACIONES")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#f8f9fa")
fig.suptitle("ML-Based Breast Cancer Subtype Classification\nTCGA Gene Expression Data",
             fontsize=16, fontweight="bold", y=0.98)

# ── Gráfico 1: Distribución de subtipos ──
ax1 = fig.add_subplot(2, 3, 1)
subtype_counts = df["subtype"].value_counts()
colores = [COLORES_SUBTIPOS[s] for s in subtype_counts.index]
bars = ax1.bar(subtype_counts.index, subtype_counts.values,
               color=colores, edgecolor="white")
for bar, val in zip(bars, subtype_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha="center", fontsize=10, fontweight="bold")
ax1.set_title("Tumor Subtype\nDistribution", fontweight="bold")
ax1.set_ylabel("Number of samples")
ax1.set_xticklabels(subtype_counts.index, rotation=15, fontsize=8)
ax1.set_facecolor("#f0f0f0")

# ── Gráfico 2: Comparación de modelos ──
ax2 = fig.add_subplot(2, 3, 2)
nombres = list(resultados.keys())
accuracies = [resultados[n]["accuracy"] for n in nombres]
cv_means = [resultados[n]["cv_mean"] for n in nombres]
x = np.arange(len(nombres))
width = 0.35
bars1 = ax2.bar(x - width/2, accuracies, width,
                label="Test Accuracy", color="#3498db", alpha=0.8)
bars2 = ax2.bar(x + width/2, cv_means, width,
                label="CV Accuracy", color="#e74c3c", alpha=0.8)
for bar, val in zip(bars1, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
ax2.set_title("Model Comparison\nTest vs Cross-Validation", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace(" ", "\n") for n in nombres], fontsize=8)
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0.8, 1.05)
ax2.legend(fontsize=8)
ax2.set_facecolor("#f0f0f0")

# ── Gráfico 3: Confusion Matrix ──
ax3 = fig.add_subplot(2, 3, 3)
cm = confusion_matrix(y_test, mejor_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax3, linewidths=0.5)
ax3.set_title(f"Confusion Matrix\n({mejor_nombre})", fontweight="bold")
ax3.set_ylabel("True Label")
ax3.set_xlabel("Predicted Label")
ax3.set_xticklabels(le.classes_, rotation=15, fontsize=7)
ax3.set_yticklabels(le.classes_, rotation=0, fontsize=7)

# ── Gráfico 4: PCA — separación de subtipos ──
ax4 = fig.add_subplot(2, 3, 4)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
for subtype in df["subtype"].unique():
    mask = df["subtype"].values == subtype
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=COLORES_SUBTIPOS[subtype],
                label=subtype, alpha=0.7, s=40, edgecolors="white",
                linewidth=0.3)
ax4.set_title(f"PCA — Tumor Subtype Separation\n(Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
              fontweight="bold")
ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax4.legend(fontsize=7)
ax4.set_facecolor("#f0f0f0")

# ── Gráfico 5: Feature importance (Random Forest) ──
ax5 = fig.add_subplot(2, 3, 5)
rf_model = resultados["Random Forest"]["modelo"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_genes = [genes[i] for i in indices]
top_importances = importances[indices]
colores_genes = ["#e74c3c" if i < 5 else "#3498db" if i < 10
                 else "#2ecc71" if i < 15 else "#9b59b6"
                 for i in indices]
bars = ax5.barh(top_genes[::-1], top_importances[::-1],
                color=colores_genes[::-1], edgecolor="white")
ax5.set_title("Top 10 Most Important Genes\n(Random Forest)", fontweight="bold")
ax5.set_xlabel("Feature Importance")
ax5.set_facecolor("#f0f0f0")

# ── Gráfico 6: Resumen clínico ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")

best_acc = resultados[mejor_nombre]["accuracy"]
best_cv = resultados[mejor_nombre]["cv_mean"]

resumen = [
    ("ML CLASSIFICATION SUMMARY", "#2c3e50", 12, "bold"),
    ("", "#2c3e50", 9, "normal"),
    (f"Dataset: 200 tumors, 20 genes", "#2c3e50", 9, "normal"),
    (f"Task: 4-class subtype classification", "#2c3e50", 9, "normal"),
    ("", "#2c3e50", 8, "normal"),
    ("MODEL PERFORMANCE:", "#2c3e50", 10, "bold"),
    (f"🏆 Best: {mejor_nombre}", "#2c3e50", 9, "normal"),
    (f"   Test accuracy:  {best_acc*100:.1f}%", "#27ae60", 10, "bold"),
    (f"   CV accuracy:    {best_cv*100:.1f}%", "#27ae60", 10, "bold"),
    ("", "#2c3e50", 8, "normal"),
    ("CLINICAL SUBTYPES:", "#2c3e50", 10, "bold"),
    ("🔵 Luminal A → Hormone therapy", "#3498db", 8, "normal"),
    ("🟢 Luminal B → Hormone + chemo", "#2ecc71", 8, "normal"),
    ("🔴 HER2-enriched → Trastuzumab", "#e74c3c", 8, "normal"),
    ("🟣 Triple Negative → Chemo + immunotherapy", "#9b59b6", 8, "normal"),
]

y_pos = 0.97
for texto, color, size, weight in resumen:
    ax6.text(0.05, y_pos, texto, transform=ax6.transAxes,
             fontsize=size, color=color, fontweight=weight,
             verticalalignment="top")
    y_pos -= 0.072

ax6.set_facecolor("#eaf4fb")
ax6.patch.set_visible(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/mnt/user-data/outputs/ml_tumor_classification.png",
            bbox_inches="tight", facecolor="#f8f9fa")
plt.close()
print("\n✅ Visualización guardada.")

# ─────────────────────────────────────────────
# MÓDULO 5: PREDICCIÓN CLÍNICA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MÓDULO 5: PREDICCIÓN CLÍNICA EN NUEVOS PACIENTES")
print("=" * 60)

def clasificar_tumor(expresion_genes, modelo, scaler, le, genes):
    """
    Clasifica el subtipo tumoral de un nuevo paciente
    a partir de su perfil de expresión génica.
    """
    X_nuevo = np.array(expresion_genes).reshape(1, -1)
    X_nuevo_scaled = scaler.transform(X_nuevo)
    subtype_pred = modelo.predict(X_nuevo_scaled)[0]
    subtype_nombre = le.inverse_transform([subtype_pred])[0]
    proba = modelo.predict_proba(X_nuevo_scaled)[0]

    tratamientos = {
        "Luminal_A":       "Hormone therapy (tamoxifen/aromatase inhibitor). Low chemotherapy benefit.",
        "Luminal_B":       "Hormone therapy + chemotherapy. Consider CDK4/6 inhibitors.",
        "HER2_enriched":   "Trastuzumab + pertuzumab + chemotherapy. HER2-targeted therapy.",
        "Triple_Negative": "Chemotherapy + immunotherapy (pembrolizumab). Consider PARP inhibitors if BRCA+."
    }

    print(f"\n🧬 TUMOR CLASSIFICATION REPORT")
    print(f"   Predicted subtype:  {subtype_nombre}")
    print(f"   Confidence:         {max(proba)*100:.1f}%")
    print(f"\n   Probability per subtype:")
    for clase, prob in zip(le.classes_, proba):
        barra = "█" * int(prob * 20)
        print(f"   {clase:<20} {barra} {prob*100:.1f}%")
    print(f"\n   💊 Recommended treatment:")
    print(f"   {tratamientos[subtype_nombre]}")

# Clasificar 3 casos de prueba
print("\n📋 CLASSIFYING NEW PATIENT TUMORS:")

# Paciente 1: perfil Luminal A
caso1 = [8.5, 8.2, 7.9, 8.1, 8.3,   # Luminal alto
         2.1, 2.3, 2.0, 2.2, 2.1,   # HER2 bajo
         1.8, 2.0, 1.9, 2.1, 1.7,   # Basal bajo
         2.9, 3.1, 2.8, 3.0, 2.7]   # Proliferación baja

# Paciente 2: perfil HER2
caso2 = [2.0, 1.8, 2.1, 1.9, 2.0,   # Luminal bajo
         9.1, 8.8, 9.2, 8.9, 9.0,   # HER2 alto
         2.0, 1.9, 2.1, 1.8, 2.0,   # Basal bajo
         6.5, 6.8, 6.2, 6.9, 6.4]   # Proliferación alta

# Paciente 3: perfil Triple Negativo
caso3 = [1.9, 2.1, 1.8, 2.0, 1.9,   # Luminal bajo
         2.0, 1.8, 2.1, 1.9, 2.0,   # HER2 bajo
         9.2, 8.9, 9.1, 9.3, 8.8,   # Basal alto
         7.1, 7.3, 6.9, 7.2, 7.0]   # Proliferación alta

for i, caso in enumerate([caso1, caso2, caso3], 1):
    print(f"\n{'='*50}")
    print(f"PATIENT {i}")
    clasificar_tumor(caso, mejor_modelo, scaler, le, genes)

# ─────────────────────────────────────────────
# EXPORTAR RESULTADOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPORTANDO RESULTADOS")
print("=" * 60)

resultados_df = pd.DataFrame({
    "model": list(resultados.keys()),
    "test_accuracy": [resultados[n]["accuracy"] for n in resultados],
    "cv_accuracy_mean": [resultados[n]["cv_mean"] for n in resultados],
    "cv_accuracy_std": [resultados[n]["cv_std"] for n in resultados]
})
resultados_df.to_csv("/mnt/user-data/outputs/ml_model_results.csv", index=False)

feature_imp_df = pd.DataFrame({
    "gene": genes,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
feature_imp_df.to_csv("/mnt/user-data/outputs/gene_importance.csv", index=False)

print("\n✅ ml_model_results.csv exportado")
print("✅ gene_importance.csv exportado")

print("\n" + "=" * 60)
print("✅ ANÁLISIS COMPLETADO")
print(f"   200 tumors classified")
print(f"   3 ML models compared")
print(f"   Best model: {mejor_nombre} ({resultados[mejor_nombre]['accuracy']*100:.1f}% accuracy)")
print(f"   3 new patients classified with treatment recommendation")
print("=" * 60)
