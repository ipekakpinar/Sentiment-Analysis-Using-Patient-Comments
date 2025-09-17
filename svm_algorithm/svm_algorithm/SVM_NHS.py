import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv("filtered_comments.csv")

# Metin ve etiketler
X = df['commentText']
y = LabelEncoder().fit_transform(df['sentiment'])  # negative=0, neutral=1, positive=2

# Veriyi %80 eğitim - %20 test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF dönüşümü
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM modeli ve eğitim
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vec, y_train)

# Tahmin
y_pred = svm_model.predict(X_test_vec)

# 🔹 Sınıf Bazlı Rapor (tek satır metin raporu)
print("🔸 Sınıf Bazlı Rapor:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))

# 🔹 Genel Metrikler
accuracy = accuracy_score(y_test, y_pred)
macro_p = precision_score(y_test, y_pred, average='macro')
macro_r = recall_score(y_test, y_pred, average='macro')
macro_f = f1_score(y_test, y_pred, average='macro')

weighted_p = precision_score(y_test, y_pred, average='weighted')
weighted_r = recall_score(y_test, y_pred, average='weighted')
weighted_f = f1_score(y_test, y_pred, average='weighted')

print("\n🔸 Genel Metrikler:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall: {macro_r:.4f}")
print(f"Macro F1: {macro_f:.4f}")
print(f"Weighted Precision: {weighted_p:.4f}")
print(f"Weighted Recall: {weighted_r:.4f}")
print(f"Weighted F1: {weighted_f:.4f}")

# 🔹 1. Sınıf Bazlı Grafik: Precision, Recall, F1
print("\n🔸 NHS Veri Seti Sınıf Bazlı Grafik:")
class_labels = ['negative', 'neutral', 'positive']
prec = precision_score(y_test, y_pred, average=None)
rec = recall_score(y_test, y_pred, average=None)
f1s = f1_score(y_test, y_pred, average=None)

x = range(len(class_labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar([i - width for i in x], prec, width, label='Precision')
rects2 = ax.bar(x, rec, width, label='Recall')
rects3 = ax.bar([i + width for i in x], f1s, width, label='F1-Score')

ax.set_ylabel('Skor')
ax.set_title('NHS Veri Seti - Sınıf Bazında Precision, Recall ve F1-Skor')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')
ax.bar_label(rects3, padding=3, fmt='%.4f')

fig.tight_layout()
plt.show()

# 🔹 2. Confusion Matrix Grafiği
print("\n🔸 NHS Veri Seti - Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print(cm_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('NHS Veri Seti - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 🔹 3. Genel Metrikler Tablosu ve Grafiği (Macro vs Weighted Avg vs Accuracy)
print("\n🔸 Genel Metrikler Karşılaştırma Tablosu (Grafik İçin):")
metric_table = pd.DataFrame({
    "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
               "Weighted Precision", "Weighted Recall", "Weighted F1"],
    "Value": [accuracy, macro_p, macro_r, macro_f, weighted_p, weighted_r, weighted_f]
})
print(metric_table)

plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Metric', data=metric_table, palette='viridis')
plt.title('Genel Metriklerin Karşılaştırılması')
plt.xlabel('Değer')
plt.ylabel('Metrik')
plt.show()