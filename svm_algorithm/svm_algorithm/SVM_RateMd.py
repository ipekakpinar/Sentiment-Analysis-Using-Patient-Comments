import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv("dengelenmis_dosya (1).csv")

# Liste gibi görünen string'leri cümle haline getir
df['review'] = df['review'].apply(ast.literal_eval)
df['review'] = df['review'].apply(lambda tokens: ' '.join(tokens))

# Özellikler ve etiketler
X = df['review']
y = df['label'].astype(int)

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM eğitimi
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vec, y_train)

# Tahmin
y_pred = svm_model.predict(X_test_vec)

# Raporu yazdır
print(classification_report(
    y_test,
    y_pred,
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))

# Sınıf bazlı metrikler
labels = ['negative', 'neutral', 'positive']
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Bar grafiği çiz
x = range(len(labels))
width = 0.2

plt.figure(figsize=(10, 6))
bars1 = plt.bar([i - width for i in x], precision, width=width, label='Precision')
bars2 = plt.bar(x, recall, width=width, label='Recall')
bars3 = plt.bar([i + width for i in x], f1, width=width, label='F1-Score')
plt.xticks(ticks=x, labels=labels)
plt.ylabel("Score")
plt.title("RateMd Veri Seti - Sınıf Bazlı Precision, Recall, F1-Score")
plt.legend()

# Bar üstlerine değer yaz
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("RateMd Veri Seti - Confusion Matrix")
plt.tight_layout()
plt.show()
