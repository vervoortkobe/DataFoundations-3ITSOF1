import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')

# Voorbereiden van de dataset
X = df[['prijs', 'bereidingstijd', 'bezoeken']]  # features
y = df['beoordeling']  # variabele om te voorspellen

# Data opsplitsen in training en test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FIT
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)

# PREDICT
y_pred = dt_model.predict(X_test)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Nauwkeurigheid: {accuracy:.2f}")

print("\nClassificatie Rapport:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15, 10))

# Beslissingsboom
plt.subplot(2, 2, 1)
plot_tree(dt_model, feature_names=['prijs', 'bereidingstijd', 'bezoeken'], 
          class_names=['Laag', 'Hoog'], filled=True, rounded=True)
plt.title('Beslissingsboom Visualisatie')

# Feature importance plot
plt.subplot(2, 2, 2)
importances = pd.DataFrame({
    'feature': ['prijs', 'bereidingstijd', 'bezoeken'],
    'importance': dt_model.feature_importances_
})
importances = importances.sort_values('importance', ascending=False)
plt.bar(importances['feature'], importances['importance'])
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.ylabel('Importance Score')

# Confusion Matrix
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Voorspelde waarde')
plt.ylabel('Werkelijke waarde')

# Vergelijking werkelijke vs voorspelde waarden
plt.subplot(2, 2, 4)
comparison_df = pd.DataFrame({
    'Werkelijk': y_test,
    'Voorspeld': y_pred
})
comparison_df['correct'] = comparison_df['Werkelijk'] == comparison_df['Voorspeld']
sns.countplot(data=comparison_df, x='Werkelijk', hue='correct')
plt.title('Werkelijke vs Voorspelde Waarden')
plt.xlabel('Werkelijke Beoordeling')
plt.ylabel('Aantal')

plt.tight_layout()
plt.show()