import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset simuleren
data = pd.DataFrame({
    'prijs': [1, 2, 3, 2, 1],
    'bereidingstijd': [15, 30, 45, 20, 10],
    'bezoeken': [5, 2, 1, 3, 7],
    'beoordeling': [1, 0, 0, 1, 1]
})

df = pd.DataFrame(data)

# De eerste rijen van mijn dataset bekijken
print(df.head())

# Mijn dataset opslaan als een csv (optioneel)
df.to_csv("beoordelingen.csv", index=False)

# Data splitsen
X = data[['prijs', 'bereidingstijd', 'bezoeken']]
y = data['beoordeling']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Resultaten van splitsing weergeven
print("Trainingsset: ")
print(X_train)
print("Testset: ")
print(X_test)

# Model bouwen
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Voorspellingen doen
y_pred = model.predict(X_test)

# Resultaten
print("Nauwkeurigheid van het model:", accuracy_score(y_test, y_pred))