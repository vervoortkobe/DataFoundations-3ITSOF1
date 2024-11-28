import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Maak een aangepaste kleurenmap
colors = ['blue', 'lightgray', 'red']
n_bins = 100  # aantal kleurstappen
custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

# Stap 1: Dataset laden en verkennen
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Correlatiematrix maken en visualiseren
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Correlatiematrix van de Iris Dataset')
plt.tight_layout()
plt.show()

# Stap 2: Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=iris.feature_names
)

# Dataset splitsen
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Stap 3: Model trainen
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Stap 4: Analyseer de resultaten
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Stap 5: Visualiseren
# Scatter plot voor de eerste twee kenmerken
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=custom_cmap)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatter plot van Iris data')
plt.colorbar(scatter)
plt.show()

# SVM beslissingsgrenzen visualiseren (voor de eerste twee kenmerken)
def plot_decision_boundaries(X, y, model, feature_indices=[0, 1]):
    h = 0.02  # mesh stepsize
    X_array = X.iloc[:, feature_indices].values
    x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1
    y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Fix: Create mesh points in correct order
    mesh_points = np.c_[xx.ravel(), yy.ravel()]  # Changed from yy, xx to xx, yy
    mesh_points = np.column_stack([
        np.zeros(mesh_points.shape[0]),
        np.zeros(mesh_points.shape[0]),
        mesh_points  # Now the petal length and width will be in positions 2 and 3
    ])
    mesh_points = pd.DataFrame(mesh_points, columns=iris.feature_names)
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.3)  # Changed from yy, xx to xx, yy
    scatter = plt.scatter(X.iloc[:, feature_indices[0]], X.iloc[:, feature_indices[1]], 
                         c=y, cmap=custom_cmap)
    plt.xlabel('Kroonbladlengte (Petal length)')
    plt.ylabel('Kroonbladbreedte (Petal width)')
    plt.title('SVM Beslissingsgrenzen: Iris Dataset')
    plt.show()

# Plot de beslissingsgrenzen
plot_decision_boundaries(X_scaled, y, svm, feature_indices=[3, 2])

# Voorspelling maken voor een nieuw voorbeeld
new_flower = pd.DataFrame([[5.0, 3.4, 1.5, 0.2]], columns=iris.feature_names)
new_flower_scaled = pd.DataFrame(
    scaler.transform(new_flower),
    columns=iris.feature_names
)
prediction = svm.predict(new_flower_scaled)
print("\nVoorspelling voor nieuwe bloem:", iris.target_names[prediction[0]])

"""
Welke kenmerken zijn het meest informatief voor de classificatie van bloemen?
In de Iris-dataset zijn er vier kenmerken (features) die worden gebruikt voor de classificatie van bloemen.
Van deze kenmerken zijn kroonbladlengte en kroonbladbreedte het meest informatief voor de classificatie van de verschillende Iris-soorten. 
Dit komt doordat deze kenmerken meestal een duidelijkere scheiding/beter verschil aantonen tussen de verschillende klassen (soorten) in de dataset.

Hoe kan de correlatiematrix helpen bij het vereenvoudigen of verbeteren van het model?
Aangezien de correlatiematrix de correlatie tussen de verschillende kenmerken in de dataset aantoont, 
kan deze helpen bij het vereenvoudigen/verbeteren van het model op volgende manieren:
1. Identificatie van redundante kenmerken:
Als twee kenmerken sterk gecorreleerd zijn, kan het nuttig zijn om één van hen te verwijderen. 
Dit vermindert de complexiteit van het model zonder al te veel informatie te verliezen.
2. Selectie van kenmerken:
Door te kijken naar de correlatie tussen kenmerken en de doelvariabele (de klasse), 
kun je bepalen welke kenmerken het meest informatief zijn voor de classificatie. 
Dit kan helpen bij het selecteren van de beste subset van kenmerken voor het model.
3. Verbetering van modelprestaties:
Door alleen de meest informatieve kenmerken te gebruiken, kan het model sneller trainen en betere prestaties leveren, 
omdat het minder ruis en irrelevante informatie bevat.
4. Visualisatie van relaties:
De correlatiematrix kan helpen bij het visualiseren van de relaties tussen kenmerken, 
wat nuttig kan zijn voor het begrijpen van de data en het maken van beslissingen over feature engineering.
"""