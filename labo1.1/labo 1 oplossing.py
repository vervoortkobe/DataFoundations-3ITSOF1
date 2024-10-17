import pandas as pd
import matplotlib.pyplot as plt

# Stap 2: Data Importeren
data = pd.read_csv('austpop.csv')

# Stap 3: Data Verkennen
variable_to_analyze = 'Vic'

# Bereken de centrale tendens
mean = data[variable_to_analyze].mean()
median = data[variable_to_analyze].median()
mode = data[variable_to_analyze].mode().values[0]

# Bereken de spreiding
std_dev = data[variable_to_analyze].std()
iqr = data[variable_to_analyze].quantile(0.75) - data[variable_to_analyze].quantile(0.25)

# Genereer een histogram
plt.hist(data[variable_to_analyze], bins=10, edgecolor='k')
plt.xlabel(variable_to_analyze)
plt.ylabel('Aantal waarnemingen')
plt.title(f'Histogram van {variable_to_analyze}')
plt.show()

# Genereer een boxplot
plt.boxplot(data[variable_to_analyze])
plt.ylabel(variable_to_analyze)
plt.title(f'Boxplot van {variable_to_analyze}')
plt.show()

# Interpretatie
print(f"Centrale tendens - Gemiddelde: {mean}, Mediaan: {median}, Modus: {mode}")
print(f"Spreiding - Standaardafwijking: {std_dev}, Interkwartielafstand (IQR): {iqr}")
