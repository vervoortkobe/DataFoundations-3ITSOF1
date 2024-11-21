import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')

print("\nGemiddelde waarden per kolom:")
print(df.mean())

print("\nMediaan waarden per kolom:")
print(df.median())

# Maak een figure met subplots
plt.figure(figsize=(15, 10))

# Histogram van beoordelingen
plt.subplot(2, 2, 1)
plt.hist(df['beoordeling'], bins=2, rwidth=0.8)
plt.title('Verdeling van Beoordelingen')
plt.xlabel('Beoordeling (0=Laag, 1=Hoog)')
plt.ylabel('Aantal restaurants')
plt.xticks([0, 1])  # Set x-axis ticks to only show 0 and 1

# Boxplot van bereidingstijd per prijsklasse
plt.subplot(2, 2, 2)
sns.boxplot(x='prijs', y='bereidingstijd', data=df)
plt.title('Bereidingstijd per Prijsklasse')
plt.xlabel('Prijsklasse')
plt.ylabel('Bereidingstijd (minuten)')

# Scatter plot: bereidingstijd vs bezoeken, gekleurd op prijs
plt.subplot(2, 2, 3)
scatter = plt.scatter(df['bereidingstijd'], df['bezoeken'], c=df['prijs'], cmap='viridis')
plt.colorbar(scatter, label='Prijsklasse')
plt.title('Bereidingstijd vs Bezoeken')
plt.xlabel('Bereidingstijd (minuten)')
plt.ylabel('Aantal bezoeken')

# Gemiddeld aantal bezoeken per prijsklasse
plt.subplot(2, 2, 4)
df.groupby('prijs')['bezoeken'].mean().plot(kind='bar')
plt.title('Gemiddeld Aantal Bezoeken per Prijsklasse')
plt.xlabel('Prijsklasse')
plt.ylabel('Gemiddeld aantal bezoeken')

plt.tight_layout()

print("\nGemiddelde bereidingstijd per prijsklasse:")
print(df.groupby('prijs')['bereidingstijd'].mean())

print("\nGemiddelde beoordeling per prijsklasse:")
print(df.groupby('prijs')['beoordeling'].mean())

print("\nCorrelatiematrix:")
print(df.corr())

plt.show()