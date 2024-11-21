import pandas as pd
import numpy as np
import random

# Seed zetten, zodat we hetzelfde resultaat krijgen om te testen
# np.random.seed(1)

# Aantal te generaten records
n_records = 100

# Dataset genereren
prijs = np.random.randint(1, 4, n_records)  # prijs 1-3
bereidingstijd = np.random.randint(10, 61, n_records)  # tijd 10-60 min
bezoeken = np.random.randint(1, 21, n_records)  # bezoeken 1-20

# Beoordeling bepalen op basis van andere variabelen
beoordeling = []
for i in range(n_records):
    # Regels voor hoge beoordeling (1):
    # - Goedkope restaurants (prijs=1): moeten snel zijn (<30 min) OF veel bezoeken hebben (>10)
    # - Medium restaurants (prijs=2): moeten redelijk snel zijn (<45 min) EN redelijk wat bezoeken hebben (>5)
    # - Dure restaurants (prijs=3): moeten veel bezoeken hebben (>15) EN de klanten niet te lang laten wachten (<50 min)
    
    if (prijs[i] == 1 and (bereidingstijd[i] < 30 or bezoeken[i] > 10)) or \
       (prijs[i] == 2 and bereidingstijd[i] < 45 and bezoeken[i] > 5) or \
       (prijs[i] == 3 and bereidingstijd[i] < 50 and bezoeken[i] > 15):
        beoordeling.append(1)
    else:
        beoordeling.append(0)

data = {
    'prijs': prijs,
    'bereidingstijd': bereidingstijd,
    'bezoeken': bezoeken,
    'beoordeling': beoordeling
}

df = pd.DataFrame(data)

print("\nEerste paar rijen van de dataset:")
print(df.head())

print("\nBasisstatistieken van de dataset:")
print(df.describe())

df.to_csv('dataset.csv', index=False)
print("\nDataset is opgeslagen als 'dataset.csv'")