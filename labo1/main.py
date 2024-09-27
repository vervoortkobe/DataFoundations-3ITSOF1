import pandas as pd
import matplotlib.pyplot as plt

# Stap 1: Data verzamelen en importeren
print(pd.read_csv('austpop.csv'))
df = pd.read_csv('austpop.csv')

# Stap 2: Data verkennen
# a. Bereken de centrale tendens (gemiddelde, mediaan, modus) van de variabele.
kolomnaam = 'Qld'

gemiddelde = df[kolomnaam].mean()

mediaan = df[kolomnaam].median()

modus = df[kolomnaam].mode()[0]

print(f'Gemiddelde: {gemiddelde}')
print(f'Mediaan: {mediaan}')
print(f'Modus: {modus}')

# b. Bereken de spreiding (standaardafwijking, interkwartielafstand) van de variabele.
standaardafwijking = df[kolomnaam].std()

variabele = df[kolomnaam].dropna()

q1 = variabele.quantile(0.25)
q3 = variabele.quantile(0.75)

iqr = q3 - q1

print(f'Standaardafwijking: {standaardafwijking}')
print(f"Eerste kwartiel (Q1): {q1}")
print(f"Derde kwartiel (q3): {q3}")
print(f"Interkwartielafstand (IQR): {iqr}")

# c. Genereer een histogram en een boxplot om de verdeling van de variabele te visualiseren.
variabele = df[kolomnaam].dropna()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(variabele, bins=10, color='blue', alpha=0.7)
plt.title(f'Histogram van {kolomnaam}')
plt.xlabel('Waarden')
plt.ylabel('Frequentie')

plt.subplot(1, 2, 2)
plt.boxplot(variabele)
plt.title(f'Boxplot van {kolomnaam}')
plt.ylabel('Waarden')

plt.tight_layout()
plt.show()

# Stap 3: Interpretatie (dit kan je opnemen in je portfolio)
"""
Zoals we kunnen zien op de boxplot en het histogram is de data van variabele 'Vic' niet normaal 
verdeeld. We zien wel degelijk uitschieters. We zien in de boxplot dat de mediaan eerder rond de 
2500 ligt. De spreiding van de data is groot, dit zien we aan de lengte van de boxplot. De 
standaardafwijking is 1177.
"""

# Stap 4: Rapportage (dit kan je eventueel opnemen in je portfolio)
"""
Bij het vergelijken van de variabelen 'Vic' en 'Qld' zien we dat het gemiddelde, de mediaan en de modus van 'Qld' hoger liggen dan die van 'Vic'. De interkwartielafstand (IQR) van 'Qld' is kleiner (1137 tegenover 1984), wat wijst op een meer geconcentreerde verdeling van de data rond de mediaan. Het histogram van 'Qld' toont een scheve verdeling met de meeste waarden onder 1000, wat op links-scheefheid wijst. 'Vic' heeft een bredere spreiding, wat duidt op meer variatie binnen de dataset. Deze verschillen kunnen wijzen op regionale of demografische verschillen in de dataset.

Vic:
Gemiddelde: 1663.7777777777778
Mediaan: 1413.0
Modus: 683
Standaardafwijking: 913.125782378553

Qld:
Gemiddelde: 2847.3333333333335
Mediaan: 2656.0
Modus: 1409
Standaardafwijking: 1177.1143742219786
"""