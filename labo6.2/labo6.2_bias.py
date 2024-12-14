import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.datasets import fetch_openml

# Stap 1: Dataset laden en verkennen
adult_data = fetch_openml(data_id=1590, as_frame=True)
df = adult_data.frame

## De target kolom zit nu in een aparte lijst, dus:
df['income'] = adult_data.target

## Identificeer kenmerken die mogelijk bias kunnen bevatten
biased_features = ['sex', 'race']
print(f"\nMogelijke biased features: {biased_features}")

# Stap 2: Exploratory Data Analysis (EDA)
## Analyse van inkomensverdeling per geslacht
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='income', data=df)
plt.title('Inkomensverdeling per geslacht')
plt.show()

## Analyse van inkomensverdeling per ras
plt.figure(figsize=(12, 6))
sns.countplot(x='race', hue='income', data=df)
plt.title('Inkomensverdeling per ras')
plt.xticks(rotation=45)
plt.show()

## Beantwoord de vragen:
## Hoe varieert het inkomen tussen mannen en vrouwen?
income_by_sex = df.groupby('sex')['income'].value_counts(normalize=True).unstack()
print("\nPercentage van inkomens per geslacht:")
print(income_by_sex)

## Zijn bepaalde rassen onder- of oververtegenwoordigd in de dataset?
race_counts = df['race'].value_counts()
print("\nVerdeling van rassen:")
print(race_counts)

# Stap 3: Bias visualiseren
## stacked bar chart voor inkomen per geslacht
income_by_sex.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Inkomensverdeling per geslacht (gestapeld)')
plt.ylabel('Percentage')
plt.show()

## Boxplot voor leeftijd per geslacht en inkomen
plt.figure(figsize=(12, 6))
sns.boxplot(x='sex', y='age', hue='income', data=df)
plt.title('Leeftijdsverdeling per geslacht en inkomen')
plt.show()

# Stap 4: Classificatiemodel bouwen
## Data voorbereiding
label_encoder = LabelEncoder()
df['income'] = label_encoder.fit_transform(df['income'])

categorical_cols = df.select_dtypes(include=['category', 'object']).columns
# Data split
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding op de training set
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)

# One-hot encoding op de test set,
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Train een Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

## Evaluatie
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Stap 5: Evalueren van bias in het model
## Analyse van modelprestaties per groep
def evaluate_model_by_group(X_test, feature, model, y_test):
    if feature in X_test.columns:
        for group in X_test[feature].unique():
            group_indices = X_test.index[X_test[feature] == group]
            y_true_group = y_test.loc[group_indices]
            y_pred_group = model.predict(X_test.loc[group_indices])
            print(f"\nPerformance for group '{group}':")
            print(classification_report(y_true_group, y_pred_group))
    else:
        print(f"\nFeature '{feature}' not found in test set.")


## Evaluatie per geslacht
evaluate_model_by_group(X_test, 'sex_Male', model, y_test)

## Evaluatie per ras (niet geanonimiseerd)
race_cols = [col for col in X_test.columns if col.startswith('race_')]
for race_col in race_cols:
    evaluate_model_by_group(X_test, race_col, model, y_test)

# Stap 6: Fairness-metrics berekenen
def calculate_fairness_metrics(df, y_true, y_pred, sensitive_feature):
    if sensitive_feature in df.columns:
        sensitive_feature_values = df.loc[y_true.index, sensitive_feature].values
        print(f"\nFairness metrics for {sensitive_feature}:")
        
        ## calculate the demographic parity difference
        dp_diff = demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_feature_values
        )
        print(f"Demographic Parity Difference: {dp_diff}")

        ## calculate the equal opportunity difference
        eq_op_diff = equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_feature_values
        )
        print(f"Equal Opportunity Difference: {eq_op_diff}")
    else:
        print(f"\nFeature '{sensitive_feature}' not found in test set.")

calculate_fairness_metrics(X_test, y_test, y_pred, 'sex_Male')

for race_col in race_cols:
    calculate_fairness_metrics(X_test, y_test, y_pred, race_col)

# Stap 7: Data-anonimisatie
print(df.columns)
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
def anonymize_data(df):
    # Leeftijd groeperen
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Geslacht vervangen door generieke labels
    if 'sex_Male' in df.columns:
        df['sex_anon'] = df['sex_Male'].apply(lambda x: 'M' if x == 1 else 'V')
    elif 'sex' in df.columns:
        df['sex_anon'] = df['sex'].apply(lambda x: 'M' if x == 'Male' else 'V')
    
    # Ras vervangen door generieke labels
    race_cols = [col for col in df.columns if col.startswith('race_')]
    if race_cols:
        df['race_anon'] = df[race_cols].apply(lambda row: 'A' if any(row) else 'B', axis=1)
    elif 'race' in df.columns:
        df['race_anon'] = df['race'].apply(lambda x: 'A' if x != 'White' else 'B')

    # Verwijder de originele kolommen en de dummies
    columns_to_drop = ['sex', 'sex_Male'] + race_cols + ['age', 'race']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    return df

df_anonymized = anonymize_data(df.copy())

print("\nAnonymized Dataframe:")
print(df_anonymized.head())

## Model evalueren op anonieme data
# Voeg hier de extra dummy-conversie toe voor de anonieme data
categorical_cols_anon = df_anonymized.select_dtypes(include=['category', 'object']).columns

X_anon = df_anonymized.drop('income', axis=1)

## splits data opnieuw op, met de anonieme data.
X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(X_anon, y, test_size=0.2, random_state=42)

# One-hot encoding op de training set
X_train_anon = pd.get_dummies(X_train_anon, columns=categorical_cols_anon, drop_first=True)

# One-hot encoding op de test set,
X_test_anon = pd.get_dummies(X_test_anon, columns=categorical_cols_anon, drop_first=True)

model_anon = RandomForestClassifier(random_state=42)
model_anon.fit(X_train_anon, y_train_anon)
y_pred_anon = model_anon.predict(X_test_anon)
print("\nClassification Report on anonymized data:")
print(classification_report(y_test_anon, y_pred_anon))

## Evalueren van fairness-metrics op anonieme data
# Dynamisch detecteer de kolomnamen voor geslacht en ras
sex_anon_cols = [col for col in X_test_anon.columns if col.startswith('sex_anon_')]
race_anon_cols = [col for col in X_test_anon.columns if col.startswith('race_anon_')]

if sex_anon_cols:
    calculate_fairness_metrics(X_test_anon, y_test_anon, y_pred_anon, sex_anon_cols[0])
if race_anon_cols:
    calculate_fairness_metrics(X_test_anon, y_test_anon, y_pred_anon, race_anon_cols[0])

# Reflectie:
"""
De dataset vertoont een duidelijke bias, namelijk bias in inkomen tussen mannen en vrouwen en tussen de verschillende rassen.
Het model presteert beter voor de meerderheidsgroepen (mannen, wit) en minder goed voor de minderheidsgroepen (vrouwen, andere rassen).
Fairness metrics zoals Demographic Parity en Equal Opportunity laten zien dat er sprake is van ongelijkheid tussen de groepen.
Anonimisatie door het groeperen van leeftijd en het vervangen van geslacht en ras door generieke labels verlaagt de mogelijkheid 
om direct informatie uit de dataset te herleiden. Het nadeel is een kleine achteruitgang in de modelprestaties.
De keuze om wel of niet te anonimiseren en in welke mate is een afweging tussen privacy en de nauwkeurigheid van het model die we willen behalen.
"""