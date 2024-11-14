import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Stap 1: Metadata verzamelen en toevoegen
df = pd.read_excel("Online Retail.xlsx")
print(df.head())

# Stap 2: bouw een data catalogus
## 1. Data catalogus maken
metadata = pd.read_json("metadata.json").T
print(metadata)

## 2. Interactie met de data catalogus
def search_metadata(description_keyword=None, importance_level=None):
  if description_keyword:
      description_filter = metadata['description'].str.contains(description_keyword, case=False, na=False)
  else:
      description_filter = pd.Series([True] * len(metadata), index=metadata.index)
  
  # Filteren op belangrijkheidsniveau
  if importance_level:
      importance_filter = metadata['importance'] == importance_level
  else:
      importance_filter = pd.Series([True] * len(metadata), index=metadata.index)

  # Toepassen van beide filters
  result = metadata[description_filter & importance_filter]
  return result

print(search_metadata(description_keyword="sales", importance_level="hoog"))

# Stap 3: data lineage logboek
## 1. Transformaties & Logboek
transformation_log = []

initial_row_count = len(df)
df = df.drop_duplicates()
rows_removed = initial_row_count - len(df)
transformation_log.append({
    "transformation": "Remove duplicates",
    "columns_affected": [],
    "rows_affected": rows_removed
})

df['total_sales'] = df['UnitPrice'] * df['Quantity']
transformation_log.append({
    "transformation": "Add 'total_sales' column",
    "columns_affected": ['total_sales'],
    "rows_affected": len(df)
})

initial_row_count = len(df)
df = df[df['Quantity'] > 10]
rows_filtered = initial_row_count - len(df)
transformation_log.append({
    "transformation": "Filter records where Quantity > 10",
    "columns_affected": ['Quantity'],
    "rows_affected": rows_filtered
})

for log in transformation_log:
    print(log)

# Stap 4: data lineage visualiseren
G = nx.DiGraph()

G.add_node("Initial Data", description="Beginpunt van de data")
G.add_node("Remove Duplicates", description="Verwijder duplicaten")
G.add_node("Add Total Sales", description="Voeg total_sales kolom toe")
G.add_node("Filter Quantity > 10", description="Filter records met Quantity > 10")

G.add_edge("Initial Data", "Remove Duplicates")
G.add_edge("Remove Duplicates", "Add Total Sales")
G.add_edge("Add Total Sales", "Filter Quantity > 10")

pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
plt.title("Data Lineage Visualisatie")
plt.show()