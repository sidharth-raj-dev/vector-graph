import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define features and their attributes for similarity calculation
features_data = {
    'Virtual Machines': {'Category': 'Compute', 'Model': 'IaaS', 'Purpose': 'Infrastructure'},
    'App Service': {'Category': 'Compute', 'Model': 'PaaS', 'Purpose': 'Logic'},
    'Azure Functions': {'Category': 'Compute', 'Model': 'PaaS', 'Purpose': 'Logic'},
    'AKS': {'Category': 'Compute', 'Model': 'PaaS', 'Purpose': 'Infrastructure'},
    'Virtual Network': {'Category': 'Networking', 'Model': 'IaaS', 'Purpose': 'Infrastructure'},
    'Load Balancer': {'Category': 'Networking', 'Model': 'PaaS', 'Purpose': 'Infrastructure'},
    'VPN Gateway': {'Category': 'Networking', 'Model': 'PaaS', 'Purpose': 'Infrastructure'},
    'Blob Storage': {'Category': 'Storage', 'Model': 'PaaS', 'Purpose': 'Data'},
    'File Storage': {'Category': 'Storage', 'Model': 'PaaS', 'Purpose': 'Data'},
    'SQL Database': {'Category': 'Database', 'Model': 'PaaS', 'Purpose': 'Data'},
    'Cosmos DB': {'Category': 'Database', 'Model': 'PaaS', 'Purpose': 'Data'},
    'Entra ID': {'Category': 'Identity', 'Model': 'SaaS', 'Purpose': 'Security'},
    'Key Vault': {'Category': 'Security', 'Model': 'PaaS', 'Purpose': 'Security'},
    'Azure Monitor': {'Category': 'Management', 'Model': 'SaaS', 'Purpose': 'Governance'},
    'Azure Advisor': {'Category': 'Management', 'Model': 'SaaS', 'Purpose': 'Governance'},
    'Azure Policy': {'Category': 'Governance', 'Model': 'SaaS', 'Purpose': 'Governance'},
    'Resource Locks': {'Category': 'Governance', 'Model': 'SaaS', 'Purpose': 'Governance'},
    'Cost Management': {'Category': 'Management', 'Model': 'SaaS', 'Purpose': 'Governance'},
    'ARM Templates': {'Category': 'Management', 'Model': 'SaaS', 'Purpose': 'Infrastructure'}
}

# Convert to DataFrame
df = pd.DataFrame.from_dict(features_data, orient='index')

# Function to calculate similarity score
def calculate_similarity(row1, row2):
    score = 0
    if row1['Category'] == row2['Category']:
        score += 0.5
    if row1['Model'] == row2['Model']:
        score += 0.3
    if row1['Purpose'] == row2['Purpose']:
        score += 0.2
    return score

# Create similarity matrix
feature_names = list(features_data.keys())
n = len(feature_names)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            sim_matrix[i, j] = calculate_similarity(df.iloc[i], df.iloc[j])

# Create the graph
G = nx.Graph()

for i in range(n):
    G.add_node(feature_names[i], category=df.iloc[i]['Category'])

# Add edges only for similarity above a certain threshold to keep the graph clean
threshold = 0.4
for i in range(n):
    for j in range(i + 1, n):
        if sim_matrix[i, j] >= threshold:
            G.add_edge(feature_names[i], feature_names[j], weight=sim_matrix[i, j])

# Color mapping for categories
categories = df['Category'].unique()
color_map = plt.get_cmap('tab20', len(categories))
cat_to_color = {cat: color_map(i) for i, cat in enumerate(categories)}
node_colors = [cat_to_color[G.nodes[node]['category']] for node in G.nodes()]

# Draw the graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)

# Draw edges with varying width based on weight
weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# Create a legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=cat_to_color[cat], markersize=10) for cat in categories]
plt.legend(handles=legend_elements, title='Categories', loc='upper right')

plt.title('Similarity Graph of Azure Fundamentals (AZ-900) Features', fontsize=15)
plt.axis('off')
plt.savefig('azure_features_similarity_graph.png', bbox_inches='tight')

# List the frequently used features
print("Frequently used Azure Features:")
for cat in categories:
    print(f"\n{cat}:")
    print(", ".join(df[df['Category'] == cat].index.tolist()))
