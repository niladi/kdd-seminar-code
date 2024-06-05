# %%

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Example adjacency matrix
adj_matrix = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Draw the graph
nx.draw(G, with_labels=True, node_color="skyblue", node_size=700, edge_color="gray")
plt.show()

# %%
