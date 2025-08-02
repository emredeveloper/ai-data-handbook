import networkx as nx
import matplotlib.pyplot as plt

# Create a sample graph
G = nx.Graph()

# Add nodes (we'll create a star-like graph with one central node)
nodes = ['Center', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
G.add_nodes_from(nodes)

# Add edges (connecting all nodes to the center)
edges = [('Center', node) for node in nodes if node != 'Center']
G.add_edges_from(edges)

# Add some additional connections to make the graph more interesting
G.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'E'), ('F', 'G')])

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800)
plt.title("Sample Graph for Centrality Measures")
plt.show()

# Calculate centrality measures
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)

# Print results
print("Betweenness Centrality:")
for node, value in sorted(betweenness.items(), key=lambda x: -x[1]):
    print(f"{node}: {value:.3f}")

print("\nCloseness Centrality:")
for node, value in sorted(closeness.items(), key=lambda x: -x[1]):
    print(f"{node}: {value:.3f}")

print("\nEigenvector Centrality:")
for node, value in sorted(eigenvector.items(), key=lambda x: -x[1]):
    print(f"{node}: {value:.3f}")


# Create a barbell graph
G = nx.barbell_graph(5, 2)

# Calculate centrality measures
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=500)

# Print results
print("\nBarbell Graph Results:")
print("Betweenness Centrality for bridge nodes:", 
      [betweenness[n] for n in [4,5]])  # The connecting nodes
print("Closeness Centrality for bridge nodes:", 
      [closeness[n] for n in [4,5]])
print("Eigenvector Centrality for bridge nodes:", 
      [eigenvector[n] for n in [4,5]])