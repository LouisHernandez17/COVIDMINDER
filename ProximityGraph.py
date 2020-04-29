import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

Covid = pd.read_csv('data/csv/04-17-2020.csv')
Centers = pd.read_csv('CenterStates.csv', index_col='state')
Borders = pd.read_csv('data/borders/Borders.txt', index_col='ST1ST2')
Abbreviations = pd.read_csv('data/borders/USStatesAbbreviationstxt.txt')[:50]
Abbreviations = Abbreviations[Abbreviations["us state"] != 'Alaska']
Abbreviations = Abbreviations[Abbreviations["us state"] != 'Hawaii']
Abbreviations = dict(Abbreviations.to_dict('split')['data'])
States = [x for x in Abbreviations]
G = nx.Graph()
G.add_nodes_from(States)
for i in range(1, len(States)):
    state1 = States[i]
    Abb1 = Abbreviations[state1]
    for j in range(i):
        state2 = States[j]
        Abb2 = Abbreviations[state2]
        if (Borders.index == Abb1 + "-" + Abb2).any():
            weight = Borders['LENGTH'].loc[Abb1 + "-" + Abb2]
            G.add_edge(state1, state2, weight=weight)
        elif (Borders.index == Abb2 + '-' + Abb1).any():
            weight = Borders['LENGTH'].loc[Abb2 + "-" + Abb1]
            G.add_edge(state1, state2, weight=weight)
pos = dict(
    [(state, (Centers['longitude'].loc[Abbreviations[state]], Centers['latitude'].loc[Abbreviations[state]])) for state
     in G.nodes])
weights = [G[u][v]['weight'] for u, v in G.edges]

Cases = dict([(state, (Covid[Covid['Province_State'] == state])['Confirmed'].sum()) for state in G.nodes])
Deaths = dict([(state, (Covid[Covid['Province_State'] == state])['Deaths'].sum()) for state in G.nodes])
nx.set_node_attributes(G, Cases, 'Cases')
nx.set_node_attributes(G, Deaths, 'Deaths')
edges = nx.draw_networkx_edges(G, pos, edge_cmap=plt.cm.jet, edge_color=weights, width=5)
nodes = nx.draw_networkx_nodes(G, pos, node_shape='s', node_color='w', node_size=100)
labels = nx.draw_networkx_labels(G, pos, labels=Abbreviations, font_color='b', font_size=8)
plt.axis('off')
# plt.axis('equal')
cb=plt.colorbar(edges, label='Length of border', orientation='horizontal')
cb.outline.set_edgecolor('w')
cb.set_label("Length of border",color="w")
plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='white')
cb.ax.xaxis.set_tick_params(color='w')
plt.savefig('Graph_Transparent.png',transparent=True)
plt.show()

