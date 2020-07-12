import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np


def RVine():
    """Class that containts the RVine tree.
    """

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.graph_list = []
        self.tree_list = []
        self.current_graph = initialize_graph()

    def estimate_rvine(self):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """
        num_current_nodes = len(self.current_graph.nodes())
        while num_current_nodes >= 2:
            self.current_tree = nx.minimum_spanning_tree(self.current_graph)
            self.tree_list.append(self.current_tree)
            self.current_graph = cm_flow_estimation(self.current_tree)
            self.graph_list.append(self.current_graph)

    def initialize_graph(self):
        """Initializes first tree with input distributions. Each node is one dimension of the
        distribution.
        """
        # initialization of the first tree
        current_graph = nx.Graph()

        # generate the 2-combinations of all nodes of the first tree.
        var_num = self.data.shape[1]
        node_list = [str(i) for i in range(0, var_num)]
        paired_nodes = combinations(node_list, 2)

        for e in paired_nodes:
            current_graph.add_edge(*e)

        # distribute data pairs onto edges of the first tree, compute tau
        # and weights for each edge
        for edge in current_graph.edges():
            n0, n1 = edge
            edge_data = {n0: self.data[:, n0], n1: self.data[:, n1]}

            current_graph[n0][n1]['edge_data'] = edge_data
            ktau = scipy.stats.kendalltau(edge_data[n0], edge_data[n1])
            current_graph[n0][n1]['kendalltau'] = ktau
            current_graph[n0][n1]['weight'] = 1 - np.abs(ktau)

        self.graph_list.append(current_graph)

    def cm_flow_estimation(self):
        """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
        Created new graph from these edges as nodes.
        """
        for edge in self.current_tree.edges():
            n0, n1 = edge
            v0 = self.current_tree[n0][n1]['edge_data'][n0]
            v1 = self.current_tree[n0][n1]['edge_data'][n1]

            train_cm_flow(v0, v1)
            # paired_data.append(np.array([v0, v1]).transpose())

        raise NotImplementedError

    def sample_multivariate_copula(self, num_samples=1000):
        """Returns samples from estimated multivariate copula.
        """
        raise NotImplementedError

    def plot(self, tree_num=1):
        """Plots R-vine tree structure.
        """
        raise NotImplementedError

# if __name__ == '__main__':
#     rvine_object =
