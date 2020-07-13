import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np
import torch

from utils import split_train_val_test
from CM_Flow import train_and_plot


class RVine():
    """Class that containts the RVine tree.
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.graph_list = []
        self.tree_list = []

    def estimate_rvine(self):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """
        def cm_flow_estimation(self):
            """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
            Created new graph from these edges as nodes.
            """
            new_graph = nx.Graph()
            new_nodes = []
            nodes_dict = {}
            for ee, edge in enumerate(self.current_tree.edges()):
                print(edge)
                n0, n1 = edge
                v0 = self.current_tree[n0][n1]['edge_data'][n0]
                v1 = self.current_tree[n0][n1]['edge_data'][n1]
                dataset, data_loaders = create_dataset(v0, v1, self.args)
                print(dataset.trn.shape)
                trained_cm_flow = train_and_plot(self.args, dataset, data_loaders, disable_tqdm=False, error_bars=False)
                exit()
                # what do i do here? copula_distr = trained_cm_flow.sample_copula(v0, v1)
                new_nodes.append(ee)
                nodes_dict[ee] = {'copula_distr': copula_distr, 'model': trained_cm_flow}
            paired_nodes = combinations(new_nodes, 2)
            for e in paired_nodes:
                new_graph.add_edge(*e)
            for edge, ee in enumerate(new_graph.edges()):
                new_graph[ee] = nodes_dict[ee]
            return new_graph

        num_current_nodes = len(self.current_graph.nodes())
        while num_current_nodes >= 2:
            print('start estimation')
            self.current_tree = nx.minimum_spanning_tree(self.current_graph, weight='weight')
            self.tree_list.append(self.current_tree)
            self.current_graph = cm_flow_estimation(self)
            self.graph_list.append(self.current_graph)

    def initialize_graph(self):
        """Initializes first tree with input distributions. Each node is one dimension of the
        distribution.
        """
        # initialization of the first tree
        self.current_graph = nx.Graph()

        # generate the 2-combinations of all nodes of the first tree.
        var_num = self.data.shape[1]
        node_list = [i for i in range(0, var_num)]
        paired_nodes = combinations(node_list, 2)

        for e in paired_nodes:
            self.current_graph.add_edge(*e)

        # distribute data pairs onto edges of the first tree, compute tau
        # and weights for each edge
        for edge in self.current_graph.edges():
            n0, n1 = edge
            edge_data = {n0: self.data[:, n0], n1: self.data[:, n1]}
            self.current_graph[n0][n1]['edge_data'] = edge_data
            ktau, __ = scipy.stats.kendalltau(edge_data[n0], edge_data[n1])
            self.current_graph[n0][n1]['kendalltau'] = ktau
            self.current_graph[n0][n1]['weight'] = 1 - np.abs(ktau)

        self.graph_list.append(self.current_graph)

    def sample_multivariate_copula(self, num_samples=1000):
        """Returns samples from estimated multivariate copula.
        """
        raise NotImplementedError

    def plot(self, tree_num=1):
        """Plots R-vine tree structure.
        """
        raise NotImplementedError


class Rvine_data():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, dim1, dim2):
        self.xx = np.concatenate([dim1.reshape(-1, 1), dim2.reshape(-1, 1)], axis=1)
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.astype(np.float32)
        self.val = val.astype(np.float32)
        self.tst = tst.astype(np.float32)


def create_dataset(dim1, dim2, args):
    dataset = Rvine_data(dim1, dim2)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_tensor = torch.from_numpy(dataset.trn)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader,
                    'test_loader': test_loader}
    return dataset, data_loaders
