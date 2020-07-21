import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np
import torch
import re

from utils import split_train_val_test
from utils.visualizer import visualize_joint
from utils.load_and_save import load_model
from CM_Flow import train_and_plot, build_model


def model_loader(self, edge):
    model_name = re.sub('[, ()]', '', str(edge))
    load_model(self.model, self.args.experiment_saved_models, 'best_epoch_model',
               model_name)
    self.model.eval()


class RVine():
    """Class that containts the RVine tree.
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.graph_list = []
        self.tree_list = []
        self.model = build_model(self.args)

    def estimate_rvine(self):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """
        def cm_flow_estimation(self, num_current_nodes):
            """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
            Created new graph from these edges as nodes.
            """
            new_graph = nx.Graph()
            for ee, edge in enumerate(self.current_tree.edges()):
                n0, n1 = edge
                v0 = self.current_tree[n0][n1]['edge_data'][n0]
                v1 = self.current_tree[n0][n1]['edge_data'][n1]
                print('vo', v0.shape)
                dataset, data_loaders = create_dataset(v0, v1, self.args)
                print('Start CM Flow training for tree {}, edge {}'.format(len(self.tree_list), edge))
                if num_current_nodes > 2:
                    best_dict = train_and_plot(self.args, dataset, data_loaders, disable_tqdm=True, rvine=True)
                else:
                    best_dict = train_and_plot(self.args, dataset, data_loaders, disable_tqdm=True, rvine=False)
                self.model = load_model(self.model, self.args.experiment_saved_models, 'train_model',
                                        best_dict['best_validation_epoch'])
                self.model.eval()
                copula_distr = self.model.log_density_RealNVP(torch.cat([v0.reshape(-1, 1), v1.reshape(-1, 1)], axis=1)).detach().numpy()
                print('copula distr', copula_distr.shape)
                self.current_tree[n0][n1]['edge_data']['copula_distr'] = copula_distr
                new_graph.add_node(edge, copula_distr=copula_distr, best_dict=best_dict)
            return new_graph

        while len(self.current_graph.nodes()) >= 2:
            self.current_tree = nx.maximum_spanning_tree(self.current_graph, weight='weight')
            self.tree_list.append(self.current_tree)
            self.current_graph = cm_flow_estimation(self, len(self.current_graph.nodes()))
            paired_nodes = combinations(list(self.current_graph.nodes), 2)
            for e in paired_nodes:
                self.current_graph.add_edge(*e)
            for edge in self.current_graph.edges():
                n0, n1 = edge
                edge_data = {n0: self.current_graph.nodes[n0]['copula_distr'], n1: self.current_graph.nodes[n1]['copula_distr']}
                self.current_graph[n0][n1]['edge_data'] = edge_data
                ktau, __ = scipy.stats.kendalltau(edge_data[n0],
                                                  edge_data[n1])
                self.current_graph[n0][n1]['weight'] = np.abs(ktau)

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
            edge_data = {n0: torch.tensor(self.data[:, n0]).float(), n1: torch.tensor(self.data[:, n1]).float()}
            self.current_graph[n0][n1]['edge_data'] = edge_data
            ktau, __ = scipy.stats.kendalltau(edge_data[n0], edge_data[n1])
            self.current_graph[n0][n1]['weight'] = np.abs(ktau)

        self.graph_list.append(self.current_graph)

    def sample_multivariate_copula(self, num_samples=100000):
        """Returns samples from estimated multivariate copula.
        """
        with torch.no_grad():
            samples_dict = {}
            tree = self.tree_list[0]
            for edge in tree.edges():
                model_loader(self, edge)
                copula_samples = self.model.sample(num_samples=num_samples)
                samples_dict[edge] = np.exp(self.model.log_density_RealNVP(copula_samples).detach().numpy())

            for tt, tree in enumerate(self.tree_list[1:]):
                for edge in tree.edges():
                    n0, n1 = edge
                    model_loader(self, edge)
                    inputs = torch.tensor(np.concatenate([samples_dict[n0], samples_dict[n1]], axis=1))
                    samples_dict[edge] = np.exp(self.model.log_density_RealNVP(inputs).detach().numpy())
                if tt == len(self.tree_list) - 2:
                    normal_distr = torch.distributions.normal.Normal(0, 1)
                    copula_samples = normal_distr.cdf(copula_samples)
                    visualize_joint(copula_samples, self.args, 'last_copula_rvine')

    def simulate_distribution(self, num_samples=100000):
        with torch.no_grad():
            samples_dict = {}
            tree = self.tree_list[-1]
            for edge in tree.edges():
                n0, n1 = edge
                model_loader(self, edge)
                copula_samples = self.model.sample(num_samples=num_samples)
                samples_dict[n0] = copula_samples[:, 0].reshape(-1, 1)
                samples_dict[n1] = copula_samples[:, 1].reshape(-1, 1)
            for tt, tree in enumerate(reversed(self.tree_list[:-1])):
                for edge in tree.edges():
                    n0, n1 = edge
                    model_loader(self, edge)
                    print(samples_dict[edge].shape)
                    copula_samples = self.model.sample(num_samples=num_samples)
                    samples_dict[n0] = copula_samples[:, 0]
                    samples_dict[n1] = copula_samples[:, 1]
            normal_distr = torch.distributions.normal.Normal(0, 1)
            copula_samples = normal_distr.cdf(copula_samples)
            visualize_joint(copula_samples, self.args, 'sample_copula_first_tree')
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
        trn = torch.from_numpy(trn)
        val = torch.from_numpy(val)
        tst = torch.from_numpy(tst)

        self.trn = trn.float()
        self.val = val.float()
        self.tst = tst.float()


def create_dataset(dim1, dim2, args):
    dataset = Rvine_data(dim1, dim2)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    # train_tensor = torch.from_numpy(dataset.trn)
    train_dataset = torch.utils.data.TensorDataset(dataset.trn)

    # valid_tensor = torch.from_numpy(dataset.val)
    valid_dataset = torch.utils.data.TensorDataset(dataset.val)

    # test_tensor = torch.from_numpy(dataset.tst)
    test_dataset = torch.utils.data.TensorDataset(dataset.tst)

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
