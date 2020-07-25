import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np
import torch
import re

from utils import split_train_val_test
from utils.visualizer import visualize_joint
from utils.load_and_save import load_model
from RealNVP import train_and_plot as RealNVP_train_and_plot, build_model as RealNVP_build_model
from DDSF import train_and_plot as DDSF_train_and_plot, build_model as DDSF_build_model


def model_loader(model, args, edge, epoch, add_name):
    edge_str = re.sub('[, ()]', '', str(edge))
    model_load_name = 'best_epoch_model' + edge_str + add_name
    load_model(model, args.experiment_saved_models, model_load_name,
               epoch)
    model.eval()


def train_copula_flow(args, model, dataset, data_loaders, conditional_copula, num_current_nodes, save_name, add_name):
    save_name = re.sub('[, ()]', '', str(save_name)) + add_name
    args.conditional_copula = conditional_copula
    rvine = True if num_current_nodes > 2 else False
    __, best_dict, __ = RealNVP_train_and_plot(args, dataset, data_loaders, disable_tqdm=True, rvine=rvine, save_name=save_name)
    return best_dict


def train_marginal_flow(args, model, dataset, data_loaders, save_name, add_name):
    save_name = re.sub('[, ()]', '', str(save_name)) + add_name
    __, best_dict, __ = DDSF_train_and_plot(args, dataset, data_loaders, disable_tqdm=True, rvine=True, save_name=save_name)
    return best_dict


class RVine():
    """Class that containts the RVine tree.
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.graph_list = []
        self.tree_list = []
        self.args.conditional_copula = False
        self.model_uncon = RealNVP_build_model(args)
        self.args.conditional_copula = True
        self.model_con = RealNVP_build_model(args)
        self.model_marg = DDSF_build_model(args)

    def estimate_rvine(self):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """

        def initialize_graph():
            """Initializes first tree with input distributions. Each node is one dimension of the
            distribution.
            """
            # create fully connected graph
            self.current_graph = nx.complete_graph(self.data.shape[1])

            # distribute data pairs onto edges of the first tree, compute tau
            # and weights for each edge

            for node in self.current_graph.nodes():
                dataset, data_loaders = create_dataset_1dim(self.data[:, node].float().reshape(-1, 1), self.args)

                print('Train Marginal Flow for tree {}, node {}'.format(len(self.tree_list), node))

                best_dict = train_marginal_flow(self.args,
                                                self.model_uncon,
                                                dataset,
                                                data_loaders,
                                                save_name=node,
                                                add_name='marginal')
                model_loader(self.model_marg, self.args, node, best_dict['best_validation_epoch'], add_name='marginal')
                with torch.no_grad():
                    transformed_inputs = self.model_marg.transform(self.data[:, node].float().reshape(-1, 1))
                self.current_graph.nodes[node]['cond_distr'] = transformed_inputs
                self.current_graph.nodes[node]['best_dict'] = best_dict


            for edge in self.current_graph.edges():
                n0, n1 = edge

                ktau, __ = scipy.stats.kendalltau(self.current_graph.nodes[n0]['cond_distr'],
                                                  self.current_graph.nodes[n1]['cond_distr'])
                self.current_graph[n0][n1]['weight'] = np.abs(ktau)

            self.graph_list.append(self.current_graph)

        def cm_flow_estimation(num_current_nodes):
            """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
            Created new graph from these edges as nodes.
            """
            self.new_graph = nx.Graph()
            if num_current_nodes > 2:
                for paired_edge in paired_tree_edges:
                    common_node = set(paired_edge[0]).intersection(paired_edge[1])
                    if len(common_node) == 1:
                        add_new_node(common_node, paired_edge[0], num_current_nodes)
                        add_new_node(common_node, paired_edge[1], num_current_nodes)
            if num_current_nodes == 2:
                for edge in self.current_tree.edges:
                    add_new_node(edge[0], edge, num_current_nodes)

        def add_new_node(common_node, edge, num_current_nodes):
            n0, n1 = edge
            if n0 == common_node:
                v0 = self.current_tree.nodes[n0]['cond_distr']
                v1 = self.current_tree.nodes[n1]['cond_distr']
            else:
                v0 = self.current_tree.nodes[n1]['cond_distr']
                v1 = self.current_tree.nodes[n0]['cond_distr']
            dataset, data_loaders = create_dataset(v0, v1, self.args)

            print('Train CM Flow for tree {}, edge {}'.format(len(self.tree_list), edge))

            best_dict_uncon = train_copula_flow(self.args,
                                                self.model_uncon,
                                                dataset,
                                                data_loaders,
                                                False,
                                                num_current_nodes,
                                                save_name=edge,
                                                add_name='cop_uncon')

            best_dict_con = train_copula_flow(self.args,
                                              self.model_con,
                                              dataset,
                                              data_loaders,
                                              True,
                                              num_current_nodes,
                                              save_name=edge,
                                              add_name='cop_con')

            # Estimate F(u_1|u_2)? or c(u_1|u_2)?
            model_loader(self.model_con, self.args, edge, best_dict_con['best_validation_epoch'], add_name='cop_con')

            self.model_con.eval()
            with torch.no_grad():
                cond_distr = self.model_con.transform(inputs=v1.reshape(-1, 1), cond_inputs=v0.reshape(-1, 1))
                self.new_graph.add_node(edge,
                                        cond_distr=cond_distr,
                                        best_dict_uncon=best_dict_uncon,
                                        best_dict_con=best_dict_con,
                                        common_node=common_node)
                if num_current_nodes == 2:
                    cond_distr = self.model_con.transform(inputs=v1.reshape(-1, 1), cond_inputs=v0.reshape(-1, 1))
                    visualize_joint(torch.cat([v0.reshape(-1, 1), cond_distr], axis=1), self.args, name='output_last_copula')

        # initialize graph and transform marginals using marginal flows
        initialize_graph()

        while len(self.current_graph.nodes()) >= 1:
            self.current_tree = nx.maximum_spanning_tree(self.current_graph, weight='weight')
            self.tree_list.append(self.current_tree)
            paired_tree_edges = combinations(list(self.current_tree.edges), 2)
            cm_flow_estimation(len(self.current_graph.nodes()))
            self.current_graph = self.new_graph
            paired_nodes = combinations(list(self.current_graph.nodes), 2)
            for e in paired_nodes:
                self.current_graph.add_edge(*e)
            for edge in self.current_graph.edges():
                n0, n1 = edge
                ktau, __ = scipy.stats.kendalltau(self.current_graph.nodes[n0]['cond_distr'],
                                                  self.current_graph.nodes[n1]['cond_distr'])
                self.current_graph[n0][n1]['weight'] = np.abs(ktau)
            self.graph_list.append(self.current_graph)

    def density(self, inputs):
        with torch.no_grad():
            log_prob = 0
            # distribute data pairs onto edges of the first tree, compute tau
            # and weights for each edge
            for ii in range(len(self.tree_list)):
                if ii == 0:
                    for node in self.tree_list[ii].nodes():
                        best_dict = self.tree_list[ii].nodes[node]['best_dict']
                        model_loader(self.model_marg, self.args, node, best_dict['best_validation_epoch'], add_name='marginal')
                        transformed_input = self.model_marg.transform(inputs=inputs[:, node].float().reshape(-1, 1))

                    for edge in self.tree_list[ii].edges():
                        # @Todo: add DDSF transformation
                        n0, n1 = edge
                        self.tree_list[ii].nodes[n0]['cond_distr'] = inputs[:, n0].float()
                        self.tree_list[ii].nodes[n1]['cond_distr'] = inputs[:, n1].float()
                else:
                    for edge in self.tree_list[ii - 1].edges():
                        # Estimate conditional distributions
                        n0, n1 = edge
                        common_node = self.tree_list[ii].nodes[edge]['common_node']
                        if n0 in common_node or ii == len(self.tree_list) - 1:
                            v0 = self.tree_list[ii - 1].nodes[n0]['cond_distr']
                            v1 = self.tree_list[ii - 1].nodes[n1]['cond_distr']
                        elif n1 in common_node:
                            v0 = self.tree_list[ii - 1].nodes[n1]['cond_distr']
                            v1 = self.tree_list[ii - 1].nodes[n0]['cond_distr']
                        else:
                            raise ValueError('No common node found.')

                        best_dict_con = self.tree_list[ii].nodes[edge]['best_dict_con']
                        model_loader(self.model_con, self.args, edge, best_dict_con['best_validation_epoch'], add_name='cop_con')
                        transformed_input = self.model_con.transform(inputs=v1.reshape(-1, 1),
                                                                     cond_inputs=v0.reshape(-1, 1))

                        self.tree_list[ii].nodes[edge]['cond_distr'] = transformed_input
                    for node in self.tree_list[ii]:
                        # Estimate copula density for previous tree
                        n0, n1 = node
                        v0 = self.tree_list[ii - 1].nodes[n0]['cond_distr'].reshape(-1, 1)
                        v1 = self.tree_list[ii - 1].nodes[n1]['cond_distr'].reshape(-1, 1)

                        best_dict_uncon = self.tree_list[ii].nodes[edge]['best_dict_uncon']
                        model_loader(self.model_uncon, self.args, edge, best_dict_uncon['best_validation_epoch'], add_name='cop_uncon')
                        copula_density = self.model_uncon.log_density(inputs=torch.cat([v0, v1], axis=1))

                        log_prob += copula_density
            assert log_prob.shape[0] == inputs.shape[0]
            prob = torch.exp(log_prob)
            assert torch.min(prob) >= 0 and torch.max(prob) <= 1
        return prob

    def simulate_distribution(self, num_samples=100000):
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


class Rvine_data_1dim():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, inputs):
        self.xx = inputs
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.float()
        self.val = val.float()
        self.tst = tst.float()


def create_dataset_1dim(inputs, args):
    dataset = Rvine_data_1dim(inputs)
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
