import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import os

from utils import split_train_val_test, js_divergence
from utils.visualizer import visualize_joint
from utils.load_and_save import load_model
from RealNVP import train_and_plot as RealNVP_train_and_plot, build_model as RealNVP_build_model
from DDSF import train_and_plot as DDSF_train_and_plot, build_model as DDSF_build_model


def model_loader(model, args, edge, epoch, add_name, send_to_device=True):
    edge_str = re.sub('[, ()]', '', str(edge))
    model_load_name = 'best_epoch_model' + edge_str + add_name
    load_model(model, args.experiment_saved_models, model_load_name, epoch)
    if send_to_device:
        model.to(args.device)
    model.eval()


def train_copula_flow(args, model, dataset, data_loaders, conditional_copula, num_current_nodes, save_name, add_name):
    save_name = re.sub('[, ()]', '', str(save_name)) + add_name
    args.conditional_copula = conditional_copula
    # rvine = True if num_current_nodes > 2 else False
    __, best_dict, __ = RealNVP_train_and_plot(args, dataset, data_loaders, disable_tqdm=True, rvine=True, save_name=save_name)
    return best_dict


def train_marginal_flow(args, model, dataset, data_loaders, save_name, add_name):
    """ Trains marginal flow and saved the results with the given save_name.
    """
    save_name = re.sub('[, ()]', '', str(save_name)) + add_name
    __, best_dict, __ = DDSF_train_and_plot(args, dataset, data_loaders, disable_tqdm=True, rvine=True, save_name=save_name)
    return best_dict


def flatten(nested_tuple):
    for i in nested_tuple:
        yield from [i] if not isinstance(i, tuple) else flatten(i)


def initialize_graph(self):
    """Initializes first tree with input distributions. Each node is one dimension of the
    distribution.
    """
    # create fully connected graph
    self.current_graph = nx.complete_graph(self.data.shape[1])

    # distribute data pairs onto edges of the first tree, compute tau
    # and weights for each edge

    for node in self.current_graph.nodes():
        # Prepare dataset for node
        dataset, data_loaders = create_dataset_1dim(self.data[:, node:node + 1].float(), self.args)

        assert not np.isnan(torch.sum(self.data[:, node:node + 1].float()).cpu()), '{}'.format(self.data[:, node:node + 1].float()[:10])

        # Unless marginal flows are disables, transform distributions using the marginal flow
        if not self.args.disable_marginal:

            print('Train Marginal Flow for tree {}, node {}'.format(len(self.tree_list), node))

            # Train marginal flow
            best_dict = train_marginal_flow(args=self.args,
                                            model=self.model_marg,
                                            dataset=dataset,
                                            data_loaders=data_loaders,
                                            save_name=node,
                                            add_name='marginal')

            # Load best model for marginal flow
            model_loader(self.model_marg, self.args, node, best_dict['best_validation_epoch'], add_name='marginal')

            # Transform inputs using the trained marginal flow
            with torch.no_grad():
                self.data = self.data.to(self.args.device)
                transformed_inputs = self.model_marg.transform(self.data[:, node:node + 1].float())
                self.data = self.data.cpu()
                self.current_graph.nodes[node]['best_dict'] = best_dict # @Todo: am i using this?

        # if marginal flows are disabled, do not transform the inputs
        else:
            self.data = self.data.to(self.args.device)
            transformed_inputs = self.data[:, node:node + 1].float().to(self.args.device)
            self.data = self.data.cpu()

        # save transformed inputs in graph node
        self.current_graph.nodes[node]['cond_distr'] = transformed_inputs

        assert not np.isnan(torch.sum(transformed_inputs).cpu()), '{}'.format(transformed_inputs[:10])

    # for each node pair, compute kendalls tau between the transformed inputs
    for edge in self.current_graph.edges():
        n0, n1 = edge

        ktau, __ = scipy.stats.kendalltau(self.current_graph.nodes[n0]['cond_distr'].cpu(),
                                          self.current_graph.nodes[n1]['cond_distr'].cpu())

        assert not np.isnan(ktau), '{}'.format(ktau)

        self.current_graph[n0][n1]['weight'] = np.abs(ktau)

    self.graph_list.append(self.current_graph)

    return self


def cm_flow_estimation(self, num_current_nodes, plots):
    """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
    Created new graph from these edges as nodes.
    """
    self.new_graph = nx.Graph()
    self.traversed_edges = []
    if num_current_nodes > 2:
        for paired_edge in self.paired_tree_edges:
            common_node = set(paired_edge[0]).intersection(paired_edge[1])
            if len(common_node) == 1:
                if paired_edge[0] not in self.traversed_edges:
                    self = add_new_node(self, common_node, paired_edge[0], num_current_nodes, plots)
                    self.traversed_edges.extend([paired_edge[0], tuple(reversed(paired_edge[0]))])
                if paired_edge[1] not in self.traversed_edges:
                    self = add_new_node(self, common_node, paired_edge[1], num_current_nodes, plots)
                    self.traversed_edges.extend([paired_edge[1], tuple(reversed(paired_edge[1]))])
    if num_current_nodes == 2:
        for edge in self.current_tree.edges:
            self = add_new_node(self, edge[0], edge, num_current_nodes, plots)
    return self


def add_new_node(self, common_node, edge, num_current_nodes, plots):
    v0, v1, edge = assign_distr_to_nodes(edge, common_node, self.current_tree)

    dataset, data_loaders = create_dataset(v0, v1, self.args)

    edge_str = re.sub('[, ()]', '', str(edge))
    visualize_joint(dataset.trn, self.args, name='rvine_input_dataset_{}'.format(edge_str))

    print('Train conditional CM Flow for tree {}, edge {}, unconditional node: {}'.format(len(self.tree_list), edge, next(flatten(edge))))

    best_dict_con = train_copula_flow(self.args,
                                      self.model_con,
                                      dataset,
                                      data_loaders,
                                      True,
                                      num_current_nodes,
                                      save_name=edge,
                                      add_name='cop_con')

    model_loader(self.model_con, self.args, edge, best_dict_con['best_validation_epoch'], add_name='cop_con')

    with torch.no_grad():
        cond_distr = self.model_con.transform(inputs=v1.reshape(-1, 1), cond_inputs=v0.reshape(-1, 1))
        self.new_graph.add_node(edge,
                                cond_distr=cond_distr,
                                best_dict_con=best_dict_con,
                                common_node=common_node)
        if plots:
            uniform_inputs = self.norm.cdf(torch.cat([v0.reshape(-1, 1), cond_distr], axis=1).cpu())
            edge_str = re.sub('[, ()]', '', str(edge))
            visualize_joint(uniform_inputs, self.args, name='rvine_con_transform_{}'.format(edge_str))

            cond_inputs = torch.tensor(np.random.normal(size=(100000, 1))).float()
            con_samples = self.model_con.sample_copula(num_samples=100000, num_inputs=1, cond_inputs=cond_inputs, device=self.args.device)
            visualize_joint(con_samples.cpu(), self.args, name='rvine_con_copula_{}'.format(edge_str))

    return self


def assign_distr_to_nodes(edge, common_node, current_tree):
    n0, n1 = edge
    if n0 == common_node or n0 in common_node:
        v0 = current_tree.nodes[n0]['cond_distr']
        v1 = current_tree.nodes[n1]['cond_distr']
        edge = n1, n0
    elif n1 == common_node or n1 in common_node:
        v0 = current_tree.nodes[n1]['cond_distr']
        v1 = current_tree.nodes[n0]['cond_distr']
    else:
        raise ValueError('No common node found.')
    return v0, v1, edge


class RVine():
    """Class that containts the RVine tree.
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.graph_list = []
        self.tree_list = []
        self.num_inputs = data.shape[1]

        # Initialize conditional copula Flow
        self.args.conditional_copula = True
        self.model_con = RealNVP_build_model(args)
        self.model_con.to(args.device)

        # Initialize marginal flow
        self.model_marg = DDSF_build_model(args)
        self.model_marg.to(args.device)

        # Initialize empty results dictionary
        self.results_dict = {}

    def estimate_rvine(self, plots=True):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """

        # initialize graph and transform marginals using marginal flows
        self = initialize_graph(self)

        # get normal distribution for transformations
        self.norm = scipy.stats.norm(loc=0, scale=1)

        # create new tree as long as current graph has 1 or more nodes
        while len(self.current_graph.nodes()) >= 1:

            # calculate the tree which maximizes the k-tau dependency of the graph
            self.current_tree = nx.maximum_spanning_tree(self.current_graph, weight='weight', algorithm='prim')

            self.tree_list.append(self.current_tree)

            # create a list of all edges which share a node
            self.paired_tree_edges = []
            for node in self.current_tree.nodes:
                edges = list(self.current_tree.edges(nbunch=node, data=False))
                if len(edges) == 2:
                    self.paired_tree_edges.append(tuple(edges))
                elif len(edges) > 2:
                    self.paired_tree_edges.extend(combinations(edges, 2))

            # estimate the copulas on the edges
            self = cm_flow_estimation(self, len(self.current_graph.nodes()), plots)
            self.current_graph = self.new_graph
            paired_nodes = combinations(list(self.current_graph.nodes), 2)
            # @Todo: this is the problem!
            for e in paired_nodes:
                self.current_graph.add_edge(*e)
            for edge in self.current_graph.edges():
                n0, n1 = edge
                ktau, __ = scipy.stats.kendalltau(self.current_graph.nodes[n0]['cond_distr'].cpu(),
                                                  self.current_graph.nodes[n1]['cond_distr'].cpu())
                self.current_graph[n0][n1]['weight'] = np.abs(ktau)
            self.graph_list.append(self.current_graph)

    def sample(self, num_samples=1000, transform=False):
        with torch.no_grad():
            # first: sample multivariate uniform distribution. then, transform the samples accordingly.
            samples = torch.Tensor(num_samples, self.num_inputs).normal_()

            # for each tree, find out which variable was transformed and transform it 'back'
            for ii in reversed(range(1, len(self.tree_list))):
                print('tree number', ii)
                # dim to be transformed: the one that has no common edge in the previous tree,
                # the common edge is the condtional input
                for node in self.tree_list[ii].nodes():
                    n0, n1 = node
                    common_node = self.tree_list[ii].nodes[node]['common_node']
                    unconditioned_node = next(flatten(common_node))
                    conditioned_node = next(flatten(node))
                    print('common node', common_node)
                    print('conditioned node', conditioned_node)

                    v0 = samples[:, unconditioned_node:unconditioned_node + 1]
                    v1 = samples[:, conditioned_node:conditioned_node + 1]

                    best_dict_con = self.tree_list[ii].nodes[node]['best_dict_con']
                    model_loader(self.model_con,
                                 self.args, node,
                                 best_dict_con['best_validation_epoch'],
                                 add_name='cop_con',
                                 send_to_device=True)
                    # inverse H-function
                    transformed_marginal = self.model_con.transform(inputs=v1, cond_inputs=v0, mode='inverse', device=self.args.device)
                    samples[:, conditioned_node:conditioned_node + 1] = transformed_marginal

        if transform:
            normal_distr = torch.distributions.normal.Normal(0, 1)
            samples = normal_distr.cdf(samples)
        return samples

    def jsd_vinecopula(self, args, rvine_estimate, true_rvine, obs=100000):
        """Returns JS-Divergence of the predicted Copula and the true Copula
        """
        with torch.no_grad():
            # Define distributions
            # normal_distr = scipy.stats.norm(0, 1)
            # true_cop_distr = datasets.distributions.Copula_Distr(args=args, transform=False)

            # Samples from both distributinos
            samples_pred = self.sample(num_samples=obs, transform=True)
            samples_target = true_rvine.simulate(obs)

            # Estimate Copula distr
            # RealNVP outputs the density directly, but not the transformation to
            # uniform marginals. Thus, an estimation with Gaussian KDE is simpler.
            pred_distr = scipy.stats.gaussian_kde(samples_pred.cpu().numpy().T)
            true_rvine = scipy.stats.gaussian_kde(samples_target.T)
            # Note, that uniform samples means the transformed samples

            # Prob X in both distributions
            prob_X_in_p = pred_distr.pdf(samples_pred.cpu().numpy().T).T
            prob_X_in_q = true_rvine.pdf(samples_pred.cpu().numpy().T).T

            # Prob Y in both distributions
            prob_Y_in_q = true_rvine.pdf(samples_target.T).T
            prob_Y_in_p = pred_distr.pdf(samples_target.T).T

            if np.isnan(np.sum(prob_X_in_q)):
                prob_X_in_p = prob_X_in_p[~np.isnan(prob_X_in_q)]
                prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_X_in_q)]
                prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_X_in_q)]
                prob_X_in_q = prob_X_in_q[~np.isnan(prob_X_in_q)]

            if np.isnan(np.sum(prob_Y_in_q)):
                prob_X_in_p = prob_X_in_p[~np.isnan(prob_Y_in_q)]
                prob_X_in_q = prob_X_in_q[~np.isnan(prob_Y_in_q)]
                prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_Y_in_q)]
                prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_Y_in_q)]

            assert np.min(samples_pred.cpu().numpy()) >= 0
            assert np.min(samples_target) >= 0
            assert np.min(prob_X_in_p) >= 0
            assert np.min(prob_X_in_q) >= 0
            assert np.min(prob_Y_in_p) >= 0
            assert np.min(prob_Y_in_q) >= 0, '%r' % (np.min(prob_Y_in_q))

            divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                       prob_X_in_q=prob_X_in_q,
                                       prob_Y_in_p=prob_Y_in_p,
                                       prob_Y_in_q=prob_Y_in_q)

            print('MC-JSD Vine Copula: {}'.format(divergence))

            self.results_dict['MC_JSD Vine Copula'] = divergence
            return divergence

    def plot(self, filename=None):
        """ @Todo: change description
        Plot the regular vine structure after sequential estimation
        via function 'modeling'.

        Parameter
        ---------

        ntrees : int, optional. The first ntrees of all the vine trees
                 will be plotted. Default is `0', meaning plotting all
                 the vine trees.

        filename : string, optional. Default is an empty string
                   indicating direct output to screen. The plot will
                   output to the specified directory if a file name
                   with extension is given.
        """
        save_path = os.path.join(self.args.figures_path, 'tree_structure' + '.pdf')
        num_trees = len(self.tree_list)
        if num_trees == 1:
            plt.title("Tree_1")
            nx.draw(self.tree_list[0])
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            mfrow = (num_trees + 1) / 2
            mfcol = 2

            fig = plt.figure(figsize=(8, 6))
            for i in range(num_trees):
                if i < len(self.tree_list):
                    plt.subplot(mfrow, mfcol, i + 1)
                    plt.title("Tree_" + str(i + 1))
                    nx.draw(self.tree_list[i], with_labels=True)
            fig.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')


class Rvine_data():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, dim1, dim2):
        self.xx = torch.cat([dim1.reshape(-1, 1), dim2.reshape(-1, 1)], axis=1)
        trn, val = split_train_val_test(self.xx, only_val=True)

        self.trn = trn.float().cpu()
        self.val = val.float().cpu()


def create_dataset(dim1, dim2, args):
    dataset = Rvine_data(dim1, dim2)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    # train_tensor = torch.from_numpy(dataset.trn)
    train_dataset = torch.utils.data.TensorDataset(dataset.trn)

    # valid_tensor = torch.from_numpy(dataset.val)
    valid_dataset = torch.utils.data.TensorDataset(dataset.val)

    # test_tensor = torch.from_numpy(dataset.tst)
    # test_dataset = torch.utils.data.TensorDataset(dataset.tst)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader}
    return dataset, data_loaders


class Rvine_data_1dim():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, inputs):
        self.xx = inputs
        trn, val = split_train_val_test(self.xx, only_val=True)

        self.trn = trn.float()
        self.val = val.float()


def create_dataset_1dim(inputs, args):
    dataset = Rvine_data_1dim(inputs)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_dataset = torch.utils.data.TensorDataset(dataset.trn)

    valid_dataset = torch.utils.data.TensorDataset(dataset.val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader}
    return dataset, data_loaders
