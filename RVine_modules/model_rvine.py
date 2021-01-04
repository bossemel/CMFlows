# This module is inspired by: Yuan, Zhenfei and Taizhong Hu. "pyvine: The Python Package for
# Regular Vine Copula Modeling, Sampling and Testing." Commun. Math. Stat., 13 Sept. 2019,
# pp. 1-34, doi:10.1007/s40304-019-00195-2.

import networkx as nx
from itertools import combinations
import scipy.stats
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import os
import torch.optim as optim

from utils import split_train_val_test, js_divergence, gaussian_change_of_var_ND
from utils.visualizer import visualize_joint
from utils.load_and_save import load_model
from NSF import build_model as build_model_nsf
# from RealNVP import train_and_plot as RealNVP_train_and_plot, build_model as RealNVP_build_model
from DDSF import build_model as build_model_ddsf
from RealNVP import build_model as build_model_rnvp
from experiment_runner import train_val
from utils import calc_jsd, normalize_torch
eps = 0.0001


def model_loader(model, args, edge, epoch, add_name, send_to_device=True):
    """Loads model weights of a given epoch.

    Params:
        model: the model to load
        args: parsed arguments
        edge: edge for which the model was trained
        add_name: additional name, usually 'uncon' or 'con'
        send_to_device: whether to send the model to the device (Cuda or CPU)
    """
    edge_str = re.sub('[, ()]', '', str(edge))
    model_load_name = 'best_epoch_model' + edge_str + add_name
    load_model(model, args.experiment_saved_models, model_load_name, epoch)
    if send_to_device:
        model.to(args.device)
    model.eval()


def flatten(nested_tuple):
    """Flattens a touble.
    """
    for i in nested_tuple:
        yield from [i] if not isinstance(i, tuple) else flatten(i)


def transform_marginals(self, node):
    with torch.no_grad():
        self.marg_flow.eval()
        self.data = self.data.to(self.args.device)
        if self.args.marg_flow == 'NSF':
            transformed_inputs = self.marg_flow.flow.transform_to_noise(self.data[:, node:node + 1].float())

        elif self.args.marg_flow == 'DDSF':
            assert not np.isnan(self.data[:, node:node + 1].sum().cpu())
            transformed_inputs = self.marg_flow.transform_to_noise(self.data[:, node:node + 1].float())
        self.data = self.data.cpu()
        return transformed_inputs


def marg_flow(self, dataset, data_loaders, node):
    print('Train Marginal Flow for tree {}, node {}'.format(len(self.tree_list), node))

    # Initialize marginal flow
    if self.args.marg_flow == 'NSF':
        self.marg_flow = build_model_nsf(self.args, flow_type='marg_flow')
    elif self.args.marg_flow == 'DDSF':
        self.marg_flow = build_model_ddsf(self.args)

    self.marg_flow.to(self.args.device)
    self.marg_flow.state = dict()
    self.marg_flow.train()
    self.args.optimizer = optim.Adam(self.marg_flow.parameters(), lr=self.args.lr_m, weight_decay=self.args.weight_decay_m)
    self.args.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.args.optimizer, self.args.epochs) #, args.num_training_steps, 0)

    # Train marginal flow
    best_dict, __ = train_val(args=self.args,
                              model=self.marg_flow,
                              dataset=dataset,
                              data_loaders=data_loaders,
                              save_name=re.sub('[, ()]', '', str(node)) + 'marginal',
                              model_name='marg_flow_rv',
                              rvine=True,
                              disable_tqdm=True)

    # Load best model for marginal flow
    model_loader(self.marg_flow, self.args, node, best_dict['best_validation_epoch'], add_name='marginal')

    # Transform inputs using the trained marginal flow
    transformed_inputs = transform_marginals(self, node)

    return transformed_inputs


def initialize_graph(self):
    """Initializes first tree with input distributions. Each node is one dimension of the
    distribution.
    """
    # create fully connected graph
    current_graph = nx.complete_graph(self.data.shape[1])

    # distribute data pairs onto edges of the first tree, compute tau
    # and weights for each edge

    for node in current_graph.nodes():
        # Prepare dataset for node
        dataset, data_loaders = create_dataset_1dim(self.data[:, node:node + 1].float(), self.args)
        node_str = re.sub('[, ()]', '', str(node))
        visualize_joint(np.concatenate([self.data[:, node:node + 1], self.data[:, node:node + 1]], axis=1), self.args.figures_path, name='rvine_pre_marginal_{}'.format(node_str))
        assert not np.isnan(torch.sum(self.data[:, node:node + 1].float()).cpu()), '{}'.format(self.data[:, node:node + 1].float()[:10])

        # Unless marginal flows are disables, transform distributions using the marginal flow
        if not self.args.disable_marginal:
            transformed_inputs = marg_flow(self, dataset, data_loaders, node)

        # if marginal flows are disabled, do not transform the inputs with the marginal flow, but cast them to
        # the real number line with the inverse Gaussian CDF
        else:
            self.data = self.data.to(self.args.device)
            transformed_inputs = torch.tensor(self.norm.ppf(self.data[:, node:node + 1].cpu())).to(self.args.device).float()
            self.data = self.data.cpu()

        # save transformed inputs in graph node
        current_graph.nodes[node]['node_data'] = transformed_inputs
        assert not np.isnan(torch.sum(transformed_inputs).cpu()), '{}'.format(transformed_inputs[:10])

    return current_graph


def add_ktau(current_graph):
    # for each node pair, compute kendalls tau between the transformed inputs
    for edge in current_graph.edges():
        n0, n1 = edge
        ktau, __ = scipy.stats.kendalltau(current_graph.nodes[n0]['node_data'].cpu(),
                                          current_graph.nodes[n1]['node_data'].cpu())
        assert not np.isnan(ktau), '{}'.format(ktau)
        current_graph[n0][n1]['weight'] = np.abs(ktau)
        current_graph[n1][n0]['weight'] = np.abs(ktau)
    return current_graph


def build_next_tree(self, current_tree, num_current_nodes, plots):
    """Adds attributes 'trained_cm_model' (or name of saved model) and 'copula' to each edge of the current tree.
    Created new graph from these edges as nodes.

    Params:
        num_current_nodes: size of the current tree
        plots: boolean to indicate whether plots should be created
    """
    new_graph = nx.Graph()
    traversed_edges = []

    for node in current_tree.nodes():
        neighbor_list = current_tree.neighbors(node)
        neighbor_list = [n for n in neighbor_list]
        print('node', node, 'neighbors', neighbor_list)
        if len(neighbor_list) >= 2:
            for neighbor in neighbor_list:
                if (node, neighbor) not in traversed_edges:
                    print('creating new node for node neighboar pair', node, neighbor)
                    new_graph = add_new_node(self, new_graph, node, (node, neighbor), plots)
                    traversed_edges.extend([(node, neighbor), tuple(reversed((node, neighbor)))])
        if num_current_nodes == 2:
            print('creating new node for node neighboar pair', node, neighbor_list[0])
            new_graph = add_new_node(self, new_graph, node, (node, neighbor_list[0]), plots)
            num_current_nodes -= 1

    return new_graph


def cop_flow_transform(self, uncon_node_data, cond_node_data):
    self.cop_flow.eval()
    if self.args.cop_flow == 'NSF':
        node_data = self.cop_flow.flow.transform_to_noise(inputs=uncon_node_data.reshape(-1, 1), context=cond_node_data.reshape(-1, 1))
    elif self.args.cop_flow == 'RealNVP':
        node_data = self.cop_flow.transform_to_noise(inputs=uncon_node_data.reshape(-1, 1), context=cond_node_data.reshape(-1, 1))
    else:
        raise ValueError('Unknown copula flow type.')
    return node_data


def plot_cop_flow(self, node_data, cond_node_data, edge):
    gaussian_inputs = torch.cat([node_data, cond_node_data.reshape(-1, 1)], axis=1).cpu()
    uniform_inputs = self.norm.cdf(torch.cat([node_data, cond_node_data.reshape(-1, 1)], axis=1).cpu())
    edge_str = re.sub('[, ()]', '', str(edge))
    visualize_joint(uniform_inputs, self.args.figures_path, name='rvine_con_transform_uniform_{}'.format(edge_str))
    #visualize_joint(gaussian_inputs, self.args.figures_path, name='rvine_con_transform_gaussian_{}'.format(edge_str))

    context = torch.tensor(np.random.normal(size=(10000, 1))).float()
    con_samples = self.cop_flow.sample_copula(num_samples=10000, num_inputs=1, context=context, device=self.args.device)
    visualize_joint(con_samples.cpu(), self.args.figures_path, name='rvine_con_copula_{}'.format(edge_str))


def add_new_node(self, new_graph, common_node, edge, plots):
    """Trains copula flow betweens two nodes and transforms inputs to create new node.

    Params:
        common_node: number of the common node, which is unconditional
        edge: edge to train flow for
        plots: boolean indicating whether to create plots
    """
    uncon_node_data, cond_node_data, edge = assign_distr_to_nodes(edge, common_node, self.current_tree, tree_num=len(self.tree_list))
    edge_str = re.sub('[, ()]', '', str(edge))

    dataset, data_loaders = create_dataset(uncon_node_data, cond_node_data, self.args)

    normal_distr = torch.distributions.normal.Normal(0, 1)
    visualize_joint(normal_distr.cdf(dataset.trn.cpu()), self.args.figures_path, name='cop_flow_input_dataset_uni_{}'.format(edge_str))

    print('Train conditional CM Flow for tree {}, edge {}, unconditional node: {}'.format(len(self.tree_list), edge, next(flatten(edge))))

    # Initialize conditional copula Flow
    args = self.args
    args.conditional_copula = True
    if self.args.cop_flow == 'NSF':
        self.cop_flow = build_model_nsf(args, flow_type='cop_flow')
    elif self.args.cop_flow == 'RealNVP':
        self.cop_flow = build_model_rnvp(args)

    self.cop_flow.to(self.args.device)
    self.cop_flow.state = dict()
    self.cop_flow.train()
    self.args.optimizer = optim.Adam(self.cop_flow.parameters(), lr=self.args.lr_c, weight_decay=self.args.weight_decay_c)
    self.args.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.args.optimizer, self.args.epochs) #, args.num_training_steps, 0)

    best_dict_con, __ = train_val(args=self.args,
                                  model=self.cop_flow,
                                  dataset=dataset,
                                  data_loaders=data_loaders,
                                  save_name=re.sub('[, ()]', '', str(edge)) + 'cop_con',
                                  model_name='cop_flow_rv',
                                  transform_inputs=False,
                                  disable_tqdm=True,
                                  rvine=True)

    model_loader(self.cop_flow, self.args, edge, best_dict_con['best_validation_epoch'], add_name='cop_con')

    with torch.no_grad():
        _uncon_node = uncon_node_data.clone()
        _cond_node = cond_node_data.clone()
        node_data = cop_flow_transform(self, _uncon_node, _cond_node)
        new_graph.add_node(edge,
                           node_data=node_data,
                           best_dict_con=best_dict_con,
                           common_node=common_node)
        if plots:
            plot_cop_flow(self, node_data, cond_node_data, edge)

    return new_graph


def assign_distr_to_nodes(edge, common_node, current_tree, tree_num=0):
    """Gives the appropriate data given the edge and the common node

    Params:
        edge: edge tuple
        common_node: node of the edge tuple which is unconditional
        current_tree: current tree

    Returns:
        cond_node_data, uncon_node_data: data for both edges, cond_node_data being the common node
        edge: edge, possible switched around to allow the very first entry to be the conditional
    """
    n0, n1 = edge
    print('n0, n1, common node', n0, n1, common_node)
    print('Tree {}, common node {}'.format(tree_num, common_node))
    if n0 == common_node or n0 in common_node:
        cond_node_data = current_tree.nodes[n0]['node_data'].detach().clone()
        uncon_node_data = current_tree.nodes[n1]['node_data'].detach().clone()
        edge = n1, n0
    elif n1 == common_node or n1 in common_node:
        cond_node_data = current_tree.nodes[n1]['node_data'].detach().clone()
        uncon_node_data = current_tree.nodes[n0]['node_data'].detach().clone()
    else:
        raise ValueError('No common node found.')
    return uncon_node_data, cond_node_data, edge


def add_edges(graph):
    paired_nodes = combinations(list(graph.nodes), 2)
    for e in paired_nodes:
        graph.add_edge(*e)
    return graph


def inverse_transform(self, uncon_node_data, cond_node_data):
    if self.args.cop_flow == 'NSF':
        transformed_marginal, __ = self.cop_flow.flow._transform.inverse(inputs=uncon_node_data.to(self.args.device), context=cond_node_data.to(self.args.device))
    elif self.args.cop_flow == 'RealNVP':
        transformed_marginal = self.cop_flow.forward(inputs=uncon_node_data.to(self.args.device), context=cond_node_data.to(self.args.device), mode='inverse')[0]
    return transformed_marginal


def find_uncon_con(tree, node):
    n0, n1 = node
    common_node = tree.nodes[node]['common_node']
    if not isinstance(common_node, int):
        con_input_node = next(flatten(common_node))
    else:
        con_input_node = common_node
    uncon_input_node = next(flatten(node))
    return uncon_input_node, con_input_node


class RVine():
    """Class that containts the RVine tree.
    """
    def __init__(self, args, num_inputs):
        self.args = args
        self.graph_list = []
        self.tree_list = []
        self.num_inputs = num_inputs #data.shape[1]

        # Initialize empty results dictionary
        self.results_dict = {}

    def fit(self, data, plots=True):
        """Sequentially estimates the best tree by minimum spanning algorithm
        and estimates the copula between nodes using CM Flows.
        """
        self.data = data
        # get normal distribution for transformations
        self.norm = scipy.stats.norm()
        # initialize graph and transform marginals using marginal flows
        self.current_graph = initialize_graph(self)
        self.current_graph = add_ktau(self.current_graph)

        self.graph_list.append(self.current_graph)

        # create new tree as long as current graph has 1 or more nodes
        while len(self.current_graph.nodes()) >= 1:

            # calculate the tree which maximizes the k-tau dependency of the graph
            self.current_tree = nx.maximum_spanning_tree(self.current_graph, weight='weight', algorithm='prim')
            self.tree_list.append(self.current_tree)

            # estimate the copulas on the edges
            self.new_graph = build_next_tree(self, self.current_tree, len(self.current_graph.nodes()), plots)

            self.current_graph = self.new_graph.copy()
            self.current_graph = add_edges(self.current_graph)
            self.current_graph = add_ktau(self.current_graph)
            self.graph_list.append(self.current_graph)

    def sample(self, num_samples=100000, transform=False):
        """Samples from the trained R-Vine.

        Params:
            num_sampels: how many samples to create
            transform: whether to transform the outputs using the normal distr. cdf

        Returns:
            samples
        """
        with torch.no_grad():
            # first: sample multivariate uniform distribution. then, transform the samples accordingly.
            current_tree_samples = torch.Tensor(num_samples, self.num_inputs).normal_()
            #transformed = []
            #current_tree_samples = samples.clone()
            next_tree_samples = current_tree_samples.clone()

            # for each tree, find out which variable was transformed and transform it 'back'
            for ii in reversed(range(1, len(self.tree_list))):
                # print('tree number', ii)
                # dim to be transformed: the one that has no common edge in the previous tree,
                # the common edge is the condtional input

                for node in self.tree_list[ii].nodes():
                    #print('tree', ii, 'node', node)
                    uncon_input_node, con_input_node = find_uncon_con(self.tree_list[ii], node)

                    #print('sampling uncon node {}, con node {}'.format(uncon_input_node, con_input_node))
                    cond_node_data = current_tree_samples[:, con_input_node:con_input_node + 1]
                    uncon_node_data = current_tree_samples[:, uncon_input_node:uncon_input_node + 1]

                    best_dict_con = self.tree_list[ii].nodes[node]['best_dict_con']
                    model_loader(self.cop_flow,
                                 self.args, node,
                                 best_dict_con['best_validation_epoch'],
                                 add_name='cop_con',
                                 send_to_device=True)
                    self.cop_flow.to(self.args.device)
                    self.cop_flow.eval()

                    # inverse H-function
                    transformed_marginal = inverse_transform(self, uncon_node_data, cond_node_data)
                    next_tree_samples[:, uncon_input_node:uncon_input_node + 1] = transformed_marginal

                current_tree_samples = next_tree_samples.clone()

        if transform:
            normal_distr = torch.distributions.normal.Normal(0, 1)
            current_tree_samples = normal_distr.cdf(current_tree_samples)
        return current_tree_samples

    def pdf_uniform(self, inputs, context=None):
        """Samples from the trained R-Vine.

        Params:
            num_sampels: how many samples to create
            transform: whether to transform the outputs using the normal distr. cdf

        Returns:
            samples
        """
        with torch.no_grad():
            current_tree_inputs = inputs.copy()
            pdf = torch.ones((current_tree_inputs.shape[0]))
            #counted = []
            next_tree_inputs = current_tree_inputs.copy()

            for ii in range(1, len(self.tree_list)):
                #counted = []

                for node in self.tree_list[ii].nodes():
                    print('pdf tree', ii, 'node', node)
                    uncon_input_node, con_input_node = find_uncon_con(self.tree_list[ii], node)

                    #if True: #uncon_input_node not in counted:
                        #print('sampling uncon node {}, con node {}'.format(uncon_input_node, con_input_node))
                    uncon_node_data = current_tree_inputs[:, uncon_input_node:uncon_input_node + 1]#.to(self.args.device)
                    cond_node_data = current_tree_inputs[:, con_input_node:con_input_node + 1]#.to(self.args.device)

                    best_dict_con = self.tree_list[ii].nodes[node]['best_dict_con']
                    model_loader(self.cop_flow,
                                 self.args, node,
                                 best_dict_con['best_validation_epoch'],
                                 add_name='cop_con',
                                 send_to_device=True)
                    self.cop_flow.to(self.args.device)
                    self.cop_flow.eval()

                    pdf *= self.cop_flow.pdf_uniform(uncon_node_data, context=cond_node_data)
                    #pdf /= self.norm.pdf(uncon_node_data).reshape(-1,)
                    assert pdf.shape == (current_tree_inputs.shape[0],)

                        #counted.append(uncon_input_node)

                    # add ppf here!
                    uncon_node_data = torch.from_numpy(self.norm.ppf(uncon_node_data)).float().to(self.args.device)
                    cond_node_data = torch.from_numpy(self.norm.ppf(cond_node_data)).float().to(self.args.device)
                    transformed_inputs = self.cop_flow.transform_to_noise(uncon_node_data,
                                                                          cond_node_data)
                    transformed_inputs_uni = self.norm.cdf(transformed_inputs.cpu().numpy())
                    next_tree_inputs[:, uncon_input_node:uncon_input_node + 1] = transformed_inputs_uni
                    normal_distr = torch.distributions.normal.Normal(0, 1)
                    visualize_joint(normal_distr.cdf(torch.cat([uncon_node_data[:, 0:1], transformed_inputs], axis=1)).cpu(), self.args.figures_path, name='transform_{}'.format(node))
                current_tree_inputs = next_tree_inputs.copy()

            assert torch.min(pdf) >= 0
            return pdf.cpu().numpy()

    # def pdf_uniform(self, inputs, device=None):
    #     with torch.no_grad():
    #         return gaussian_change_of_var_ND(inputs, self.pdf_normal, self.args.device)

    def jsd_vinecopula(self, args, true_cop_distr, num_samples=10000, visualize=True):
        """Returns JS-Divergence of the predicted Copula and the true Copula.

        Params:
            args: passsed arguments
            true_cop_distr: rvine from which the dataset was created
            num_samples: how many num_sampleservations to create

        Returns:
            divergence: estimated JS-divergence
        """
        with torch.no_grad():
            # Samples from both distributinos
            samples_pred_uni = self.sample(num_samples=num_samples, transform=True)
            samples_target_uni = true_cop_distr.simulate(num_samples)

            # @Todo: remove before submitting code
            if visualize:
                visualize_joint(samples_pred_uni[:, :2].cpu(), self.args.figures_path, name='samples_pred01')
                visualize_joint(samples_target_uni[:, :2], self.args.figures_path, name='samples_target01')
                visualize_joint(samples_pred_uni[:, 1:3].cpu(), self.args.figures_path, name='samples_pred12')
                visualize_joint(samples_target_uni[:, 1:3], self.args.figures_path, name='samples_target12')
                visualize_joint(samples_pred_uni[:, 2:4].cpu(), self.args.figures_path, name='samples_pred23')
                visualize_joint(samples_target_uni[:, 2:4], self.args.figures_path, name='samples_target23')

                visualize_joint(torch.cat([samples_pred_uni[:, 0:1], samples_pred_uni[:, 2:3]], axis=1).cpu(), self.args.figures_path, name='samples_pred02')
                visualize_joint(np.concatenate([samples_target_uni[:, 0:1], samples_target_uni[:, 2:3]], axis=1), self.args.figures_path, name='samples_target02')
                visualize_joint(torch.cat([samples_pred_uni[:, 1:2], samples_pred_uni[:, 3:4]], axis=1).cpu(), self.args.figures_path, name='samples_pred13')
                visualize_joint(np.concatenate([samples_target_uni[:, 1:2], samples_target_uni[:, 3:4]], axis=1), self.args.figures_path, name='samples_target13')
                visualize_joint(torch.cat([samples_pred_uni[:, 0:1], samples_pred_uni[:, 3:4]], axis=1).cpu(), self.args.figures_path, name='samples_pred03')
                visualize_joint(np.concatenate([samples_target_uni[:, 0:1], samples_target_uni[:, 3:4]], axis=1), self.args.figures_path, name='samples_target03')

            if not args.error_bars:
                # @Todo: do with change of var
                calc_jsd(args, test_dict={}, samples_pred=samples_pred_uni[:, :2].cpu(), samples_target=samples_target_uni[:, :2], name='01')
                calc_jsd(args, test_dict={}, samples_pred=samples_pred_uni[:, 1:3].cpu(), samples_target=samples_target_uni[:, 1:3], name='12')
                calc_jsd(args, test_dict={}, samples_pred=samples_pred_uni[:, 2:4].cpu(), samples_target=samples_target_uni[:, 2:4], name='23')
                calc_jsd(args, test_dict={}, samples_pred=torch.cat([samples_pred_uni[:, 0:1], samples_pred_uni[:, 2:3]], axis=1).cpu(),
                         samples_target=np.concatenate([samples_target_uni[:, 0:1], samples_target_uni[:, 2:3]], axis=1), name='02')
                calc_jsd(args, test_dict={}, samples_pred=torch.cat([samples_pred_uni[:, 1:2], samples_pred_uni[:, 3:4]], axis=1).cpu(),
                         samples_target=np.concatenate([samples_target_uni[:, 1:2], samples_target_uni[:, 3:4]], axis=1), name='13')
                calc_jsd(args, test_dict={}, samples_pred=torch.cat([samples_pred_uni[:, 0:1], samples_pred_uni[:, 3:4]], axis=1).cpu(),
                         samples_target=np.concatenate([samples_target_uni[:, 0:1], samples_target_uni[:, 3:4]], axis=1), name='03')

                # calc_jsd(args, test_dict={}, samples_pred=samples_pred_uni.cpu(),
                #          samples_target=samples_target_uni, name='full')

            assert torch.max(samples_pred_uni) <= 1
            assert torch.min(samples_pred_uni) >= 0

            assert np.max(samples_target_uni) <= 1
            assert np.min(samples_target_uni) >= 0

            # Prob X in both distributions
            prob_X_in_p = self.pdf_uniform(samples_pred_uni.cpu().numpy())
            prob_X_in_q = true_cop_distr.pdf(samples_pred_uni.cpu().numpy())

            # Prob Y in both distributions
            prob_Y_in_p = self.pdf_uniform(samples_target_uni)
            prob_Y_in_q = true_cop_distr.pdf(samples_target_uni)

            print('prob_X_in_p', prob_X_in_p.mean())
            print('prob_X_in_q', prob_X_in_q.mean())
            print('prob_Y_in_p', prob_Y_in_p.mean())
            print('prob_Y_in_q', prob_Y_in_q.mean())

            assert np.min(prob_X_in_p) >= 0
            assert np.min(prob_X_in_q) >= 0
            assert np.min(prob_Y_in_p) >= 0
            assert np.min(prob_Y_in_q) >= 0

            assert prob_X_in_p.shape == (num_samples,), '{}'.format(prob_X_in_p.shape)
            assert prob_X_in_q.shape == (num_samples,)
            assert prob_Y_in_p.shape == (num_samples,)
            assert prob_Y_in_q.shape == (num_samples,)

            divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                       prob_X_in_q=prob_X_in_q,
                                       prob_Y_in_p=prob_Y_in_p,
                                       prob_Y_in_q=prob_Y_in_q)

            print('MC-JSD Vine Copula: {}'.format(divergence))

            self.results_dict['MC_JSD Vine Copula'] = divergence

            # RealNVP outputs the density directly, but not the transformation to
            # uniform marginals. Thus, an estimation with Gaussian KDE is simpler.
            pred_distr = scipy.stats.gaussian_kde(samples_pred_uni.cpu().numpy().T)
            true_rvine = scipy.stats.gaussian_kde(samples_target_uni.T)
            # Note, that uniform samples means the transformed samples

            # Prob X in both distributions
            prob_X_in_p = pred_distr.pdf(samples_pred_uni.cpu().numpy().T).T
            prob_X_in_q = true_rvine.pdf(samples_pred_uni.cpu().numpy().T).T

            # Prob Y in both distributions
            prob_Y_in_q = true_rvine.pdf(samples_target_uni.T).T
            prob_Y_in_p = pred_distr.pdf(samples_target_uni.T).T
            divergence_2 = js_divergence(prob_X_in_p=prob_X_in_p,
                                         prob_X_in_q=prob_X_in_q,
                                         prob_Y_in_p=prob_Y_in_p,
                                         prob_Y_in_q=prob_Y_in_q)

            print('KDE MC-JSD Vine Copula: {}'.format(divergence_2))

            self.results_dict['KDE MC_JSD Vine Copula'] = divergence_2

            return divergence

    def plot(self):
        """Plot the regular vine structure after sequential estimation.
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


def create_dataset(uncon_node_data, con_node_data, args):
    """Creates a two dimensional dataset as needed for CM Flows given the data from each edge.

    Params:
        uncon_node_data: data from first dimension
        con_node_data: data from second dimension
        args: passed arguments

    Returns:
        dataset: full dataset
        data_loaders: train and validation set data loaders. Test set is not needed at this stage.
    """
    dataset = Rvine_data(uncon_node_data, con_node_data)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_dataset = torch.utils.data.TensorDataset(dataset.trn)
    valid_dataset = torch.utils.data.TensorDataset(dataset.val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader}
    return dataset, data_loaders


class Rvine_data_1dim():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, inputs):
        self.xx = normalize_torch(inputs)
        trn, val = split_train_val_test(self.xx, only_val=True)

        self.trn = trn.float()
        self.val = val.float()


def create_dataset_1dim(inputs, args):
    """Creates a one dimensional dataset as needed for CM Flows given the data from each edge.

    Params:
        inputs: inputs data
        args: passed arguments

    Returns:
        dataset: full dataset
        data_loaders: train and validation set data loaders. Test set is not needed at this stage.
    """
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
