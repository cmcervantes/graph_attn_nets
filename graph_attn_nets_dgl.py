import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_func
import torch.nn.init as torch_init

from dgl import DGLGraph
from dgl.data import citation_graph

from collections import namedtuple
from time import time
import numpy as np


class GATLayer(torch_nn.Module):
    """
    Graph Attention Network layer; computes attention values over
    each node's neighborhood (adjacent nodes) and returns a node
    representation combining the node's original embedding with
    the attention-weighted neighbors
    """

    def __init__(self, graph, input_embed_len, output_embed_len, attn_dropout_proba=0.0):
        """
        Initializes the GATLayer for the given graph

        :param graph: graph on which this layer will be applied
        :param input_embed_len: the length of the input node embeddings
        :param output_embed_len: length of the node embeddings this layer
            will output
        :param attn_dropout_proba: probability that a normalized attention
            weight will be dropped
        """
        super(GATLayer, self).__init__()

        self._graph = graph

        # create a simple linear projection from the input node
        # embeddings to the output embeddings
        self._node_lin_transf = torch_nn.Linear(input_embed_len, output_embed_len, bias=False)

        # create the internal attention parameters (transforming the
        # concatenated node features to a raw scalar)
        self._attn_lin_transf = torch_nn.Linear(2 * output_embed_len, 1, bias=False)

        # create a dropout sub-module to apply to the attention weights
        self._attn_dropout = torch_nn.Dropout(attn_dropout_proba)

        # initialize our parameters
        torch_init.xavier_normal_(self._node_lin_transf.weight)
        torch_init.xavier_normal_(self._attn_lin_transf.weight)
    #enddef

    def _calc_edge_attn_fn(self, edges):
        """
        Function which calculates (raw) edge attention values
        :param edges: graph edges
        :return: dictionary of {'e': <edge_attn_vals>}
        """
        # concatenate the transformed node embeddings
        z_cat = torch.cat([edges.src['z'], edges.dst['z']], dim=1)

        # return the raw attention scalar, passed through a nonlinearity
        return {'e': torch_func.leaky_relu(self._attn_lin_transf(z_cat))}
    #enddef

    def _message_fn(self, edges):
        """
        Function for message collection (as part of the message passing mechanism)

        :param edges: graph edges
        :return: dictionary of {'z': <projected_node_embeds>, 'e': <edge_attn_vals>}
        """
        # message function to collect projected node embeddings and
        # raw attention values
        return {'z': edges.src['z'], 'e': edges.data['e']}
    #enddef

    def _reduce_fn(self, nodes):
        """
        Function for message reduction (as part of the message passing mechanism)

        :param nodes: graph nodes
        :return: dictionary of {'h': <output_embeddings>}
        """
        # calculate attention weights as the softmax of the raw values across
        # neighborhoods;        [n_nodes]
        attn_weights = self._attn_dropout(torch_func.softmax(nodes.mailbox['e'], dim=1))

        # return the new node representation: an attention weighted sum
        # of each node's neighborhood (including itself)
        return {'h': torch.sum(attn_weights * nodes.mailbox['z'], dim=1)}
    #enddef

    def forward(self, h):
        """
        Torch module forward pass function; computes the layer's output node
        embeddings, given the layer's input node embeddings

        :param h: input node embeddings
        :return: output node embeddings
        """
        # project the node into the output space, applying dropout as we do so
        self._graph.ndata['z'] = self._node_lin_transf(h)

        # use the built-in apply_edges functionality to
        # calculate edge attention values for all edges
        self._graph.apply_edges(self._calc_edge_attn_fn)

        # update the nodes according to the message and reduce
        # functions, for this pass
        self._graph.update_all(self._message_fn, self._reduce_fn)
        return self._graph.ndata.pop('h')
    #enddef

    def swap_graph(self, graph):
        """
        Helper function to switch out the layer's internal graph
        :param graph: Replacement for the current internal graph
        """
        self._graph = graph
    #enddef
#endclass


class MultiHeadGATLayer(torch_nn.Module):
    """
    Multi-headed Graph Attention Network layer, which combines
    multiple GATLayers
    """

    def __init__(self, graph, input_embed_len, output_embed_len, n_heads,
                 attn_dropout_proba=0.0, merge_mode='cat'):
        """
        Initializes the multi-headed graph attention layer, for the given graph

        :param graph: graph on which this layer will be applied
        :param input_embed_len: the length of the input node embeddings
        :param output_embed_len: length of the node embeddings this layer
            will output
        :param n_heads: number of heads (parallel GATLayers) to use
        :param attn_dropout_proba: probability that a normalized attention
            weight will be dropped
        :param merge_mode: how the multiple attention heads will be combined;
            one of ['cat', 'avg']
        """
        super(MultiHeadGATLayer, self).__init__()

        # create as many GATLayers as were specified
        self._attn_heads = torch_nn.ModuleList()
        for _ in range(n_heads):
            self._attn_heads.append(GATLayer(graph, input_embed_len, output_embed_len, attn_dropout_proba))

        # store our merge mode
        self._merge_mode = merge_mode
    #enddef

    def forward(self, h):
        """
        Torch module forward pass function; combines the GATLayers' output node
        embeddings, given the inputs

        :param h: input node embeddings
        :return: output node embeddings
        """
        # collect the outputs
        head_outputs = [attn_head(h) for attn_head in self._attn_heads]

        # merge them based on our merge mode (default is avg)
        if self._merge_mode == 'cat':
            return torch.cat(head_outputs, dim=1)
        return torch.mean(torch.stack(head_outputs), dim=0)
    #enddef

    def swap_graph(self, graph):
        """
        Helper function to switch out the layer's internal graph
        :param graph: Replacement for the current internal graph
        """
        for gat_layer in self._attn_heads:
            gat_layer.swap_graph(graph)
    #enddef
#endclass


class GAT(torch_nn.Module):
    """
    Graph Attention Network model, which applies the multi-headed GAT layer
    (over inputs) with a single GAT layer (over intermediate embeddings) to predict
    node classes
    """

    def __init__(self, graph, input_embed_len, hidden_embed_len, n_labels, n_heads,
                 input_dropout_proba=0.0, attn_dropout_proba=0.0, intermed_dropout_proba=0.0):
        """
        Initializes the GAT model, given the graph

        :param graph: graph on which this layer will be applied
        :param input_embed_len: the length of the input node embeddings
        :param hidden_embed_len: length of the node embeddings the multi-headed
            GAT layer will output
        :param n_labels: number of labels
        :param n_heads: number of heads (parallel GATLayers) to use
        :param input_dropout_proba: probability that an input feature will be dropped
        :param attn_dropout_proba: probability that a normalized attention
            weight will be dropped
        :param intermed_dropout_proba: probability that one of the features of the
            intermediate node embeddings will be dropped
        """
        super(GAT, self).__init__()

        # the first layer uses the multi-headed GAT approach to transform node embeddings
        # from input_embed_len to hidden_embed_len (attending over neighborhoods as it does so)
        self._multi_headed_gat_layer = MultiHeadGATLayer(
            graph, input_embed_len, hidden_embed_len, n_heads, attn_dropout_proba)

        # the second layer transforms those new embeddings into the label space, which
        # can be done with any kind of layer, but for simplicity we'll reuse our attention setup
        self._label_proj_layer = GATLayer(graph, hidden_embed_len * n_heads, n_labels)

        # define two dropout sub-modules for the inputs and intermediate embeddings, respectively
        self._input_dropout = torch_nn.Dropout(input_dropout_proba)
        self._intermed_dropout = torch_nn.Dropout(intermed_dropout_proba)
    #enddef

    def forward(self, h):
        """
        Torch module forward pass function; passing the inputs through the multi-headed
        GAT layer, a nonlinearity, and projects those intermediate representations into
        the label space

        :param h: input node embeddings
        :return: label distribution (raw logits)
        """
        # in each step, we want to
        #   1) apply dropout over inputs
        #   2) attend over input embeddings' neighborhoods
        #   3) pass those embeddings through a nonlinearity
        #   4) apply dropout to those intermediate embeddings
        #   5) project those new embeddings into the label space
        intermed = torch_func.elu(self._multi_headed_gat_layer(self._input_dropout(h)))
        return self._label_proj_layer(self._intermed_dropout(intermed))
    #enddef
#endclass


def load_citation_graph(graph_name):
    """
    Loads one of the DGL-hosted citation graph datasets

    :param graph_name: name of the citation graph to load; one of
        ['cora', 'citeseer', 'pubmed']
    :return: namedtuple for the citation graph dataset; attributes:
        [graph, features, labels, mask]
    """
    # retrieve the dataset
    if graph_name == 'cora':
        dataset = citation_graph.load_cora()
    elif graph_name == 'citeseer':
        dataset = citation_graph.load_citeseer()
    elif graph_name == 'pubmed':
        dataset = citation_graph.load_pubmed()
    else:
        raise ValueError("Unknown citation graph name <{:s}>; "
                         "Expected one of [cora, citeseer, pubmed]".format(graph_name))
    #endif

    # return the datasets' components
    dataset_tuple = namedtuple("citation_graph", ["graph", "features", "labels", "mask"])
    return dataset_tuple(DGLGraph(dataset.graph), torch.FloatTensor(dataset.features),
                         torch.LongTensor(dataset.labels), torch.BoolTensor(dataset.train_mask))
#enddef


def train_eval_citation_graph():
    import matplotlib.pyplot as plt
    import networkx as nx

    hidden_embed_len = 8
    n_heads = 8
    n_epochs = 100
    lrn_rate = 5e-3
    train_weight_decay = 5e-4
    input_dropout_proba, attn_dropout_proba, intermed_dropout_proba = 0.6, 0.6, 0.6

    # load the dataset, collecting data-dependent variables
    data_vars = load_citation_graph('cora')
    n_nodes = data_vars.features.size()[0]
    input_embed_len = data_vars.features.size()[1]
    n_labels = int(max(data_vars.labels)) + 1

    '''
    fig, ax = plt.subplots()

    herecolors = ['#48DAD0', '#673A93', '#C41C33', '#3F59A7',
                  '#EC610E', '#FAB800', '#70943C']
    rand_indices = [i for i in range(n_nodes)]
    np.random.shuffle(rand_indices)
    rand_indices = set(rand_indices[:1000])
    rand_mask = [i in rand_indices for i in range(n_nodes)]


    rand_nodes = data_vars.graph.nodes()[rand_mask]
    subgraph = data_vars.graph.subgraph(rand_nodes).to_networkx().to_undirected()
    #full_graph = data_vars.graph.to_networkx().to_undirected()
    positions = nx.spring_layout(subgraph, k=0.0625)

    gold_colors = [herecolors[l] for l in data_vars.labels[rand_mask]]
    nx.draw(subgraph, node_color=gold_colors, with_labels=False,
            node_size=10, ax=ax, pos=positions)
    plt.show()
    quit()
    '''


    # create a new model
    gat_model = GAT(data_vars.graph, input_embed_len=input_embed_len,
                    hidden_embed_len=hidden_embed_len, n_labels=n_labels,
                    n_heads=n_heads, input_dropout_proba=input_dropout_proba,
                    attn_dropout_proba=attn_dropout_proba,
                    intermed_dropout_proba=intermed_dropout_proba)

    # create an optimizer for the model's parameters
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=lrn_rate,
                                 weight_decay=train_weight_decay)

    # train
    for epoch_idx in range(n_epochs):
        # run our model, getting 1) raw logits 2) log probabilities, and
        # 3) calculating negative log likelihood loss over our training samples
        logits = gat_model(data_vars.features)
        log_prob = torch_func.log_softmax(logits, 1)
        loss = torch_func.nll_loss(log_prob[data_vars.mask], data_vars.labels[data_vars.mask])

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch_idx + 1) % 10 == 0:
            print("Epoch: {:02d}; Loss: {:.03f}".format(epoch_idx+1, loss.item()))
    #endfor

    eval_indices = [i for i in range(n_nodes) if not data_vars.mask[i]]
    np.random.shuffle(eval_indices)
    eval_indices = set(eval_indices[:1000])
    eval_mask = [i in eval_indices for i in range(n_nodes)]

    # evaluate
    gat_model.eval()
    with torch.no_grad():
        logits = gat_model(data_vars.features)[eval_mask]
        y = data_vars.labels[eval_mask]
        _, y_pred = torch.max(logits, dim=1)
        pred_labels = list(y_pred.numpy())
        print("Accuracy: {:.2f}%".format(100.0 * torch.sum(y_pred == y).item() * 1.0 / len(y)))
    #endwith

    # We're going to draw the evaluation graph twice; once with the gold
    # label colorings and once with the predicted
    eval_nodes = data_vars.graph.nodes()[eval_mask]
    eval_graph = data_vars.graph.subgraph(eval_nodes).to_networkx().to_undirected()

    herecolors = ['#48DAD0', '#673A93', '#C41C33', '#3F59A7',
                  '#EC610E', '#FAB800', '#70943C']
    figs, axes = plt.subplots(nrows=1, ncols=2)
    positions = nx.spring_layout(eval_graph, k=0.0625)

    gold_colors = [herecolors[l] for l in data_vars.labels[eval_mask]]
    pred_colors = [herecolors[l] for l in pred_labels]
    axes[0].set_title("Gold")
    nx.draw(eval_graph, node_color=gold_colors, with_labels=False,
            node_size=10, ax=axes[0], pos=positions)
    axes[1].set_title("Predicted")
    nx.draw(eval_graph, node_color=pred_colors, with_labels=False,
            node_size=10, ax=axes[1], pos=positions)
    plt.show()
#enddef


def load_ppi(batch_size):
    """
    Loads the DGL-hosted PPI dataset

    :param batch_size: number of samples in each batch
    :return: namedtuple for the PPI graph dataset; attributes:
        [input_embed_len, n_labels, train_graph, train_loader, test_loader]
    """
    from dgl.data.ppi import LegacyPPIDataset
    from torch.utils.data import DataLoader
    from dgl import batch as dgl_batch

    def collate_fn(sample):
        """
        Helper function for the torch dataloader
        """
        graphs, feats, labels = map(list, zip(*sample))
        graph = dgl_batch(graphs)
        feats = torch.from_numpy(np.concatenate(feats))
        labels = torch.from_numpy(np.concatenate(labels))
        return graph, feats, labels
    #enddef

    # create and return the dataset
    train_data = LegacyPPIDataset(mode='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(LegacyPPIDataset(mode='test'), batch_size=batch_size,
                             collate_fn=collate_fn)
    data_tuple = namedtuple("data_vars", ["input_embed_len", "n_labels", "train_graph",
                                          "train_loader", "test_loader"])
    return data_tuple(train_data.features.shape[1], train_data.labels.shape[1],
                      train_data.graph, train_loader, test_loader)
#enddef


class ThreeLayerGAT(torch_nn.Module):
    """
    Graph Attention Network model for a three-layer GAT as specified by the
    inductive learning setup in the original GAT paper; two multi-headed layers
    are used before passing the resulting embeddings to a final layer which predicts
    node classes
    """

    def __init__(self, graph, input_embed_len, hidden_embed_len, n_labels,
                 n_start_heads, n_end_heads):
        """
        Initializes the GAT model, given the graph

        :param graph: graph on which this layer will be applied
        :param input_embed_len: the length of the input node embeddings
        :param hidden_embed_len: length of the node embeddings the multi-headed
            GAT layer will output
        :param n_labels: number of labels
        :param n_start_heads: number of heads (parallel GATLayers) to use in the
            first two layers
        :param n_end_heads: number of heads (parallel GATLayers) to use in the
            final layer
        """
        super(ThreeLayerGAT, self).__init__()

        # the first layer uses the multi-headed GAT approach to transform node embeddings
        # from input_embed_len to hidden_embed_len (attending over neighborhoods as it does so)
        self._multi_headed_gat_layer_1 = MultiHeadGATLayer(
            graph, input_embed_len, hidden_embed_len, n_start_heads, merge_mode='cat')
        multiheaded_emebed_len = hidden_embed_len * n_start_heads

        # the second layer also uses the multi-headed GAT approach, but does not change the
        # length of the node representations
        self._multi_headed_gat_layer_2 = MultiHeadGATLayer(
            graph, multiheaded_emebed_len, hidden_embed_len, n_start_heads, merge_mode='cat')

        # the final layer transforms those new embeddings into the label space, again using
        # the multi-headed GAT approach
        self._multi_headed_gat_layer_3 = MultiHeadGATLayer(
            graph, multiheaded_emebed_len, n_labels, n_end_heads, merge_mode='avg')
    #enddef

    def forward(self, h):
        """
        Torch module forward pass function; passing the inputs through the multi-headed
        GAT layers to predict labels

        :param h: input node embeddings
        :return: label distribution (raw logits)
        """
        intermed_1 = torch_func.elu(self._multi_headed_gat_layer_1(h))
        intermed_2 = torch_func.elu(self._multi_headed_gat_layer_2(intermed_1))
        return self._multi_headed_gat_layer_3(intermed_2)
    #enddef

    def swap_graph(self, graph):
        """
        Helper function to switch out the layer's internal graph
        :param graph: Replacement for the current internal graph
        """
        self._multi_headed_gat_layer_1.swap_graph(graph)
        self._multi_headed_gat_layer_2.swap_graph(graph)
        self._multi_headed_gat_layer_3.swap_graph(graph)
    #enddef
#endclass


def train_eval_ppi(train=True, eval=True):
    from sklearn.metrics import f1_score as sk_f1

    hidden_embed_len = 256
    n_start_heads = 4
    n_end_heads = 6
    n_epochs = 10
    lrn_rate = 5e-3
    batch_size = 2

    # load the dataset
    data_vars = load_ppi(batch_size)

    if train:
        # create a new model
        gat_model = ThreeLayerGAT(data_vars.train_graph, input_embed_len=data_vars.input_embed_len,
                                  hidden_embed_len=hidden_embed_len, n_labels=data_vars.n_labels,
                                  n_start_heads=n_start_heads, n_end_heads=n_end_heads)

        # create an optimizer for the model's parameters
        optimizer = torch.optim.Adam(gat_model.parameters(), lr=lrn_rate)

        # train
        start_ts = time()
        for epoch_idx in range(n_epochs):
            gat_model.train()
            losses = []
            for batch_idx, data in enumerate(data_vars.train_loader):
                subgraph, feats, labels = data

                # swap the model's graph for this batch
                gat_model.swap_graph(subgraph)

                # predict and compute Binary Cross Entropy loss
                logits = gat_model(feats.float())
                loss = torch_nn.BCEWithLogitsLoss()(logits, labels.float())
                losses.append(loss.item())

                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #endfor

            elapsed_time = time() - start_ts
            print("Epoch: {:02d}; Avg. Loss: {:.03f}; Time Elapsed: {:d}m {:d}s".format(
                epoch_idx+1, np.mean(losses), int(elapsed_time // 60), int(elapsed_time % 60)))
        #endfor

        # save the model
        torch.save(gat_model.state_dict(), "models/three_layer_gat.pt")
    #endif

    if eval:
        # load where we're not training
        gat_model = ThreeLayerGAT(graph=None, input_embed_len=data_vars.input_embed_len,
                                  hidden_embed_len=hidden_embed_len, n_labels=data_vars.n_labels,
                                  n_start_heads=n_start_heads, n_end_heads=n_end_heads)
        gat_model.load_state_dict(torch.load("models/three_layer_gat.pt"))


        # evaluate
        test_f1s = []
        gat_model.eval()
        with torch.no_grad():
            for batch_idx, test_data in enumerate(data_vars.test_loader):
                subgraph, feats, labels = test_data
                gat_model.swap_graph(subgraph)
                logits = gat_model(feats.float())
                y_pred = np.where(logits.data.numpy() >= 0, 1, 0)
                test_f1s.append(sk_f1(labels.data.numpy(), y_pred, average='micro'))
            #endfor
        #endwith
        print("Micro-Averaged F1: {:.2f}%".format(100.0 * np.mean(test_f1s)))
    #endif
#enddef

if __name__ == '__main__':
    #train_eval_ppi()
    train_eval_citation_graph()