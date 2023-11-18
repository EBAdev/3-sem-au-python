from typing import List, Tuple, Union, Dict
import networkx as nx

Num = Union[int, float]
Arc = Tuple[int, int]


def create_max_flow_network(
    nodes: List[int],
    edges: List[Tuple[int, int, Num]],
    source_node: int,
    sink_node: int,
):
    G = nx.DiGraph()
    G.add_nodes_from(nodes, source_node=False, sink_node=False)
    G.nodes[source_node]["source_node"] = True
    G.nodes[sink_node]["sink_node"] = True
    print(edges)
    for edge in edges:
        G.add_edge(edge[0], edge[1], capacity=edge[2], flow=0)
        G.add_edge(edge[1], edge[0], capacity=0, flow=0)
    for idx, layer in enumerate(nx.bfs_layers(G, source_node)):
        for n in layer:
            G.nodes[n]["layer"] = idx
    return G


def check_labels(num_nodes: int, node_labels: Dict) -> Union[bool, ValueError]:
    if len(node_labels) != num_nodes:
        return ValueError(
            "\nThe number of nodes given and the amount of node labels does not match.\nThere should be a label for each node."
        )
    return True


def breath_first_search(network: nx.DiGraph, source, sink):
    T = nx.DiGraph()
    T = network.copy()
    T.remove_edges_from(network.edges())

    # add all edges to the tree with positive capacity
    for n in list(T.nodes()):
        for edge in network.edges(n):  # all edges going away from n
            edge_data = network.edges[edge]
            if edge_data["capacity"] == 0:
                continue
            T.add_edge(*edge, capacity=edge_data["capacity"], flow=edge_data["flow"])
    # add edge from sink to tree to create cycle if there is a path
    # T.add_edge(sink, source)

    return T


def push_flow_along_path(network: nx.DiGraph, shortest_path):
    network_copy = network.copy()
    path = [(*arc, network_copy.edges[arc]) for arc in shortest_path]
    lowest_capacity = min([arc[2]["capacity"] for arc in path])
    for arc in path:
        network_copy.edges[arc[0], arc[1]]["capacity"] += -lowest_capacity
        network_copy.edges[arc[0], arc[1]]["flow"] += lowest_capacity
    return (network_copy, lowest_capacity)


def flow_to_sink(network: nx.DiGraph, sink):
    incident_edges = []
    for n in network.neighbors(sink):
        n_edges = list(network.edges(n))
        for edge in n_edges:
            if edge[1] == sink:
                incident_edges += [(edge[0], edge[1], network.edges[edge[0], edge[1]])]
                break
    flows = [edge[2]["flow"] for edge in incident_edges]
    return sum(flows)


def max_flow_residual_network_to_latex(
    network: nx.DiGraph,
    caption: str = "",
    node_labels: Dict = {},
    layer_key: str = "layer",
    scale: int = 2,
    latex_label: str = "",
) -> str:
    """
    Function to return a Tikz figure of the residual network, given a maximum flow network.
    #### Parameters
    1. network:nx.DiGraph [required]
            * A directed network corresponding to the maximum flow problem.
              Each node in the network must contain an attribute of value integer corresponding to the layer
              of the node in a breath first search.
    2. caption:str
            * The caption of the LaTeX figure, if none is provided a caption will not be written.
    3. node_labels:Dict
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format.
    4. layer_key:str = "layer"
            * The attribute name in the node generator of the network, which the layer integer.
    5. scale:int = 2
            * a integer value which changes the size of the picture.
    6. latex_label:str =""
            * The label used by latex to reference the graph
    #### Useful knowledge
    The nodes will be placed on a horizontal line, however the edges are not taken into account. Therefore it might be nesseary to move some nodes up or down. We can also bend lines using the option [bend right=int] at the start of the edge, and we can move labels below a line by setting the parameter below.
    """
    G = nx.Graph()
    G.add_nodes_from(network.nodes(data=True))  # make undirected graph

    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    for edge in network.edges(data=True):
        if edge[0] > edge[1]:
            continue  # if we have caught the reverse edge
        G.add_edge(
            edge[0],
            edge[1],
            out_residual=edge[2]["capacity"],
            in_residual=edge[2]["flow"],
        )
    edge_labels = {}
    edge_label_opt = {}

    pos = nx.multipartite_layout(G, scale=scale, subset_key=layer_key)

    for arc in G.edges():
        data = G.edges[arc[0], arc[1]]
        label = (
            "$r_{"
            + str(arc[0])
            + "}="
            + str(data["out_residual"])
            + "\\quad r_{"
            + str(arc[1])
            + "}="
            + str(data["in_residual"])
            + "$"
        )
        edge_labels.update({arc: label})
        edge_label_opt.update({arc: "[pos=.5,scale=0.50,above,sloped]"})

    return nx.to_latex(
        G,
        pos=pos,
        as_document=False,
        caption=caption,
        figure_wrapper="\\begin{{figure}}[H]\n\\centering\n{content}{caption}{label}\n\\end{{figure}}",
        node_label=node_labels,
        tikz_options="scale=" + str(scale),
        edge_label=edge_labels,
        edge_label_options=edge_label_opt,
        latex_label=latex_label,
    )


def bfs_network_to_latex(
    network: nx.DiGraph,
    caption: str = "",
    node_labels: Dict = {},
    layer_key: str = "layer",
    scale: int = 2,
    latex_label: str = "",
) -> str:
    """
    Function to return a Tikz figure of the breath first search network.
    #### Parameters
    1. network:nx.DiGraph [required]
            * A directed network corresponding to the maximum flow problem.
              Each node in the network must contain an attribute of value integer corresponding to the layer
              of the node in a breath first search.
    2. caption:str
            * The caption of the LaTeX figure, if none is provided a caption will not be written.
    3. node_labels:Dict
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format.
    4. layer_key:str = "layer"
            * The attribute name in the node generator of the network, which the layer integer.
    5. scale:int = 2
            * a integer value which changes the size of the picture.
    6. latex_label:str =""
            * The label used by latex to reference the graph
    #### Useful knowledge
    The nodes will be placed on a horizontal line, however the edges are not taken into account. Therefore it might be nesseary to move some nodes up or down. We can also bend lines using the option [bend right=int] at the start of the edge, and we can move labels below a line by setting the parameter below.
    """
    G = nx.DiGraph()
    G.add_nodes_from(network.nodes(data=True))  # make directed graph

    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    for edge in network.edges(data=True):
        if edge[2]["capacity"] == 0:  # if arc has no flow
            continue  # if we have caught the reverse edge
        G.add_edge(
            edge[0],
            edge[1],
            out_residual=edge[2]["capacity"],
            current_flow=edge[2]["flow"],
        )
    edge_labels = {}
    edge_label_opt = {}

    pos = nx.multipartite_layout(G, scale=scale, subset_key=layer_key)

    for arc in G.edges():
        data = G.edges[arc[0], arc[1]]
        label = (
            "$r_{"
            + str(arc[0])
            + "}="
            + str(data["out_residual"])
            + "\\quad f_{"
            + str(arc[0])
            + ","
            + str(arc[1])
            + "}="
            + str(data["current_flow"])
            + "$"
        )
        edge_labels.update({arc: label})
        edge_label_opt.update({arc: "[pos=.5,scale=0.50,above,sloped]"})

    return nx.to_latex(
        G,
        pos=pos,
        as_document=False,
        caption=caption,
        figure_wrapper="\\begin{{figure}}[H]\n\\centering\n{content}{caption}{label}\n\\end{{figure}}",
        node_label=node_labels,
        tikz_options="scale=" + str(scale),
        edge_label=edge_labels,
        edge_label_options=edge_label_opt,
        latex_label=latex_label,
    )


def max_flow_network_simplex(
    nodes: List[int],
    edges: List[Tuple[int, int, Num]],
    source_node: int,
    sink_node: int,
    node_labels: Dict = {},
):
    """
    Function to calculate the max flow network from source node to sink node, using ford fulkersons algoritm.
    #### Parameters
    1. nodes: List[int] [required]
            * A list of integers starting from 1 to m representing where m is number of nodes.
    2. edges: List[Tuple[int,int,Num]] [required]
            * A list of edges in the max flow network where each element is a tuple of tree digits. The first is the arcs origin, the second is the arcs destination, and the third is the capacity along the arc.
    3. source_node:int [required]
            * The source node number.
    4. sink_node:int [required]
            * The sink node number..
    5. node_labels: Dict = {}
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format, used for printing.
    """
    G = create_max_flow_network(nodes, edges, source_node, sink_node)
    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    T = breath_first_search(G, source_node, sink_node)

    len_of_paths = [
        len(path) for path in nx.all_simple_edge_paths(T, source_node, sink_node)
    ]

    iteration_counter = 0
    while len_of_paths != []:
        iteration_counter += 1
        shortest_path_idx = len_of_paths.index(min(len_of_paths))

        for idx, path in enumerate(nx.all_simple_edge_paths(T, source_node, sink_node)):
            if idx == shortest_path_idx:
                shortest_path = path
                break
        flow_push = push_flow_along_path(G, shortest_path)
        G = flow_push[0]
        added_flow_to_sink = flow_push[1]
        T = breath_first_search(G, source_node, sink_node)

        len_of_paths = [
            len(path) for path in nx.all_simple_edge_paths(T, source_node, sink_node)
        ]

    return G


if __name__ == "__main__":
    # example
    nodes_inp = [1, 2, 3, 4, 5, 6, 7]

    edge_inp = [
        (1, 2, 5),
        (1, 4, 7),
        (1, 3, 4),
        (2, 4, 1),
        (2, 5, 3),
        (3, 6, 4),
        (4, 3, 2),
        (4, 5, 4),
        (4, 6, 5),
        (5, 7, 9),
        (6, 5, 1),
        (6, 7, 6),
    ]
    so = 1
    si = 7
    labels = {
        1: "tivoli",
        2: "busstop",
        3: "skole",
        4: "arbejde",
        5: "kirke",
        6: "havn",
        7: "hjem",
    }

    optimal = max_flow_network_simplex(nodes_inp, edge_inp, 1, 7)
