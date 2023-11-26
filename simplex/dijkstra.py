from typing import Union, List, Tuple, Dict
import networkx as nx
import numpy as np


Num = Union[float, int]


def create_dijkstra_network(
    nodes: List[int], arcs: List[Tuple[int, int, Num]], path_to_node: int
) -> nx.DiGraph:
    """
    For displaying the nodes in certain layers, we use bfs this takes a list of "source nodes" which is given here. Change this to change what we start at.
    """
    G = nx.Graph()
    sources = [path_to_node]

    for node in nodes:
        G.add_node(node, path_to_source=None)
    G.nodes[path_to_node]["path_to_source"] = 0

    for arc in arcs:
        G.add_edge(arc[0], arc[1], length=arc[2], incident=False, optimal=False)
    for neigbor in nx.all_neighbors(G, path_to_node):
        G.edges[neigbor, path_to_node]["incident"] = True
    for idx, layer in enumerate(nx.bfs_layers(G, sources)):
        for node in layer:
            G.nodes[node]["layer"] = idx
    return G


def negative_length(arcs: List[Tuple[int, int, Num]]) -> bool:
    for edge in arcs:
        if edge[2] < 0:
            return True
    return False


def check_labels(num_nodes: int, node_labels: Dict) -> Union[bool, ValueError]:
    if len(node_labels) != num_nodes:
        return ValueError(
            "\nThe number of nodes given and the amount of node labels does not match.\nThere should be a label for each node."
        )
    return True


def update_path(G: nx.Graph, shortest_edge: Tuple[int, int, Num]):
    H = G.copy()
    if H.nodes[shortest_edge[0]]["path_to_source"] is None:
        node_to_add = shortest_edge[0]
    else:
        node_to_add = shortest_edge[1]

    # update p-val and incidence of entering node
    H.nodes[node_to_add]["path_to_source"] = shortest_edge[2]
    H.remove_edges_from(G.edges())
    for edge in G.edges(data=True):
        opt = edge[2]["optimal"]
        if (shortest_edge[0], shortest_edge[1]) == (edge[0], edge[1]):
            opt = True
        H.add_edge(
            edge[0], edge[1], length=edge[2]["length"], optimal=opt, incident=False
        )
    nodes_pathed = [
        node[0] for node in H.nodes(data=True) if node[1]["path_to_source"] is not None
    ]

    # find all neighbors of added node and update their incidency.
    for pnode in nodes_pathed:
        for node in H.nodes():
            if (
                pnode not in list(nx.neighbors(G, node))  # if not a neighbor
                or H.nodes[node]["path_to_source"] is not None  # of if already pathed
            ):
                continue

            H.edges[node, pnode]["incident"] = True
    return H


def dijkstra_network_to_latex(
    G: nx.Graph,
    node_labels: Dict,
    caption: str = "",
    layer_key: str = "layer",
    scale: int = 2,
    latex_label: str = "",
):
    """
    Function to return a Tikz figure of the djikstra-problem.
    #### Parameters
    1. network:nx.DiGraph [required]
            * A undirected network corresponding to the djikstra problem.
              Each node in the network must contain an attribute of value integer corresponding to the layer
              of the node in a breath first search with origin of the suppliers.
    2. node_labels:Dict
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format.
    3. caption:str
            * The caption of the LaTeX figure, if none is provided a caption will not be written.
    4. layer_key:str = "layer"
            * The attribute name in the node generator of the network, which the layer integer.
    5. scale:int = 2
            * a integer value which changes the size of the picture.
    6. latex_label:str =""
            * The label used by latex to reference the graph
    #### Useful knowledge
    The nodes will be placed on a horizontal line, however the edges are not taken into account. Therefore it might be nesseary to move some nodes up or down. We can also bend lines using the option [bend right=int] at the start of the edge, and we can move labels below a line by setting the parameter below.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for edge in G.edges(data=True):
        if edge[2]["optimal"] is False:
            continue
        H.add_edge(edge[0], edge[1], length=edge[2]["length"])

    edge_labels = {}
    edge_label_opt = {}
    pos = nx.multipartite_layout(H, scale=scale, subset_key=layer_key)

    for edge in H.edges():
        data = H.edges[edge[0], edge[1]]
        label = (
            "$c_{"
            + str(edge[0])
            + ","
            + str(edge[1])
            + "}="
            + str(data["length"])
            + "$"
        )
        edge_labels.update({edge: label})
        edge_label_opt.update({edge: "[pos=.5,scale=0.50,above,sloped]"})
    return nx.to_latex(
        H,
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


def dijkstra_shortest_path(
    nodes: List[int],
    arcs: List[Tuple[int, int, Num]],
    path_to_node: int,
    node_labels: Dict = {},
    LaTex: bool = False,
):
    """
    Function to calculate shortest paths from all nodes to one node using the djikstra algorithm.
    Note that this requires only positive costs
    #### Parameters
    1. nodes: List[int] [required]
            * A list of integers each representing a node.
    2. arcs: List[Tuple[int,int,Num]] [required]
            * A list of the edges in the bellman-ford problem where each element is a tuple of tree digits. The first is the arcs origin, the second is the arcs destination, and the third is the cost of traversal along the arc, i.e. the length.
    3. path_to_node:int[required]
            * An integer value corresponding to the node to which paths are calculated
    4. node_labels: Dict = {}
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format, used for printing.
    5. LaTex:bool = False
            * A boolean value used to determine if latex printing is needed.
    """
    G = create_dijkstra_network(nodes, arcs, path_to_node)

    if negative_length(arcs) is True:
        raise ValueError(
            "The provided values cannot be used with djikstas algoritm.\nAt least one of the provided lengths of the arcs is negative."
        )
    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    pathed_nodes = [
        node for node in G.nodes(data=True) if node[1]["path_to_source"] is not None
    ]
    while len(pathed_nodes) != G.number_of_nodes():
        incident_edges = [
            edge for edge in G.edges(data=True) if edge[2]["incident"] == True
        ]
        to_minimize = []
        for edge in incident_edges:
            if G.nodes[edge[0]]["path_to_source"] is not None:
                p_val = G.nodes[edge[0]]["path_to_source"]

            else:
                p_val = G.nodes[edge[1]]["path_to_source"]

            to_minimize.append((edge[0], edge[1], edge[2]["length"] + p_val))
        shortest_edge = to_minimize[
            [tmin[2] for tmin in to_minimize].index(
                min([tmin[2] for tmin in to_minimize])
            )
        ]
        G = update_path(G, shortest_edge)
        pathed_nodes = [
            node for node in G.nodes(data=True) if node[1]["path_to_source"] is not None
        ]
        if LaTex:
            print(dijkstra_network_to_latex(G, node_labels, "test"))
    return G


if __name__ == "__main__":
    nodes_inp = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    arcs_inp = [
        (9, 2, 4),
        (9, 8, 8),
        (8, 2, 11),
        (8, 7, 1),
        (8, 1, 7),
        (2, 3, 8),
        (3, 1, 2),
        (3, 4, 7),
        (3, 6, 4),
        (1, 7, 6),
        (7, 6, 2),
        (6, 4, 14),
        (6, 5, 10),
        (4, 5, 9),
    ]
    path_to_node = 9
    dijkstra_shortest_path(nodes_inp, arcs_inp, path_to_node, LaTex=False)
