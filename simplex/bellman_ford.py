from typing import Union, List, Tuple, Dict
import networkx as nx
import numpy as np


Num = Union[float, int]


def create_bellman_ford_network(
    nodes: List[int], arcs: List[Tuple[int, int, Num]], path_to_node: int
) -> nx.DiGraph:
    """
    For displaying the nodes in certain layers, we use bfs this takes a list of "source nodes" which is given here. Change this to change what we start at.
    """
    G = nx.DiGraph()
    sources = [path_to_node]

    for node in nodes:
        G.add_node(node)

    for arc in arcs:
        G.add_edge(arc[0], arc[1], length=arc[2])

    for idx, layer in enumerate(nx.bfs_layers(G, sources)):
        for node in layer:
            G.nodes[node]["layer"] = idx
    return G


def negative_cycle(G: nx.DiGraph) -> bool:
    """
    Determine if there is a negative cycle, if it exists, return it otherwise return False
    """
    for node in G.nodes():
        try:
            nx.find_negative_cycle(G, node, "length")
        except nx.exception.NetworkXError:  # if there was no negative cycle
            continue
        else:
            return nx.find_negative_cycle(
                G, node, "length"
            )  # return the negative cycle
    return False


def get_shortest_paths(G: nx.DiGraph, path_to_node: int, shortest_path, pathed_nodes):
    """
    Function to determine a vector of the shortest paths, and their costs.
    """
    for node in G.nodes():  # for every node
        connected_nodes = [n for n in nx.neighbors(G, node)]  # get all neigbors
        to_minimize = []
        for pnode in pathed_nodes:  # if pathed node is not in connected nodes
            if pnode not in connected_nodes:
                continue
            to_minimize.append(
                (
                    node,
                    pnode,
                    G.edges[node, pnode]["length"] + shortest_path[pnode - 1][2],
                )
            )
        if not to_minimize:  # if no values were appended
            continue
        min_len = to_minimize[
            [tmin[2] for tmin in to_minimize].index(
                min([tmin[2] for tmin in to_minimize])
            )
        ]  # get the lowest length in to_minimize

        if shortest_path[min_len[0] - 1][2] > min_len[2]:  # if new paths are better
            shortest_path[min_len[0] - 1] = (
                min_len[0],
                min_len[1],
                min_len[2],
            )  # update the path in shortest path
        if node not in pathed_nodes:
            pathed_nodes.append(min_len[0])
    return (shortest_path, pathed_nodes)


def check_labels(num_nodes: int, node_labels: Dict) -> Union[bool, ValueError]:
    if len(node_labels) != num_nodes:
        return ValueError(
            "\nThe number of nodes given and the amount of node labels does not match.\nThere should be a label for each node."
        )
    return True


def bellman_ford_network_to_latex(
    G: nx.DiGraph,
    node_labels: Dict,
    caption: str = "",
    layer_key: str = "layer",
    scale: int = 2,
    latex_label: str = "",
):
    """
    Function to return a Tikz figure of the bellman-ford-problem.
    #### Parameters
    1. network:nx.DiGraph [required]
            * A directed network corresponding to the bellman-ford problem.
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
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    for node in H.nodes(data=True):
        if node[0] == node[1]["path to"]:
            continue
        H.add_edge(
            node[0],
            node[1]["path to"],
            length=G.edges[node[0], node[1]["path to"]]["length"],
        )

    edge_labels = {}
    edge_label_opt = {}
    pos = nx.multipartite_layout(H, scale=scale, subset_key=layer_key)

    for arc in H.edges():
        data = H.edges[arc[0], arc[1]]
        label = (
            "$c_{" + str(arc[0]) + "," + str(arc[1]) + "}=" + str(data["length"]) + "$"
        )
        edge_labels.update({arc: label})
        edge_label_opt.update({arc: "[pos=.5,scale=0.50,above,sloped]"})
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


def bellman_ford_shortest_path(
    nodes: List[int],
    arcs: List[Tuple[int, int, Num]],
    path_to_node: int,
    node_labels: Dict = {},
    LaTex: bool = False,
):
    """
    Function to calculate shortest paths from all nodes to one node using the bellman-ford algorithm.
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
    G = create_bellman_ford_network(nodes, arcs, path_to_node)

    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    if negative_cycle(G) is not False:
        return ("There is a negative cycle", negative_cycle(G))
    # if there are no negative cycles the longest a path can be is number of nodes - 1
    longest_path_num = G.number_of_nodes() - 1

    shortest_path = [
        (i + 1, np.nan, np.inf) if i + 1 != path_to_node else (i + 1, path_to_node, 0)
        for i in range(G.number_of_nodes())
    ]  # The inital shortest paths are infinity since we can only go from 6 to 6 in zero arcs
    pathed_nodes = [path_to_node]  # therefore we have only pathed the original node

    for iteration in range(longest_path_num):
        update = get_shortest_paths(G, path_to_node, shortest_path, pathed_nodes)
        shortest_path = update[0]
        pathed_nodes = update[1]

        for path in shortest_path:
            G.nodes[path[0]]["path to"] = path[1]
            G.nodes[path[0]]["shortest path"] = path[2]

    if LaTex:
        print(
            bellman_ford_network_to_latex(
                G,
                node_labels,
                "test",
            )
        )
    return shortest_path


if __name__ == "__main__":
    nodes_inp = [1, 2, 3, 4, 5, 6]
    arcs_inp = [
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 3),
        (2, 5, 4),
        (3, 2, 2),
        (3, 4, 4),
        (3, 5, 5),
        (4, 6, 9),
        (5, 4, 1),
        (5, 6, 6),
    ]
    bellman_ford(nodes_inp, arcs_inp, 6, LaTex=True)
