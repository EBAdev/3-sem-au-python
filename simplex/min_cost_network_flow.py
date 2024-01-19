import networkx as nx
import numpy as np
from typing import List, Tuple, Union, Dict

Num = Union[float, int]


def np_mat_to_latex_pmatrix(a):
    """Returns a LaTeX pmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{pmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{pmatrix}"]
    return "\n".join(rv)


def create_min_cost_network(nodes, arcs) -> nx.DiGraph:
    G = nx.DiGraph()
    total_supply = 0
    source_nodes = [node[0] for node in nodes if node[1] > 0]
    for node in nodes:
        G.add_node(node[0], supply=node[1])
        total_supply += node[1]

    if total_supply != 0:
        raise ValueError(
            "The provided network is not feasible\nin order to fix this, make sure that aggregate supplies are 0. (Supply = Demand)."
        )
    for arc in arcs:
        G.add_edge(arc[0], arc[1], cost=arc[2], flow=0)

    for idx, layer in enumerate(nx.bfs_layers(G, source_nodes)):
        for node in layer:
            G.nodes[node]["layer"] = idx

    return G


def north_west_method(G: nx.DiGraph) -> Tuple[nx.DiGraph, List[Tuple[int, int, Num]]]:
    """
    Function to determine a starting flow-folution to a transport problem, returns a tree solution.
    #### Parameters
    1. G:nx.DiGraph
            * A directed graph containing a transportation problem, with n nodes and m edges.
            Each node must have a attribute keyed supply corresponding to the the supply of that node (can be negative).
    #### Returns
    A directed graph representing the obtained tree solution.

    """
    T = nx.DiGraph()
    T.add_nodes_from(G.nodes(data=True))

    supply = [node[1]["supply"] for node in G.nodes(data=True) if node[1]["supply"] > 0]
    demand = [
        -node[1]["supply"] for node in G.nodes(data=True) if node[1]["supply"] < 0
    ]

    # we always start at the NW corner (i,j)=(1,len(supply))
    i = 0
    j = 0

    bfs = []
    while len(bfs) < (len(supply) + len(demand)) - 1:
        s = supply[i]
        d = demand[j]
        # if s<d => all supply will be givent to d
        # if d<s => all demand must have been met
        # in either case, the bottelneck must be min(s,d).
        v = min(s, d)

        # determine remaining supply & demand:
        supply[i] -= v
        demand[j] -= v

        # Save the amount we could send (v) to the arc (i,j)
        bfs.append((i + 1, len(supply) + j + 1, v))  # fix so we count from 1
        # if supply was bottelneck and there is more supply to give
        if supply[i] == 0 and i < len(supply):
            i += 1
        elif demand[j] == 0 and j < len(demand):
            j += 1

    for arc in bfs:
        T.add_edge(arc[0], arc[1], flow=arc[2], cost=G.edges[arc[0], arc[1]]["cost"])

    return T


def get_initial_flow(T: nx.DiGraph) -> List[Tuple[int, int, Num]]:
    H = T.copy()
    flow_update = []

    while len(list(H.edges())) > 0:
        for node in list(H.nodes(data=True)):
            if nx.degree(H, node[0]) == 1:  # if node is leaf
                if list(H.edges(node[0])):  # if leaf has a forward arc
                    if node[1]["supply"] < 0:
                        raise ValueError(
                            "The network is infeasible.\nWhen determining the initial tree solution, the supply of atleast one leaf was negative.\nSince there is no supplier for a leaf the network is infeasible"
                        )

                    arc = list(H.edges(node[0]))[0]  # get the arc of the leaf
                    flow_update.append(
                        (arc[0], arc[1], node[1]["supply"])
                    )  # add the flow to the flow vector
                    H.nodes[arc[1]]["supply"] += node[1][
                        "supply"
                    ]  # update flow at the incident node
                    H.remove_edge(*arc)  # remove the edge from consideration
                else:
                    if node[1]["supply"] > 0:  # if node is supplier
                        raise ValueError(
                            "There is a positive supply but nowhere to send it"
                        )
                    connected_arcs = [edge for edge in H.edges if edge[1] == node[0]]
                    neighbor = (connected_arcs[0][0], H.nodes[connected_arcs[0][0]])

                    if neighbor[1]["supply"] < -node[1]["supply"]:
                        continue
                    flow_update.append(
                        (neighbor[0], node[0], -node[1]["supply"])
                    )  # send flow along edge
                    H.nodes[neighbor[0]]["supply"] += node[1][
                        "supply"
                    ]  # update available flow at supplier
                    H.remove_edge(neighbor[0], node[0])
    return flow_update


def update_network_flow(
    Graph: nx.DiGraph, Tree: nx.DiGraph, flow_update: List[Tuple[int, int, Num]]
) -> Tuple[nx.DiGraph, nx.DiGraph]:
    G = Graph.copy()
    T = Tree.copy()

    for flow in flow_update:
        for arc in T.edges():
            # if we are at the right flow variable
            if (flow[0], flow[1]) == arc:
                T.edges[arc[0], arc[1]]["flow"] += flow[2]  # update flow in tree
                G.edges[arc[0], arc[1]]["flow"] += flow[2]  # update flow in total graph
                break  # there is only one arc which matches the flow arc

    return (G, T)


def get_dual_solution(
    G: nx.DiGraph,
    T: nx.DiGraph,
    variable_to_zero: int,
) -> np.ndarray:
    """
    Since incidens coefficient matrix A in network problem does not have full rank.
    A variable corresponding to a node will be set to zero to solve equations.
    If the variable is not provided we use the last node.
    """

    A = -1 * nx.incidence_matrix(T, oriented=True).toarray().T  # get incidence matrix

    A = np.row_stack(
        (A, [0 if i + 1 != variable_to_zero else 1 for i in range(A.shape[1])])
    )  # add row setting variable to zero

    b = np.array(
        [[arc[2]["cost"] for arc in T.edges(data=True)] + [0]]
    ).T  # determine bounds of linear systems
    dual_solution = np.linalg.solve(A, b)

    return dual_solution


def get_reduced_costs(
    arcs: List[Tuple[int, int, Num]], dual_solution: np.ndarray
) -> List[Tuple[int, int, Num]]:
    dual_list = (dual_solution.T)[0]

    # r_{ij} = c_{ij} - (y_i - y_j)
    reduced_costs = [
        (arc[0], arc[1], arc[2] - (dual_list[arc[0] - 1] - dual_list[arc[1] - 1]))
        for arc in arcs
    ]

    return reduced_costs


def optimal_reduced_costs(reduced_costs: List[Tuple[int, int, Num]]) -> bool:
    for entry in reduced_costs:
        if entry[2] < 0:
            return False
    return True


def get_min_reduced_costs(
    reduced_costs: List[Tuple[int, int, Num]]
) -> Tuple[int, int, Num]:
    r_costs = [cost[2] for cost in reduced_costs]
    leaving_arc_idx = r_costs.index(min(r_costs))
    leaving_arc = reduced_costs[leaving_arc_idx]
    return leaving_arc


def get_network_optimization(
    Graph: nx.DiGraph, Tree: nx.DiGraph, lowest_reduced_cost: Tuple[int, int, Num]
) -> Tuple[List[Tuple[int, int, Num]], Tuple[int, int]]:
    T = Tree.copy()
    G = Graph.copy()

    T.add_edge(
        lowest_reduced_cost[0],
        lowest_reduced_cost[1],
        flow=0,
        cost=G.edges[lowest_reduced_cost[0], lowest_reduced_cost[1]]["cost"],
    )

    cycle = nx.find_cycle(T, lowest_reduced_cost[1], orientation="ignore")
    backwards_path = [
        (c[0], c[1], T.edges[c[0], c[1]]) for c in cycle if c[2] == "reverse"
    ]

    if backwards_path == []:  # network unbound
        return (False, cycle)

    theta = min([c[2]["flow"] for c in backwards_path])
    leaving_arc_idx = [c[2]["flow"] for c in backwards_path].index(theta)
    leaving_arc = (
        backwards_path[leaving_arc_idx][0],
        backwards_path[leaving_arc_idx][1],
    )

    flow_update = [
        (c[0], c[1], -theta) if c[2] == "reverse" else (c[0], c[1], theta)
        for c in cycle
    ]
    return (flow_update, leaving_arc, cycle)


def cycle_fig_tex(
    G: nx.DiGraph,
    cycle,
    node_labels,
    scale,
    caption,
    latex_label,
    transportation_problem,
) -> str:
    C = nx.DiGraph()
    C.add_nodes_from(G.nodes())
    labels = node_labels.copy()
    for c in cycle:
        C.add_edge(c[0], c[1], direction=c[2], flow=G.edges[c[0], c[1]]["flow"])

    edge_labels = {}
    edge_label_opt = {}
    for label in labels:
        labels.update({label: "$" + labels[label] + "$"})
    if transportation_problem is False:
        pos = nx.multipartite_layout(G, scale=scale, subset_key="layer")
    else:
        suppliers = [node[0] for node in G.nodes(data=True) if node[1]["supply"] > 0]
        pos = nx.bipartite_layout(G, suppliers, scale=scale)

    for arc in C.edges():
        data = C.edges[arc[0], arc[1]]

        if data["direction"] == "reverse":
            label = (
                "$f_{"
                + node_labels[arc[0]]
                + ","
                + node_labels[arc[1]]
                + "}="
                + str(data["flow"])
                + "-\\delta$"
            )
        else:
            label = (
                "$f_{"
                + node_labels[arc[0]]
                + ","
                + node_labels[arc[1]]
                + "}="
                + str(data["flow"])
                + "+\\delta$"
            )
        edge_labels.update({arc: label})
        edge_label_opt.update({arc: "[pos=.5,scale=0.50,above,sloped]"})

    return nx.to_latex(
        C,
        pos=pos,
        as_document=False,
        caption=caption,
        figure_wrapper="\\begin{{figure}}[H]\n\\centering\n{content}{caption}{label}\n\\end{{figure}}",
        node_label=labels,
        tikz_options="scale=" + str(scale),
        edge_label=edge_labels,
        edge_label_options=edge_label_opt,
        latex_label=latex_label,
    )


def table_method(G: nx.DiGraph, tree_sol):
    T = nx.DiGraph()
    T.add_nodes_from(G.nodes(data=True))
    edges = [edge for edge in G.edges(data=True) if (edge[0], edge[1]) in tree_sol]

    T.add_edges_from(edges)
    C = T.copy()

    while len(C.edges) != 0 and len(C.nodes) != 0:
        for supply_node in list(C.nodes(True)):
            if supply_node[1]["supply"] <= 0:
                continue

            for arc in list(C.edges(supply_node[0], data=True)):
                if nx.degree(C, arc[1]) == 1 or nx.degree(C, arc[0]) == 1:
                    demand_node = (arc[1], C.nodes[arc[1]])
                    if (
                        -demand_node[1]["supply"]
                        <= supply_node[1]["supply"]
                        # if demand less curret flow is less than the available supply
                    ):
                        T.edges[supply_node[0], demand_node[0]]["flow"] = -demand_node[
                            1
                        ]["supply"]
                        # demand clears
                        # update supply
                        C.nodes[supply_node[0]]["supply"] += demand_node[1]["supply"]
                        C.nodes[demand_node[0]]["supply"] = 0
                        C.remove_edge(supply_node[0], demand_node[0])
                        C.remove_node(demand_node[0])
                    elif (
                        supply_node[1]["supply"]
                        < -demand_node[1][
                            "supply"
                        ]  # if supply is less than the current demand
                    ):
                        T.edges[supply_node[0], demand_node[0]]["flow"] += supply_node[
                            1
                        ][
                            "supply"
                        ]  # send all supply.
                        # update supply
                        C.nodes[demand_node[0]]["supply"] += C.nodes[supply_node[0]][
                            "supply"
                        ]
                        C.nodes[supply_node[0]]["supply"] = 0
                        C.remove_node(supply_node[0])
    return T


def min_cost_network_flow_simplex(
    nodes: List[Tuple[int, Num]],
    arcs: List[Tuple[int, int, Num]],
    tree_solution: List[Tuple[int, int]] = None,
    transportation_problem: bool = False,
    variable_to_zero: int = -1,
    node_labels: Dict = None,
    LaTeX: bool = False,
):
    """
    Function to calculate the minimum cost network flow from suppliers to demanders.
    #### Parameters
    1. nodes: List[Tuple[int,Num]] [required]
            * A of tuples, each representing a node, the second element in the tuple should be the supply of that node.
            (can be negative)
    2. arcs: List[Tuple[int,int,Num]] [required]
            * A list of the edges in the min cost network flow problem where each element is a tuple of tree digits. The first is the arcs origin, the second is the arcs destination, and the third is the cost of traversal along the arc.
    3. tree_solution:List[Tuple[int,int]] = [] [required if not transportation problem]
            * A list of directed arcs in a spanning tree of the MCNF graph.
    4. transportation_problem:bool = False
            * A bool which should be True if the MCNF problem is a transportation problem, if so the algoritm will use the North West corner method to determine the inital basis.
    5. node_labels: Dict = {}
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format, used for printing.
    """
    G = create_min_cost_network(nodes, arcs)

    if node_labels is None:
        node_labels = {}
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})

    if check_labels(G.number_of_nodes(), node_labels) is not True:
        raise check_labels(G.number_of_nodes(), node_labels)

    if transportation_problem is True and tree_solution is None:
        T = north_west_method(G)
        for arc in T.edges(data=True):
            G.edges[arc[0], arc[1]]["flow"] = arc[2]["flow"]

    elif transportation_problem is True and len(tree_solution) == len(G.nodes) - 1:
        T = table_method(G, tree_solution)

        for arc in T.edges(data=True):
            G.edges[arc[0], arc[1]]["flow"] = arc[2]["flow"]

    elif not transportation_problem and len(tree_solution) == len(G.nodes) - 1:
        tree_solution = [arc for arc in arcs if (arc[0], arc[1]) in tree_solution]
        T = create_min_cost_network(nodes, tree_solution)

        flow_update = get_initial_flow(T)
        G, T = update_network_flow(G, T, flow_update)

    else:
        raise ValueError(
            "There was a problem with the provided tree solution, it does not span the provided min-cost-network."
        )

    if variable_to_zero == -1:
        variable_to_zero = G.number_of_nodes()

    dual_solution = get_dual_solution(G, T, variable_to_zero)
    reduced_costs = get_reduced_costs(arcs, dual_solution)

    nit = 0
    while optimal_reduced_costs(reduced_costs) is not True:
        nit += 1
        lowest_reduced_cost = get_min_reduced_costs(reduced_costs)

        entering_arc = (lowest_reduced_cost[0], lowest_reduced_cost[1])

        network_optimization = get_network_optimization(G, T, lowest_reduced_cost)

        if network_optimization[0] is False:
            raise ValueError(
                "When adding "
                + str(entering_arc)
                + " to basis, a cycle with no reverse arcs was discovered. This means that the network is unbound."
            )

        flow_update = network_optimization[0]
        leaving_arc = network_optimization[1]
        cycle = network_optimization[2]
        T.add_edge(
            entering_arc[0],
            entering_arc[1],
            cost=G.edges[entering_arc[0], entering_arc[1]]["cost"],
            flow=0,
        )
        G, T = update_network_flow(G, T, flow_update)
        T.remove_edge(*leaving_arc)

        dual_solution = get_dual_solution(G, T, variable_to_zero)
        reduced_costs = get_reduced_costs(arcs, dual_solution)

    return T


def check_labels(num_nodes: int, node_labels: Dict) -> Union[bool, ValueError]:
    if len(node_labels) != num_nodes:
        return ValueError(
            "\nThe number of nodes given and the amount of node labels does not match.\nThere should be a label for each node."
        )
    return True


def min_cost_network_flow_to_latex(
    G: nx.DiGraph,
    transportation_problem: bool = False,
    caption: str = "",
    node_labels: Dict = {},
    layer_key: str = "layer",
    scale: int = 2,
    latex_label: str = "",
) -> str:
    """
    Function to return a Tikz figure of the min cost network flow problem.
    #### Parameters
    1. network:nx.DiGraph [required]
            * A directed network corresponding to the min cost network flow problem.
              Each node in the network must contain an attribute of value integer corresponding to the layer
              of the node in a breath first search with origin of the suppliers.
    2. transportation_problem:bool = False
            * A bool which should be True if the MCNF problem is a transportation problem, if so the algoritm will use the North West corner method to determine the inital basis.
    3. caption:str
            * The caption of the LaTeX figure, if none is provided a caption will not be written.
    4. node_labels:Dict
            * A dictionary matching the integer value of all of the nodes, to a corresponding string in LaTeX format.
    5. layer_key:str = "layer"
            * The attribute name in the node generator of the network, which the layer integer.
    8. scale:int = 2
            * a integer value which changes the size of the picture.
    7. latex_label:str =""
            * The label used by latex to reference the graph
    #### Useful knowledge
    The nodes will be placed on a horizontal line, however the edges are not taken into account. Therefore it might be nesseary to move some nodes up or down. We can also bend lines using the option [bend right=int] at the start of the edge, and we can move labels below a line by setting the parameter below.
    """

    if node_labels == {}:
        for i in range(G.number_of_nodes()):
            node_labels.update({i + 1: str(i + 1)})
    labels = node_labels.copy()
    for label in labels:
        labels.update({label: "$" + labels[label] + "$"})
    if check_labels(G.number_of_nodes(), labels) is not True:
        raise check_labels(G.number_of_nodes(), labels)

    edge_labels = {}
    edge_label_opt = {}
    if transportation_problem is False:
        pos = nx.multipartite_layout(G, scale=scale, subset_key=layer_key)
    else:
        suppliers = [node[0] for node in G.nodes(data=True) if node[1]["supply"] > 0]
        pos = nx.bipartite_layout(G, suppliers, scale=scale)
    for arc in G.edges():
        data = G.edges[arc[0], arc[1]]
        label = (
            "$c_{"
            + node_labels[arc[0]]
            + ","
            + node_labels[arc[1]]
            + "}="
            + str(data["cost"])
            + "\\quad f_{"
            + node_labels[arc[0]]
            + ","
            + node_labels[arc[1]]
            + "}="
            + str(data["flow"])
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
        node_label=labels,
        tikz_options="scale=" + str(scale),
        edge_label=edge_labels,
        edge_label_options=edge_label_opt,
        latex_label=latex_label,
    )


if __name__ == "__main__":
    nodes_inp = [
        (1, 2),
        (2, 0),
        (3, -2),
        (4, 2),
        (5, 1),
        (6, -2),
        (7, 0),
        (8, -1),
    ]
    arcs_inp = [
        (1, 2, 1),
        (1, 4, 1),
        (2, 3, 3),
        (3, 8, 5),
        (3, 5, 2),
        (4, 3, 5),
        (4, 5, 1),
        (5, 6, 1),
        (6, 7, 4),
        (7, 5, 1),
        (8, 4, 1),
        (8, 6, 1),
    ]
    tree_sol_inp = [(1, 2), (2, 3), (3, 8), (4, 3), (5, 6), (6, 7), (8, 6)]

    transp_nodes_inp = [
        (1, 50),
        (2, 60),
        (3, 50),
        (4, 50),
        (5, -30),
        (6, -20),
        (7, -70),
        (8, -30),
        (9, -60),
    ]
    transp_arcs_inp = [
        (1, 5, 16),
        (1, 6, 16),
        (1, 7, 13),
        (1, 8, 22),
        (1, 9, 17),
        (2, 5, 14),
        (2, 6, 14),
        (2, 7, 13),
        (2, 8, 19),
        (2, 9, 15),
        (3, 5, 19),
        (3, 6, 19),
        (3, 7, 20),
        (3, 8, 23),
        (3, 9, 50),
        (4, 5, 50),
        (4, 6, 12),
        (4, 7, 50),
        (4, 8, 15),
        (4, 9, 11),
    ]
    optimal = min_cost_network_flow_simplex(
        transp_nodes_inp, transp_arcs_inp, None, True, LaTeX=True
    )
