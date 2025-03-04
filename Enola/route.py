import json
import math
import copy
from .codegen import CodeGen, global_dict
from networkx import maximal_independent_set, Graph
from typing import Sequence, Mapping, Any


global_dict["full_code"] = True

def direction_vector(dx, dy):
    if dx == 0:  # Vertical movement (up or down)
        return (0, 1) if dy > 0 else (0, -1)
    elif dy == 0:  # Horizontal movement (left or right)
        return (1, 0) if dx > 0 else (-1, 0)
    else:
        # For diagonal or other movements, return the direction tuple directly (no normalization)
        return (dx // abs(dx), dy // abs(dy))
        
        
def apply_mov_for_map(movement, map):
    res_map = copy.deepcopy(map)
    for movs in movement:
        # print(movs)
        for qubit, (ox,oy), (nx,ny) in movs:
            try: 
                assert res_map[qubit] == (ox,oy), f"origin loc not equal"
                res_map[qubit] = (nx,ny)
            except:
                raise ValueError(f"{movement} \n with \n {map} \n in {qubit}")
    return res_map
    
def compatible_2D(a: list[int], b: list[int]) -> bool:
    """
    Checks if two 2D points are compatible based on specified rules.

    Parameters:
    a (list[int]): A list of four integers representing the first point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].
    b (list[int]): A list of four integers representing the second point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].

    Returns:
    bool: True if the points are compatible, False otherwise.
    """
    assert len(a) == 4 and len(b) == 4, "Both arguments must be lists with exactly four elements."

    # Check compatibility for the first two elements of each point
    if a[0] == b[0] and a[2] != b[2]:
        return False
    if a[2] == b[2] and a[0] != b[0]:
        return False
    if a[0] < b[0] and a[2] >= b[2]:
        return False
    if a[0] > b[0] and a[2] <= b[2]:
        return False

    # Check compatibility for the last two elements of each point
    if a[1] == b[1] and a[3] != b[3]:
        return False
    if a[3] == b[3] and a[1] != b[1]:
        return False
    if a[1] < b[1] and a[3] >= b[3]:
        return False
    if a[1] > b[1] and a[3] <= b[3]:
        return False

    return True

def maximalis_solve_sort(n: int, edges: list[tuple[int]], nodes: set[int]) -> list[int]:
    """
    Finds a maximal independent set from the given graph nodes using a sorted approach.

    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): list of edges in the graph, where each edge is a tuple of two nodes.
    nodes (set[int]): Set of nodes to consider for the maximal independent set.

    Returns:
    list[int]: list of nodes in the maximal independent set.
    """
    # Initialize conflict status for each node
    is_node_conflict = [False for _ in range(n)]
    
    # Create a dictionary to store neighbors of each node
    node_neighbors = {i: [] for i in range(n)}
    
    # Populate the neighbors dictionary
    for edge in edges:
        node_neighbors[edge[0]].append(edge[1])
        node_neighbors[edge[1]].append(edge[0])
    
    result = []
    for i in nodes:
        if is_node_conflict[i]:
            continue
        else:
            result.append(i)
            for j in node_neighbors[i]:
                is_node_conflict[j] = True
    return result

def maximalis_solve(nodes:list[int], edges:list[tuple[int]])-> list[int]:
    """
    Wrapper function to find a maximal independent set using the Graph class.

    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): list of edges in the graph.

    Returns:
    list[int]: list of nodes in the maximal independent set.
    """
    G = Graph()
    for i in nodes:
        G.add_node(i)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Use a library function to find the maximal independent set
    result = maximal_independent_set(G, seed=0) 
    return result

def get_movements(current_map: list, next_map: list, window_size=None) -> map:
    """
    Determines the movements of qubits between two maps.

    Parameters:
    current_map (list): list of current positions of qubits.
    next_map (list): list of next positions of qubits.
    window_size (optional): Size of the window for movement calculations.

    Returns:
    map: A dictionary with qubit movements.
    """
    movements = {}
    # Determine movements of qubits
    for qubit, current_position in enumerate(current_map):
        next_position = next_map[qubit]
        if current_position != next_position:
            move_details = current_position + next_position
            movements[qubit] = move_details
    return movements

def map_to_layer(map: list) -> dict[str, list]:
    """
    Converts a list of qubit positions to a layer dictionary.

    Parameters:
    map (list): list of qubit positions.

    Returns:
    map: Dictionary representing the layer configuration.
    """
    return {
        "qubits": [{
            "id": i,
            "a": 0,
            "x": map[i][0],
            "y": map[i][1],
            "c": map[i][0],
            "r": map[i][1],
        } for i in range(len(map))],
        "gates": []
    }

def gates_in_layer(gate_list:list[list[int]])->list[dict[str, int]]:
    res = []
    for i in range(len(gate_list)-1,-1,-1):
        assert len(gate_list[i]) == 2
        res.append({'id':i,'q0':gate_list[i][0],'q1':gate_list[i][1]})
    return res

class QuantumRouter:
    def __init__(self, num_qubits: int, 
                 before_gate_maps: list[list[list[int]]], 
                 qft_gate_list: list[list[int]], 
                 arch_size: list[int], 
                 real_gate_list: list[list[list[int]]],
                 routing_strategy: str = "maximalis"
                 ) -> None:
        """
        Initialize the QuantumRouter object with the given parameters.
        
        Parameters:
        num_qubits (int): Number of qubits.
        before_gate_maps (list[list[list[int]]]): maps before the gate execute.
        gate_list (list[list[int]]): list of two-qubit gates.
        arch_size (list[int]): Architecture size as [x, y].
        routing_strategy (str): Strategy used for routing.
        """
        self.num_qubits = num_qubits
        self.validate_embeddings(before_gate_maps)
        self.before_gate_maps = before_gate_maps
        self.gate_maps = []
        
        if real_gate_list:
            assert len(real_gate_list) == len(qft_gate_list)
        else:
            real_gate_list = qft_gate_list
        assert len(before_gate_maps) == len(qft_gate_list), "The number of before gate maps should match the number of two-qubit gates in gate_list."
        self.qft_gate_list = qft_gate_list
        self.real_gate_list = real_gate_list

        self.validate_real_gates()
        self.validate_architecture_size(arch_size)
        self.arch_size = arch_size
        self.routing_strategy = routing_strategy
        self.movement_list = []

    def validate_real_gates(self):
        for i, gates in enumerate(self.real_gate_list):
            for gate in gates:
                q0, q1 = gate
                if [q0,q1] not in self.qft_gate_list[i] and [q1,q0] not in self.qft_gate_list:
                    raise ValueError(f"qft list not contain this gate list")
    def validate_embeddings(self, before_gate_maps: list[list[list[int]]]) -> None:
        """
        Validate the maps to ensure they contain locations for all qubits.
        
        Parameters:
        before_gate_maps (list[list[list[int]]]): maps before the gate execute.
        """
        for map in before_gate_maps:
            assert len(map) == self.num_qubits, f"Each embedding must contain locations for all {self.num_qubits} qubits."
            for loc in map:
                assert len(loc) == 2, "Each location must be a list containing exactly two coordinates: [x, y]."

    def validate_architecture_size(self, arch_size: list[int]) -> None:
        """
        Validate the architecture size to ensure it can accommodate all qubits.
        Parameters:
        arch_size (list[int]): Architecture size as [x, y].
        """
        assert len(arch_size) == 2, "Architecture size should be specified as a list with two elements: [x, y]."
        assert arch_size[0] * arch_size[1] >= self.num_qubits, (
            f"The product of the architecture dimensions x and y must be at least {self.num_qubits} to accommodate all qubits; "
            f"currently, it is {arch_size[0] * arch_size[1]}."
        )

    def initialize_program(self) -> None:
        """
        Initialize the program with the initial layer and gates.
        """
        layers = [map_to_layer(self.before_gate_maps[0])]
        # initial_layer = map_to_layer(self.before_gate_maps[0])
        # initial_layer["gates"] = gates_in_layer(self.gate_list[0])
        # layers.append(initial_layer)
        return self.generate_program(layers)

    def generate_program(self, layers: list[dict[str, Any]]) -> Sequence[Mapping[str, Any]]:
        """
        Generate the program from the given layers.
        
        Parameters:
        layers (list[dict[str, Any]]): list of layers.
        
        Returns:
        str: The generated program.
        """
        data = {
            "no_transfer": False,
            "layers": layers,
            "n_q": self.num_qubits,
            "g_q": self.qft_gate_list,
            "n_x": self.arch_size[0],
            "n_y": self.arch_size[1],
            "n_c": self.arch_size[0],
            "n_r": self.arch_size[1]
        }
        code_gen = CodeGen(data)
        program = code_gen.builder(no_transfer=False)
        return program.emit_full()

    def process_all_maps(self) -> None:
        """
        Process all maps to resolve movements and update the program.
        """
        for current_pos in range(len(self.before_gate_maps) - 1):
            movements, gate_map = self.move_to_excute_gate(current_pos)
            assert len(movements) > 0, "there should be some movements between embeddings"
            self.gate_maps.append(gate_map)
            self.movement_list.append(movements)
            movements = self.move_to_before_map(current_pos, gate_map)
            assert len(movements) > 0, "there should be some movements between embeddings"
            self.movement_list.append(movements)

        movements, gate_map = self.move_to_excute_gate(-1)
        assert len(movements) > 0, "there should be some movements between embeddings"
        self.gate_maps.append(gate_map)
        self.movement_list.append(movements)

    def move_to_excute_gate(self,current_pos:int):
        gate_map = self.get_gate_maps(current_pos)
        movements = get_movements(self.before_gate_maps[current_pos], gate_map)
        sorted_movements = sorted(
            movements.keys(),
            key=lambda k: (direction_vector(movements[k][2] - movements[k][0], movements[k][3] - movements[k][1]),  # Direction vector (Δx, Δy)
                        math.dist(movements[k][:2], movements[k][2:]))  # distance
        )
        
        violations = self.check_violations(sorted_movements, movements)
        move_sequences = self.handle_violations(violations, movements, sorted_movements, current_pos)
        # print(current_pos)
        gate_map = copy.deepcopy(self.before_gate_maps[current_pos])
        for movements in move_sequences:
            for move in movements:
                # print(move)
                qubit = move[0]
                assert move[1] == gate_map[qubit], f"qubit should move form {gate_map[qubit]}, now in {move[1]}"
                gate_map[qubit] = move[2]
        # print(f"gate maps:{gate_maps}")
        return move_sequences, gate_map

    def get_gate_maps(self, current_pos:int ):
        gate_maps = copy.deepcopy(self.before_gate_maps[current_pos])
        for q0, q1 in self.qft_gate_list[current_pos]:
            gate_maps[q0], gate_maps[q1] = gate_maps[q1], gate_maps[q0]
        return gate_maps
    def move_to_before_map(self, current_pos:int, gate_map):
        movements = get_movements(gate_map, self.before_gate_maps[current_pos+1])
        sorted_movements = sorted(
            movements.keys(),
            key=lambda k: (direction_vector(movements[k][2] - movements[k][0], movements[k][3] - movements[k][1]),  # Direction vector (Δx, Δy)
                        math.dist(movements[k][:2], movements[k][2:]))  # distance
        )
        
        violations = self.check_violations(sorted_movements, movements)
        move_sequences = self.handle_violations(violations, movements, sorted_movements)
        return move_sequences

    def dual_move_to_excute_gate(self, current_pos: int):
        pass
    def solve_violations(self, movements, violations, sorted_keys, current_pos):
        """
        Resolves violations in qubit movements based on the routing strategy.

        Parameters:
        movements (dict): Dictionary of qubit movements.
        violations (list): list of violations to be resolved.
        sorted_keys (list): list of qubit keys sorted based on priority.
        layer (dict): Dictionary representing the current layer configuration.

        Returns:
        tuple: remaining movements, unresolved violations and movement sequence to finish movement this time
        """
        if self.routing_strategy == "maximalis":
            resolution_order = maximalis_solve(sorted_keys, violations)
        else:
            resolution_order = maximalis_solve_sort(self.num_qubits, violations, sorted_keys)
        # print(f'Resolution Order: {resolution_order}')
        move_sequence =[]
        # if 2*len(resolution_order) != len(movements) and current_pos:
        #     # print(current_pos,len(resolution_order),len(movements))
        #     print(f"{current_pos} move still live, it have:\nmovements={movements}\n violations={violations}\n gate list:\n{self.gate_list[current_pos]}\n")
        for qubit in resolution_order:
            sorted_keys.remove(qubit)

            move = movements[qubit]
            # print(self.momvents)
            move_sequence.append([qubit,(move[0],move[1]),(move[2],move[3])])
            # print(f'Move qubit {qubit} from ({move[0]}, {move[1]}) to ({move[2]}, {move[3]})')
            # Remove resolved violations
            violations = [v for v in violations if qubit not in v]
            del movements[qubit]
            if current_pos != None:
                # print(f"qubit:{qubit}, pos:{current_pos}, gate:{self.gate_list[current_pos]}")
                for q0,q1 in self.qft_gate_list[current_pos]:
                    if q0 == qubit:
                        sorted_keys.remove(q1)
                        violations = [v for v in violations if q1 not in v]
                        del movements[q1]
                        break
                    elif q1 == qubit:
                        sorted_keys.remove(q0)
                        violations = [v for v in violations if q0 not in v]
                        del movements[q0]
                        break
                    else:
                        pass

        return movements, violations, move_sequence

    def resolve_movements(self, current_pos: int) -> list[int, tuple[int, int], tuple[int, int]]:
        """
        Resolve movements between the current and next embeddings.
        
        Parameters:
        current_pos (int): The current position in the embeddings list.
        
        Returns:
        The list for the resolved movements.
        """
        next_pos = current_pos + 1
        movements = get_movements(self.before_gate_maps[current_pos], self.before_gate_maps[next_pos])
        sorted_movements = sorted(movements.keys(), key=lambda k: math.dist(movements[k][:2], movements[k][2:]))
        violations = self.check_violations(sorted_movements, movements)
        move_sequences = self.handle_violations(violations, movements, sorted_movements, current_pos)
        return move_sequences

    def handle_violations(self, violations: list[tuple[int, int]], remained_mov_map: dict[int, tuple[int, int, int, int]], sorted_movements: list[int], current_pos=None) -> list[int, tuple[int, int], tuple[int, int]]:
        """
        Handle violations and return the movement sequence accordingly.
        
        Parameters:
        violations (list[tuple[int, int]]): list of violations.
        movements (dict[int, tuple[int, int, int, int]]): Movements between embeddings.
        sorted_movements (list[int]): Sorted list of movements.
        current_pos (int): The current position in the embeddings list.
        
        Returns:
        list[int, tuple[int, int], tuple[int, int]]: movement sequences.
        """
        movement_sequence =[]
        while remained_mov_map:
            remained_mov_map, violations, movement = self.solve_violations(remained_mov_map, violations, sorted_movements,current_pos)
            movement_sequence.append(movement)

        return movement_sequence

    def check_violations(self, sorted_movements: list[int], remained_mov_map: dict[int, tuple[int, int, int, int]]) -> list[tuple[int, int]]:
        """
        Check for violations between movements.
        
        Parameters:
        sorted_movements (list[int]): Sorted list of movements.
        movements (dict[int, tuple[int, int, int, int]]): Movements between embeddings.
        
        Returns:
        list[tuple[int, int]]: list of violations.
        """
        violations = []
        for i in range(len(sorted_movements)):
            for j in range(i + 1, len(sorted_movements)):
                if not compatible_2D(remained_mov_map[sorted_movements[i]], remained_mov_map[sorted_movements[j]]):
                    violations.append((sorted_movements[i], sorted_movements[j]))
        return violations

    def update_layer(self, layer, movements):
        new_layer = copy.deepcopy(layer)
        for qubit, current_pos, next_pos in movements:
            assert layer["qubits"][qubit]["id"] == qubit, "some error happen during layer generation"
            assert layer["qubits"][qubit]["x"] == current_pos[0], f"layer have problem with location of qubit {qubit}, in x-axis"
            assert layer["qubits"][qubit]["y"] == current_pos[1], f"layer have problem with location of qubit {qubit}, in y-axis"

            new_layer["qubits"][qubit]["a"] = 1
            layer["qubits"][qubit]["x"] = next_pos[0]
            layer["qubits"][qubit]["y"] = next_pos[1]
            layer["qubits"][qubit]["c"] = next_pos[0]
            layer["qubits"][qubit]["r"] = next_pos[1]
        return new_layer

    
    def using_qubit(self, pos:int):
        using_qubit = []
        for gate in self.real_gate_list[pos]:
            q0, q1 = gate
            using_qubit.append(q0)
            using_qubit.append(q1)
        return set(using_qubit)

    def modify_mov_for_real(self, gate_pos:int, move_pos:int):
        assert self.movement_list, "should finish the calculation of movements"
        use_qubits = self.using_qubit(gate_pos)
        # print(use_qubits)
        mov_for_gate = []
        mov_for_move = []
        
        for movs in self.movement_list[move_pos]:
            temp_mov_for_gate = []
            temp_mov_for_move = []
            for mov in movs:
                qubit, _, _ = mov
                if qubit in use_qubits:
                    temp_mov_for_gate.append(mov)
                else:
                    temp_mov_for_move.append(mov)
            
            if temp_mov_for_gate:
                mov_for_gate.append(temp_mov_for_gate)
            if temp_mov_for_move:
                mov_for_move.append(temp_mov_for_move)
        
        return mov_for_gate, mov_for_move
        
    def save_program(self, filename: str) -> None:
        """
        Save the generated program to a file.
        Parameters:
        filename (str): The filename to save the program.
        """
        print(filename)
        assert filename.endswith('.json'), "program should be saved to a .json file"
        assert len(self.movement_list) == len(self.before_gate_maps)+len(self.gate_maps)-1, f"before generate program, movement should be finished, now: {len(self.movement_list),len(self.before_gate_maps),len(self.gate_maps)}"
        program = []
        for i, before_map in enumerate(self.before_gate_maps):
            # print(i)
            layers=[map_to_layer(before_map)]
            
            mov_for_gate, mov_for_move = self.modify_mov_for_real(i,i*2)
            # print(mov_for_move)
            # print(mov_for_gate)
            real_gate_map = apply_mov_for_map(mov_for_gate, before_map) # if move_for_gate have no move, real_gate_map should equal to gate_maps[i]
            
            if self.real_gate_list[i]:
                # move to gate map
                for mov in mov_for_gate:
                    layers.append(self.update_layer(map_to_layer(before_map),mov))

                layers.append(map_to_layer(real_gate_map))
                
                layers[-1]["gates"] = gates_in_layer(self.real_gate_list[i])
                
            if mov_for_move and all(len(inner) for inner in mov_for_move):
                for mov in mov_for_move:
                    layers.append(self.update_layer(map_to_layer(real_gate_map),mov))
                layers.append(map_to_layer(self.gate_maps[i]))

                
            
            # gate map to next before map
            if i+1 < len(self.before_gate_maps):
                for mov in self.movement_list[i*2+1]:
                    layers.append(self.update_layer(map_to_layer(self.gate_maps[i]),mov))
                layers.append(map_to_layer(self.before_gate_maps[i+1]))


            # print(f"layers: {layers}")
            if i == 0:
                program += self.generate_program(layers)
            else:
                program += self.generate_program(layers)[2:]
        with open(filename, 'w') as file:
            json.dump(program, file)

    def run(self) -> None:
        """
        Run the QuantumRouter to initialize, process embeddings.
        """
        self.movement_list = []
        self.process_all_maps()