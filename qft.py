from qiskit import QuantumCircuit,qasm2, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from IPython.display import display
from numpy import pi
import copy
import random

basis_gate_set=["cz", "id", "u1", "u2", "u3"]

def linear_map(n:int, col = True):
    if col:
        m = [(i,0) for i in range(n)]
    else :
        m = [(0,i) for i in range(n)]
    return m
import copy
def fold_map(n:int, width: int, start_x = 0, start_y = 0):
    assert n >= 2
    if width%2 :
        width-=1
    x_dir = 1 # 1 to right, -1 to left
    y_dir = 1 # 1 to up, -1 to down
    bottom_line = True
    m = []
    pos = (start_x,start_y)
    new_pos = pos
    for i in range(n):
        m.append(new_pos)
        if i%2:
            new_pos = (pos[0], pos[1]+ y_dir*1)
            y_dir *= -1
        else:
            new_pos = (pos[0]+ x_dir*1, pos[1])
            if new_pos[0] in [start_x,width+start_x]:
                bottom_line = False
                x_dir *= -1
                y_dir *= -1
            elif new_pos[0] in [start_x+1,width+start_x-1] and not bottom_line:
                bottom_line = True
                y_dir *= -1
        pos = new_pos

    if m[-1][0] == m[-2][0] and m[-1][0] in [start_x,width+start_x]:
        assert m[-1][1] -m[-2][1] == 1, f"qubit {n-1} in {m[-1][0],m[-1][1]}, and qubit {n-2} in {m[-2][0],m[-2][1]}"
        m[-1] = (m[-2][0],m[-2][1]-1)
    return m
class QFT:
    def __init__(self, num_qubits: int) -> None:
        """
        Initialize the Quantum Fourier Transform (QFT) class with a specified number of qubits.

        :param num_qubits: int
            The number of qubits for the quantum circuit.
        """
        self.num_qubits = num_qubits
        self.full_circuit = self._initialize_circuit()
        self.cz_circuit = get_cz_circuit(self.full_circuit)
        self.gate_list = self._split_gates()
        self.ifQft = True
        self.qaoa: list[bool] = [True for _ in range(len(self.gate_list))]
        self.maps = None

    def _initialize_circuit(self) -> tuple[QuantumCircuit, list[list[int]]]:
        """
        Private method to create and return a Quantum Circuit for QFT.

        Returns:
            QuantumCircuit
                The initialized quantum circuit with QFT applied.
        """
        quantum_circuit = QuantumCircuit(self.num_qubits)

        # Apply Controlled-Z gates in a specific pattern across all qubits
        for i in range(self.num_qubits):
            quantum_circuit.h(i)
            for j in range(i+1,self.num_qubits):
                theta = pi/(2**(j-i))
                quantum_circuit.cp(theta,i,j)

        return transpile(quantum_circuit, basis_gates=basis_gate_set, optimization_level= 0)

    def _split_gates(self):
        gates_list = []
        dag = circuit_to_dag(self.cz_circuit)
        for layer in dag.layers():
            layer_as_circuit = dag_to_circuit(layer['graph'])
            gates = []
            for ins in layer_as_circuit.data:
                gates.append([i._index for i in ins.qubits])
            gates_list.append(gates)
        return gates_list

    def draw(self) -> None:
        display(self.full_circuit.draw('mpl'))

    def LNN_maps(self,width =0,) -> list[list[int]]:
        maps = []
        if width>=2:
            current_map = fold_map(self.num_qubits,width, 0, 1)
        else:
            if width < 0 :
                current_map = linear_map(self.num_qubits, True)
            else:
                current_map = linear_map(self.num_qubits, False)
        
        for i, gates in enumerate(self.gate_list):
            if i%2 == 0:
                maps.append(current_map)
            else:
                new_map = swap_qubits_by_move(current_map, gates)
                maps.append(new_map)
                current_map = copy.deepcopy(new_map)
        self.maps = maps
        return maps

    def remove_gate(self, num_lay:int, num_gate = 1):
        assert self.maps, "should initial the maps"
        assert self.ifQft, "before move gate, the circuit should still be qft"
        pick_layers = random.sample(range(len(self.gate_list)//2), num_lay)
        
        new_gate_list = []
        del_gate_list = []
        ignore_gate_list = []
        new_maps = []
        current_map = self.maps[0]
        for i in range(0, len(self.gate_list), 2):
            if i//2 in pick_layers:
                if num_gate < len(self.gate_list[i]):
                    del_gates = random.sample(self.gate_list[i], num_gate)
                else:
                    del_gates = self.gate_list[i]
                
                del_gate_list.append(del_gates)
                gates = [gate for gate in self.gate_list[i] if gate not in del_gates]
                if gates:
                    ignore_gate_list.extend([True,True,False])
                    new_gate_list.append(gates)
                    new_gate_list.append(gates)
                    new_gate_list.append(del_gates)
                    
                    new_maps.append(current_map)
                    
                    next_map = swap_qubits_by_move(current_map, gates)
                    current_map = copy.deepcopy(next_map)
                    
                    new_maps.append(current_map)
                    
                    next_map = swap_qubits_by_move(current_map, del_gates)
                    current_map = copy.deepcopy(next_map)
                    new_maps.append(current_map)
                    
                else:
                    ignore_gate_list.extend([False])
                    new_gate_list.append(del_gates)
                    
                    new_maps.append(current_map)
                    next_map = swap_qubits_by_move(current_map, del_gates)
                    current_map = copy.deepcopy(next_map)
            else:
                ignore_gate_list.extend([True,True])
                new_gate_list.append(self.gate_list[i])
                new_gate_list.append(self.gate_list[i])
                
                gates = self.gate_list[i]
                new_maps.append(current_map)
                next_map = swap_qubits_by_move(current_map, gates)
                current_map = copy.deepcopy(next_map)
                new_maps.append(current_map)
                
        self.gate_list = new_gate_list
        self.maps = new_maps
        self.qaoa = ignore_gate_list
        self.ifQft = False
        # print(del_gate_list)
    
    def remove_layers(self, n:int):
        pass
    def export_to_qasm(self, filename: str, full = True) -> None:
        """
        Export the quantum circuit to an OpenQASM file.

        :param filename: str
            The filename for the QASM output.
            full: bool
            Export full circuit or cz circuit
        """
        
        if not filename.endswith('.qasm'):
            raise ValueError("The filename must end with '.qasm' to ensure proper QASM file format.")
        if self.ifQft:
            if full:
                qasm2.dump(self.full_circuit, filename)
            else:
                qasm2.dump(self.cz_circuit, filename)
        else:
            qc = QuantumCircuit(self.num_qubits)
            for i, gates in enumerate(self.gate_list):
                if self.qaoa[i]:
                    for gate in gates:
                        qc.cz(gate[0],gate[1])
            qasm2.dump(qc, filename)
        print(f"QASM file '{filename}' has been created.")

def get_2q_gates_list(circ: QuantumCircuit)-> list[tuple[int]]:
    """
    Extracts a list of 2-qubit gate operations from the quantum circuit.

    Args:
        circ: A quantum circuit object containing the data.

    Returns:
        List of tuples containing the indices of qubits involved in 2-qubit gates.
    """
    # Use a list comprehension for concise and efficient extraction
    return [
        (ins.qubits[0]._index, ins.qubits[1]._index)
        for ins in circ.data
        if ins.operation.num_qubits == 2
    ]

def get_qubits_num(gate_2q_list: list[tuple[int]])-> int:
    """
    Calculates the total number of qubits needed based on the 2-qubit gate list.

    Args:
        gate_2q_list: A list of tuples with qubit indices involved in 2-qubit gates.

    Returns:
        Total number of qubits required for the circuit.
    """
    # Using max() directly with a generator expression
    return max(max(gate) for gate in gate_2q_list) + 1

def get_cz_circuit(circ: QuantumCircuit)-> QuantumCircuit:
    """
    Converts a list of 2-qubit gates into a quantum circuit and its DAG representation.

    Args:
        gate_2q_list: A list of tuples with qubit indices for 2-qubit gates.

    Returns:
        A tuple containing the quantum circuit and its corresponding DAG.
    """
    gate_2q_list = get_2q_gates_list(circ)
    # Get the number of qubits required for the circuit
    qubit_num = get_qubits_num(gate_2q_list)
    # Initialize a quantum circuit with the required number of qubits
    circ = QuantumCircuit(qubit_num)
    # Add CZ gates for each pair of qubits specified in the gate list
    for two_qubit_gate in gate_2q_list:
        circ.cz(two_qubit_gate[0], two_qubit_gate[1])
    return circ

def swap_qubits_by_move(locations, qubits_pair):
    new_loc = copy.deepcopy(locations)
    for q0,q1 in qubits_pair:
        new_loc[q0], new_loc[q1] = new_loc[q1], new_loc[q0]
    return new_loc