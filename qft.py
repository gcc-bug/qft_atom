from qiskit import QuantumCircuit,qasm2, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from IPython.display import display
from numpy import pi
import copy

basis_gate_set=["cz", "id", "u1", "u2", "u3"]
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

        return transpile(quantum_circuit, basis_gates=basis_gate_set, optimization_level=3)

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
    
    def export_to_qasm(self, filename: str) -> None:
        """
        Export the quantum circuit to an OpenQASM file.

        :param filename: str
            The filename for the QASM output.
        """
        if not filename.endswith('.qasm'):
            raise ValueError("The filename must end with '.qasm' to ensure proper QASM file format.")
        qasm2.dump(self.full_circuit, filename)
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