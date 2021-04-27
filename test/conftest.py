import tensorflow as tf
from typing import Any, Dict
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from c3.parametermap import ParameterMap
from c3.generator.generator import Generator
from c3.generator.devices import Crosstalk
from c3.c3objs import Quantity
import pytest


@pytest.fixture()
def get_test_circuit() -> QuantumCircuit:
    """fixture for sample Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture()
def get_bell_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit to make Bell
    State |11> + |00>

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X and 2 Measures
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture()
def get_bad_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit with
    unsupported operations

    Returns
    -------
    QuantumCircuit
        A circuit with a Conditional
    """
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    qc.x(q[0]).c_if(c, 0)
    qc.measure(q, c)
    return qc


@pytest.fixture()
def get_6_qubit_circuit() -> QuantumCircuit:
    """fixture for 6 qubit Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with an X on qubit 1
    """
    qc = QuantumCircuit(6, 6)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    return qc


@pytest.fixture()
def get_result_qiskit() -> Dict[str, Dict[str, Any]]:
    """Fixture for returning sample experiment result

    Returns
    -------
    Dict[str, Dict[str, Any]]
            A dictionary of results for physics simulation and perfect gates
            A result dictionary which looks something like::

            {
            "name": name of this experiment (obtained from qobj.experiment header)
            "seed": random seed used for simulation
            "shots": number of shots used in the simulation
            "data":
                {
                "counts": {'0x9: 5, ...},
                "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                },
            "status": status string for the simulation
            "success": boolean
            "time_taken": simulation time of this single experiment
            }

    """
    # Result of physics based sim for applying X on qubit 0 in 6 qubits
    perfect_counts = {"110000": 1000}

    counts_dict = {
        "c3_qasm_perfect_simulator": perfect_counts,
    }
    return counts_dict


@pytest.fixture()
def get_xtalk_pmap() -> ParameterMap:
    xtalk = Crosstalk(
        name="crosstalk",
        channels=["TC1", "TC2"],
        crosstalk_matrix=Quantity(
            value=[[1, 0], [0, 1]],
            min_val=[[0, 0], [0, 0]],
            max_val=[[1, 1], [1, 1]],
            unit="",
        ),
    )

    gen = Generator(devices={"crosstalk": xtalk})
    pmap = ParameterMap(generator=gen)
    pmap.set_opt_map([[["crosstalk", "crosstalk_matrix"]]])
    return pmap


@pytest.fixture()
def get_test_signal() -> Dict:
    return {
        "TC1": {"values": tf.linspace(0, 100, 101)},
        "TC2": {"values": tf.linspace(100, 200, 101)},
    }
