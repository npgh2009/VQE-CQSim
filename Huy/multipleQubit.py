# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:19:47 2020

@author: lul
"""

import numpy as np
from random import random
from scipy.optimize import minimize
from itertools import product

import qiskit
#import qiskit.aqua.operators as operator
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import Pauli, pauli_group
from qiskit.circuit.library import EfficientSU2


def hamiltonian_operator(H_weight, n = 2):
    """

    Parameters
    ----------
    n : number of qubits in circuit, default to 2
    H_weight : n-dimensional array, each dimension has 4 component
                corresponding to I, X, Y, Z Pauli matrix.
                
    Returns
    -------
    matrix representing hamiltonian

    """
    op_list = np.array(pauli_group(2)).reshape(4,4)
    result = np.zeros((4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            result += (H_weight[i][j] * op_list[i][j].to_matrix())   
    
    return result

scale = 10
n = 2
H_weight = np.ones((4,4))
#H_weight = scale*np.random.random((4,4))
H = hamiltonian_operator(H_weight, 2)

w, v = np.linalg.eigh(H)
reference_energy = min(np.real(w))
print('The exact ground state energy is: {}'.format(reference_energy))



def quantum_state_preparation(circuit, parameters, n = 2):
    """

    Parameters
    ----------
    circuit: qiskit circuit
    parameters: 8*3 array
                each row contains rx and ry parameters
                
    Returns
    -------
    qiskit circuit

    """

    q = circuit.qregs
    #U3_0, U3_1
    circuit.rx(parameters[0][0], q[i])
    circuit.ry(parameters[0][1], q[i])
        
    circuit.cx(q[1], q[0])
    
    return circuit


def quantum_state_preparation_qiskit(circuit, parameters, n = 2):
    """

    Parameters
    ----------
    circuit: qiskit circuit
    parameters: n*2 array
                each row contains rx and ry parameters
                
    Returns
    -------
    qiskit circuit

    """

    q = circuit.qregs
    for i in range(len(q)):
        circuit.rx(parameters[i][0], q[i])
        circuit.ry(parameters[i][1], q[i])
        
    circuit.cx(q[1], q[0])
    
    return circuit


def vqe_circuit(parameters, measure, n = 2):
    """

    Parameters
    ----------
    parameters: n*2 array
                each row contains rx and ry parameters
    measure: n array (or tuple)
                contain the corresponding measurement gate for each qubit
                
    Returns
    -------
    qiskit circuit

    """
    

    q = qiskit.QuantumRegister(n)
    c = qiskit.ClassicalRegister(n)
    circuit = qiskit.QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    for i in range(n):
        if measure[i] == 'Z':
            circuit.measure(q[i], c[i])
        elif measure[i] == 'X':
            circuit.u(np.pi/2, 0, np.pi, q[i])
            circuit.measure(q[i], c[i])
        elif measure[i] == 'Y':
            circuit.u(np.pi/2, 0, np.pi/2, q[i])
            circuit.measure(q[i], c[i])
        else:
            raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

    return circuit


def quantum_module(parameters, measure, n = 2):
    """

    Parameters
    ----------
    parameters: n*2 array
                each row contains rx and ry parameters
    measure: n array (or tuple)
                contain the corresponding measurement gate for each qubit
                
    Returns
    -------
    expectation value for corresponding measurement

    """
    
    circuit = vqe_circuit(parameters, measure)
    
    shots = 2048
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        sign = +1
        if measure_result == '1':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
        
    return expectation_value

### Generate all possible measurement and arrange them into (4,4) array
pauli = ['X', 'Y', 'Z']
measure_array = np.array(list(product(pauli, pauli))).reshape(3,3,n)


def vqe(parameters_list):
    
    parameters = parameters_list.reshape(n,2)
    classical_adder = 0
    
    for i in range(3):
        for j in range(3):
            classical_adder += H_weight[i][j] * quantum_module(parameters,
                                                measure_array[i][j])
    
    return classical_adder

parameters_list = np.pi*np.ones(n*2)
tol = 1e-3 # tolerance for optimization precision.


vqe_result = minimize(vqe, parameters_list, method="COBYLA", tol=tol, options={'maxiter': 10000})
#print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))