# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:19:47 2020

@author: lul
"""

import numpy as np
from random import random
import time
from scipy.optimize import minimize
from itertools import product

import qiskit
#import qiskit.aqua.operators as operator
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import Pauli, pauli_group
from qiskit.circuit.library import EfficientSU2

start_time = time.time()

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

scale = 1
n = 2
#np.random.seed(113)
#H_weight = np.ones((4,4))
H_weight = scale*np.random.random((4,4))
H = hamiltonian_operator(H_weight, 2)

w, v = np.linalg.eigh(H)
reference_energy = min(np.real(w))
print('The exact ground state energy is: {}'.format(reference_energy))


def state_preparation_qiskit(parameters):
    
    #parameters 8x3 array => 24 parameters
    
    q = qiskit.QuantumRegister(2)
    c = qiskit.ClassicalRegister(2)
    circuit = qiskit.QuantumCircuit(q, c)
    
    #U3_0, U3_1
    circuit.u3(*parameters[0], q[0])
    circuit.u3(*parameters[1], q[1])
    
    #first cnot
    circuit.cx(q[1], q[0])
    
    #U3_2, U3_3
    circuit.u3(*parameters[2], q[0])
    circuit.u3(*parameters[3], q[1])
    
    #second cnot
    circuit.cx(q[0], q[1])
    
    #U3_4, U3_5
    circuit.u3(*parameters[4], q[0])
    circuit.u3(*parameters[5], q[1])

    #third cnot
    circuit.cx(q[1], q[0])
    
    #U3_6, U3_7
    circuit.u3(*parameters[6], q[0])
    circuit.u3(*parameters[7], q[1])
    
    return circuit

def state_preparation_shende(parameters):
    
    #parameters 15x1
    
    q = qiskit.QuantumRegister(2)
    c = qiskit.ClassicalRegister(2)
    circuit = qiskit.QuantumCircuit(q, c)
    
    #c, d gates
    circuit.u3(*parameters[0:3], q[0])
    circuit.u3(*parameters[3:6], q[1])
    
    #first cnot
    circuit.cx(q[1], q[0])
    
    #rz, ry
    circuit.rz(parameters[6], q[0])
    circuit.ry(parameters[7], q[1])
    
    #second cnot
    circuit.cx(q[0], q[1])

    #I, ry
    circuit.ry(parameters[8], q[1])
    
    #third cnot
    circuit.cx(q[1], q[0])
    
    #a, b gates
    circuit.u3(*parameters[9:12], q[0])
    circuit.u3(*parameters[12:15], q[1])
    
    return circuit
    
    
def single_expectation_value(circuit_type, parameters, measure):
    
    if circuit_type == 'qiskit':
        circuit = state_preparation_qiskit(parameters)
    elif circuit_type == 'shende':
        circuit = state_preparation_shende(parameters)
    
    q = circuit.qregs[0]
    c = circuit.cregs[0]

    for i in range(2):
        if measure[i] == 'Z':
            circuit.measure(q[i], c[i])
        elif measure[i] == 'X':
            circuit.u(np.pi/2, 0, np.pi, q[i])
            circuit.measure(q[i], c[i])
        elif measure[i] == 'Y':
            circuit.u(np.pi/2, 0, np.pi/2, q[i])
            circuit.measure(q[i], c[i])
    
    shots = 1024
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        if (measure_result == '00') or (measure_result == '11'):
            sign = +1
        if (measure_result == '01') or (measure_result == '10'):
            sign = -1
        expectation_value += sign * counts[measure_result]
        
    expectation_value = expectation_value / shots
        
    return expectation_value


pauli = ['X', 'Y', 'Z']
measure_array = np.array(list(product(pauli, pauli))).reshape(3,3,n)

cost = []

def total_expectation_value(parameters_list, circuit_type):
    
    if circuit_type == 'qiskit':        
        parameters = parameters_list.reshape(8, 3)
    elif circuit_type == 'shende':  
        parameters = parameters_list
    
    total_expectation = 0
    
    for i in range(3):
        for j in range(3):
            total_expectation += H_weight[i][j] * \
                single_expectation_value(circuit_type, parameters,
                                                measure_array[i][j])
    
    cost.append(total_expectation)
    
    return total_expectation

class costFn():
    def __init__(self):
        self.cost = []
    
    def addValue(self, x):
        self.cost.append(x)
        
    def values(self):
        return self.cost


def vqe(circuit_type, optimizer):
    global cost
    if circuit_type == 'qiskit':
        parameters_list = 0.1*np.ones(8*3)   
    elif circuit_type == 'shende':
        parameters_list = 0.1*np.ones(15)
    
    tol = 1e-3 # tolerance for optimization precision.
    vqe_result = minimize(total_expectation_value,
        parameters_list, args = (circuit_type), method=optimizer)
    
    return vqe_result
    

circuit_type = 'shende' #qiskit or shende
optimizer = 'Powell' #scipy minimize
print('Using quantum circuit in %s with optimizer %s' % (circuit_type, optimizer))
vqe_result = vqe(circuit_type, optimizer)

#print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))
print("--- %s seconds ---" % (time.time() - start_time))