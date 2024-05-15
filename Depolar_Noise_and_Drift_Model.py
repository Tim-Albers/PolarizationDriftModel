# IMPORTS --------------------------------------------------------

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.operators import Operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from argparse import ArgumentParser


# FUNCTIONS --------------------------------------------------------

def random_innovation_matrix(sigma_p):
    """Generate a random unitary matrix representing a Jones matrix for polarization drift."""
    # Random variables for the matrix exponential components
    theta = np.random.normal(0, sigma_p) # Random angle (sigma should not be squared)
    a = np.random.normal(0, 1, 3) # Random vector
    a /= np.linalg.norm(a)  # Normalize to get a random unit vector
    # Pauli matrices as basic rotation generators
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Construct the weighted Pauli matrix
    weighted_pauli = a[0] * sigma_x + a[1] * sigma_y + a[2] * sigma_z
    # Calculate the unitary matrix using matrix exponential
    jones_matrix = expm(-1j * theta * weighted_pauli)
    return Operator("CustomJones", jones_matrix)

def simulate_polarization_drift(qubit, sigma_p, steps):
    """Simulate polarization drift on a qubit using a sequence of Jones matrices."""
    for _ in range(steps):
        unitary_op = random_innovation_matrix(sigma_p)
        qapi.operate(qubit, unitary_op)


# DRIFT MODEL --------------------------------------------------------

# Take an average of the fidelity over multiple runs
def average_fidelity_drift_model(num_runs, sigma_p, steps):
    """Calculate the average fidelity of the qubit state after polarization drift over multiple runs."""
    fidelities = []
    for _ in range(num_runs):
        qubit = qapi.create_qubits(1)
        qubit_init = qapi.create_qubits(1)
        simulate_polarization_drift(qubit[0], sigma_p, steps)
        fidelities.append(ns.qubits.fidelity(qubit[0], reference_state=ns.s0, squared=True))
    return np.mean(fidelities)

# Calculate the average fidelities for different polarization drift standard deviations
def calculate_average_fidelities_drift_model(num_runs, delta_ps, T, v_fiber, channel_length):
    """Calculate the average fidelities for different polarization drift standard deviations."""
    fidelities = []
    for delta_p in delta_ps:
        sigma_p = np.sqrt(2 * delta_p * T)
        steps = int(channel_length / (v_fiber * T))
        fidelities.append(average_fidelity_drift_model(num_runs=num_runs, sigma_p=sigma_p, steps=steps))
    return fidelities

def calculate_average_fidelities_channel_length(num_runs, delta_p, T, v_fiber, channel_lengths):
    """Calculate the average fidelities for different channel lengths."""
    fidelities = []
    for length in channel_lengths:
        steps = int(length / (v_fiber * T))
        fidelities.append(average_fidelity_drift_model(num_runs=num_runs, sigma_p=np.sqrt(2 * delta_p * T), steps=steps))
    return fidelities


# DEPOLARIZING NOISE MODEL --------------------------------------------

def simulate_depolar_model(qubit, depolar_prob, time_independent=True):
    """Apply depolarizing drift to a qubit using DepolarNoiseModel."""
    # Set up the depolarizing model
    depolar_model = DepolarNoiseModel(depolar_rate=depolar_prob, time_independent=time_independent)
    # Apply the noise model
    if time_independent:
        # If time-independent, delta_time does not matter
        depolar_model.error_operation([qubit])
    else:
        # Else, specify the time each operation applies (e.g., 1 ns per step)
        depolar_model.error_operation([qubit], delta_time=1)  # 1 ns per step for simplicity

def average_fidelity_depolar_model(num_runs, depolar_prob, time_independent=True):
    """Calculate the average fidelity of the qubit state after depolarizing drift over multiple runs."""
    fidelities = []
    for _ in range(num_runs):
        qubit = qapi.create_qubits(1)
        qubit_init = qapi.create_qubits(1)
        simulate_depolar_model(qubit[0], depolar_prob, time_independent)
        fidelities.append(ns.qubits.fidelity(qubit[0], reference_state=ns.s0, squared=True))
    return np.mean(fidelities)

def calculate_average_fidelities_depolar_model(num_runs, depolar_probs, time_independent=True):
    """Calculate the average fidelities for different depolarization probabilities."""
    fidelities = []
    for depolar_prob in depolar_probs:
        fidelities.append(average_fidelity_depolar_model(num_runs=num_runs, depolar_prob=depolar_prob, time_independent=time_independent))
    return fidelities


# FITTING a--------------------------------------------------------

def F(L, a):
    return 0.5 * (1 + np.exp(-a * L))

def fit_a(L, fidelities):
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(F, L, fidelities)
    return popt[0]

# MAIN --------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--delta_p", type=float, help="Polarization linewidth (Hz)")
    args = parser.parse_args()
    delta_p = args.delta_p if args.delta_p else 1000
    L = np.linspace(0, 500, 1000) # Channel lengths (km)
    #delta_p = 1000 # Polarization linewidth (Hz)
    T = 1e-6 # Time between drift operations (s)
    v_fiber = 2e5 # Speed of light in fiber (km/s)
    num_runs = 1000 # Number of runs for averaging

    # Calculate the average fidelities for different channel lengths
    print("Request recieved")
    print(f"-------------------------\ndelpta_p = {delta_p} Hz\n-------------------------\n")
    print("Calculating average fidelities for different channel lengths...")
    fidelities = calculate_average_fidelities_channel_length(num_runs, delta_p, T, v_fiber, L)

    # Fit the curve
    print("Fitting the curve...")
    a = fit_a(L, fidelities)
    print("Done!")
    print("\nResults:")
    print("-------------------------")
    print(f"delta_p = {delta_p} Hz")
    print(f"a = {a}")

    # Plot the results
    plt.figure()
    plt.plot(L, fidelities, label="Simulated fidelities")
    plt.plot(L, F(L, a), label=f"Fitted curve")
    plt.xlabel("Channel length (km)")
    plt.ylabel("Average Fidelity")
    plt.legend()
    plt.grid()
    plt.savefig(f"fidelity_vs_length(dp={delta_p}).png")
    plt.show()