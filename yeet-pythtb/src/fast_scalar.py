import numpy as np
from numba import njit, prange
from numba.types import float64, complex128


@njit(parallel=True, fastmath=True)
def solve_all(
    dim_k, per, orb, norb, nsta, site_energies, hst, hind, hR, k_list, eig_vectors=False
):
    nk = len(k_list)  # number of k points and hoppings
    # Initialize arrays eigvecs/eigvals
    eigvals = np.zeros((nsta, nk), dtype=float64)
    eigvecs = np.zeros((nsta, nk, norb), dtype=complex128)

    # Generate the Hamiltonian for all k points
    H_k = generate_H(dim_k, per, orb, norb, site_energies, hst, hind, hR, k_list)

    # Solve the Hamiltonian for all k points
    if eig_vectors:
        for k in prange(len(k_list)):
            eigvals[:, k], eigvecs[:, k, :] = np.linalg.eigh(H_k[k])
    else:
        for k in prange(len(k_list)):
            eigvals[:, k] = np.linalg.eigvalsh(H_k[k])

    return (eigvals, eigvecs)


@njit(parallel=True, fastmath=True)
def generate_H(dim_k, per, orb, norb, site_energies, hst, hind, hR, k_list):
    nk, nhop = len(k_list), len(hst)  # number of k points and hoppings
    H_k = np.zeros((nk, norb, norb), dtype=complex128)
    # @njit(parallel=True) friendly implementation of
    # H_k = np.repeat(np.diag(site_energies+0j)[None, :, :], nkp, axis=0)
    for k in prange(nk):
        np.fill_diagonal(H_k[k], site_energies)
    all_i, all_j = hind.T
    amp = np.zeros((nk, nhop), dtype=np.complex128)
    if dim_k > 0:
        # r_{ij} = \tau_j - \tau_i + R
        r_ij = -orb[all_i, :] + orb[all_j, :] + hR
        r_ij = r_ij[:, per]  # Neglect non-periodic components
        # In parallel, lots of stuff doesn't work...
        # Can we find a way to do this without a loop that works in parallel?
        phase = np.exp(2 * np.pi * 1j * (k_list @ r_ij.T))
        for k in prange(nk):
            amp[k, :] = hst * phase[k, :]
    else:
        # In 0-dim case there is no phase factor
        for k in prange(nk):
            amp[k, :] = hst

    # Numba doesn't support np.add.at, need loop
    for k in prange(nk):
        for h, (i, j) in enumerate(zip(all_i, all_j)):
            H_k[k, i, j] += amp[k, h]
            H_k[k, j, i] += np.conjugate(amp[k, h])
    return H_k
