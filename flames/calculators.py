import math
from typing import Union

import ase
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import jit, prange, set_num_threads
from scipy import special  # For erfc

NUM_THREADS_TO_USE = 4
set_num_threads(NUM_THREADS_TO_USE)


@jit(nopython=True, fastmath=True)
def _getNMax_jit(cell, volume, R_cutoff):
    """
    Numba-compiled version of _getNMax.
    """
    a, b, c = cell
    cross_bc = np.cross(b, c)
    cross_ac = np.cross(a, c)
    cross_ab = np.cross(a, b)

    norm_cross_bc = np.linalg.norm(cross_bc)
    norm_cross_ac = np.linalg.norm(cross_ac)
    norm_cross_ab = np.linalg.norm(cross_ab)

    h_a = volume / norm_cross_bc if norm_cross_bc > 1e-9 else 1e18
    h_b = volume / norm_cross_ac if norm_cross_ac > 1e-9 else 1e18
    h_c = volume / norm_cross_ab if norm_cross_ab > 1e-9 else 1e18

    # Numba doesn't like np.inf, use a large number instead
    # Can't create a numpy array and then call .astype on it
    N_max_arr = np.array([R_cutoff / h_a, R_cutoff / h_b, R_cutoff / h_c])

    # Use np.ceil and manual int casting
    return (
        int(np.ceil(N_max_arr[0])),
        int(np.ceil(N_max_arr[1])),
        int(np.ceil(N_max_arr[2])),
    )


@jit(nopython=True, fastmath=True, parallel=True)
def _realspace_loop_jit(
    n_atoms,
    positions,
    charges,
    cell,
    Nx_max,
    Ny_max,
    Nz_max,
    alpha,
    R_cutoff,
):
    """
    The full Numba-compiled N^2 real-space loop.
    """
    real_energies = np.zeros(n_atoms)
    cell_T = cell.T  # Transpose for easier dot product logic

    # --- Loop over all N*N atom pairs ---
    for i in prange(n_atoms):
        i_energy_raw = 0.0
        pos_i = positions[i]

        for j in range(n_atoms):
            qi_qj = charges[i] * charges[j]
            rij_vec = pos_i - positions[j]

            # --- Loop over all N_cell periodic images ---
            for nx in range(-Nx_max, Nx_max + 1):
                for ny in range(-Ny_max, Ny_max + 1):
                    for nz in range(-Nz_max, Nz_max + 1):

                        # --- Skip n=0 term if i==j ---
                        if i == j and nx == 0 and ny == 0 and nz == 0:
                            continue

                        # --- Calculate Cartesian translation vector n ---
                        # nv = n.T @ cell = cell.T @ n
                        n_vec = np.array([float(nx), float(ny), float(nz)])
                        nv_cart_x = (
                            cell_T[0, 0] * n_vec[0]
                            + cell_T[0, 1] * n_vec[1]
                            + cell_T[0, 2] * n_vec[2]
                        )
                        nv_cart_y = (
                            cell_T[1, 0] * n_vec[0]
                            + cell_T[1, 1] * n_vec[1]
                            + cell_T[1, 2] * n_vec[2]
                        )
                        nv_cart_z = (
                            cell_T[2, 0] * n_vec[0]
                            + cell_T[2, 1] * n_vec[1]
                            + cell_T[2, 2] * n_vec[2]
                        )

                        # --- Calculate full r_ij + n vector and its norm ---
                        rv_x = rij_vec[0] + nv_cart_x
                        rv_y = rij_vec[1] + nv_cart_y
                        rv_z = rij_vec[2] + nv_cart_z

                        r_norm_sq = rv_x * rv_x + rv_y * rv_y + rv_z * rv_z
                        r_norm = np.sqrt(r_norm_sq)

                        # --- Apply cutoff and r>0 check ---
                        if r_norm <= R_cutoff and r_norm > 1e-9:

                            # --- Use math.erfc (Numba knows this one) ---
                            term = math.erfc(alpha * r_norm) / r_norm
                            i_energy_raw += qi_qj * term

        # Apply 0.5 prefactor
        real_energies[i] = 0.5 * i_energy_raw

    return real_energies


@jit(nopython=True, fastmath=True, parallel=True)
def _reciprocal_loop_jit(
    n_atoms,
    positions,
    charges,
    kv_cartesian,  # (N_k_vectors, 3)
    Ak_terms,  # (N_k_vectors,)
):
    """
    Numba-compiled version of the reciprocal space loop.
    Avoids creating the massive (N_atoms, N_k_vectors) matrix.
    """
    n_k_vectors = kv_cartesian.shape[0]
    recip_energies_raw = np.zeros(n_atoms)

    # We need to handle complex numbers. Numba can do this.
    Q_vector = np.zeros(n_k_vectors, dtype=np.complex128)

    # --- Pass 1: Calculate Q(k) = sum_j q_j * exp(i * k.r_j) ---
    for k_idx in prange(n_k_vectors):
        k_vec = kv_cartesian[k_idx]

        Qk_real = 0.0
        Qk_imag = 0.0

        for j in range(n_atoms):
            pos_j = positions[j]
            k_dot_r = k_vec[0] * pos_j[0] + k_vec[1] * pos_j[1] + k_vec[2] * pos_j[2]

            # exp(i*x) = cos(x) + i*sin(x)
            cos_kr = np.cos(k_dot_r)
            sin_kr = np.sin(k_dot_r)

            Qk_real += charges[j] * cos_kr
            Qk_imag += charges[j] * sin_kr

        Q_vector[k_idx] = Qk_real + 1j * Qk_imag

    Q_conj_vector = np.conjugate(Q_vector)

    # --- Pass 2: Calculate per-atom energy ---
    # E_recip_i = q_i * sum_k [ A(k) * Re( exp(i*k.r_i) * Q(k)* ) ]
    for i in prange(n_atoms):
        pos_i = positions[i]
        q_i = charges[i]
        sum_over_k = 0.0

        for k_idx in range(n_k_vectors):
            k_vec = kv_cartesian[k_idx]
            Ak = Ak_terms[k_idx]
            Qk_conj = Q_conj_vector[k_idx]  # This is Q(k)*

            k_dot_r = k_vec[0] * pos_i[0] + k_vec[1] * pos_i[1] + k_vec[2] * pos_i[2]

            # exp(i*k.r_i)
            exp_kri = np.cos(k_dot_r) + 1j * np.sin(k_dot_r)

            # term_in_brackets = exp(i*k.r_i) * Q(k)*
            term_in_brackets = exp_kri * Qk_conj

            sum_over_k += Ak * np.real(term_in_brackets)

        recip_energies_raw[i] = q_i * sum_over_k

    return recip_energies_raw


class EwaldSum(Calculator):
    """
    A generalized, vectorized Ewald summation calculator for ASE.

    This class calculates the electrostatic energy of a periodic structure
    using the Ewald summation method. It is generalized for any unit cell
    and uses numpy vectorization for speed.

    It calculates per-atom energies, and the total energy is the sum of these.

    The energy is E = E_real + E_reciprocal + E_self.
    """

    implemented_properties = [
        "energy",
        "free_energy",
        "energies",
        # 'forces', 'stress', # to be implemented
    ]

    def __init__(self, R_cutoff: float, G_cutoff_N: float, alpha: float, **kwargs) -> None:
        """
        Initializes the EwaldSum calculator.

        Args:
            R_cutoff: The cutoff radius for the real-space sum (Angstroms).
            G_cutoff_N: The *integer* cutoff for the reciprocal-space sum.
                        (sum over nx^2 + ny^2 + nz^2 <= G_cutoff_N^2).
            alpha: The Ewald splitting parameter (in 1/Angstroms).
                   A common choice is 5.0 / L (for a cell of length L).
        """

        Calculator.__init__(self, **kwargs)

        self.R_cutoff = R_cutoff
        self.G_cutoff_N = G_cutoff_N
        self.alpha = alpha

    def _getNMax(self, cell, volume) -> tuple[int, int, int]:
        """
        Calculates the number of maximum unit cells based on the "height"
        of the cell perpendicular to each pair of vectors and the Real
        space cutoff (R_cutoff)

        Returns
        -------
        N_max: tuple[int, int, int]
            The nx_max, ny_max, and nz_max numbers of unit cells

        """
        a, b, c = cell
        cross_bc = np.cross(b, c)
        cross_ac = np.cross(a, c)
        cross_ab = np.cross(a, b)

        # Add safety checks for 2D/1D systems
        norm_cross_bc = np.linalg.norm(cross_bc)
        norm_cross_ac = np.linalg.norm(cross_ac)
        norm_cross_ab = np.linalg.norm(cross_ab)

        h_a = volume / norm_cross_bc if norm_cross_bc > 1e-9 else np.inf
        h_b = volume / norm_cross_ac if norm_cross_ac > 1e-9 else np.inf
        h_c = volume / norm_cross_ab if norm_cross_ab > 1e-9 else np.inf

        N_max = np.ceil(self.R_cutoff / np.array([h_a, h_b, h_c])).astype(int).tolist()

        return N_max

    def _realspaceEnergyOld(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the real-space energy (vectorized) per atom.
        E_real_i = 0.5 * sum_j,n' [ q_i*q_j * erfc(alpha*|r_ij + n|) / |r_ij + n| ]

        Returns:
            np.ndarray: Array of size (N_atoms) with raw real-space energy
                        per atom (in e^2/A).
        """
        n_atoms = len(structure)
        cell = structure.cell.array
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()
        real_energies = np.zeros(n_atoms)

        # Find max translation vectors (n_x, n_y, n_z) needed
        Nx_max, Ny_max, Nz_max = self._getNMax(cell=cell, volume=structure.cell.volume)

        # Create grid of n-vectors [nx, ny, nz]
        n_x_range = np.arange(-Nx_max, Nx_max + 1, dtype=int)
        n_y_range = np.arange(-Ny_max, Ny_max + 1, dtype=int)
        n_z_range = np.arange(-Nz_max, Nz_max + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_x_range, n_y_range, n_z_range, indexing="ij")

        # (N_cells, 3) array of n-vectors
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # (N_cells, 3) array of Cartesian translation vectors
        nv_vectors_cart = np.dot(n_vectors_flat, cell)

        # (N_cells) boolean array identifying n=0 vector
        n_is_zero = ~np.any(n_vectors_flat, axis=1)

        for i in range(n_atoms):
            i_energy_raw = 0.0
            for j in range(n_atoms):
                qi_qj = charges[i] * charges[j]
                rij_vec = positions[i] - positions[j]

                # (N_cells, 3) array of all translated r_ij vectors
                rv_vectors = rij_vec + nv_vectors_cart

                # (N_cells) array of distances
                r_norms = np.linalg.norm(rv_vectors, axis=1)

                if i == j:
                    # For self-interaction, only consider n != 0
                    r_norms = r_norms[~n_is_zero]

                # Apply cutoff
                r_norms_inside_cutoff = r_norms[r_norms <= self.R_cutoff]

                # Filter out r=0 (e.g., overlapping atoms at n=0)
                r_norms_valid = r_norms_inside_cutoff[r_norms_inside_cutoff > 1e-9]

                if r_norms_valid.size > 0:
                    with np.errstate(divide="ignore"):
                        terms = special.erfc(self.alpha * r_norms_valid) / r_norms_valid
                        i_energy_raw += qi_qj * np.sum(terms)

            # The per-atom energy is 1/2 of its sum with all other atoms
            real_energies[i] = 0.5 * i_energy_raw

        return real_energies

    def _realspaceEnergy(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the real-space energy (vectorized) per atom.

        This is now a wrapper that calls the Numba-compiled JIT function.
        """
        # 1. Extract raw numpy arrays
        n_atoms = len(structure)
        cell = structure.cell.array
        volume = structure.cell.volume
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        # 2. Get loop bounds from the first JIT function
        Nx_max, Ny_max, Nz_max = _getNMax_jit(cell, volume, self.R_cutoff)

        # 3. Call the main JIT loop function
        # The first time this runs, it will take a second to compile.
        # Every run after that will be extremely fast.
        real_energies = _realspace_loop_jit(
            n_atoms,
            positions,
            charges,
            cell,
            Nx_max,
            Ny_max,
            Nz_max,
            self.alpha,
            self.R_cutoff,
        )

        return real_energies

    def _reciprocalEnergy(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the reciprocal-space energy (vectorized) per atom.

        This is now a wrapper that calls the Numba-compiled JIT function.
        """
        n_atoms = len(structure)
        volume = structure.cell.volume
        reciprocal_cell_matrix = 2.0 * np.pi * np.linalg.inv(structure.cell.array).T
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        Nmax = int(np.ceil(self.G_cutoff_N))
        G_cutoff_N_sq = self.G_cutoff_N**2

        # Create grid of n-vectors [nx, ny, nz] for the sum
        n_range = np.arange(-Nmax, Nmax + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_range, n_range, n_range, indexing="ij")
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # --- Filter n-vectors to remove k=0
        n_is_not_zero = np.any(n_vectors_flat, axis=1)
        n_vectors_nonzero = n_vectors_flat[n_is_not_zero]

        n_norm_sq = np.sum(n_vectors_nonzero**2, axis=1)
        n_vectors_valid = n_vectors_nonzero[n_norm_sq <= G_cutoff_N_sq]

        if n_vectors_valid.shape[0] == 0:
            return np.zeros(n_atoms)  # No k-vectors in cutoff

        # (N_k_vectors, 3) array of Cartesian k-vectors
        kv_cartesian = np.dot(n_vectors_valid, reciprocal_cell_matrix)

        # (N_k_vectors) array of k-magnitudes squared
        k_norm_sq = np.sum(kv_cartesian**2, axis=1)

        # A(k) term = (1/k^2) * exp(-k^2 / (4*alpha^2))
        Ak_terms = (1.0 / k_norm_sq) * np.exp(-k_norm_sq / (4.0 * self.alpha**2))

        # --- Hand off to Numba for the big O(N*Nk) loops ---
        recip_energies_raw = _reciprocal_loop_jit(
            n_atoms, positions, charges, kv_cartesian, Ak_terms
        )

        # Apply prefactor
        rec_energy = (2 * np.pi / volume) * recip_energies_raw

        return rec_energy

    def _reciprocalEnergyOld(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the reciprocal-space energy (vectorized) per atom.
        E_recip_i = q_i * (2*pi/V) * sum_k!=0 [ A(k) * Re( exp(i*k.r_i) * Q(k)* ) ]

        Returns:
            np.ndarray: Array of size (N_atoms) with raw recip. energy
                        per atom (in e^2/A).
        """
        n_atoms = len(structure)
        volume = structure.cell.volume
        reciprocal_cell_matrix = 2.0 * np.pi * np.linalg.inv(structure.cell.array).T
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        Nmax = int(np.ceil(self.G_cutoff_N))
        G_cutoff_N_sq = self.G_cutoff_N**2

        # Create grid of n-vectors [nx, ny, nz]
        n_range = np.arange(-Nmax, Nmax + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_range, n_range, n_range, indexing="ij")
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # --- Filter n-vectors ---
        # 1. Remove n=0 term
        n_is_not_zero = np.any(n_vectors_flat, axis=1)
        n_vectors_nonzero = n_vectors_flat[n_is_not_zero]

        # 2. Apply spherical cutoff in index space
        n_norm_sq = np.sum(n_vectors_nonzero**2, axis=1)
        n_vectors_valid = n_vectors_nonzero[n_norm_sq <= G_cutoff_N_sq]

        if n_vectors_valid.shape[0] == 0:
            return np.zeros(n_atoms)  # No k-vectors in cutoff

        # --- Calculate k-vectors and A(k) term ---
        # (N_k_vectors, 3) array of Cartesian k-vectors
        kv_cartesian = np.dot(n_vectors_valid, reciprocal_cell_matrix)

        # (N_k_vectors) array of k-magnitudes squared
        k_norm_sq = np.sum(kv_cartesian**2, axis=1)

        # A(k) term = (1/k^2) * exp(-k^2 / (4*alpha^2))
        # Note: This is A(k) *without* the (2*pi/V) prefactor
        Ak_terms = (1.0 / k_norm_sq) * np.exp(-k_norm_sq / (4.0 * self.alpha**2))

        # --- Calculate Per-Atom Term ---
        # We need: q_i * Re( exp(i*k.r_i) * Q(k)* )

        # (N_atoms, N_k_vectors) array of exp(i * k.r_i)
        k_dot_r_matrix = np.dot(positions, kv_cartesian.T)
        exp_k_dot_r = np.exp(1j * k_dot_r_matrix)

        # (N_k_vectors) array of Q(k) = sum_j q_j * exp(i * k.r_j)
        Q_vector = np.dot(charges, exp_k_dot_r)

        # (N_k_vectors) array of Q(k)* (conjugate)
        Q_conj_vector = np.conjugate(Q_vector)

        # (N_atoms, N_k_vectors) array of exp(i*k.r_i) * Q(k)*
        term_in_brackets = exp_k_dot_r * Q_conj_vector

        # (N_atoms, N_k_vectors) array of Re[ ... ]
        real_term = np.real(term_in_brackets)

        # (N_atoms) array: sum_k [ Ak * Re(...) ]
        sum_over_k = np.dot(real_term, Ak_terms)

        # (N_atoms) array: q_i * sum_k [ ... ]
        recip_energies_raw = charges * sum_over_k

        # Apply prefactor
        rec_energy = (2 * np.pi / volume) * recip_energies_raw

        return rec_energy

    def _selfEnergy(self, charges: np.ndarray) -> np.ndarray:
        """
        Calculates the self-energy correction term per atom.
        E_self_i = - (alpha / sqrt(pi)) * q_i^2

        Returns:
            np.ndarray: Array of size (N_atoms) with raw self-energy
                        per atom (in e^2/A).
        """
        # E_self_i = - (alpha / sqrt(pi)) * q_i^2
        return -(self.alpha / np.sqrt(np.pi)) * (charges**2)

    def calculate(
        self,
        atoms: Union[ase.Atoms, None] = None,
        properties: list[str] = ["energy", "energies"],
        system_changes=all_changes,
    ):

        Calculator.calculate(self, atoms, properties, system_changes)

        # Coulomb constant Ke = 1 / (4 * pi * e0) in eV . Angs / e^2
        Ke = units.C * units.m * units._e**2 / (4 * np.pi * units._eps0)

        charges = self.atoms.get_initial_charges()  # type: ignore

        # --- Calculate all per-atom components (raw, in e^2/A) ---
        real_energies_raw = self._realspaceEnergy(self.atoms)  # type: ignore
        recip_energies_raw = self._reciprocalEnergy(self.atoms)  # type: ignore
        self_energies_raw = self._selfEnergy(charges)

        volume = self.atoms.cell.volume  # type: ignore
        total_charge = np.sum(charges)
        neutrality_energy_raw = 0.0

        if abs(total_charge) > 1e-9:  # Check if system is non-neutral
            neutrality_energy_raw = -(np.pi / (2.0 * volume * self.alpha**2)) * (total_charge**2)

        # --- Sum them to get the total per-atom array (raw) ---
        total_energies_raw = real_energies_raw + recip_energies_raw + self_energies_raw

        # --- Convert to eV ---
        total_energies_ev = total_energies_raw * Ke
        neutrality_energy_ev = neutrality_energy_raw * Ke

        # --- Store results ---
        if "energies" in properties:
            self.results["energies"] = total_energies_ev

        if "energy" in properties:
            total_energy_ev = np.sum(total_energies_ev)

            # Add the global neutrality correction
            final_total_energy = total_energy_ev + neutrality_energy_ev

            self.results["energy"] = final_total_energy
            self.results["free_energy"] = final_total_energy


class CustomLennardJones(Calculator):
    """
    Custom Lennard Jones potential calculator based on the ASE calculator interface.
    This method is intended to be as close as possible to RASPA2 implementation.

    The fundamental definition of this potential is a pairwise energy:

    ``u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )``

    For convenience, we'll use d_ij to refer to "distance vector" and
    ``r_ij`` to refer to "scalar distance". So, with position vectors `r_i`:

    ``r_ij = | r_j - r_i | = | d_ij |``

    Therefore:

    ``d r_ij / d d_ij = + d_ij / r_ij``
    ``d r_ij / d d_i  = - d_ij / r_ij``

    The derivative of u_ij is:

    ::

        d u_ij / d r_ij
        = (-24 epsilon / r_ij) ( 2 sigma^12/r_ij^12 - sigma^6/r_ij^6 )

    We can define a "pairwise force"

    ``f_ij = d u_ij / d d_ij = d u_ij / d r_ij * d_ij / r_ij``

    The terms in front of d_ij are combined into a "general derivative".

    ``du_ij = (d u_ij / d d_ij) / r_ij``

    We do this for convenience: `du_ij` is purely scalar The pairwise force is:

    ``f_ij = du_ij * d_ij``

    The total force on an atom is:

    ``f_i = sum_(j != i) f_ij``

    There is some freedom of choice in assigning atomic energies, i.e.
    choosing a way to partition the total energy into atomic contributions.

    We choose a symmetric approach (`bothways=True` in the neighbor list):

    ``u_i = 1/2 sum_(j != i) u_ij``

    The total energy of a system of atoms is then:

    ``u = sum_i u_i = 1/2 sum_(i, j != i) u_ij``

    Differentiating `u` with respect to `r_i` yields the force,
    independent of the choice of partitioning.

    ::

        f_i = - d u / d r_i = - sum_ij d u_ij / d r_i
            = - sum_ij d u_ij / d r_ij * d r_ij / d r_i
            = sum_ij du_ij d_ij = sum_ij f_ij

    This justifies calling `f_ij` pairwise forces.

    The stress can be written as ( `(x)` denoting outer product):

    ``sigma = 1/2 sum_(i, j != i) f_ij (x) d_ij = sum_i sigma_i ,``
    with atomic contributions

    ``sigma_i  = 1/2 sum_(j != i) f_ij (x) d_ij``

    Another consideration is the cutoff. We have to ensure that the potential
    goes to zero smoothly as an atom moves across the cutoff threshold,
    otherwise the potential is not continuous. In cases where the cutoff is
    so large that u_ij is very small at the cutoff this is automatically
    ensured, but in general, `u_ij(rc) != 0`.

    This implementation offers two ways to deal with this:

    Either, we shift the pairwise energy

    ``u'_ij = u_ij - u_ij(rc)``

    which ensures that it is precisely zero at the cutoff. However, this means
    that the energy effectively depends on the cutoff, which might lead to
    unexpected results! If this option is chosen, the forces discontinuously
    jump to zero at the cutoff.

    An alternative is to modify the pairwise potential by multiplying
    it with a cutoff function that goes from 1 to 0 between an onset radius
    ro and the cutoff rc. If the function is chosen suitably, it can also
    smoothly push the forces down to zero, ensuring continuous forces as well.
    In order for this to work well, the onset radius has to be set suitably,
    typically around 2*sigma.

    In this case, we introduce a modified pairwise potential:

    ``u'_ij = fc * u_ij``

    The pairwise forces have to be modified accordingly:

    ``f'_ij = fc * f_ij + fc' * u_ij``

    Where `fc' = d fc / d d_ij`.

    This approach is taken from Jax-MD (https://github.com/google/jax-md),
    which in turn is inspired by HOOMD Blue
    (https://glotzerlab.engin.umich.edu/hoomd-blue/).

    """

    implemented_properties = ["energy", "energies"]
    default_parameters = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "rc": None,
        "ro": None,
        "smooth": False,
    }
    nolabel = True

    def __init__(self, lj_parameters: dict, **kwargs):
        """
        Parameters
        ----------
        lj_parameters : dict
            Dictionary containing the Lennard-Jones parameters.
            The parameters should be in the form:
            "O": {
                "sigma": 3.03315,  # In Angstroms
                "epsilon": 48.1581 # In Kelvin
                }
        vdw_cutoff : float, optional
            Cutoff distance for the van der Waals interactions.
            Default is 12.0 Angstroms.
        """

        Calculator.__init__(self, **kwargs)

        self.lj_params: dict = lj_parameters
        self.vdw_cutoff = kwargs.get("vdw_cutoff", 12.0)
        self.shifted = kwargs.get("shifted", True)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        np.seterr(invalid="ignore")

        nAtoms = len(self.atoms)  # type: ignore

        if "labels" in self.atoms.arrays.keys():  # type: ignore
            # Replace missing labels with atomic type
            for i, label in enumerate(self.atoms.arrays["labels"]):  # type: ignore
                if label == 0:
                    self.atoms.arrays["labels"][i] = self.atoms.symbols[i]  # type: ignore
        else:
            self.atoms.arrays["labels"] = self.atoms.symbols  # type: ignore

        # Preallocate arrays
        sigmas = np.empty((nAtoms, nAtoms))
        epsilons = np.empty((nAtoms, nAtoms))

        sigma_vec = np.array(
            [self.lj_params[s]["sigma"] for s in self.atoms.arrays["labels"]]  # type: ignore
        )
        epsilon_vec = np.array(
            [self.lj_params[s]["epsilon"] for s in self.atoms.arrays["labels"]]  # type: ignore
        )

        # Use broadcasting instead of loops
        sigmas = (sigma_vec[:, None] + sigma_vec[None, :]) / 2
        epsilons = np.sqrt(epsilon_vec[:, None] * epsilon_vec[None, :])

        rij = self.atoms.get_all_distances(mic=True)  # type: ignore

        # We must avoid division by zero for i=j pairs.
        # Set diagonal to infinity so energy contribution becomes zero.
        np.fill_diagonal(rij, np.inf)

        # Calculate the energy for *all* pairs
        # (This is vectorized and fast)
        s_over_r = sigmas / rij
        s_over_r_6 = s_over_r**6
        energy = 4 * epsilons * (s_over_r_6**2 - s_over_r_6)

        if self.shifted:
            # --- Shifted Potential Logic ---
            # Calculate the energy shift at the cutoff distance
            s_over_rc = sigmas / self.vdw_cutoff
            s_over_rc_6 = s_over_rc**6
            energy_shift = 4 * epsilons * (s_over_rc_6**2 - s_over_rc_6)

            # Apply cutoff AND shift
            # First, set everything outside the cutoff to 0
            energy[rij > self.vdw_cutoff] = 0.0

            # Create a mask for interactions *inside* the cutoff
            mask = (rij > 0) & (rij <= self.vdw_cutoff)

            # Subtract the shift from all interactions *inside* the cutoff
            energy[mask] -= energy_shift[mask]

        # NOW, apply the cutoff:
        # Set energy to 0 for all pairs *outside* the cutoff.
        energy[rij > self.vdw_cutoff] = 0.0

        # Sum the energy matrix and divide by 2 to avoid double counting since the energy matrix is symmetric
        energy /= 2

        # Convert from K to eV
        energy *= units.kB

        self.results["energy"] = energy.sum()
        self.results["energies"] = energy.sum(axis=1)
        self.results["free_energy"] = energy.sum()
