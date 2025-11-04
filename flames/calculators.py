import ase
import numpy as np
from scipy import special  # For erfc
from typing import Union
from ase.calculators.calculator import Calculator, all_changes

class EwaldSumCalc(Calculator):
    """
    A generalized, vectorized Ewald summation calculator.

    This class calculates the electrostatic energy of a periodic structure
    using the Ewald summation method. It is generalized for any unit cell
    and uses numpy vectorization for speed.

    The energy is E = E_real + E_reciprocal + E_self.
    """  

    implemented_properties = [
        'energy', 'free_energy'
         # 'energies', 'forces', 'stress', # to be inplemented
    ]

    def __init__(self,
                 R_cutoff: float,
                 G_cutoff_N: float,
                 alpha: float,
                 **kwargs) -> None:
        """
        Initializes the EwaldSum calculator.

        Args:
            structure: The ase.Atoms object.
            R_cutoff: The cutoff radius for the real-space sum (Angstroms).
            G_cutoff_N: The *integer* cutoff for the reciprocal-space sum.
                        (sum over nx^2 + ny^2 + nz^2 <= G_cutoff_N^2).
            alpha: The Ewald splitting parameter (in 1/Angstroms).
                   A common choice is 5.0 / L (for a cell of length L).
        """

        Calculator.__init__(self, **kwargs)

        #self.structure = structure
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
        
        h_a = volume / np.linalg.norm(cross_bc)
        h_b = volume / np.linalg.norm(cross_ac)
        h_c = volume / np.linalg.norm(cross_ab)

        N_max = np.ceil(self.R_cutoff / np.array([h_a, h_b, h_c])).astype(int).tolist()

        return N_max

    def _realspaceEnergy(self, structure: ase.Atoms) -> float:
        """
        Calculates the real-space energy (vectorized).
        E_real = 0.5 * sum_i,j sum_n' [ q_i*q_j * erfc(alpha*|r_ij + n|) / |r_ij + n| ]
        """

        cell = structure.cell.array
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        real_energy = 0.0

        # Find max translation vectors (n_x, n_y, n_z) needed
        # We find the "height" of the cell perpendicular to each pair of vectors
        Nx_max, Ny_max, Nz_max = self._getNMax(cell=cell, volume=structure.cell.volume)

        # Create grid of n-vectors [nx, ny, nz]
        n_x_range = np.arange(-Nx_max, Nx_max + 1, dtype=int)
        n_y_range = np.arange(-Ny_max, Ny_max + 1, dtype=int)
        n_z_range = np.arange(-Nz_max, Nz_max + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_x_range, n_y_range, n_z_range, indexing='ij')
        
        # (N_cells, 3) array of n-vectors
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)
        
        # (N_cells, 3) array of Cartesian translation vectors
        nv_vectors_cart = np.dot(n_vectors_flat, cell)

        # (N_cells) boolean array identifying n=0 vector
        n_is_zero = ~np.any(n_vectors_flat, axis=1)

        for i in range(len(structure)):
            for j in range(len(structure)):
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
                    # RuntimeWarning: divide by zero encountered in divide
                    # This is OK, we handle it by filtering r_norms_valid
                    with np.errstate(divide='ignore'):
                        terms = special.erfc(self.alpha * r_norms_valid) / r_norms_valid
                        real_energy += qi_qj * np.sum(terms)

        # Divide by 2 to correct for double counting
        return real_energy / 2.0

    def _reciprocalEnergy(self, structure: ase.Atoms) -> float:
        """
        Calculates the reciprocal-space energy (vectorized).
        E_rec = (2*pi / V) * sum_k!=0 [ (1/k^2) * exp(-k^2 / (4*alpha^2)) * |Q(k)|^2 ]
        """

        reciprocal_cell_matrix = 2.0 * np.pi * np.linalg.inv(structure.cell.array).T
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        Nmax = int(np.ceil(self.G_cutoff_N))
        G_cutoff_N_sq = self.G_cutoff_N**2

        # Create grid of n-vectors [nx, ny, nz]
        n_range = np.arange(-Nmax, Nmax + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_range, n_range, n_range, indexing='ij')
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # --- Filter n-vectors ---
        # 1. Remove n=0 term
        n_is_not_zero = np.any(n_vectors_flat, axis=1)
        n_vectors_nonzero = n_vectors_flat[n_is_not_zero]

        # 2. Apply spherical cutoff in index space
        n_norm_sq = np.sum(n_vectors_nonzero**2, axis=1)
        n_vectors_valid = n_vectors_nonzero[n_norm_sq <= G_cutoff_N_sq]
        
        if n_vectors_valid.shape[0] == 0:
            return 0.0  # No k-vectors in cutoff

        # --- Calculate k-vectors and A(k) term ---
        # (N_valid_cells, 3) array of Cartesian k-vectors
        kv_cartesian = np.dot(n_vectors_valid, reciprocal_cell_matrix)
        
        # (N_valid_cells) array of k-magnitudes squared
        k_norm_sq = np.sum(kv_cartesian**2, axis=1)
        k_norms = np.sqrt(k_norm_sq)

        # A(k) term = (1/k^2) * exp(-k^2 / (4*alpha^2))
        Ak_terms = (1.0 / k_norm_sq) * np.exp(-k_norm_sq / (4.0 * self.alpha**2))

        # --- Calculate Structure Factor Q(k) ---
        # Q(k) = sum_i q_i * exp(i * k . r_i)
        # We use the Cartesian dot product: k . r_i
        
        # (N_atoms, N_valid_cells) array of k.r values
        k_dot_r_matrix = np.dot(positions, kv_cartesian.T)
        
        # (N_atoms, N_valid_cells) array of exp(i * k.r)
        exp_k_dot_r = np.exp(1j * k_dot_r_matrix)

        # (N_valid_cells) array of Q(k) = sum_i q_i * exp(i * k.r_i)
        Q_vector = np.dot(charges, exp_k_dot_r)

        # (N_valid_cells) array of |Q(k)|^2
        Q2_terms = np.absolute(Q_vector)**2

        # --- Sum Energy ---
        # E_rec = (2*pi / V) * sum_k [ A(k) * |Q(k)|^2 ]
        rec_energy = (2.0 * np.pi / structure.cell.volume) * np.sum(Ak_terms * Q2_terms)

        return rec_energy

    def _selfEnergy(self, charges) -> float:
        """
        Calculates the self-energy correction term.
        E_self = - (alpha / sqrt(pi)) * sum_i q_i^2
        """
        # E_self = - (alpha / sqrt(pi)) * sum(q_i^2)
        return -( self.alpha / np.sqrt(np.pi)) * np.sum(charges**2)

    def _get_total_energy(self, structure: ase.Atoms) -> float:
        """
        Calculates the total Ewald electrostatic energy.
        """

        COULOMB_CONSTANT_eV_A = 14.399645353 # This converts energy from (e^2 / A) to (eV)

        # Calculate raw energies in (e^2 / A)
        real_energy = self._realspaceEnergy(structure)
        rec_energy = self._reciprocalEnergy(structure)
        self_energy = self._selfEnergy(structure.get_initial_charges())

        total_energy = real_energy + rec_energy + self_energy

        # Convert it to eV
        total_energy *= COULOMB_CONSTANT_eV_A

        return total_energy
    
    def calculate(self, 
                  atoms: Union[ase.Atoms, None] = None,
                  properties: list[str] = ['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            self.results['energy'] = self._get_total_energy(atoms)  # type: ignore
            self.results['free_energy'] = self.results['energy']
