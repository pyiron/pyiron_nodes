from pyiron_workflow import as_function_node
from typing import Optional
import ase as _ase

@as_function_node("crack_param_dict")
def anisotropic_crack_params_plane_strain(
    C
):
    '''
    Takes in a rotated elasticity tensor and returns the material parameters from linear elastic fracture mechanics for plane strain.
    
    Please refer to G. C. Sih, H. Liebowitz, Mathematical theories of brittle fracture, in: H. Liebowitz (Ed.), 
    Fracture: An Advanced Treatise (Mathematical Fundamentals vol 2), Academic Press, New York, 1968, Ch. 2, pp. 67–190

    '''

    import numpy as np
    
    QQ = np.array([[C[0,0], C[0,5], C[0,4]], [C[0,5], C[5,5], C[4,5]], [C[0,4], C[4,5], C[4,4]]])
    R = np.array([[C[0,5], C[0,1], C[0,3]], [C[5,5], C[1,5], C[3,5]], [C[4,5], C[1,4], C[3,4]]])
    T = np.array([[C[5,5], C[1,5], C[3,5]], [C[1,5], C[1,1], C[1,3]], [C[3,5], C[1,3], C[3,3]]])
    N1 = -1 * np.dot(np.linalg.inv(T),np.transpose(R))
    N2 = np.linalg.inv(T)
    N3 = np.dot(np.dot(R, np.linalg.inv(T)), np.transpose(R)) - QQ
    NN1 = np.concatenate((N1, N2), axis=1)
    NN2 = np.concatenate((N3, np.transpose(N1)), axis=1)
    N = np.concatenate((NN1, NN2), axis=0)

    #--- finding eigenvector and eigen values, ...
    [v, u] = np.linalg.eig(N) # v - eigenvalues, v - eigenvectors
    a1 = [[u[0,0]], [u[1,0]], [u[2,0]]]
    pp1 = v[0]
    b1 = np.dot(np.transpose(R)+np.dot(pp1, T),a1)

    a2 = [[u[0,2]], [u[1,2]], [u[2,2]]]
    pp2 = v[2]
    b2 = np.dot(np.transpose(R)+np.dot(pp2, T),a2)

    a3 = [[u[0,4]], [u[1,4]], [u[2,4]]]
    pp3 = v[4]
    b3 = np.dot(np.transpose(R)+np.dot(pp3, T),a3)
    AA = np.concatenate((a1, a2, a3), axis=1)
    BB = np.concatenate((b1, b2, b3), axis=1)

    p = np.array([pp1, pp2, pp3])

    AB = np.concatenate((AA, BB), axis=0)
    J = np.zeros(np.shape(u))
    for i in range(3):
        J[i, i+3] = 1
        J[i+3, i] = 1
    AB_n = np.zeros(np.shape(u), dtype=complex)    
    for i in range(3):
        AB_n[:, i] = AB[:, i] / np.sqrt(np.matmul(AB[:, i].T, np.matmul(J, AB[:, i])))

    A = AB_n[0:3, 0:3]
    B = AB_n[3:6, 0:3]    
    B_inv = np.linalg.inv(B)

    return {'A': A, 'B_inv': B_inv, 'p': p}

@as_function_node("cracked_structure")
def displace_atoms_crack_aniso_plane_strain(
    atoms: _ase.Atoms, 
    K_I: Optional[int|float],
    K_II: Optional[int|float],
    K_III: Optional[int|float],
    crack_params: Optional[dict]
):
    '''
    Returns a structure with a crack inserted using the anisotropic linear elastic solution in plane strain.
    The mathematical center of the crack tip is assumed to be at the center of the simulation box.

    atoms: input structure
    K_I, K_II, K_III: stress intensity factors under mode-I, II and III loading (MPa*sqrt(m))
    crack_params: material parameters from linear elastic fracture mechanics for plane strain
    
    Please refer to G. C. Sih, H. Liebowitz, Mathematical theories of brittle fracture, in: H. Liebowitz (Ed.), 
    Fracture: An Advanced Treatise (Mathematical Fundamentals vol 2), Academic Press, New York, 1968, Ch. 2, pp. 67–190

    '''

    import numpy as np
    import math

    A = crack_params['A']
    B_inv = crack_params['B_inv']
    p = crack_params['p']

    crack_struct = atoms.copy()
    
    pos_xyz = crack_struct.get_positions()
    X_c = crack_struct.cell[0][0]/2
    Y_c = crack_struct.cell[1][1]/2
    K_vector = [K_II*100, K_I*100, K_III*100]              # mode I and II are swapped below hence this order to keep the conventional modes
    
    for iii in range(len(pos_xyz)):
        x1 = pos_xyz[iii, 0] - X_c
        x2 = pos_xyz[iii, 1] - Y_c
        x3 = pos_xyz[iii, 2]    
        r = np.sqrt(x1**2 + x2**2)
        teta = math.atan2(x2, x1)
        p_diag = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            p_diag[i, i] = np.sqrt(np.cos(teta) + p[i] * np.sin(teta))
        disp = np.sqrt(2 * r/np.pi) * np.real(A @ p_diag @ B_inv) @ K_vector
        crack_struct.positions[iii] += disp
    
    return crack_struct