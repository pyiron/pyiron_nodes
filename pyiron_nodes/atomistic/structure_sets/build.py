from pyiron_workflow import as_function_node, as_macro_node
from typing import Optional
import numpy as np
import itertools
@as_function_node("structure")
def SurfaceSegregation(

    host_element: str,
    substitution_element: str,
    lateral_repeat: int,
    number_of_layers: int,
    vacuum_size: float

):

    """
    Gives the set of all symmetry inequivalent surfaces with segregated atoms in the first layer for fcc111
    """
    import pyiron_nodes as pn
    from pyiron_atomistics import _StructureFactory

    slab = _StructureFactory().surface(host_element, 'fcc111', (lateral_repeat, lateral_repeat, number_of_layers), vacuum_size, center=True)
    z_min_layer = slab.positions[0, 2] + 0.5
    z_top_layer = slab.positions[-1, 2] - 0.5
    inds_first_layer = np.argwhere(slab.positions[:, 2] < z_min_layer).flatten()
    species = [0, 1]    
    species_symbols = [host_element, substitution_element]
    struct_list = []
    struct_set = set()
    for subset in itertools.combinations_with_replacement(species, lateral_repeat**2):
        perms = set(itertools.permutations(subset))
        for p in perms:
            slab = _StructureFactory().surface(host_element, 'fcc111', (lateral_repeat, lateral_repeat, number_of_layers), vacuum_size, center=True)
            for layer_index, cell_index in zip(inds_first_layer, p):
                slab[layer_index] = species_symbols[cell_index] 
            occ = LRPoccupations(slab, z_min_layer=z_min_layer , el_host=host_element)
            occ_vec = occ.get_occ_vector()
            p_str = ''.join([str(pp) for pp in occ_vec])
            if p_str not in struct_set:
                struct_set.add(p_str)
                struct_list.append(slab.copy())
    return struct_list

            



class LRPoccupations:
    '''
    Class to fit 2D surface structures (layers).
    '''
    def __init__(self, structure, n_sym=6, i_structure_step=0, z_min_layer=11., z_top_layer=6. , el_host='Mg'):
        # self._job = job
        self.n_sym = n_sym
        self._ref_vec = self.get_ref_vector()
        self._z_min_layer = z_min_layer
        self._z_top_layer = z_top_layer
        # atoms in position < z_min_layer and > z_top_layer will be selected
        self._el_host = el_host

        self.structure = structure 
        self.set_ind_lst()
        self.set_indices()    

        self.n_tensor = None

    def get_ref_vector(self):
        phi = np.linspace(0, 2*np.pi, self.n_sym, endpoint=False)
        vec_ref = np.array([-np.cos(phi), np.sin(phi), phi * 0]).T
        return vec_ref

    def get_resorted_list(self, nbrs):
        resorted_list = []
        for vec in nbrs.vecs:
            r_list = []
            for v_r in self._ref_vec:
                r_list.append(np.argmax([v@v_r for v in vec]))
            resorted_list.append(r_list)    
        return resorted_list    

    def get_sorted_indices(self, nbrs):
        sorted_list = self.get_resorted_list(nbrs)
        return np.array([nbrs.indices[i][sorted_list[i]] for i in range(len(sorted_list))])

    def get_occupations(self):
        ind_layer_1 = np.argwhere((self.structure.positions[:, 2] < self._z_min_layer) & (self.structure.positions[:, 2] > self._z_top_layer)).flatten()
        self.struct_layer = self.structure[ind_layer_1]
        nbrs = self.struct_layer.get_neighbors(num_neighbors=self.n_sym)
        indices = self.struct_layer.indices
        nbr_indices_sorted = self.get_sorted_indices(nbrs)

        symbols = self.struct_layer.get_species_symbols()
        if (len(symbols) == 1) & (self._el_host not in symbols):
            indices = indices * 0 + 1   

        self.nbr_indices = indices[nbr_indices_sorted]
        self.occupations = np.append(np.array([indices]).T, self.nbr_indices, axis=1) 
        self.indices = indices

        return self.occupations    

    def to_index(self, vec):
        assert (self.n_sym == 6)
        o0, o1, o2, o3, o4, o5 = vec
        return 32*o0 + 16*o1 + 8*o2 + 4*o3 + 2*o4 + o5         

    def to_inequivalent_index(self, vec):
        i_site = vec[0]
        i_nbrs = self.to_index(vec[1:])
        return self.new_to_old_index[i_site*64 + self.inequivalent_index[i_nbrs]]

    def mirror(self, vec):
        return np.array(vec)[::-1]
        vec_new[1] = vec_new[5]
        vec_new[2] = vec_new[4]
        return vec_new

    def rotation(self, vec):
        sym_vecs = []
        for i in range(6):
            sym_vecs.append(self.to_index(np.roll(vec, i)))
            sym_vecs.append(self.to_index(np.roll(self.mirror(vec), i)))
        return np.unique(sym_vecs)    
    
    def set_indices(self):    
        inequivalent_index = np.zeros_like(self.ind_lst)
        for i in range(len(self.ind_lst)):
            if inequivalent_index[i] == 0:
                rot_inds = self.rotation(self.arg_dict[i])
                for i_r in rot_inds:
                    inequivalent_index[i] = i_r
        self.inequivalent_index = inequivalent_index            
            
        new_index = np.unique(inequivalent_index) 
        new_index = np.append(new_index, new_index + 64)

        self.new_to_old_index = {n: i for i, n in enumerate(new_index)}

        self.new_index = new_index       

    def set_ind_lst(self):
        ind_lst = []
        arg_lst = []
        arg_dict = {}
        species = [0, 1]
        for vec_c in itertools.combinations_with_replacement(species, 6):
            for v in set(itertools.permutations(vec_c)):
                arg_lst.append(v)
                index = self.to_index(v)
                ind_lst.append(index)
                arg_dict[index] = v
                
        self.ind_lst = np.array(ind_lst)
        self.arg_dict = arg_dict       

    def get_occ_vector(self):
        occs = self.get_occupations()
        occ_lst = np.zeros(len(self.new_to_old_index.keys())).astype(int)
        for o in occs:
            occ_lst[self.to_inequivalent_index(o)] += 1

        self.occ_lst = occ_lst
        return occ_lst   

    def get_occ_basis(self, index):
        if index >= 64:
            o = 1
        else:
            o = 0
        return np.array([o] + list(self.arg_dict[index%64]))

    def set_e_tensor(self, tensor_fit):
        self.full_inequivalent_index = np.append(self.inequivalent_index, 64 + self.inequivalent_index)

        self._tensor_fit = tensor_fit
        self.e_tensor = np.zeros((2,2,2,2,2,2,2))
        self.n_tensor = np.zeros((2,2,2,2,2,2,2)).astype(int)
        for i in range(128):
            self.e_tensor[tuple(self.get_occ_basis(i))] = tensor_fit[self.new_to_old_index[self.full_inequivalent_index[i]]]
            self.n_tensor[tuple(self.get_occ_basis(i))] = self.new_to_old_index[self.full_inequivalent_index[i]]
 
    def save_tensors(self, file_name='e_surf_Ca'):
        np.save(file_name, self.e_tensor)
        np.save('n_tensor', self.n_tensor)   
                           