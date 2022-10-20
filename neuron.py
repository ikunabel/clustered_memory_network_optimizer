import nest


soma_params = {
    # passive parameters
    "C_m": 0.330880516,  # [nF] Capacitance
    "g_C": 0.0,  # soma has no parent
    "g_L": 0.015135876677,  # [uS] Leak conductance
    "e_L": -75.00046388760929,  # [mV] leak reversal
    # ion channel params
    "gbar_Na": 22.036637488000,  # [nS] Na maximal conductance
    "e_Na": 50.0,  # [mV] Na reversal
    "gbar_K": 9.871382640833,  # [nS] K maximal conductance
    "e_K": -85.0,  # [mV] K reversal
}
dend_params = {
    # passive parameters
    "C_m": 0.002124942,  # [nF] Capacitance
    "g_C": 0.000552611865,  # [uS] Coupling with parent
    "g_L": 0.000072191148,  # [uS] Leak conductance
    "e_L": -75.0,  # [mV] Leak reversal
}

soma_params_pas = {
    key: val
    for key, val in soma_params.items()
    if key in ["C_m", "g_C", "g_L", "e_L"]
}


class L5pyr_simp_sym:

    """
    simplified 5 leaf compartment version of the layer 5 pyramidal neuron
    """

    def __init__(self, n_dend=5, plot=False):
        self.type = "L5pyr_simp"
        self.n_dend = n_dend

        if not n_dend == 5:
            assert ValueError("we can only use 5 dendrites at the moment")

        self.leaf_idxs = [i for i in range(1, self.n_dend + 1)]
        self.v_comps = [f"v_comp{i}" for i in self.leaf_idxs]
        self.w_dend = [1.0] * len(self.leaf_idxs)
        self.exc_rec = [0, 1, 2, 3, 4, 5]
        self.inh_rec = [6, 7, 8, 9, 10, 11]

    def create(
        self,
        n_neurons,
        passive_soma=False,
        AMPA_params={},
        AMPA_NMDA_params={},
        GABA_params={},
    ):
        if passive_soma:
            soma_params_ = soma_params_pas
        else:
            soma_params_ = soma_params

        # define the model with its compartments
        cm_pop = nest.Create("cm_default", n_neurons)

        cm_pop.compartments = [
            {"parent_idx": -1, "params": soma_params_},
            {"parent_idx": 0, "params": dend_params},
            {"parent_idx": 0, "params": dend_params},
            {"parent_idx": 0, "params": dend_params},
            {"parent_idx": 0, "params": dend_params},
            {"parent_idx": 0, "params": dend_params},
        ]
        # add the dendritic receptors
        cm_pop.receptors = [
            # AMPA receptors to all dendritic compartments
            {"comp_idx": 0, "receptor_type": "AMPA", "params": AMPA_params},
            {
                "comp_idx": 1,
                "receptor_type": "AMPA_NMDA",
                "params": AMPA_NMDA_params,
            },
            {
                "comp_idx": 2,
                "receptor_type": "AMPA_NMDA",
                "params": AMPA_NMDA_params,
            },
            {
                "comp_idx": 3,
                "receptor_type": "AMPA_NMDA",
                "params": AMPA_NMDA_params,
            },
            {
                "comp_idx": 4,
                "receptor_type": "AMPA_NMDA",
                "params": AMPA_NMDA_params,
            },
            {
                "comp_idx": 5,
                "receptor_type": "AMPA_NMDA",
                "params": AMPA_NMDA_params,
            },
            # AMPA+NMDA receptors to all dendritic compartments
            {"comp_idx": 0, "receptor_type": "GABA", "params": GABA_params},
            {"comp_idx": 1, "receptor_type": "GABA", "params": GABA_params},
            {"comp_idx": 2, "receptor_type": "GABA", "params": GABA_params},
            {"comp_idx": 3, "receptor_type": "GABA", "params": GABA_params},
            {"comp_idx": 4, "receptor_type": "GABA", "params": GABA_params},
            {"comp_idx": 5, "receptor_type": "GABA", "params": GABA_params},
        ]

        return cm_pop


class Single_comp:

    """
    simplified 5 leaf compartment version of the layer 5 pyramidal neuron
    """

    def __init__(self):
        self.type = "Single_comp"
        self.n_dend = 1

        self.leaf_idxs = [i for i in range(1, self.n_dend + 1)]
        self.v_comps = [f"v_comp{i}" for i in self.leaf_idxs]
        self.w_dend = [1.0] * len(self.leaf_idxs)
        self.exc_rec = [0]
        self.inh_rec = [1]

    def create(
        self, n_neurons, passive_soma=False, AMPA_params={}, GABA_params={}
    ):
        if passive_soma:
            soma_params_ = soma_params_pas
        else:
            soma_params_ = soma_params

        # define the model with its compartments
        cm = nest.Create("cm_default", n_neurons)
        cm.compartments = [{"parent_idx": -1, "params": soma_params_}]
        # add the dendritic receptors
        cm.receptors = [
            # AMPA receptors to all dendritic compartments
            {"comp_idx": 0, "receptor_type": "AMPA", "params": AMPA_params},
            # AMPA+NMDA receptors to all dendritic compartments
            {"comp_idx": 0, "receptor_type": "GABA", "params": GABA_params},
        ]

        return cm


if __name__ == "__main__":
    T_neuron = L5pyr_simp_sym()
    print(type(T_neuron))
    breakpoint()
    pop = T_neuron.create(5)
