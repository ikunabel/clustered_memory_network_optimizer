from mmap import mmap
import numpy as np
import nest
from modularity import clustered_connections
import scipy.stats as stats


class Network:
    # change to take 1 dictionary as input
    # change to: provide inhibitory and excitatory models seperatly
    def __init__(
        self,
        r,
        g,
        w,
        iw_fac,
        rho,
        n_cl,
        n_dend,
        n,
        mod,
        c,
        exc_neuron,
        inh_neuron,
    ):

        # neuron
        self.exc_model = exc_neuron
        self.inh_model = inh_neuron
        self.ipop_weight_factor = iw_fac

        # synapse parameters
        self.delay = 1.5
        # reservoir parameters
        self.N = n
        self.N_E = int(0.8 * self.N)
        self.N_I = int(0.2 * self.N)
        self.g = g
        self.w_res = w
        self.Q = n_cl
        self.B = n_dend
        self.mod = mod
        self.c = c
        self.rho = rho
        # input parameters
        self.inp_rate = r
        self.w_inp = w

    def connect_excitatory2(self):

        # drawing a random clustered connection matrix
        self.adj_matrix, self.cluster_pos, t = clustered_connections(
            n_neurons=self.N_E,
            n_clusters=self.Q,
            density=self.rho,
            modularity=self.mod,
        )
        W = self.adj_matrix * self.w_res

        # determine the upper boarders of the ids of the clusters
        cluster_range = [pos[0][1] + 1 for pos in self.cluster_pos]

        def get_ec(neuron_id):
            # returns the embedded cluster of the provided neuron_id
            return np.where(np.array(cluster_range) >= neuron_id)[0][0]

        self.debug = 0

        def choose_dendrec(pre_ec, post_ec, pref_cl):
            # returns the randomly chosen branch to connect the presynaptic neuron to based on the
            if pre_ec in pref_cl and list(pref_cl).count(pre_ec) != self.B:
                if np.random.rand() < self.c:  # success
                    return self.exc_model.exc_rec[1:][
                        np.random.choice(np.where(pref_cl == pre_ec)[0])
                    ]
                else:
                    return self.exc_model.exc_rec[1:][
                        np.random.choice(np.where(pref_cl != pre_ec)[0])
                    ]
            else:
                return np.random.choice(self.exc_model.exc_rec[1:])

        def get_rec_ids(pre_ids, post_ec, pref_cl):
            rec_ids = []
            dend_hist = {
                f"{dendrec}": {f"{cl_id}": 0 for cl_id in range(self.Q)}
                for dendrec in self.exc_model.exc_rec
            }
            for pre_id in pre_ids:
                pre_ec = get_ec(pre_id)
                dendrec_id = choose_dendrec(pre_ec, post_ec, pref_cl)
                rec_ids.append(dendrec_id)
                dend_hist[str(dendrec_id)][str(pre_ec)] += 1
            return rec_ids, dend_hist

        self.dend_hists = {}

        for i, post in enumerate(self.E_pop):
            post_id = post.get("global_id")
            # determine number of the embedded cluster 0,1,2,3,Q-1
            post_ec = get_ec(post_id)

            # draw the prefered cluster for the branches
            prob_vec = [(1 - self.mod) / (self.Q - 1)] * self.Q
            prob_vec[post_ec] = self.mod
            pref_cl = np.random.choice(self.Q, self.B, p=prob_vec)
            weights = W[
                :, i
            ]  # weight row of all the neurons that neuron connect to post neuron
            nonzero_indices = np.where(weights != 0)[
                0
            ]  # only non-zero connections need to be considered
            weights = weights[nonzero_indices]
            ones = np.ones(len(nonzero_indices))
            pre_ids = self.E_pop[nonzero_indices].get("global_id")
            post_ids = np.ones(len(nonzero_indices), dtype=int) * post_id

            # rec_ids =  [choose_dendrec(get_ec(pre_id), post_ec, pref_cl) for pre_id in pre_ids]
            rec_ids, dend_hist = get_rec_ids(pre_ids, post_ec, pref_cl)

            self.dend_hists[f"{post_id}"] = dend_hist

            nest.Connect(
                pre_ids,
                post_ids,
                "one_to_one",
                syn_spec={
                    "synapse_model": "static_synapse",
                    "weight": weights,
                    "delay": self.delay * ones,
                    "receptor_type": rec_ids,
                },
            )
        # debug_test = [[list(np.sort(nest.GetConnections(self.E_pop, self.E_pop[ii]).get('receptor'))).count(i) for i in [2,4,6,8,10]] for ii in range(1,100)]

    def connect_excitatory(self):

        if self.exc_model.N_dend != self.Q:
            raise ValueError(
                "Number of clusters must be number of dendritic compartments"
            )

        # drawing possilbe random dendritic connection to (counter act that compartments are not equal)
        comp_id_mat = [0] * self.N_E
        for i in range(len(comp_id_mat)):
            comp_id_mat[i] = np.random.permutation(
                np.array(self.exc_model.exc_rec[1:])
            )
        comp_id_mat = np.array(comp_id_mat)

        # drawing a random clustered connection matrix
        self.adj_matrix, self.cluster_pos, t = clustered_connections(
            n_neurons=self.N_E,
            n_clusters=self.Q,
            density=self.rho,
            modularity=self.mod,
        )
        W = self.adj_matrix * self.w_res
        # determine the upper boarders of the ids of the clusters
        cluster_range = []
        for pos in self.cluster_pos:
            cluster_range.append(pos[0][1])

        for i, pre in enumerate(self.E_pop):
            cluster_id = None
            # determine id of the cluster that pre belongs to;  cluster_id [1, 2, ..., Q]
            for k, cl_range in enumerate(cluster_range):
                if pre.get("global_id") <= cl_range + 1:
                    cluster_id = k + 1
                    break
            weights = W[
                :, i
            ]  # weight vector of all the neurons that neuron i is connected to
            nonzero_indices = np.where(weights != 0)[
                0
            ]  # only non-zero connections need to be considered
            weights = weights[nonzero_indices]
            ones = np.ones(len(nonzero_indices))
            post = self.E_pop[nonzero_indices]

            # connect to prefered dendrite with prob c and with equal probability to all other DENDRITIC compartments
            prob_vec = np.ones(self.Q) * ((1 - self.c) / (self.Q - 1))
            prob_vec[cluster_id - 1] = self.c
            # id_vec = np.random.choice(ex_rec[1:], len(nonzero_indices), p = prob_vec)
            choice_vec = np.random.choice(
                [0, 1, 2, 3, 4], len(nonzero_indices), p=prob_vec
            )

            id_vec = [
                comp_id_mat[nonzero_indices][ii][choice]
                for ii, choice in enumerate(choice_vec)
            ]

            pre_array = np.ones(len(nonzero_indices), dtype=np.int64) * pre.get(
                "global_id"
            )  # array of the pre neuron to connect 'one_to_one'

            nest.Connect(
                pre_array,
                post,
                "one_to_one",
                syn_spec={
                    "synapse_model": "static_synapse",
                    "weight": [
                        weight * self.exc_model.w_dend[choice_vec[i]]
                        for i, weight in enumerate(weights)
                    ],
                    "delay": self.delay * ones,
                    "receptor_type": id_vec,
                },
            )
        print("\n configured all excitatory connections \n")

    def build_reservoir(self, inhibition=None):

        # creating the populations
        self.E_pop = self.exc_model.create(self.N_E)
        self.I_pop = self.inh_model.create(self.N_I)

        self.res = self.E_pop + self.I_pop

        # connecting the two subpopulations
        # EE
        self.connect_excitatory2()  # connected with clustered connectivity
        # EI, II, IE
        # fixed indegrees
        K_EI = int(0.1 * self.N_E)
        K_IE = int(0.1 * self.N_I)
        K_II = int(0.1 * self.N_I)
        conn_params_EI = {
            "rule": "fixed_indegree",
            "indegree": K_EI,
            "allow_autapses": False,
            "allow_multapses": False,
        }
        conn_params_IE = {
            "rule": "fixed_indegree",
            "indegree": K_IE,
            "allow_autapses": False,
            "allow_multapses": False,
        }
        conn_params_II = {
            "rule": "fixed_indegree",
            "indegree": K_II,
            "allow_autapses": False,
            "allow_multapses": False,
        }

        if inhibition == "SOMA":
            rec_ids_IE = self.exc_model.inh_rec[
                0
            ]  # inhibition to somatic compatments of the exc neurons
        elif inhibition == "RAND_DEND":
            rec_ids_IE = np.random.choice(
                self.exc_model.inh_rec[1:], (len(self.E_pop), K_IE)
            )  # inhibition to dendritic compatments of the exc neurons
        else:
            ValueError(
                "type of inhibition not supoorted \n only: RAND_DEND or SOMA"
            )

        rec_ids_EI = self.inh_model.exc_rec[0]
        rec_ids_II = self.inh_model.inh_rec[0]

        nest.Connect(
            self.E_pop,
            self.I_pop,
            conn_params_EI,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.w_res * self.ipop_weight_factor,
                "delay": self.delay,
                "receptor_type": rec_ids_EI,
            },
        )
        nest.Connect(
            self.I_pop,
            self.E_pop,
            conn_params_IE,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.w_res * self.g,
                "delay": self.delay,
                "receptor_type": rec_ids_IE,
            },
        )
        # no inh_weight fac for inh
        nest.Connect(
            self.I_pop,
            self.I_pop,
            conn_params_II,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.w_res * self.g,
                "delay": self.delay,
                "receptor_type": rec_ids_II,
            },
        )
        print("reservoir fully connected")

    def rand_initial_state(self, t_stop=1000.0):
        # feeding noise into the network for 1s to create a random initial state
        noise = nest.Create(
            "noise_generator", 1, {"stop": t_stop, "mean": 0.0, "std": 0.4}
        )
        nest.Connect(noise, self.res)

    def bg_noise(self, t_sim, t_start=1000.0):
        # poisson input to drive the reservoir connected to the somatic compartments
        self.t_sim = t_sim + 2000.0

        self.bg_neurons = nest.Create("poisson_generator", 1)
        #
        # increased input strength
        #
        nest.SetStatus(
            self.bg_neurons,
            {"rate": self.inp_rate * 0.1 * self.N_E, "start": t_start},
        )  # equivalent to 0.1*self.N_E possion processes with rate inp_rate
        nest.Connect(
            self.bg_neurons,
            self.E_pop,
            syn_spec={
                "delay": 0.1,
                "weight": self.w_inp,
                "receptor_type": self.exc_model.exc_rec[0],
            },
        )  # connected to soma
        nest.Connect(
            self.bg_neurons,
            self.I_pop,
            syn_spec={
                "delay": 0.1,
                "weight": self.w_inp * self.ipop_weight_factor,
                "receptor_type": self.inh_model.exc_rec[0],
            },
        )  # connected to soma

    def record_voltages(self, time=10000.0):
        # randomly selects N neurons to record the voltage from
        rec_ids = self.E_pop.get("global_id")
        start = 2000.0
        self.mm = nest.Create(
            "multimeter",
            params={
                "interval": 1.0,
                "start": start,
                "stop": start + time,
                "record_from": ["v_comp0"] + self.exc_model.v_comps,
            },
        )
        nest.Connect(self.mm, rec_ids)

    def rec_plottrace(self, N, time=3000.0):
        rec_ids = np.sort(np.random.choice(self.E_pop.get("global_id"), N))
        start = 2000.0
        self.mm_plot = nest.Create(
            "multimeter",
            params={
                "interval": 0.1,
                "start": start,
                "record_from": ["v_comp0"] + self.exc_model.v_comps,
            },
        )
        nest.Connect(self.mm_plot, rec_ids)

    def get_plottrace(self):
        data = {}
        data = nest.GetStatus(self.mm_plot)[0]["events"]
        data["leaf_idxs"] = self.exc_model.leaf_idxs
        data["v_comps"] = self.exc_model.v_comps
        return data

    def get_data(self, rec_spikes, rec_voltages, rec_sequence, rec_inp):
        # saving simulation raw data
        if rec_spikes and not rec_voltages and rec_sequence:
            data = self.sr.get("events")
        elif rec_spikes and rec_voltages:
            data = self.sr.get("events")
            data["sample_neurons"] = nest.GetStatus(self.mm)[0]["events"]
            data["sample_neurons"]["leaf_idxs"] = self.exc_model.leaf_idxs
            data["sample_neurons"]["v_comps"] = self.exc_model.v_comps
        else:
            data = self.sr.get("events")
        if rec_inp:
            data["inp"] = self.sr_inp.get("events")
        data["excitatory_ids"] = list(self.E_pop.get("global_id"))
        data["inhibitory_ids"] = list(self.I_pop.get("global_id"))
        data["t_sim"] = self.t_sim
        data["hists"] = self.dend_hists
        return data

    def record_spikes(self, rec_inp):
        self.sr = nest.Create("spike_recorder")
        nest.Connect(self.E_pop, self.sr)
        nest.Connect(self.I_pop, self.sr)
        if rec_inp:
            self.sr_inp = nest.Create("spike_recorder")
            nest.Connect(self.stim_neurons, self.sr_inp)


    def poisson_inp(self, inp_seq, inp_str, tstep):
        # gaussian bell with the peak moved over the excitatory population
        steps = len(inp_seq)
        self.t_sim = 1000.0 + steps * tstep

        self.stim_neurons = nest.Create(
            "inhomogeneous_poisson_generator", int(self.N_E / 10)
        )  # input only to excitatory

        stim_id = np.linspace(
            int(self.N_E / 100), self.N_E, int(self.N_E / 10), True
        ).reshape(int(self.N_E / 10), 1)
        zero_point = int(self.N_E / 2)
        std = float(self.N_E / 20)
        interval = 0.4 * self.N_E

        # stim_rates = np.array([stats.norm.pdf(stim_id, zero_point+ interval*val, std)*self.N_E for val in inp_seq])
        stim_rates = np.zeros([len(inp_seq), int(self.N_E / 10)])
        stim_rates[:] = np.transpose(
            stats.norm.pdf(stim_id, zero_point + interval * inp_seq[:], std)
            * self.N_E
        )
        base_rates = np.ones((len(inp_seq), int(self.N_E / 10)))

        rate_values = (
            ((1 - inp_str) * base_rates + inp_str * stim_rates)
            * self.inp_rate
            * 0.1
            * self.N_E
        ).tolist()
        # rate_values = ((base_rates + self.inp_str*stim_rates) * self.inp_rate * 0.1 * self.N_E).tolist()

        del stim_rates
        del base_rates

        rate_times = (
            np.arange(0.1, len(inp_seq) * tstep + 0.1, tstep)
            + np.ones(len(inp_seq)) * 1000.0
        )

        # cutting of the prerun steps of the input sequence

        for idx, stim_neuron in enumerate(self.stim_neurons):
            nest.SetStatus(
                stim_neuron,
                {
                    "rate_times": rate_times,
                    "rate_values": np.transpose(rate_values)[idx],
                },
            )
            nest.Connect(
                stim_neuron,
                self.E_pop[idx * 10 : (idx + 1) * 10],
                syn_spec={
                    "delay": 0.1,
                    "weight": self.w_inp,
                    "receptor_type": self.exc_model.exc_rec[0],
                },
            )  # input connected to soma

        # creating homogenous input to also to the inhibitory population
        self.bg_neurons = nest.Create("poisson_generator", 1)
        nest.SetStatus(
            self.bg_neurons,
            {"rate": self.inp_rate * 0.1 * self.N_E, "start": 1000.0},
        )  # equivalent to 0.1*self.N_E possion processes with rate inp_ra
        nest.Connect(
            self.bg_neurons,
            self.I_pop,
            syn_spec={
                "delay": 0.1,
                "weight": self.w_inp * self.ipop_weight_factor,
                "receptor_type": self.inh_model.exc_rec[0],
            },
        )  # connected to soma

    def simulate(self):
        nest.Simulate(self.t_sim)

    """
    def run(self, steps, save, skip = False):
        self.inp_str = self.params['inp_str']
        self.steps = steps
        self.inp_tstep = self.params['inp_tstep']
        self.t_sim = steps * self.inp_tstep + 1.
        # generating random sequence with prerun of 1s
       
        
        # modulated poisson input and intervall rate as state
        self.__poisson_inp(self.inp_seq)
        self.sr = nest.Create('spike_recorder')     
        nest.Connect(self.E_pop, self.sr)
        nest.Connect(self.I_pop, self.sr)   
        
        nest.Simulate(2000. + self.t_sim )
        
        # this is to also record voltage traces for the test run
        if steps > 1000 or skip:
            data = {}
        else:
            data = self.get_data(save, save_raw = False)
        data['seq'] = self.inp_seq
        data['N'] = self.N
        data['t_step'] = self.inp_tstep
        data['steps'] = self.steps
        data['t_sim'] = self.t_sim
        data['inp_str'] = self.inp_str
        res_data = self.sr.get('events')
        data['res'] = [res_data['senders'], res_data['times']]        
        pickle.dump(data, open('out_data/' + save + '/' + 
                'raw_step{}_str{}_mod{}_c{}'.format(self.inp_tstep, self.inp_str, self.mod, self.c) + '.p','wb'))
        return data
    """
