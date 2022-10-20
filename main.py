from argparse import ArgumentParser
import pickle
from network import Network
import signals as sg
import numpy as np
import json
import time
import os
import nest
import matplotlib.pyplot as plt
import capacities as cap
import signals as sg
from neuron import L5pyr_simp_sym
from neuron import Single_comp
from enum import Enum, auto


class Inhibition(Enum):
    # types of inhibition used in the network

    RAND_DEND = auto()  # inhibition to random dendritic compartment
    SOMA = auto()  # inhibition to somatic compartment


# Network and output specific settings
class Input(Enum):
    # types of inputs

    RATE_MODULATION = 'rate_modulation'
    CONST_RATE = 'const_rate'


class PProc(Enum):
    # types of postprocessing calculations which also determines the data which is recorded

    CAPACITY = 'cap'
    NETWORK_NMDA_STATS = 'stats'
    SCALING = 'scal'


def create_outdir(dirname):
    name = "out_data/" + dirname
    if not os.path.exists(dirname):
        try:
            os.mkdir(name)
            print("directory: '", name, "'  has been created ")
        except:
            pass


def simulate(
    r,
    g,
    w,
    rho,
    n_cl,
    n_dend,
    n,
    mod,
    c,
    exc_neuron,
    inh_neuron,
    inhib,
    iw_fac,
    post_proc,
    inp_type,
    job_name,
    t_sim=10000,
    n_cores=1,
    inp_str=None,
    tstep=None,
    steps=None,
    seq=None,
    rec_plottrace=False,
    rec_inp=False,
    raw_path=False,
):

    # running the nest simulation
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": n_cores})

    exc_neuron = L5pyr_simp_sym(n_dend=5)
    inh_neuron = Single_comp()

    # running the network
    network = Network(
        r=r,
        g=g,
        w=w,
        rho=rho,
        n_cl=n_cl,
        n_dend=n_dend,
        n=n,
        mod=mod,
        c=c,
        exc_neuron=exc_neuron,
        inh_neuron=inh_neuron,
        iw_fac=iw_fac,
    )

    if inhib == Inhibition.RAND_DEND:
        network.build_reservoir("RAND_DEND")
    elif inhib == Inhibition.SOMA:
        network.build_reservoir("SOMA")
    if rec_plottrace:
        network.rec_plottrace(20)

    network.rand_initial_state()

    if inp_type == Input.RATE_MODULATION:
        pre_steps = int(1000 / tstep)
        seq = np.random.uniform(low=-1.0, high=1.0, size=steps + pre_steps)
        network.poisson_inp(inp_seq=seq, inp_str=inp_str, tstep=tstep)
        seq = seq[pre_steps:]

    elif inp_type == Input.CONST_RATE:
        network.bg_noise(t_sim)
        seq = None

    rec_spikes = False
    rec_voltages = False
    rec_sequence = False  # more like save sequence
    if post_proc == PProc.CAPACITY:
        network.record_spikes(rec_inp=rec_inp)
        rec_spikes = True
        rec_sequence = True
    elif post_proc == PProc.NETWORK_NMDA_STATS:
        network.record_spikes(rec_inp)
        network.record_voltages()
        rec_spikes = True
        rec_voltages = True
    elif post_proc == PProc.SCALING:
        network.record_spikes(rec_inp)
        rec_spikes = True
    network.simulate()

    raw = network.get_data(rec_spikes, rec_voltages, rec_sequence, rec_inp)
    if inp_type == Input.RATE_MODULATION:
        raw["seq"] = seq
    elif inp_type == Input.CONST_RATE:
        raw["seq"] = None
    if rec_plottrace:
        raw["plottrace"] = network.get_plottrace()
    return raw


def post(raw, post_proc, params):

    if post_proc == PProc.CAPACITY:
        results = sg.post_proc_capacity(raw_data=raw)
    elif post_proc == PProc.NETWORK_NMDA_STATS:
        results = sg.network_nmda_stats(raw, params)
    elif post_proc == PProc.SCALING:
        results = sg.net_rate(raw, params)
    else:
        print("Error")
    return results


def run(
    r,
    g,
    w,
    rho,
    n_cl,
    n_dend,
    n,
    mod,
    c,
    exc_neuron,
    inh_neuron,
    inhib,
    iw_fac,
    post_proc,
    inp_type,
    t_sim=10000,
    job_name="",
    n_cores=1,
    inp_str=0.0,
    tstep=0.0,
    steps=None,
    rec_plottrace=False,
    rec_inp=False,
    raw_path=False,
):
    # start timer to test scaling
    if post_proc == PProc.SCALING:
        t_begin = time.time()
    # saving the input parameters in a dictionary to include them in a file
    params = {}
    params["c"] = c
    params["mod"] = mod
    params["r_bg"] = r
    params["g"] = g
    params["w"] = w
    params["rho"] = rho
    params["N_cl"] = n_cl
    params["N_dend"] = n_dend
    params["N"] = n
    params["inp_str"] = inp_str
    params["tstep"] = tstep
    params["steps"] = steps

    # creating outdir and filename
    outdir = job_name + "_" + post_proc.value + "_" + inp_type.value
    create_outdir(outdir)
    create_outdir(outdir + "/raw")
    create_outdir(outdir + "/res")
    filename = f"_r{r}_g{g}_mod{mod}_c{c}_str{inp_str}_step{tstep}_weight{w}"

    # run simulation or load rawdata
    if raw_path == False:
        raw = simulate(
            r=r,
            g=g,
            w=w,
            rho=rho,
            n_cl=n_cl,
            n_dend=n_dend,
            n=n,
            mod=mod,
            c=c,
            exc_neuron=exc_neuron,
            inh_neuron=inh_neuron,
            inhib=inhib,
            iw_fac=iw_fac,
            post_proc=post_proc,
            inp_type=inp_type,
            t_sim=t_sim,
            steps=steps,
            job_name="",
            inp_str=inp_str,
            tstep=tstep,
            n_cores=n_cores,
            rec_plottrace=rec_plottrace,
            rec_inp=rec_inp,
        )
        raw = raw | params
        with open(
            "out_data/" + outdir + "/raw/" + "raw" + filename + ".p", "wb"
        ) as f:
            pickle.dump(raw, f)

    elif raw_path == True:
        with open(
            "out_data/" + outdir + "/raw/" + "raw" + filename + ".p", "rb"
        ) as f:
            raw = pickle.load(f)
    else:
        with open(raw_path, "rb") as f:
            raw = pickle.load(f)

    # postprocessing
    results = post(raw, post_proc, params)
    results = results | params

    if (
        post_proc == PProc.CAPACITY
    ):  # save as pickle if we we run the capacity calculation
        with open("out_data/" + outdir + "/res/res" + filename + ".p", "wb") as f:
            pickle.dump(results, f)
    elif post_proc == PProc.NETWORK_NMDA_STATS:
        with open(
            "out_data/" + outdir + "/res/" + filename + ".json", "w"
        ) as f:
            json.dump(results, f)
    elif post_proc == PProc.SCALING:
        t_end = time.time()
        results["cores"] = n_cores
        results["time"] = t_end - t_begin
        with open(
            "out_data/"
            + outdir
            + "/res/"
            + filename
            + f"_c{n_cores}_nd{n_dend}"
            + ".json",
            "w",
        ) as f:
            json.dump(results, f)


if __name__ == "__main__":

    run(
        job_name="test",
        n_cores=4,
        g=14.0,
        r=15.00,
        w=0.00025,
        rho=0.1,
        n_cl=5,
        n_dend=5,
        n=1250,
        mod=0.0,
        c=0.2,
        iw_fac=1.19,
        inp_type=Input.CONST_RATE,
        t_sim=1000,
        inp_str=0.1,
        tstep=50.0,
        steps=100,
        exc_neuron=L5pyr_simp_sym(),
        inh_neuron=Single_comp(),
        inhib=Inhibition.RAND_DEND,
        post_proc=PProc.NETWORK_NMDA_STATS,
        rec_plottrace=False,
        rec_inp=False,
    )
