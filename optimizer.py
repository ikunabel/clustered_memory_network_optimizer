import main
from main import Input, run, PostProcessing, Inhibition
from neuron import L5pyr_simp_sym
from neuron import Single_comp
import conf
import optuna

def objective(trial):
    
    g_value = trial.suggest_float("g_value", 12, 20, step=0.1)
    r_value = trial.suggest_float("r_value", 10, 20, step=0.1)

    run(
        job_name="test",
        n_cores=4,
        g=g_value,
        r=r_value,
        w=0.00025,
        rho=0.1,
        n_cl=5,
        n_dend=5,
        n=1250,
        mod=0.0,
        c=0.2,
        iw_fac=1.19,
        inp_type=Input.CONST_RATE,
        t_sim=500,
        inp_str=0.1,
        tstep=50.0,
        steps=100,
        exc_neuron=L5pyr_simp_sym(),
        inh_neuron=Single_comp(),
        inhib=Inhibition.RAND_DEND,
        post_proc=PostProcessing.NETWORK_NMDA_STATS,
        rec_plottrace=False,
        rec_inp=False,
    )

    return conf.trial_results.get('avg_cv')#, conf.trial_results.get('avg_cv')


def run_optimizer():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    print(best_params)

run_optimizer()

