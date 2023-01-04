from main import Input, run, PostProcessing, Inhibition
from neuron import L5pyr_simp_sym
from neuron import Single_comp
import conf
import pandas as pd
import plotly.express as px
import optuna

def objective(trial):
    
    ratio = trial.suggest_float("ratio", 12, 20)
    backgroundFiring = trial.suggest_float("backgroundFiring", 12, 20)
    #numberClusters = trial.suggest_int("numberClusters", 0, 5)
    #modInterNeural = trial.suggest_float("modInterNeural", )
    #modIntraNeural = trial.suggest_float("modIntraNeural", )
    
    run(
        job_name="test",
        n_cores=4,
        g=ratio,
        r=backgroundFiring,
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
        post_proc=PostProcessing.NETWORK_NMDA_STATS,
        rec_plottrace=False,
        rec_inp=False,
    )
    #loss function here 
    #mean squared error
    avg_cv = conf.trial_results.get('avg_cv')
    return (1-avg_cv)**2

def run_optimizer():
    study = optuna.load_study(
        study_name="optunaStudy",
        #storage="mysql://root@127.0.0.1/optunaStudy"
        storage="/p/project/jinm60/users/ilyes-kun1/mysql/mysql-install/bin/mysql://root@localhost/optunaStudy"
    )
    study.optimize(objective, n_trials=8) 
    print(study.best_params)
    df = pd.DataFrame(study.trials)
    df.to_csv('./output/test_stats_const_rate/trials/study_trials.csv', index=False)
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_html("./output/test_stats_const_rate/plot/optimization_history.html")

run_optimizer()

