import barrier_crossing.models as bcm
import barrier_crossing.utils as bcu
import barrier_crossing.simulate as bcs
import barrier_crossing.iterate_landscape as bcla

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

import pickle

from typing import Iterable
from argparse import ArgumentParser

import jax.random as random
import jax.numpy as jnp

import barrier_crossing.plotting as plotting

import seaborn as sns

SEED = 1001
results_dir = os.environ.get("ITERATIVE_RESULT_DIR", "results")
MISC_SAVE_PATH = f"{results_dir}/misc_plots"
BATCH_SIZE = 300
BINS = 70
NUM_RECONSTRUCTIONS = 2


os.makedirs(MISC_SAVE_PATH, exist_ok = True)

def load_data_from_result(file_dir):
    dfs = []
    for file in os.listdir(file_dir):
        f = os.path.join(file_dir, file)
        if os.path.isfile(f):
            with open(f, "r") as g:
                dfs.append(pd.read_csv(g.readline()))
                
    return pd.concat(dfs)

def load_model_from_coeff_file(param_set, coeff_file, name) -> bcm.ScheduleModel:
    
    if "joint" in name:
        with open(coeff_file, "rb") as f:
            pos_coeffs, stiff_coeffs = pickle.load(f)[-1]
        
        pos_model = bcm.ScheduleModel(param_set, *bcu.get_trap_pos_from_params(param_set), pos_coeffs)
        
        stiff_model = bcm.ScheduleModel(param_set, param_set.k_s, param_set.k_s, stiff_coeffs)
        
        return bcm.JointModel(param_set, pos_model, stiff_model)
    else:
        with open(coeff_file, "rb") as f:
            coeffs = pickle.load(f)[-1]
        
        assert not jnp.iterable(coeffs[0]), f"Expected float got {coeffs[0]}"
        
        return bcm.ScheduleModel(param_set, *bcu.get_trap_pos_from_params(param_set), coeffs)
    
def make_df_names_nice(df):
    """modifies inplace""" 
    def _str_to_float(val):
        val = str(val)
        if "kt" in val.lower():
            return float(val[:-2])
        else:
            return float(val)
    nice_case = lambda x: ' '.join(word.capitalize() for word in x.split('_'))
    df["Model Name"] = df['model_name'].apply(nice_case)
    
    df["barrier_height_int"] = df["barrier_height"].apply(_str_to_float)
    
def reconstruct_with_models(param_set, models: list[bcm.ScheduleModel], names: list[str], save_name) -> tuple[float, tuple]:
    """Get bias from reconstructions."""
    
    key = random.PRNGKey(SEED)

    fig, (ax_landscape, ax_hist) = plt.subplots(1,2, figsize = (14, 6))


    reconstructed_landscapes = []
    reconstructed_labels = []
    biases = [] 
    all_works = []
    for model, name in zip(models, names):
        energies = []
        for _ in tqdm(range(NUM_RECONSTRUCTIONS)):
            key, _ = random.split(key)
            
            if isinstance(model, bcm.JointModel):
                pos_stiff_models = model.models
                            
            else:
                pos_stiff_models = (model, param_set.k_s)
            
            no_key_simulate_fn = lambda key: param_set.simulate_fn(*pos_stiff_models, key = key, regime = "brownian")
            _, (trajectories, works, _) = bcs.batch_simulate_harmonic(BATCH_SIZE, no_key_simulate_fn, key)
            
            midpoints, E = bcla.energy_reconstruction(
                            jnp.cumsum(works, axis=1), 
                            trajectories, 
                            BINS, 
                            *pos_stiff_models,
                            param_set.simulation_steps, 
                            BATCH_SIZE, 
                            param_set.beta)

            energies.append(E)
        
        energies = bcla.interpolate_inf(jnp.mean(jnp.array(energies), axis = 0))
        
        ls = [midpoints, energies]
        first_well, second_well = bcu.get_trap_pos_from_params(param_set)
        biases.append(jnp.max(bcla.landscape_discrepancies(ls, param_set.energy_fn(no_trap = True), first_well, first_well, second_well)))
        
        reconstructed_landscapes.append(ls)
        reconstructed_labels.append(name)
        all_works.append(works)
        
        
    plotting.plot_landscapes(param_set, reconstructed_landscapes, reconstructed_labels, ax = ax_landscape)
    plotting.add_axes("landscape", ax = ax_landscape)
    ax_landscape.legend()

    plotting.plot_hists(models, names, ax = ax_hist)
    ax_hist.set_title("Dissipated Work")
    ax_hist.legend()
    plt.tight_layout()
    fig.savefig(MISC_SAVE_PATH + f"/{save_name}.png")
    
    return [float(b) for b in biases]


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("file_dir")
    args = parser.parse_args()
    
    file_dir = args.file_dir
    
    df = load_data_from_result(file_dir)
    reconstruct_results = []
    for index, lb_df in df.groupby(["landscape_name", "barrier_height"]):
        with open(lb_df.params_file.iloc[0], "rb") as f:
            param_set = pickle.load(f)
            
        landscape_name, barrier_height = index
        full_name = "_".join([str(i) for i in index])
        
        models = []
        names = []
        for index, row in lb_df.iterrows():
            models.append(load_model_from_coeff_file(param_set, row["coeff_file"], name = row["model_name"]))
            names.append(row["model_name"])
        
        biases = reconstruct_with_models(param_set, models, names, save_name = f"{full_name}kt")
        reconstruct_results.append(pd.Series(data = biases, index = lb_df.index))
    
    df["bias"] = pd.concat(reconstruct_results)
    
    make_df_names_nice(df)
    
    for landscape_name, landscape_df in df.groupby("landscape_name"):
        with open(lb_df.params_file.iloc[0], "rb") as f:
            param_set = pickle.load(f)
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        sns.lineplot(data = landscape_df, x = "barrier_height_int", y="bias", hue = "Model Name", markers="o", ax = ax)
        plotting.add_energy_fn_plot(param_set, fig = fig)
        plt.tight_layout()
        ax.set_xlabel("Barrier Height (kT)")
        ax.set_ylabel("Landscape Bias (kT)")
        fig.savefig(f"{results_dir}/{landscape_name}kt.png")
        
        