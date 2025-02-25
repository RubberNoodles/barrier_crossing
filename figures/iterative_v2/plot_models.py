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
import numpy as np

import barrier_crossing.plotting as plotting
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

SEED = 42
BINS = 70
results_dir = os.environ.get("ITERATIVE_RESULT_DIR", "results")
MISC_SAVE_PATH = f"{results_dir}/misc_plots"

testing_mode = False if os.environ.get("PLOT_TESTING", None) is None else True
if testing_mode == True:
    print("Testing mode is on. Plotting with reduced batch_size and num_reconstructions. `unset PLOT_TESTING` env variable if this was not intended.")
    BATCH_SIZE = 100
    NUM_RECONSTRUCTIONS = 10
else:
    BATCH_SIZE = 1000
    NUM_RECONSTRUCTIONS = 100

os.makedirs(MISC_SAVE_PATH, exist_ok = True)

def load_data_from_result(file_dir):
    dfs = []
    count = 0
    for file in os.listdir(file_dir):
        f = os.path.join(file_dir, file)
        if os.path.isfile(f):
            with open(f, "r") as g:
                dfs.append(pd.read_csv(g.readline()))
        count +=1
        if testing_mode == True:
            if count == 5:
                break
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
    
def reconstruct_with_models(param_set: bcu.MDParameters, models: list[bcm.ScheduleModel], names: list[str], save_name) -> tuple[float, tuple]:
    """Get bias and Free Energy from Batch Simulation + reconstructions."""
    
    key = random.PRNGKey(SEED)

    fig, (ax_landscape, ax_hist) = plt.subplots(1,2, figsize = (14, 6))


    reconstructed_landscapes = []
    reconstructed_labels = []
    biases = [] 
    biases_std = []
    all_works = []
    model_fes = []
    model_fes_std = []
    for model, name in zip(models, names):
        energies = []
        fes = []
        midpoints = []
        for _ in tqdm(range(NUM_RECONSTRUCTIONS)):
            key, _ = random.split(key)
            
            if isinstance(model, bcm.JointModel):
                pos_stiff_models = model.models
                            
            else:
                pos_stiff_models = (model, param_set.k_s)
            
            no_key_simulate_fn = lambda key: param_set.simulate_fn(*pos_stiff_models, key = key, regime = "brownian")
            total_work, (trajectories, works, _) = bcs.batch_simulate_harmonic(BATCH_SIZE, no_key_simulate_fn, key)
            
            m, E = bcla.energy_reconstruction(
                            jnp.cumsum(works, axis=1), 
                            trajectories, 
                            BINS, 
                            *pos_stiff_models,
                            param_set.simulation_steps, 
                            BATCH_SIZE, 
                            param_set.beta)

            energies.append(bcla.interpolate_inf(E))
            midpoints.append(m)
            b = param_set.beta
            fes.append(-(1/b) * jnp.log(jnp.mean(jnp.exp(- b * total_work))))
        
        first_well, second_well = bcu.get_trap_pos_from_params(param_set)
        
        all_biases = []
        for m, e in zip(midpoints, energies):
            all_biases.append(jnp.max(bcla.landscape_discrepancies(
                (m, e), 
                param_set.energy_fn(no_trap = True), 
                r_min = first_well, 
                r_max = second_well)))
            
            
        # Average Landscape
        ls = bcla.average_landscape(midpoints, energies)
        
        all_biases = jnp.array(all_biases)
        bias = jnp.max(bcla.landscape_discrepancies(
            ls, 
            param_set.energy_fn(no_trap = True),
            r_min = first_well,
            r_max = second_well
        )) 
        
        biases.append(float(bias))
        biases_std.append(float(jnp.std(all_biases)))
        
        fes = jnp.array(fes)
        model_fes.append(float(jnp.mean(fes)))
        model_fes_std.append(float(jnp.std(fes)))
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
    
    return (biases, biases_std), (model_fes, model_fes_std)


def plot_landscape_bias_with_increasing_barrier_height(landscape_df, zoom = False):
    with open(landscape_df.params_file.iloc[0], "rb") as f:
        param_set = pickle.load(f)
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    sns.lineplot(data = landscape_df, x = "barrier_height_int", y="bias", hue = "Model Name", errorbar=("sd", 1), markers="o", ax = ax)
    if zoom == True:
        ul_inset = fig.add_axes([0.17, 1 - 0.4 - 0.15, 0.3,0.4])
        _b = landscape_df[landscape_df.barrier_height_int <= 17.5]
        sns.lineplot(data = _b, x = "barrier_height_int", y="bias", hue = "Model Name", errorbar=("sd", 0), markers="o", ax = ul_inset, legend=None)
        ul_inset.grid()
        ul_inset.set_xlabel("")
        ul_inset.set_ylabel("")
    else:
        plotting.add_energy_fn_plot(param_set, fig = fig)
        
    ax.legend(loc="upper right")
    ax.set_xlabel("Barrier Height (kT)")
    ax.set_ylabel("Landscape Bias (kT)")
    ax.grid()
    
    return fig


def plot_dg_with_increasing_barrier_height(landscape_df): 
    with open(landscape_df.params_file.iloc[0], "rb") as f:
        param_set = pickle.load(f)
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    sns.lineplot(data = landscape_df, x = "barrier_height_int", y="dg", hue = "Model Name", markers="o", ax = ax)
    plotting.add_energy_fn_plot(param_set, fig = fig)
    ax.legend(loc="upper right")
    ax.set_xlabel("Barrier Height (kT)")
    ax.set_ylabel("Free Energy Difference (kT)")
    ax.grid()
    
    return fig


def resample_rows_for_plotting(df: pd.DataFrame, n_iter = 3000) -> pd.DataFrame:
    """Each row, resample it with the given standard deviation `n_iter` number of times so plotting with seaborn is easy."""
    new_rows = []
    for _, row in df.iterrows():
        resampling_cols = [{} for _ in range(n_iter)] 
        new_row = row.to_dict()
        for col in df.columns:
            if col.endswith("_std"):
                original_col = col[:-4]
                resampled = np.random.normal(loc=row[original_col], scale = row[col], size = n_iter)
                for id, r in enumerate(resampled):
                    resampling_cols[id][original_col] = r
                    
        
        for resampled_row in resampling_cols:
            new_rows.append({**new_row, **resampled_row})
    
    return pd.DataFrame(new_rows)
        

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("file_dir")
    parser.add_argument("-r", "--reconstruct", action = "store_true", help = "Perform all reconstructions (enable if data.csv is not available)")
    args = parser.parse_args()
    
    file_dir = args.file_dir
    
    if args.reconstruct:
        df = load_data_from_result(file_dir)
        df = df.reset_index()
        bias_results = [[], []]
        dG_results = [[], []]
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
            
            statistics = reconstruct_with_models(param_set, models, names, save_name = f"{full_name}kt")
            (bias, bias_std), (dg, dg_std) = statistics
            bias_results[0].append(pd.Series(data = bias, index = lb_df.index))
            bias_results[1].append(pd.Series(data = bias_std, index = lb_df.index))
            dG_results[0].append(pd.Series(data = dg, index = lb_df.index))
            dG_results[1].append(pd.Series(data = dg_std, index = lb_df.index))
        
        df["bias"] = pd.concat(bias_results[0])
        df["bias_std"] = pd.concat(bias_results[1])
        df["dg"] = pd.concat(dG_results[0])
        df["dg_std"] = pd.concat(dG_results[1])
        
        df.to_csv(f"{results_dir}/data.csv")
    else:
        assert os.path.isfile(f"{results_dir}/data.csv"), f"data.csv not found."
        df = pd.read_csv(f"{results_dir}/data.csv")
        
    make_df_names_nice(df)
    resampled_df = resample_rows_for_plotting(df)
    
    for landscape_name, landscape_df in resampled_df.groupby("landscape_name"):
        landscape_bias_fig = plot_landscape_bias_with_increasing_barrier_height(landscape_df)
        landscape_bias_fig.savefig(f"{results_dir}/{landscape_name}_bias_full.png")
        
        landscape_bias_zoomed_fig = plot_landscape_bias_with_increasing_barrier_height(landscape_df, zoom = True)
        landscape_bias_zoomed_fig.savefig(f"{results_dir}/{landscape_name}_bias_zoomed.png")
        
        dg_fig = plot_dg_with_increasing_barrier_height(landscape_df)
        dg_fig.savefig(f"{results_dir}/{landscape_name}_dg.png")
        