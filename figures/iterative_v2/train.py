import barrier_crossing.models as bcm
import barrier_crossing.iterate_landscape as bcla
import barrier_crossing.loss as loss
import barrier_crossing.train as bct
import barrier_crossing.plotting as plotting
import barrier_crossing.utils as bcu

import matplotlib.pyplot as plt

import datetime
import pandas as pd

import os

import jax.random as random
import optax
import pickle

BINS = 70
MAX_ITER = 7
TRAIN_BATCH_SIZE = 3000
NUM_EPOCHS = 500
NUM_RECONSTRUCTIONS = 100

key = random.PRNGKey(1001)
learning_rate = optax.exponential_decay(0.1, NUM_EPOCHS, 0.5, end_value = 0.001)
OPTIMIZER = optax.adam(learning_rate)

OUT_DIR = "output_data/" + str(datetime.datetime.now()).replace(" ","_") + "/"
main = os.environ.get("ITERATIVE_RESULT_DIR", "results")
FILE_DIR = f"{main}/files/"
MISC_SAVE_PATH = f"{main}/misc_plots/"


    
def train_iterative_model(max_iter: int, model: bcm.ScheduleModel, name) -> bcm.ScheduleModel:
    RECONSTRUCT_BATCH_SIZE = 1000
    
    param_set = model.params

    true_simulation_fwd = lambda trap_fn, ks_fn: lambda keys: param_set.simulate_fn(
    trap_fn, 
    ks_fn,
    keys, 
    regime = "brownian",
    fwd = True)

    sim_no_E = lambda energy_fn: lambda trap_fn, ks_fn, keys: param_set.simulate_fn(
    trap_fn, 
    ks_fn,
    keys, 
    regime = "brownian",
    fwd = False,
    custom = energy_fn)

    reconstruct_fn = lambda batch_works, trajectories, position_fn, stiffness_fn, batch_size: \
    bcla.energy_reconstruction(batch_works, 
                                trajectories, 
                                BINS, 
                                position_fn, 
                                stiffness_fn, 
                                param_set.simulation_steps,
                                batch_size, 
                                param_set.beta)

    grad_no_E = lambda model, simulate_fn: lambda num_batches: loss.estimate_gradient_work(
        num_batches,
        simulate_fn,
        model)

    train_fn = lambda model, grad_fn, key: bct.train(model, OPTIMIZER, grad_fn, key, batch_size = TRAIN_BATCH_SIZE, num_epochs = NUM_EPOCHS)

    it_landscapes, it_coeffs, _ = bcla.optimize_landscape(
    max_iter,
    RECONSTRUCT_BATCH_SIZE,
    model,
    true_simulation_fwd,
    sim_no_E,
    grad_no_E,
    reconstruct_fn,
    train_fn,
    key,
    num_reconstructions = NUM_RECONSTRUCTIONS)
    
    model.pop_hist(reset = True)
    
    new_model = model.clone()
    
    for coef in it_coeffs:
        new_model.coeffs = coef
    
    fig, (ax_landscapes, ax_protocol) = plt.subplots(1,2, figsize = (11,5))
    plotting.plot_evolving(ax = ax_protocol, model = new_model, num = max_iter)
    
    plotting.plot_landscapes(param_set, ax = ax_landscapes, landscapes = [it_landscapes], labels = ["Iterated Landscape"], iterated_index = 0)
    plotting.add_axes("landscape", ax = ax_landscapes)

    ax_landscapes.legend()
    
    plt.tight_layout()
    fig.savefig(MISC_SAVE_PATH + f"train_{name}.png")
    
    return new_model
    
def train_true_model(model: bcm.ScheduleModel, name) -> bcm.ScheduleModel:
    param_set = model.params
    if isinstance(model, bcm.JointModel): 
       auto_diff_simulate_fn = lambda pos, stiff, key: param_set.simulate_fn(pos, stiff, key = key)
    else:
       auto_diff_simulate_fn = lambda pos, key: param_set.simulate_fn(pos, ks_trap_fn = False, key = key)
    
    grad_fn = lambda N: loss.estimate_gradient_work(N, auto_diff_simulate_fn, model)

    losses = bct.train(model, OPTIMIZER, grad_fn, key, TRAIN_BATCH_SIZE, NUM_EPOCHS)
    
    fig, (loss_ax, protocol_ax) = plt.subplots(1, 2, figsize = (11,5))

    plotting.plot_evolving(ax = protocol_ax, model = model, color = "k")
    
    protocol_ax.set_xlabel("Timestep")

    plotting.plot_with_stdev(losses, ax = loss_ax, axis = 1)
    loss_ax.set_title("Dissipated Work Loss")
    loss_ax.set_xlabel("Epochs")
    
    plt.tight_layout()
    fig.savefig(MISC_SAVE_PATH + f"train_{name}.png")
    
    return model

if __name__ == "__main__":
    
    os.makedirs(MISC_SAVE_PATH, exist_ok = True)
    os.makedirs(OUT_DIR, exist_ok = True)
    os.makedirs(FILE_DIR, exist_ok= True)
    
    args, p = bcu.parse_args()
    param_set = p.param_set
    
    full_name = f"{args.landscape_name}_{args.param_suffix}kt"
    
    position_model = bcm.ScheduleModel(param_set, init_pos = p.r0_init, final_pos = p.r0_final)
    stiffness_model = bcm.ScheduleModel(param_set, init_pos = p.ks_init, final_pos = p.ks_final)
    
    joint_model = bcm.JointModel(param_set, position_model, stiffness_model)
    
    iterative_models = {
        "iterative_position": position_model.clone(), 
        "iterative_joint": joint_model.clone()
    }
    
    true_models = {
        "true_position": position_model.clone(),
        "true_joint": joint_model.clone()
    }
    
    linear_model = position_model.clone()
    
    results = []
    
    params_path = OUT_DIR + "params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(param_set, f)
    
    def save_coeffs(model, name, results):
        out_path = OUT_DIR + name + ".pkl"
        results.append({
            "landscape_name": args.landscape_name,
            "barrier_height": args.param_suffix,
            "model_name": name,
            "coeff_file": out_path,
            "params_file": params_path,
            "epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "end_time": param_set.end_time
        })
        
        with open(out_path, "wb") as f:
            pickle.dump(model.coef_hist, f)
    
    for it_name, it_model in iterative_models.items():
        trained_it_model = train_iterative_model(MAX_ITER, it_model, f"{full_name}_{it_name}")
        
        save_coeffs(trained_it_model, it_name, results)
        
    
    for tr_name, tr_model in true_models.items():
        trained_tr_model = train_true_model(tr_model, f"{full_name}_{tr_name}")
        
        save_coeffs(trained_tr_model, tr_name, results)
    
    save_coeffs(linear_model, "linear", results)
    
    csv_path = os.path.realpath(OUT_DIR + "data.csv")
    pd.DataFrame(results).to_csv(csv_path)
    
    with open(FILE_DIR + f"{full_name}.txt", "w") as f:
        f.write(csv_path)
    
