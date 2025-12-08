import pickle as pkl
from counterfactuals.find_many_counterfactuals import multiple_vanilla_counterfactuals

# ---- 1. Paths to your trained models ----
DATA_PATH_GRID = "cnn_spn_models/full_run_100e_ft50/grid0/"  # ends with /

# ---- 2. Load the params used for this grid run ----
with open(DATA_PATH_GRID + "grid_params.pkl", "rb") as f:
    grid_params = pkl.load(f)

# grid_params is usually a list/array of param objects; take the first for grid0
params = grid_params[0] if isinstance(grid_params, (list, tuple)) else grid_params

# Grab values from params where possible, otherwise fall back to what we know
dataset_name = getattr(params, "dataset_name", "chexpert")
add_info = getattr(params, "use_add_info", 1)
model_n = "cnn_spn"

# This is where load_model will look; it will also get the right fold via fold_idx
model_path = DATA_PATH_GRID

# ---- 3. Call the counterfactual generator for fold 0 ----
if __name__ == "__main__":
    multiple_vanilla_counterfactuals(
        dataset_name=dataset_name,
        params=params,
        data_path_grid=DATA_PATH_GRID,
        add_info=add_info,
        model_n=model_n,
        num_imgs=5,         # how many test images you want CFs for
        path="",         # same 'path' you used in training's load_dataset
        model_path=model_path,
        learning_rate=0.01,
        num_steps=150,
        fold_idx=0,         # use 1 or 2 for other folds
        replicates=50       # number of CF samples per image
    )
