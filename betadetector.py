import pickle as pkl
import numpy as np

# ==== adjust these paths if needed ====
path_b0g0 = "cnn_spn_models/full_run_100e_ft25/grid0/fold_0/counterfactual_imgs/datab0g0SPN_1000_new.pkl"
path_b1g0 = "cnn_spn_models/full_run_100e_ft25/grid0/fold_0/counterfactual_imgs/datab1g0SPN_1000_new.pkl"

print("Loading pickles...")
p0 = pkl.load(open(path_b0g0, "rb"))   # beta = 0, gamma = 0
p1 = pkl.load(open(path_b1g0, "rb"))   # beta = 1, gamma = 0

print("=== High-level structure ===")
print("type p0:", type(p0), "len:", len(p0))
print("type p0[0]:", type(p0[0]), "len:", len(p0[0]))

print("type p0[0][0]:", type(p0[0][0]))
print("tuple length (p0[0][0]):", len(p0[0][0]))

# Inspect the content of the tuple elements
print("\n=== Inspecting tuple elements for image 0 (beta=0) ===")
t0 = p0[0][0]
for j, elem in enumerate(t0):
    if hasattr(elem, "shape"):
        print(f"  elem {j}: array with shape {elem.shape}, dtype {elem.dtype}")
    else:
        print(f"  elem {j}: type {type(elem)}")

# ==== Compare beta=0 vs beta=1 ====
print("\n=== Comparing beta=0 vs beta=1 ===")
images_b0 = p0[0]
images_b1 = p1[0]

n_imgs = min(len(images_b0), len(images_b1))
print(f"Number of images in each: {len(images_b0)} (b0), {len(images_b1)} (b1)")
print(f"Comparing first {n_imgs} images\n")

for i in range(n_imgs):
    t0 = images_b0[i]
    t1 = images_b1[i]

    print(f"--- Image {i} ---")
    if len(t0) != len(t1):
        print(f"  WARNING: tuple lengths differ: {len(t0)} vs {len(t1)}")
        continue

    for j in range(len(t0)):
        a, b = t0[j], t1[j]

        # Only compare numpy arrays / tensors
        if hasattr(a, "shape") and hasattr(b, "shape"):
            same_shape = (a.shape == b.shape)
            if not same_shape:
                print(f"  elem {j}: shape differs: {a.shape} vs {b.shape}")
                continue

            diff = np.abs(a - b)
            max_diff = diff.max() if diff.size > 0 else 0.0
            allclose = np.allclose(a, b)

            print(f"  elem {j}: shape {a.shape}, allclose={allclose}, max |Δ|={max_diff:.3e}")
        else:
            # Non-array elements (e.g., scalars, lists, strings)
            equal = (a == b)
            print(f"  elem {j}: non-array type {type(a)}, equal={equal}")
