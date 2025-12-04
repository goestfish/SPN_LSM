import numpy as np
import networkx as nx

# add NINF

try:
    import scipy
    if not hasattr(scipy, "NINF"):
        scipy.NINF = -np.inf
        print("[PATCH] Added scipy.NINF = -np.inf")
    else:
        print("[PATCH] scipy.NINF already exists")
except Exception as e:
    print("[PATCH WARNING] Failed to patch scipy.NINF:", e)


# Fix from_numpy_matrix removal

try:
    import spn.algorithms.splitting.Base as _base
    _base.from_numpy_matrix = nx.from_numpy_array
    print("[PATCH] Patched networkx.from_numpy_matrix → nx.from_numpy_array in spn.splitting.Base")
except Exception as e:
    print("[PATCH WARNING] Failed to patch from_numpy_matrix:", e)



# Remove unsupported n_clusters kwarg
import inspect
import spn.algorithms.LearningWrappers as LW

if "n_clusters" not in inspect.signature(LW.learn_parametric).parameters:
    original_func = LW.learn_parametric

    def learn_parametric_no_clusters(*args, **kwargs):
        if "n_clusters" in kwargs:
            print("[PATCH] Warning: removing unsupported kwarg n_clusters")
            kwargs.pop("n_clusters")
        return original_func(*args, **kwargs)

    LW.learn_parametric = learn_parametric_no_clusters
    print("[PATCH] Patched learn_parametric() to ignore n_clusters")