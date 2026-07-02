"""Translate baseline names and parameters from the original notebooks."""


def normalize_fixed_baseline(name, params):
    name = str(name)
    p = dict(params)
    if name == "EasyTSF":
        name = str(p.pop("model_type"))
        p = {
            "window_size": p.get("window_size", 12),
            "learning_rate": p.get("learning_rate", p.get("lr", 1e-3)),
            "weight_decay": p.get("weight_decay", 0.0),
            "epochs": p.get("epochs", p.get("max_epochs", 150)),
            "patience": p.get("patience", 15),
            **({"kernel_size": p.get("moving_avg", 3)} if name == "DLinear" else {}),
        }
    elif name == "ARKAN":
        p = {
            "window_size": p.get("window_size", 12),
            "h1": p.get("h1", 8),
            "h2": p.get("h2", 4),
            "n_hidden_layers": p.get("n_hidden_layers", 1),
            "grid": p.get("grid", 3),
            "k": p.get("k", 3),
            "learning_rate": p.get("lr", 1e-3),
            "weight_decay": p.get("weight_decay", 0.0),
            "epochs": p.get("max_epochs", 100),
            "patience": p.get("patience", 12),
        }
    elif name == "GaussianProcess" and p.get("kernel_type") == "pairwise":
        p["kernel_type"] = "rbf"
    elif name == "Chronos":
        p = {
            key: value
            for key, value in p.items()
            if key
            in {
                "model_or_factory",
                "predict_fn",
                "model_id",
                "device_map",
                "torch_dtype",
                "prediction_kwargs",
            }
        }
    if name == "AR":
        p.setdefault("period", 12)
    return name, p
