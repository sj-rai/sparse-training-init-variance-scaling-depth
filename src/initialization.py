import math
import torch

def initialize_model(model, mode="he", df=5, sparsity=0.0, variance_matched=False):
    for name, param in model.named_parameters():
        if "weight" in name:

            if len(param.shape) == 4:  # Conv2d
                fan_in = param.shape[1] * param.shape[2] * param.shape[3]
            elif len(param.shape) == 2:  # Linear
                fan_in = param.shape[1]
            else:
                fan_in = param.shape[0]

            # Adjust fan-in for sparsity if using static sparse training
            effective_fan_in = fan_in * (1 - sparsity)
            if effective_fan_in == 0:
                effective_fan_in = fan_in  # safety

            if mode == "he":
                std = math.sqrt(2.0 / effective_fan_in)
                param.data.normal_(0, std)

            elif mode == "student":

                if variance_matched == True:
                    # we dont want to change the overall scale, but still need ocassional extreme values stil matching Gaussian variance
                    # Sample from Student-t
                    dist = torch.distributions.StudentT(df)
                    w = dist.sample(param.shape)

                    # Empirical variance normalization
                    # for any df (even <= 2)
                    emp_std = w.std()
                    w = w / emp_std

                    # He scaling
                    std = math.sqrt(2.0 / effective_fan_in)
                    w = w * std

                    param.data = w
                else:
                    dist = torch.distributions.StudentT(df)
                    w = dist.sample(param.shape)

                    # Normalize Student-t variance if df > 2
                    if df > 2:
                        var = df / (df - 2)
                        w = w / math.sqrt(var)

                    std = math.sqrt(2.0 / effective_fan_in)
                    w = w * std
                    param.data = w

        elif "bias" in name:
            param.data.zero_()

def initialize_model_mask_first(model, mode="he", df=5, sparsity=0.0, variance_matched=True):

    masks = {}  # store masks if you want to reuse them later

    for name, param in model.named_parameters():

        if "weight" in name:

            # ---- Compute fan-in ----
            if len(param.shape) == 4:  # Conv2d
                fan_in = param.shape[1] * param.shape[2] * param.shape[3]
            elif len(param.shape) == 2:  # Linear
                fan_in = param.shape[1]
            else:
                fan_in = param.shape[0]

            # ---- Create mask FIRST ----
            if sparsity > 0:
                mask = (torch.rand_like(param) > sparsity).float()
            else:
                mask = torch.ones_like(param)

            masks[name] = mask

            # ---- Effective fan-in (based on active connections) ----
            active_per_neuron = fan_in * (1 - sparsity)
            if active_per_neuron <= 0:
                active_per_neuron = 1  # safety

            std = math.sqrt(2.0 / active_per_neuron)

            # ---- Initialize only active weights ----
            if mode == "he":

                w = torch.zeros_like(param)
                w[mask.bool()] = torch.randn_like(w[mask.bool()]) * std
                param.data = w

            elif mode == "student":

                dist = torch.distributions.StudentT(df)
                w = torch.zeros_like(param)

                sampled = dist.sample(w[mask.bool()].shape)

                if variance_matched:
                    sampled = sampled / sampled.std()

                sampled = sampled * std
                w[mask.bool()] = sampled

                param.data = w

        elif "bias" in name:
            param.data.zero_()

    return masks

# def student_t_he_init(param, df, effective_fan_in):
#     # Sample from Student-t
#     dist = torch.distributions.StudentT(df)
#     w = dist.sample(param.shape)

#     # --- Empirical variance normalization ---
#     # Works for ANY df (even <= 2)
#     emp_std = w.std()
#     w = w / emp_std

#     # --- He scaling ---
#     std = math.sqrt(2.0 / effective_fan_in)
#     w = w * std

#     param.data = w