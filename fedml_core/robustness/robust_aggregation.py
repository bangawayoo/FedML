"""
Computing (multi-)Krum is from https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning
"""

import torch


def vectorize_weight(state_dict):
    weight_list = []
    for (k, v) in state_dict.items():
        if is_weight_param(k):
            weight_list.append(v.flatten())
    return torch.cat(weight_list)


def load_model_weight_diff(local_state_dict, weight_diff, global_state_dict):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    recons_local_state_dict = {}
    index_bias = 0
    for item_index, (k, v) in enumerate(local_state_dict.items()):
        if is_weight_param(k):
            recons_local_state_dict[k] = weight_diff[index_bias:index_bias + v.numel()].view(v.size()) + \
                                         global_state_dict[k]
            index_bias += v.numel()
        else:
            recons_local_state_dict[k] = v
    return recons_local_state_dict


def is_weight_param(k):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)


class RobustAggregator(object):
    def __init__(self, args):
        self.defense_type = args.defense_type
        self.norm_bound = args.norm_bound  # for norm diff clipping and weak DP defenses
        self.stddev = args.stddev  # for weak DP defenses

    def norm_diff_clipping(self, local_state_dict, global_state_dict):
        vec_local_weight = vectorize_weight(local_state_dict)
        device = vec_local_weight.device
        vec_global_weight = vectorize_weight(global_state_dict).to(device)

        # clip the norm diff
        vec_diff = vec_local_weight - vec_global_weight
        weight_diff_norm = torch.norm(vec_diff).item()
        clipped_weight_diff = vec_diff / max(1, weight_diff_norm / self.norm_bound)
        clipped_local_state_dict = load_model_weight_diff(local_state_dict,
                                                          clipped_weight_diff,
                                                          global_state_dict)
        return clipped_local_state_dict

    def add_noise(self, local_weight):
        device = local_weight.device
        gaussian_noise = torch.randn(local_weight.size(),
                                     device=device) * self.stddev
        dp_weight = local_weight + gaussian_noise
        return dp_weight

    def getKrum(self, vectorized_weight):
        '''
        From https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning
        compute krum or multi-krum of input. O(dn^2)

        input : batchsize * vector dimension * n

        return
            krum : batchsize * vector dimension * 1
            mkrum : batchsize * vector dimension * 1
        '''

        n = vectorized_weight.shape[-1]
        f = n // 10  # 10% malicious points
        k = n - f - 2

        # collection distance, distance from points to points
        x = vectorized_weight.permute(0, 2, 1)
        cdist = torch.cdist(x, x, p=2)
        # find the k+1 nbh of each point
        nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
        # the point closest to its nbh
        i_star = torch.argmin(nbhDist.sum(2))
        # krum
        krum = vectorized_weight[:, :, [i_star]]
        # Multi-Krum
        mkrum = vectorized_weight[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
        return krum, mkrum
