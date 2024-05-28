import torch

from . import indicator
from ..p_utils import get_layer_metric_array_mine
import torch.nn as nn


@indicator('mine', bn=False, mode='param')
def compute_mine_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):
    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def mine(layer):
        if layer._get_name() == 'AttentionSuper' and layer.attentions is not None:
            score = layer.diversity_score()
            # print('captured!, score: ', score.item())
            return score.to(device)
        else:
            return torch.tensor(0).to(device)

    siml = get_layer_metric_array_mine(net, mine, mode)

    nonlinearize(net, signs)

    return siml
