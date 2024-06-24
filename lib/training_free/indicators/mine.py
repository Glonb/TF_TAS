import torch

from . import indicator
from ..p_utils import get_layer_metric_array_mine
import torch.nn.functional as F


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

    @torch.no_grad()
    def calculate_attention_similarity(attns):
        num_heads = attns.shape[1]
        attention_weights = attns.mean(dim=0)  # Average over the batch dimension
        similarities = torch.zeros(num_heads, num_heads)

        for i in range(num_heads):
            for j in range(num_heads):
                if i == j:
                    similarities[i, j] = 1.0
                else:
                    sim = F.cosine_similarity(attention_weights[i].flatten(), attention_weights[j].flatten(), dim=0)
                    similarities[i, j] = sim

        return torch.abs(similarities).sum()

    def mine(layer):
        if layer.attns is not None:
            score = calculate_attention_similarity(layer.attns)
            # print('captured!, score: ', score.item())
            return score.to(device)
        else:
            return torch.tensor(0).to(device)

    siml = get_layer_metric_array_mine(net, mine, mode)

    nonlinearize(net, signs)

    return siml
