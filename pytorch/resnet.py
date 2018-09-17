import torch
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
    flatten_params, flatten_stats, batch_norm


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = torch.Tensor([16, 32, 64]).mul(width).int().numpy().tolist()

    def gen_block_params(ni, no, scalar):
        if scalar:
            return {
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
            }
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count, bias=False):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no, bias)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no)}
                for i in range(count)}

    flat_vectors = flatten_params({
        'conv0': conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'conv1': conv_params(widths[2], num_classes, 1),
    })

    flat_scalars = flatten_params({
        'group0': gen_group_params(16, widths[0], n, True),
        'group1': gen_group_params(widths[0], widths[1], n, True),
        'group2': gen_group_params(widths[1], widths[2], n, True),
        'bn': bnparams(widths[2]),
    })

    flat_stats = flatten_stats({
        'group0': gen_group_stats(16, widths[0], n),
        'group1': gen_group_stats(widths[0], widths[1], n),
        'group2': gen_group_stats(widths[1], widths[2], n),
        'bn': bnstats(widths[2]),
    })

    def block(x, params, stats, base, mode, stride):
        o1 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode, 1.), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, stats, base + '.bn1', mode, 1.), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, stats, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = F.relu(batch_norm(g2, params, stats, 'bn', mode, 1.))
        o = F.conv2d(o, params['conv1'])
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        return o

    return f, flat_vectors, flat_scalars, flat_stats
