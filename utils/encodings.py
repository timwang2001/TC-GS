import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import torchac
import math
import multiprocessing


anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.

def get_binary_vxl_size(binary_vxl):
    # binary_vxl: {0, 1}
    # assert torch.unique(binary_vxl).mean() == 0.5
    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  #  + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg
    # print('binary_vxl:', Pg.item(), ttl_bit.item(), ttl_num, pos_num.item(), neg_num.item())
    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num


def multiprocess_encoder(lower, symbol, file_name, chunk_num=10):
    def enc_func(l, s, f, b_l, i):
        byte_stream = torchac.encode_float_cdf(l, s, check_input_bounds=True)
        with open(f, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream) * 8
        b_l[i] = bit_len
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    b_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        symbol_m = symbol[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=enc_func, args=(lower_m, symbol_m, file_name_m, b_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    ttl_bit_len = sum(list(b_list))
    return ttl_bit_len


def multiprocess_deoder(lower, file_name, chunk_num=10):
    def dec_func(l, f, o_l, i):
        with open(f, 'rb') as fin:
            byte_stream_d = fin.read()
        o = torchac.decode_float_cdf(l, byte_stream_d).to(torch.float32)
        o_l[i] = o
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    output_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=dec_func, args=(lower_m, file_name_m, output_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    output_list = torch.cat(list(output_list), dim=0).cuda()
    return output_list


def encoder_gaussian(x, mean, scale, Q, file_name='tmp.b'):
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert x.shape == mean.shape == scale.shape == Q.shape
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    del samples
    del mean
    del scale
    del GD
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    assert (x_int_round_idx.to(torch.int32) == x_int_round - min_value).all()
    # if x_int_round_idx.max() >= lower.shape[-1] - 1:  x_int_round_idx.max() exceed 65536 but to int6, that's why error
        # assert False

    if not use_multiprocessor:
        byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)
        with open(file_name, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream)*8
    else:
        bit_len = multiprocess_encoder(lower.cpu(), x_int_round_idx.cpu(), file_name)
    torch.cuda.empty_cache()
    return bit_len, min_value, max_value


def decoder_gaussian(mean, scale, Q, file_name='tmp.b', min_value=-100, max_value=100):
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert mean.shape == scale.shape == Q.shape
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(mean.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    if not use_multiprocessor:
        with open(file_name, 'rb') as fin:
            byte_stream_d = fin.read()
        sym_out = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).to(mean.device).to(torch.float32)
    else:
        sym_out = multiprocess_deoder(lower.cpu(), file_name, chunk_num=10).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    torch.cuda.empty_cache()
    return x


def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(((x+1)/2)).to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    # Number of bits taken by the stream
    bit_len = len(byte_stream) * 8
    # Write to a file.
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    # Read from a file.
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    # Decode from bytestream.
    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)


def encoder_anchor(anchors, anchor_digits=anchor_round_digits, file_name='anchor.b'):
    anchors = anchors.detach()
    anchors_voxel = anchors / Q_anchor
    anchor_voxel_min = (torch.min(anchors_voxel, dim=0, keepdim=True)[0])
    anchor_voxel_max = (torch.max(anchors_voxel, dim=0, keepdim=True)[0])

    anchors_voxel -= anchor_voxel_min
    anchors_voxel = anchors_voxel.to(torch.long)
    voxel_hwd = anchor_voxel_max - anchor_voxel_min + 1

    voxel_hwd = voxel_hwd.to(torch.long).squeeze()
    voxel_1 = torch.zeros(size=voxel_hwd.tolist(), device='cuda', dtype=torch.long)

    anchors_unique_values, anchors_unique_cnts = torch.unique(anchors_voxel, dim=0, return_counts=True)
    # anchors_reassembled = torch.repeat_interleave(anchors_unique_values, anchors_unique_cnts.to(torch.long), dim=0)
    _, inv = torch.unique(anchors_voxel, dim=0, return_inverse=True)
    indices = torch.argsort(inv)

    anchors_unique_values = anchors_unique_values.to(torch.long)
    mask = anchors_unique_cnts == 1
    anchors_unique_values_1 = anchors_unique_values[mask]
    anchors_unique_values_not1 = anchors_unique_values[~mask]
    anchors_unique_cnts_not1 = anchors_unique_cnts[~mask]
    voxel_1[anchors_unique_values_1[:, 0], anchors_unique_values_1[:, 1], anchors_unique_values_1[:, 2]] = 1
    voxel_1 = voxel_1.view(-1)
    prob_1 = (voxel_1.sum() / voxel_1.numel()).item()
    p = torch.zeros_like(voxel_1).to(torch.float32)
    p[...] = prob_1
    voxel_1 = voxel_1 * 2 - 1

    file_name_value_not1 = file_name.replace('.b', '_value_not1.b')
    file_name_cnts_not1 = file_name.replace('.b', '_cnts_not1.b')

    # Directly store them for simplicity
    torch.save(anchors_unique_values_not1, f=file_name_value_not1)  # assume we use digit=anchor_digits to encode.
    torch.save(anchors_unique_cnts_not1, f=file_name_cnts_not1)  # assume we use digit=16 to encode.

    bit_len_anchor_1 = encoder(voxel_1, p, file_name)
    # decoded_list = decoder(p, file_name)  # {-1, 1}
    bit_len_anchor_not1 = anchor_digits * anchors_unique_values_not1.numel() + 16 * anchors_unique_cnts_not1.numel()
    bit_len_anchor = bit_len_anchor_1 + bit_len_anchor_not1
    return bit_len_anchor, indices, voxel_hwd, anchor_voxel_min, anchor_voxel_max, prob_1


def decoder_anchor(voxel_hwd, prob_1, anchor_voxel_min, anchor_voxel_max, file_name='anchor.b'):
    voxel_1 = torch.zeros(size=voxel_hwd.tolist(), device='cuda', dtype=torch.long)
    p = torch.zeros_like(voxel_1).to(torch.float32)
    p[...] = prob_1
    decoded_list = decoder(p, file_name)  # {-1, 1}. shape=[h, w, d]
    decoded_list = (decoded_list + 1) / 2  # {0, 1}
    decoded_voxel_1 = torch.argwhere(decoded_list == 1)

    file_name_value_not1 = file_name.replace('.b', '_value_not1.b')
    file_name_cnts_not1 = file_name.replace('.b', '_cnts_not1.b')
    decoded_voxel_value_not1 = torch.load(file_name_value_not1)
    decoded_voxel_cnts_not1 = torch.load(file_name_cnts_not1)
    decodedc_voxel_not1 = torch.repeat_interleave(decoded_voxel_value_not1, decoded_voxel_cnts_not1.to(torch.long), dim=0)

    decodedc_voxel = torch.cat([decoded_voxel_1, decodedc_voxel_not1], dim=0).to(torch.float32)

    v, c = torch.unique(decodedc_voxel, dim=0, return_counts=True)
    decodedc_reassembled = torch.repeat_interleave(v, c.to(torch.long), dim=0)

    decoded_anchor = (decodedc_reassembled + anchor_voxel_min) * Q_anchor

    return decoded_anchor


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            input_min = input_mean - 15_000 * Q
            input_max = input_mean + 15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        # if anchor_round_digits == 32:
            # return anchors
        # min_v = torch.min(anchors).detach()
        # max_v = torch.max(anchors).detach()
        # scales = 2 ** anchor_round_digits - 1
        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # quantized_v = (anchors - min_v) // interval
        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None

