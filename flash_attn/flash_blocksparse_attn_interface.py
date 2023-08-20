# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py
import torch
import torch.nn as nn

import flash_attn_cuda


def convert_blockmask(blockmask, causal):
    """Convert from the 0-1 format to the format used by the CUDA code.
    0 means the block is skipped.
    nonzero means the block is not skipped.
    Argument:
        blockmask: (row, col): a 0-1 tensor
    Return:
        blockmask_converted: (col, row), dtype torch.int32: for each column, it contains the row
            indices of the nonzero blocks, padded with -1 to reach length @row.
            The indices are multiplied by 4, with the smallest bit used to encode whether
            it is the first nonzero in its row, and the 2nd smallest bit to encode whether it is
            the last nonzero in its row..
    """
    assert not causal
    # TD [2022-05-13]: The indexing and sorting is very tricky
    nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=0, stable=True, descending=True)
    nonzero_unsorted_rowidx = nonzero_sorted_rowidx.argsort(dim=0)
    last_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True).indices[:, -1]
    last_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), last_nonzero_col_per_row
    ]
    first_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True, descending=True).indices[:, 0]
    first_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), first_nonzero_col_per_row
    ]
    nonzero_idx = nonzero_sorted_rowidx * 4
    nonzero_idx[last_nonzero_col_per_row_after_sort, last_nonzero_col_per_row] += 2
    nonzero_idx[first_nonzero_col_per_row_after_sort, first_nonzero_col_per_row] += 1
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.T.contiguous().to(dtype=torch.int32)


# follow https://github.com/Dao-AILab/flash-attention/blob/6cc7342575393568f32c69aba6365e93d7701cbb/flash_attn/flash_attn_interface.py#L39
# how to call unpacked from packed kqv

#def _flash_attn_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
#                        softmax_scale, causal, return_softmax):

#std::vector<at::Tensor>
#mha_fwd_block(const at::Tensor &q,         // total_q x num_heads x head_size, total := \sum_{i=0}^{b} s_i
#              const at::Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              const at::Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              const at::Tensor &cu_seqlens_q,  // b+1
#              const at::Tensor &cu_seqlens_k,  // b+1
#              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
#              const int max_seqlen_q_,
#              const int max_seqlen_k_,
#              const float p_dropout,
#              const float softmax_scale,
#              const bool is_causal,
#              const bool return_softmax,
#              c10::optional<at::Generator> gen_) {

def _flash_blocksparse_attn_forward(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale,
                                     causal, return_softmax):
    context, softmax_lse, *rest = flash_attn_cuda.fwd_block(
        qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens,
        blockmask, max_s, max_s, dropout_p,
        softmax_scale, causal,
        return_softmax, None)
    # if context.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


#std::vector<at::Tensor>
#mha_bwd_block(const at::Tensor &dout,  // total x num_heads, x head_size
#              const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
#              const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              const at::Tensor &out,   // total_q x num_heads x head_size
#              const at::Tensor &softmax_lse_,     // b x h x s softmax logsumexp
#              at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
#              at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
#              const at::Tensor &cu_seqlens_q,  // b+1
#              const at::Tensor &cu_seqlens_k,  // b+1
#              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
#              const int max_seqlen_q_,
#              const int max_seqlen_k_,          // max sequence length to choose the kernel
#              const float p_dropout,         // probability to drop
#              const float softmax_scale,
#              const bool is_causal,
#              c10::optional<at::Generator> gen_
def _flash_blocksparse_attn_backward(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, blockmask,
                                      dropout_p, max_s, softmax_scale, causal):

    # follow https://github.com/Dao-AILab/flash-attention/blob/6cc7342575393568f32c69aba6365e93d7701cbb/flash_attn/flash_attn_interface.py#L59
    dqkv = torch.empty_like(qkv)

    #dqkv, dp, softmax_d =

    flash_attn_cuda.bwd_block(dout,
                              qkv[:, 0], qkv[:, 1], qkv[:, 2], #qkv,
                              out,
                              #S_dmask,
                              softmax_lse,
                              dqkv[:, 0], dqkv[:, 1], dqkv[:, 2],
                              cu_seqlens, cu_seqlens,
                              blockmask,
                              max_s, max_s,
                              dropout_p, softmax_scale,
                              #max_s,
                              causal, None)
    # if dqkv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dqkv



class FlashBlocksparseAttnFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal=causal,
            return_softmax=False
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = _flash_blocksparse_attn_backward(
            dout, qkv, context, context, softmax_lse, cu_seqlens, blockmask, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class FlashBlocksparseAttnFunWithS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass is gonna regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal=causal,
            return_softmax=True
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context, S_dmask, softmax_lse

    @staticmethod
    def backward(ctx, dout, _dS_dmask_ignored, _dsoftmax_sum_ignored):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dqkv = _flash_blocksparse_attn_backward(
            dout, qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


def flash_blocksparse_attn_func(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale=None,
                                 causal=False, return_attn_probs=False, convert_mask=True):
    """dropout_p should be set to 0.0 during evaluation
    """
    func = FlashBlocksparseAttnFun if not return_attn_probs else FlashBlocksparseAttnFunWithS
    if convert_mask:
        blockmask = convert_blockmask(blockmask, causal=causal)
    return func.apply(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal)

