import numpy as np
from nntile.layer.attention import Attention
from nntile.tensor import TensorMoments, Tensor, TensorTraits
from transformers.models.t5.modeling_t5 import T5Attention as T5AttentionTorch
from nntile.model.t5_config import T5ConfigNNTile


class T5Attention(Attention):
    def __init__(
        self,
        x_q: TensorMoments,
        x_k: TensorMoments,
        x_v: TensorMoments,
        y: TensorMoments,
        w_q: TensorMoments,
        w_k: TensorMoments,
        w_v: TensorMoments,
        w: TensorMoments,
        q_transposed: TensorMoments,
        q: TensorMoments,
        k_transposed: TensorMoments,
        k: TensorMoments,
        v_transposed: TensorMoments,
        v: TensorMoments,
        a: TensorMoments,
        a_maxsumexp: Tensor,
        a_sumprod_slice: Tensor,
        b: TensorMoments,
        b_transposed: TensorMoments,
        in_proj_bias_q: TensorMoments,
        in_proj_bias_k: TensorMoments,
        in_proj_bias_v: TensorMoments,
        out_proj_bias: TensorMoments,
        mask=None,
        redux: bool = False,
    ):
        super().__init__(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            y=y,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            w=w,
            q_transposed=q_transposed,
            q=q,
            k_transposed=k_transposed,
            k=k,
            v_transposed=v_transposed,
            v=v,
            a=a,
            a_maxsumexp=a_maxsumexp,
            a_sumprod_slice=a_sumprod_slice,
            b=b,
            b_transposed=b_transposed,
            in_proj_bias_q=in_proj_bias_q,
            in_proj_bias_k=in_proj_bias_k,
            in_proj_bias_v=in_proj_bias_v,
            out_proj_bias=out_proj_bias,
            mask=mask,
            scale_attn=False,
            redux=redux,
        )

    @staticmethod
    def generate_simple(
        x_q: TensorMoments,
        x_k: TensorMoments,
        x_v: TensorMoments,
        n_head: int,
        n_head_tile: int,
        next_tag: int,
        inner_dim: int = None,
        inner_dim_tile: int = None,
        bias=False,
        mask=None,
        redux: bool = False,
    ):
        # Get sizes
        n_batch_tile = x_q.value.basetile_shape[2]
        n_emb, n_seq, n_batch = x_q.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x_q.value.basetile_shape

        inner_dim = n_emb if inner_dim is None else inner_dim
        inner_dim_tile = n_emb_tile if inner_dim_tile is None else inner_dim_tile

        head_size = inner_dim // n_head
        # Stupid check, that is not necessary, as the code shall work
        if inner_dim != head_size * n_head:
            print(n_emb, head_size, n_head)
            raise RuntimeError
        n_emb_k = x_k.value.shape[0]
        n_emb_k_tile = x_k.value.basetile_shape[0]
        if [n_seq, n_batch] != x_k.value.shape[1:]:
            raise ValueError("Invalid shape of x_k")
        if [n_seq_tile, n_batch_tile] != x_k.value.basetile_shape[1:]:
            raise ValueError("Invalid basetile shape of x_k")
        n_emb_v = x_v.value.shape[0]
        n_emb_v_tile = x_v.value.basetile_shape[0]
        if [n_seq, n_batch] != x_v.value.shape[1:]:
            raise ValueError("Invalid shape of x_v")
        if [n_seq_tile, n_batch_tile] != x_v.value.basetile_shape[1:]:
            raise ValueError("Invalid basetile shape of x_v")
        # Fixed for now
        head_size_tile = head_size
        # Define shape of each tensor
        w_q_shape = [n_head, head_size, n_emb]
        w_k_shape = [n_head, head_size, n_emb_k]
        w_v_shape = [n_head, head_size, n_emb_v]
        w_shape = [n_emb, n_head, head_size]
        q_transposed_shape = [n_head, head_size, n_seq, n_batch]
        q_shape = [head_size, n_seq, n_batch, n_head]
        k_transposed_shape = [n_head, head_size, n_seq, n_batch]
        k_shape = [head_size, n_seq, n_batch, n_head]
        v_transposed_shape = [n_head, head_size, n_seq, n_batch]
        v_shape = [head_size, n_seq, n_batch, n_head]
        a_shape = [n_seq, n_seq, n_batch, n_head]
        a_maxsumexp_shape = [2, n_seq, n_batch, n_head]
        a_sumprod_slice_shape = [n_seq, n_batch, n_head]
        b_shape = [head_size, n_seq, n_batch, n_head]
        b_transposed_shape = [n_head, head_size, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [n_head_tile, head_size_tile, n_emb_tile]
        w_k_basetile = [n_head_tile, head_size_tile, n_emb_k_tile]
        w_v_basetile = [n_head_tile, head_size_tile, n_emb_v_tile]
        w_basetile = [n_emb_tile, n_head_tile, head_size_tile]
        q_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        q_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        k_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        k_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        v_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        v_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile, n_head_tile]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile, n_head_tile]
        a_sumprod_slice_basetile = [n_seq_tile, n_batch_tile, n_head_tile]
        b_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        b_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        w_traits = TensorTraits(w_shape, w_basetile)
        q_transposed_traits = TensorTraits(q_transposed_shape, q_transposed_basetile)
        q_traits = TensorTraits(q_shape, q_basetile)
        k_transposed_traits = TensorTraits(k_transposed_shape, k_transposed_basetile)
        k_traits = TensorTraits(k_shape, k_basetile)
        v_transposed_traits = TensorTraits(v_transposed_shape, v_transposed_basetile)
        v_traits = TensorTraits(v_shape, v_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(a_maxsumexp_shape, a_maxsumexp_basetile)
        a_sumprod_slice_traits = TensorTraits(
            a_sumprod_slice_shape, a_sumprod_slice_basetile
        )
        b_traits = TensorTraits(b_shape, b_basetile)
        b_transposed_traits = TensorTraits(b_transposed_shape, b_transposed_basetile)
        # TODO change distribution
        w_q_distr = [0] * w_q_traits.grid.nelems
        w_k_distr = [0] * w_k_traits.grid.nelems
        w_v_distr = [0] * w_v_traits.grid.nelems
        w_distr = [0] * w_traits.grid.nelems
        q_transposed_distr = [0] * q_transposed_traits.grid.nelems
        q_distr = [0] * q_traits.grid.nelems
        k_transposed_distr = [0] * k_transposed_traits.grid.nelems
        k_distr = [0] * k_traits.grid.nelems
        v_transposed_distr = [0] * v_transposed_traits.grid.nelems
        v_distr = [0] * v_traits.grid.nelems
        a_distr = [0] * a_traits.grid.nelems
        a_maxsumexp_distr = [0] * a_maxsumexp_traits.grid.nelems
        a_sumprod_slice_distr = [0] * a_sumprod_slice_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        b_transposed_distr = [0] * b_transposed_traits.grid.nelems
        if bias:
            in_proj_bias_qkv_traits = TensorTraits(
                [head_size, n_head], [head_size_tile, n_head_tile]
            )
            in_proj_bias_qkv_distr = [0] * in_proj_bias_qkv_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_value.next_tag
        w_q_grad = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_grad.next_tag
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if bias:
            in_proj_bias_q_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_q_value.next_tag
            in_proj_bias_q_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_q_grad.next_tag
            bias_inproj_q = TensorMoments(
                in_proj_bias_q_value, in_proj_bias_q_grad, True
            )
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_value.next_tag
        w_k_grad = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_grad.next_tag
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if bias:
            in_proj_bias_k_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_k_value.next_tag
            in_proj_bias_k_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_k_grad.next_tag
            bias_inproj_k = TensorMoments(
                in_proj_bias_k_value, in_proj_bias_k_grad, True
            )
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_value.next_tag
        w_v_grad = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_grad.next_tag
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if bias:
            in_proj_bias_v_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_v_value.next_tag
            in_proj_bias_v_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_v_grad.next_tag
            bias_inproj_v = TensorMoments(
                in_proj_bias_v_value, in_proj_bias_v_grad, True
            )
        else:
            bias_inproj_v = None
        # w
        w_value = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        w_grad = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        w = TensorMoments(w_value, w_grad, True)
        # q_transposed
        q_transposed_value = type(x_q.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_value.next_tag
        q_transposed_grad = type(x_q.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_grad.next_tag
        q_transposed = TensorMoments(q_transposed_value, q_transposed_grad, True)
        # q
        q_value = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_value.next_tag
        q_grad = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_grad.next_tag
        q = TensorMoments(q_value, q_grad, True)
        # k_transposed
        k_transposed_value = type(x_q.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_value.next_tag
        k_transposed_grad = type(x_q.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_grad.next_tag
        k_transposed = TensorMoments(k_transposed_value, k_transposed_grad, True)
        # k
        k_value = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_value.next_tag
        k_grad = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_grad.next_tag
        k = TensorMoments(k_value, k_grad, True)
        # v_transposed
        v_transposed_value = type(x_q.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_value.next_tag
        v_transposed_grad = type(x_q.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_grad.next_tag
        v_transposed = TensorMoments(v_transposed_value, v_transposed_grad, True)
        # v
        v_value = type(x_q.value)(v_traits, v_distr, next_tag)
        next_tag = v_value.next_tag
        v_grad = type(x_q.value)(v_traits, v_distr, next_tag)
        next_tag = v_grad.next_tag
        v = TensorMoments(v_value, v_grad, True)
        # a
        a_value = type(x_q.value)(a_traits, a_distr, next_tag)
        next_tag = a_value.next_tag
        a_grad = type(x_q.value)(a_traits, a_distr, next_tag)
        next_tag = a_grad.next_tag
        a = TensorMoments(a_value, a_grad, True)
        # a_maxsumexp
        a_maxsumexp = type(x_q.value)(a_maxsumexp_traits, a_maxsumexp_distr, next_tag)
        next_tag = a_maxsumexp.next_tag
        # a_sumprod_slice
        a_sumprod_slice = type(x_q.value)(
            a_sumprod_slice_traits, a_sumprod_slice_distr, next_tag
        )
        next_tag = a_sumprod_slice.next_tag
        # b
        b_value = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_value.next_tag
        b_grad = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_grad.next_tag
        b = TensorMoments(b_value, b_grad, True)
        # b_transposed
        b_transposed_value = type(x_q.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_value.next_tag
        b_transposed_grad = type(x_q.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_grad.next_tag
        b_transposed = TensorMoments(b_transposed_value, b_transposed_grad, True)
        # Allocate tensors for bias for q, k, v and output projection
        if bias:
            out_proj_bias_traits = TensorTraits([n_emb], [n_emb_tile])
            out_proj_bias_distr = [0] * out_proj_bias_traits.grid.nelems
            out_proj_bias_value = type(x_q.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_value.next_tag
            out_proj_bias_grad = type(x_q.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_grad.next_tag
            out_proj_bias = TensorMoments(out_proj_bias_value, out_proj_bias_grad, True)
        else:
            out_proj_bias = None
        # Allocate tensor for output y
        y_traits = TensorTraits(x_q.value.shape, x_q.value.basetile_shape)
        y_value = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        # Create attention layer with all the provided data
        layer = T5Attention(
            x_q,
            x_k,
            x_v,
            y,
            w_q,
            w_k,
            w_v,
            w,
            q_transposed,
            q,
            k_transposed,
            k,
            v_transposed,
            v,
            a,
            a_maxsumexp,
            a_sumprod_slice,
            b,
            b_transposed,
            bias_inproj_q,
            bias_inproj_k,
            bias_inproj_v,
            out_proj_bias,
            mask,
            redux=redux,
        )
        # Return layer and next tag to be used
        return (layer, next_tag)

    @classmethod
    def from_torch(
        cls,
        torch_layer: T5AttentionTorch,
        x: TensorMoments,
        mask: np.ndarray,
        config: T5ConfigNNTile,
        next_tag: int,
    ):
        attn, next_tag = T5Attention.generate_simple(
            x_q=x,
            x_k=x,
            x_v=x,
            n_head=config.n_head,
            n_head_tile=config.n_head_tile,
            next_tag=next_tag,
            inner_dim=config.d_ff,
            inner_dim_tile=config.d_ff_tile,
            bias=False,
            mask=mask,
            redux=config.redux,
        )

        attn.w_q.value.from_array(
            torch_layer.q.weight.view((torch_layer.n_heads, -1, config.d_model))
            .cpu()
            .detach()
            .numpy()
        )
        attn.w_k.value.from_array(
            torch_layer.k.weight.view((torch_layer.n_heads, -1, config.d_model))
            .cpu()
            .detach()
            .numpy()
        )
        attn.w_v.value.from_array(
            torch_layer.v.weight.view(torch_layer.n_heads, -1, config.d_model)
            .cpu()
            .detach()
            .numpy()
        )

        attn.w.value.from_array(
            torch_layer.o.weight.view(config.d_model, torch_layer.n_heads, -1)
            .cpu()
            .detach()
            .numpy()
        )

        return attn, next_tag
