# http://jalammar.github.io/illustrated-gpt2/
import cntk as C
import numpy as np
from transformer import layer_normalization

def positional_encoding(token_dims:int, discount_factor:float=0.99):
    X = C.placeholder(token_dims, name='positional_encoding')
    encoder = C.layers.Recurrence(C.element_times,initial_state=1, return_full_state=True)(C.ones_like(X)*discount_factor)
    return C.stop_gradient(C.as_block(encoder, [(X,X)], 'positional_encoding','positional_encoding_'))

# def self_attention(token_dims:int, sa_dims:int, mask_opt:bool = False, as_block:bool=False, name:str='self_attention'):
#     X = C.placeholder(token_dims, dynamic_axes=(C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()) ,name=name)

#     q = C.layers.Dense(sa_dims, name=name+'_q')(X)
#     k = C.layers.Dense(sa_dims, name=name+'_k')(X)
#     v = C.layers.Dense(sa_dims, name=name+'_v')(X)

#     q_ = C.sequence.unpack(q, 0, True)
#     k_ = C.sequence.unpack(k, 0, True)
#     v_ = C.sequence.unpack(v, 0, True)

#     scores = C.times_transpose(q_, k_, name=name+'_scores')

#     if mask_opt:
#         mask = triangular_matrix_seq(2)(X)
#         inf_mask = -np.inf*(mask-0.5)
#         scores = C.element_min(scores, inf_mask)
    
#     softmax = C.softmax(scores, name=name+'_softmax')
#     attention = C.times(softmax, v_, name=name+'_attention')

#     result = C.to_sequence_like(attention, X, name=name+'_seq')

#     if as_block:
#         return C.as_block(result, [(X,X)], name,name)
    
#     return result

def gpt2_self_attention(token_dims:int, head_dims:int, mask_opt:bool = False, as_block:bool=False, name:str='self_attention'):
    X = C.placeholder(token_dims, dynamic_axes=(C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()) ,name=name)

    # q = C.layers.Dense(token_dims, name=name+'_q')(X)
    # k = C.layers.Dense(token_dims, name=name+'_k')(X)
    # v = C.layers.Dense(token_dims, name=name+'_v')(X)

    # attn_c_attn_w = C.parameter((token_dims,3*token_dims), name='attn_c_attn_w')
    # qkv = C.reshape(X@attn_c_attn_w, (3,-1), name='qkv')

    qkv = C.layers.Dense((3,token_dims), name='qkv')(X)
    q_seq, k_seq, v_seq = qkv[0], qkv[1], qkv[2]

    q_mh = C.reshape(q_seq, (head_dims,-1), name='multi_head_q')
    k_mh = C.reshape(k_seq, (head_dims,-1), name='multi_head_k')
    v_mh = C.reshape(v_seq, (head_dims,-1), name='multi_head_v')

#region split multi head attention
    q_heads = [C.squeeze(q_mh[i],name='simgle_head_q'+str(i)) for i in range(head_dims)]
    k_heads = [C.squeeze(k_mh[i],name='simgle_head_q'+str(i)) for i in range(head_dims)]
    v_heads = [C.squeeze(v_mh[i],name='simgle_head_q'+str(i)) for i in range(head_dims)]
#endregion

    attention_head = []
    for i in range(head_dims):
        q = q_heads[i]
        k = k_heads[i]
        v = v_heads[i]

#region score
        q_ = C.sequence.last(q, name='last_q'+str(i)) # q present
        k_ = C.sequence.unpack(k, 0, True, name='seq_k'+str(i)) # k seq
        v_ = C.sequence.unpack(v, 0, True, name='seq_v'+str(i)) # v seq

        scores = C.times_transpose(q_, k_)
        softmax = C.softmax(scores)
#endregion
#region sum
        attention = C.times(softmax, v_)
#endregion
        attention_head.append(attention)

#region merge attention heads
    attention = C.splice(*attention_head, name='merged_attention')
#endergion

#region project
    project = C.layers.Dense(token_dims, name='project')(attention)
#endregion

    if as_block:
        return C.as_block(project, [(X,X)], 'gpt2_self_attention', 'gpt2_self_attention')

    return project

def gpt2_block(token_dims:int, head_dims:int, name:str='gpt2_block'):
    X = C.placeholder(token_dims, dynamic_axes=(C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()) ,name=name)

    sa_layer = gpt2_self_attention(token_dims, head_dims)
    attention = sa_layer(layer_normalization(X))

    from IPython import embed;embed()
    exit()

if __name__ == '__main__':
    VOCAB_DIMS = 100
    TOKENS_DIMS = 4
    SA_DIMS = 3

    Q = C.sequence.input_variable(TOKENS_DIMS, name='input')
    PE = positional_encoding(TOKENS_DIMS)(Q)

    m = Q+PE

    print(m.eval({Q:np.arange(TOKENS_DIMS*10).reshape(1,-1,TOKENS_DIMS).astype(np.float32)}))


    VOCAB_DIMS = 50257
    TOKENS_DIMS = 768
    SEQ_DIMS = 1024
    # SA_DIMS = 3
    HEAD_DIMS = 12

    Q = C.sequence.input_variable(VOCAB_DIMS, name='vocab_input')
    E = C.layers.Embedding(TOKENS_DIMS)(Q)
    initial_pe = np.ones((TOKENS_DIMS, SEQ_DIMS))
    PE = C.parameter((TOKENS_DIMS, SEQ_DIMS), init=initial_pe ,name='positional_encoding')

    # sa_layer = gpt2_self_attention(TOKENS_DIMS, HEAD_DIMS)
    block_layer = gpt2_block(TOKENS_DIMS, HEAD_DIMS)

    from IPython import embed;embed()