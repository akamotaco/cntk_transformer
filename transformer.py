import cntk as C

CH_DIMS = 2
OUT_DIMS = IN_DIMS = 4
SA_DIMS = 3

HEAD_DIMS = 8

def self_attention(in_dims:int, in_ch:int, out_dims:int, sa_dims:int, name='self_attention'):
    sq_sa_dims = C.Constant(C.sqrt(sa_dims).eval(), name='sq_dims')

    init = C.initializer.normal(1)

    X = C.placeholder((in_ch, in_dims), name=name+'_ph')
    W_Q = C.parameter((IN_DIMS, SA_DIMS), init=init, name=name+'_q')
    W_K = C.parameter((IN_DIMS, SA_DIMS), init=init, name=name+'_k')
    W_V = C.parameter((IN_DIMS, SA_DIMS), init=init, name=name+'_v')

    q = X@W_Q
    k = X@W_K
    v = X@W_V

    # score = C.reduce_sum(q*k) # dot
    # result = C.softmax(score/SA_DIMS_SQ)

    score = C.times_transpose(q, k, name=name+'_score')
    result = C.times(C.softmax(score/sq_sa_dims), v, name=name+'_sum')

    return C.as_block(result, [(X,X)], 'self_attention', 'self_attention_')

def multi_headed_self_attention(num_of_head:int, in_dims:int, in_ch:int, out_dims:int, sa_dims:int, name='multi_headed_self_attention', as_block:bool = False):
    X = C.placeholder((in_ch, in_dims), name=name+'_ph')
    layers = []
    for i in range(num_of_head):
        layers.append(self_attention(IN_DIMS, CH_DIMS, OUT_DIMS, SA_DIMS, name=name+str(i)))
    outputs = []
    for layer in layers:
        outputs.append(layer(X))
    
    concat = C.splice(*outputs, name='concat')

    init = C.initializer.normal(1)

    W_O = C.parameter((IN_DIMS, concat.shape[1]), init=init, name=name+'_Wo')

    result = C.times_transpose(concat, W_O, name='result')

    if as_block is True:
        result = C.as_block(result, [(X,X)], 'multi_headed_self_attetion','multi_headed_self_attetion_')

    return result

from IPython import embed;embed()
exit()

import numpy as np

v = np.ones((CH_DIMS, IN_DIMS))
X = C.input_variable((CH_DIMS, IN_DIMS))
sa_layer = self_attention(IN_DIMS, CH_DIMS, OUT_DIMS, SA_DIMS)

model = sa_layer(X)
print(model.eval({model.arguments[0]:v}))

mhsa_layer = multi_headed_self_attention(HEAD_DIMS, IN_DIMS, CH_DIMS, OUT_DIMS, SA_DIMS, as_block=True)
model = mhsa_layer(X)
print(model.eval({model.arguments[0]:v}))