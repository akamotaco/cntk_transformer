# http://jalammar.github.io/illustrated-transformer/
# https://nlpinkorean.github.io/illustrated-transformer/
import cntk as C

SEQ_DIMS = 2
OUT_DIMS = IN_DIMS = 4
SA_DIMS = 3

HEAD_DIMS = 8

def self_attention_layer(in_dims:int, seq_dims:int, out_dims:int, sa_dims:int, name='self_attention') -> C.Function:
    sq_sa_dims = C.Constant(C.sqrt(sa_dims).eval(), name='sq_dims')

    init = C.initializer.normal(1)

    X = C.placeholder((seq_dims, in_dims), name=name+'_ph')
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

def multi_headed_self_attention_layer(num_of_head:int, in_dims:int, seq_dims:int, out_dims:int, sa_dims:int, name='multi_headed_self_attention', as_block:bool = False) -> C.Function:
    X = C.placeholder((seq_dims, in_dims), name=name+'_ph')
    layers = []
    for i in range(num_of_head):
        layers.append(self_attention_layer(IN_DIMS, SEQ_DIMS, OUT_DIMS, SA_DIMS, name=name+str(i)))
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

def layer_normalization(inputs:C.Function, name='layer_normalization') -> C.Function:
    X = C.placeholder(inputs.shape, name=name+'_ph')

    mu = C.reduce_mean(X, name='mu')
    sigma = C.sqrt(C.reduce_mean(C.square(X-mu)), name='sigma')

    result = (X-mu)/sigma

    block = C.as_block(result, [(X,X)], name)

    return block(inputs)

def feed_forward_layer(in_dims:int, seq_dims:int, out_dims:int, name='feed_forward', as_block:bool = False) -> C.Function:
    X = C.placeholder((seq_dims, in_dims), name=name+'_ph')

    hidden_size = in_dims*seq_dims

    ff = C.layers.Dense(hidden_size, C.relu, name=name+'_l1')(X)
    ff = C.layers.Dense(hidden_size, name=name+'_l2')(ff)

    result = C.reshape(ff, (seq_dims, in_dims), name=name+'_result')

    if as_block:
        return C.as_block(result, [(X,X)], name)
    else:
        return result

def encoder(in_dims:int, seq_dims:int, out_dims:int, head_dims:int, sa_dims:int, name:str='encoder', as_block=False) -> C.Function:
    X = C.placeholder((seq_dims, in_dims), name=name+'_ph')

    mhsa_layer = multi_headed_self_attention_layer(head_dims, in_dims, seq_dims, out_dims, sa_dims)
    ff_layer = feed_forward_layer(in_dims, seq_dims, out_dims)

    sa = layer_normalization(X + mhsa_layer(X))
    ff = layer_normalization(sa + ff_layer(sa))

    result = ff
    if as_block is True:
        return C.as_block(result, [(X,X)], name)
    else:
        return result

#region positional_encoding: https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return pos_encoding
#endregion






from IPython import embed;embed(header='end')
exit()




import numpy as np

v = np.ones((SEQ_DIMS, IN_DIMS)).astype(np.float32)
X = C.input_variable((SEQ_DIMS, IN_DIMS))
sa_layer = self_attention_layer(IN_DIMS, SEQ_DIMS, OUT_DIMS, SA_DIMS)

model = sa_layer(X)
print(model.eval({model.arguments[0]:v}))

mhsa_layer = multi_headed_self_attention_layer(HEAD_DIMS, IN_DIMS, SEQ_DIMS, OUT_DIMS, SA_DIMS, as_block=True)
model = mhsa_layer(X)
print(model.eval({model.arguments[0]:v}))

ff_layer = feed_forward_layer(IN_DIMS, SEQ_DIMS, OUT_DIMS)


sa = layer_normalization(X + mhsa_layer(X))
ff = layer_normalization(sa + ff_layer(sa))

model = ff
print(model.eval({model.arguments[0]:v}))

en_layer = encoder(IN_DIMS, SEQ_DIMS, OUT_DIMS, HEAD_DIMS, SA_DIMS, as_block=False)

model = en_layer(X)
print(model.eval({model.arguments[0]:v}))

#region training test
answer = C.input_variable((SEQ_DIMS, IN_DIMS))

loss = C.reduce_mean(C.square(model-answer))

trainer = C.Trainer(model, (loss, None), C.adam(model.parameters, 0.001, 0.001))
trainer.train_minibatch(dict(zip(loss.arguments,[v,v])))
#endregion