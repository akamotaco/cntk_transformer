# http://jalammar.github.io/illustrated-transformer/
# https://nlpinkorean.github.io/illustrated-transformer/
import cntk as C

def self_attention_layer(in_dims:int, out_dims:int, name='self_attention', as_block:bool = False, k_ph:bool=False, v_ph:bool=False) -> C.Function:
    sq_sa_dims = C.Constant(C.sqrt(out_dims).eval(), name='sq_dims')

    # init = C.initializer.normal(1)

    X = C.placeholder(in_dims, name=name+'_ph')

    if k_ph is False and v_ph is False:
        q = C.layers.Dense(out_dims, name=name+'_q')(X) # W_Q = C.parameter((in_dims, out_dims), init=init, name=name+'_q')
        k = C.layers.Dense(out_dims, name=name+'_k')(X) # W_K = C.parameter((in_dims, out_dims), init=init, name=name+'_k')
        v = C.layers.Dense(out_dims, name=name+'_v')(X) # W_V = C.parameter((in_dims, out_dims), init=init, name=name+'_v')
    elif k_ph is True and v_ph is True: # ???? test
        k_ = C.placeholder((in_dims, -3), name=name+'_k_ph')
        v_ = C.placeholder((in_dims, -3), name=name+'_v_ph')
        q = X@C.ones_like(k_) # ??????? X 와 Q, KV를 맞추는 방법은?
        k = X@k_
        v = X@v_
    else:
        raise Exception(f'k_ph:{k_ph}, v_ph:{v_ph}')

    q_ = C.sequence.unpack(q, 0, True, name=name+'_unpack_q')
    k_ = C.sequence.unpack(k, 0, True, name=name+'_unpack_k')
    v_ = C.sequence.unpack(v, 0, True, name=name+'_unpack_v')
    
    scores = C.times_transpose(q_, k_, name=name+'_score_matrix')
    div_k = scores/sq_sa_dims
    softmax = C.softmax(div_k, name=name+'_softmax')
    softmax_value = C.times(softmax, v_, name=name+'_softmax_value')

    result = C.to_sequence_like(softmax_value, X)
    
    if as_block:
        if k_ph is False and v_ph is False:
            return C.as_block(result, [(X,X)], 'self_attention', 'self_attention_')
        elif k_ph is True and v_ph is True:
            return C.as_block(result, [(X,X), (k_,k_), (v_,v_)], 'self_attention', 'self_attention_')
        else:
            raise Exception(f'k_ph:{k_ph} v_ph:{v_ph}')
    else:
        return result

def multi_headed_self_attention_layer(in_dims:int, hidden_dims:int, num_of_head:int, name='multi_headed_self_attention', as_block:bool = False, k_ph:bool=False, v_ph:bool=False) -> C.Function:
    X = C.placeholder(in_dims, name=name+'_ph')

    layers = []
    outputs = []

    if k_ph is False and v_ph is False:
        for i in range(num_of_head):
            layers.append(self_attention_layer(in_dims, hidden_dims, name=name+str(i), as_block=True))
        for layer in layers:
            outputs.append(layer(X))
    elif k_ph is True and v_ph is True:
        k_ = C.placeholder((-3, in_dims), name=name+'_k_ph') # -3: sequence axis
        v_ = C.placeholder((-3, in_dims), name=name+'_v_ph')
        for i in range(num_of_head):
            layers.append(self_attention_layer(in_dims, in_dims, name=name+str(i), as_block=True, k_ph=k_ph, v_ph=v_ph))
        for layer in layers:
            outputs.append(layer(X, k_, v_))
    else:
        raise Exception(f'k_ph:{k_ph}, v_ph:{v_ph}')
    
    concat = C.splice(*outputs, name='concat')

    result = C.layers.Dense(in_dims, name='W_o')(concat)

    # init = C.initializer.normal(1)
    # W_O = C.parameter((in_dims, hidden_dims*num_of_head), init=init, name=name+'_Wo')
    # result = C.times_transpose(concat, W_O, name='result')

    if as_block is True:
        if k_ph is False and v_ph is False:
            result = C.as_block(result, [(X,X)], 'multi_headed_self_attetion','multi_headed_self_attetion_')
        elif k_ph is True and v_ph is True:
            result = C.as_block(result, [(X,X), (k_,k_), (v_,v_)], 'multi_headed_self_attetion','multi_headed_self_attetion_')
        else:
            raise Exception(f'k_ph:{k_ph} v_ph:{v_ph}')

    return result

def layer_normalization(inputs:C.Function, name='layer_normalization') -> C.Function:
    X = C.placeholder(inputs.shape, name=name+'_ph')

    mu = C.reduce_mean(X, name='mu')
    sigma = C.sqrt(C.reduce_mean(C.square(X-mu)), name='sigma')

    result = (X-mu)/sigma

    block = C.as_block(result, [(X,X)], name)

    return block(inputs)

def feed_forward_layer(in_dims:int, hidden_dims:int, name='feed_forward', as_block:bool = False) -> C.Function:
    X = C.placeholder(in_dims, name=name+'_ph')

    ff = C.layers.Dense(hidden_dims, C.relu, name=name+'_l1')(X)
    ff = C.layers.Dense(in_dims, name=name+'_l2')(ff)

    # result = C.reshape(ff, in_dims, name=name+'_result')
    result = ff

    if as_block:
        return C.as_block(result, [(X,X)], name)
    else:
        return result

def encoder(in_dims:int, sa_dims:int, head_dims:int, hidden_dims:int, name:str='encoder', as_block=False) -> C.Function:
    X = C.placeholder(in_dims, name=name+'_ph')

    mhsa_layer = multi_headed_self_attention_layer(in_dims, sa_dims, head_dims)
    ff_layer = feed_forward_layer(in_dims, hidden_dims)

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




if __name__ == '__main__':
    IN_DIMS = 4 # size of tokens
    SA_DIMS = 3 # size of self attention
    HEAD_DIMS = 8 # size of multi-headed self attention
    HIDDEN_DIMS = 24


    import numpy as np

    v = np.array([ [1,0,0,0], [1,1,1,1], [0,1,0,0] ], np.float32) # seq
    X = C.sequence.input_variable(IN_DIMS, name='encoder_input')
    sa_layer = self_attention_layer(IN_DIMS, SA_DIMS)

    model = sa_layer(X)
    print(model.eval({model.arguments[0]:v}))

    mhsa_layer = multi_headed_self_attention_layer(IN_DIMS, SA_DIMS, HEAD_DIMS, as_block=False)
    model = mhsa_layer(X)
    print(model.eval({model.arguments[0]:v}))


    ff_layer = feed_forward_layer(IN_DIMS, HIDDEN_DIMS)


    sa = layer_normalization(X + mhsa_layer(X))
    ff = layer_normalization(sa + ff_layer(sa))

    model = ff
    print(model.eval({model.arguments[0]:v}))

    en_layer = encoder(IN_DIMS, SA_DIMS, HEAD_DIMS, HIDDEN_DIMS, as_block=False)

    model = en_layer(X)
    print(model.eval({model.arguments[0]:v}))

#################################################

    encoder = model

    # # OUT_DIMS = 5

    # # y = np.array(range(15),np.float32).reshape(-1,OUT_DIMS)
    # Y = C.input_variable(IN_DIMS, name='decoder_input') # encoder 차원과 decoer의 차원의 개수는 항상 일치해야 하는가?

    # # edal_layer = multi_headed_self_attention_layer(IN_DIMS, IN_DIMS, HEAD_DIMS, as_block=False, k_ph=True, v_ph=True)
    # edal_layer = self_attention_layer(IN_DIMS, IN_DIMS, k_ph=True, v_ph=True)
    # kv_memory = C.transpose(C.sequence.unpack(encoder.output, 0, True), (1,0))
    # m = edal_layer(Y, kv_memory, kv_memory)
    # print(m.eval({X:v.reshape(1,3,4), Y:v[0].reshape(1,4)}))



    #region training test
    # a = np.array([ [1,0,1], [0,0,0], [1,1,1]], np.float32) # for self_attention
    # answer_for_test = C.sequence.input_variable(OUT_DIMS) # for self_attentio

    answer_for_test = C.sequence.input_variable(IN_DIMS) # else

    loss = C.reduce_mean(C.square(model-answer_for_test))

    trainer = C.Trainer(model, (loss, None), C.adam(model.parameters, 0.001, 0.001))
    # trainer.train_minibatch(dict(zip(loss.arguments,[v,a]))) # for self_attention
    print(trainer.train_minibatch(dict(zip(loss.arguments,[v,v]))))
    #endregion

    from IPython import embed;embed(header='end')