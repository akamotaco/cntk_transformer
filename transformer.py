# http://jalammar.github.io/illustrated-transformer/
# https://nlpinkorean.github.io/illustrated-transformer/
import cntk as C

def triangular_matrix(mode:int = 1):
    X = C.placeholder(1)
    ones = C.ones_like(X[0])
    perm_1 = C.layers.Recurrence(C.plus, return_full_state=True)(ones)
    perm_2 = C.layers.Recurrence(C.plus, go_backwards=True, return_full_state=True)(ones)

    arr_1 = C.sequence.unpack(perm_1,0,True)
    arr_2 = C.sequence.unpack(perm_2,0,True)

    mat = C.times_transpose(arr_1, arr_2)
    mat_c = perm_1*perm_2

    mat_seq = C.to_sequence_like(mat, mat_c)
    diagonal_seq = mat_seq - mat_c

    diagonal_mat = C.sequence.unpack(diagonal_seq,0,True)

    final_mat = diagonal_mat
    if mode == 0:
        final_mat = C.equal(final_mat, 0)
    elif mode == 1:
        final_mat = C.less_equal(final_mat, 0)
    elif mode == 2:
        final_mat = C.less(final_mat, 0)
    elif mode == -1:
        final_mat = C.greater_equal(final_mat, 0)
    elif mode == -2:
        final_mat = C.greater(final_mat, 0)

    result = C.as_block(final_mat, [(X,X)], 'triangular_matrix')

    return result

def self_attention_layer(in_dims:int, out_dims:int, name='self_attention', as_block:bool = False, k_ph:bool=False, v_ph:bool=False, mask_opt:bool=False) -> C.Function:
    sq_sa_dims = C.Constant(C.sqrt(out_dims).eval(), name='sq_dims')

    X = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()), name=name+'_ph')

    if k_ph is False and v_ph is False:
        q = C.layers.Dense(out_dims, name=name+'_q')(X) # W_Q = C.parameter((in_dims, out_dims), init=init, name=name+'_q')
        k = C.layers.Dense(out_dims, name=name+'_k')(X) # W_K = C.parameter((in_dims, out_dims), init=init, name=name+'_k')
        v = C.layers.Dense(out_dims, name=name+'_v')(X) # W_V = C.parameter((in_dims, out_dims), init=init, name=name+'_v')
    elif k_ph is True and v_ph is True:
        q = C.layers.Dense(out_dims, name=name+'_q')(X)
        k = C.placeholder(out_dims, (C.Axis.default_batch_axis(), C.Axis('kv_seq')), name=name+'_k_ph')
        v = C.placeholder(out_dims, (C.Axis.default_batch_axis(), C.Axis('kv_seq')), name=name+'_v_ph')
    else:
        raise Exception(f'k_ph:{k_ph}, v_ph:{v_ph}')

    q_ = C.sequence.unpack(q, 0, True, name=name+'_unpack_q')
    k_ = C.sequence.unpack(k, 0, True, name=name+'_unpack_k')
    v_ = C.sequence.unpack(v, 0, True, name=name+'_unpack_v')
    
    scores = C.times_transpose(q_, k_, name=name+'_score_matrix')
    scaled = scores/sq_sa_dims # div_k

    if mask_opt:
        mask = triangular_matrix(2)(X)
        scaled = mask * scaled

    softmax = C.softmax(scaled, name=name+'_softmax')
    softmax_value = C.times(softmax, v_, name=name+'_softmax_value')

    result = C.to_sequence_like(softmax_value, X)
    
    if as_block:
        if k_ph is False and v_ph is False:
            return C.as_block(result, [(X,X)], 'self_attention', 'self_attention_')
        elif k_ph is True and v_ph is True:
            return C.as_block(result, [(X,X), (k,k), (v,v)], 'self_attention', 'self_attention_')
        else:
            raise Exception(f'k_ph:{k_ph} v_ph:{v_ph}')
    else:
        return result

def multi_headed_self_attention_layer(in_dims:int, hidden_dims:int, num_of_head:int, name='multi_headed_self_attention', as_block:bool = False, k_ph:bool=False, v_ph:bool=False, mask_opt:bool=False) -> C.Function:
    X = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()), name=name+'_ph')

    outputs = []

    if k_ph is False and v_ph is False:
        for i in range(num_of_head):
            layer = self_attention_layer(in_dims, hidden_dims, name=name+str(i), as_block=not as_block, mask_opt=mask_opt)
            outputs.append(layer(X))
    elif k_ph is True and v_ph is True:
        k_ = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis('kv_seq')), name=name+'_k_ph') # -3: sequence axis
        v_ = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis('kv_seq')), name=name+'_v_ph')
        for i in range(num_of_head):
            layer = self_attention_layer(in_dims, in_dims, name=name+str(i), as_block=not as_block, k_ph=k_ph, v_ph=v_ph)
            outputs.append(layer(X,k_,v_))
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
    X = C.placeholder(inputs.shape, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()), name=name+'_ph')

    mu = C.reduce_mean(X, name='mu')
    sigma = C.sqrt(C.reduce_mean(C.square(X-mu)), name='sigma')

    result = (X-mu)/sigma

    block = C.as_block(result, [(X,X)], name)

    return block(inputs)

def feed_forward_layer(in_dims:int, hidden_dims:int, name='feed_forward', as_block:bool = False) -> C.Function:
    X = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()), name=name+'_ph')

    ff = C.layers.Dense(hidden_dims, C.relu, name=name+'_l1')(X)
    ff = C.layers.Dense(in_dims, name=name+'_l2')(ff)

    # result = C.reshape(ff, in_dims, name=name+'_result')
    result = ff

    if as_block:
        return C.as_block(result, [(X,X)], name)
    else:
        return result

def encoder(in_dims:int, sa_dims:int, head_dims:int, hidden_dims:int, name:str='encoder', as_block=False) -> C.Function:
    X = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()), name=name+'_ph')

    mhsa_layer = multi_headed_self_attention_layer(in_dims, sa_dims, head_dims)
    ff_layer = feed_forward_layer(in_dims, hidden_dims)

    sa = layer_normalization(X + mhsa_layer(X))
    ff = layer_normalization(sa + ff_layer(sa))

    result = ff
    if as_block is True:
        return C.as_block(result, [(X,X)], name)
    else:
        return result

def decoder(in_dims:int, sa_dims:int, head_dims:int, hidden_dims:int, kv_memory, name:str='decoder', as_block:bool = False) -> C.Function:
    X = C.placeholder(in_dims, (C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()),name=name+'_ph')
    k_memory = C.placeholder(in_dims, (C.Axis.default_batch_axis(),C.Axis('kv_seq')), name=name+'_k_memory')
    v_memory = C.placeholder(in_dims, (C.Axis.default_batch_axis(),C.Axis('kv_seq')), name=name+'_v_memory')
    # placeholder 는 clone이 안되서 k, v를 kv로 하나의 placeholder로서 묶으면 안됨

    mhsa_layer = multi_headed_self_attention_layer(in_dims, sa_dims, head_dims, mask_opt=True)
    eda_layer = multi_headed_self_attention_layer(in_dims, sa_dims, head_dims, k_ph=True, v_ph=True)
    ff_layer = feed_forward_layer(in_dims, hidden_dims)

    sa = layer_normalization(X + mhsa_layer(X)) # w/o mask
    eda = layer_normalization(sa + eda_layer(sa, k_memory, v_memory))
    ff = layer_normalization(eda + ff_layer(eda))

    result = ff
    if as_block is True:
        return C.as_block(result, [(X,X), (k_memory,k_memory), (v_memory,v_memory)], name)
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
    TOKEN_DIMS = 4 # size of tokens (# of embedding)
    SA_DIMS = 3 # size of self attention
    HEAD_DIMS = 8 # size of multi-headed self attention
    HIDDEN_DIMS = 24

    import numpy as np

    v = np.array([ [1,0,0,0], [1,1,1,1], [0,1,0,0] ], np.float32) # seq
    X = C.sequence.input_variable(TOKEN_DIMS, name='encoder_input', sequence_axis=C.Axis('encoder_seq'))
    sa_layer = self_attention_layer(TOKEN_DIMS, SA_DIMS)

    model = sa_layer(X)
    print(model.eval({model.arguments[0]:v}))

    mhsa_layer = multi_headed_self_attention_layer(TOKEN_DIMS, SA_DIMS, HEAD_DIMS)
    model = mhsa_layer(X)
    print(model.eval({model.arguments[0]:v}))


    ff_layer = feed_forward_layer(TOKEN_DIMS, HIDDEN_DIMS)


    sa = layer_normalization(X + mhsa_layer(X))
    ff = layer_normalization(sa + ff_layer(sa))

    model = ff
    print(model.eval({model.arguments[0]:v}))

    en_layer = encoder(TOKEN_DIMS, SA_DIMS, HEAD_DIMS, HIDDEN_DIMS, as_block=False)

    model = en_layer(X)
    print(model.eval({model.arguments[0]:v}))

#################################################

#region seq test
    q1 = C.sequence.input_variable(1)
    q2 = C.sequence.input_variable(1, sequence_axis=C.Axis('a1'))

    r1 = q1*2

    a1 = np.array(range(5),np.float32).reshape(1,-1,1)
    a2 = np.array(range(3),np.float32).reshape(1,-1,1)

    u1 = C.sequence.unpack(r1,0,True)
    u2 = C.sequence.unpack(q2,0,True)

    print('===========================================')
    print(C.sequence.last(r1).eval({q1:a1}))
    print(u1.eval({q1:a1}))
    print(u2.eval({q2:a2}))
    print(C.times_transpose(u1,u2).eval({q1:a1, q2:a2}))
#endregion

    encoder = model

    # y = np.array(range(15),np.float32).reshape(3,OUT_DIMS)
    input_size = 6 #v.shape[0]
    y = np.array(range(TOKEN_DIMS*input_size),np.float32).reshape(input_size,TOKEN_DIMS)
    Y = C.sequence.input_variable(TOKEN_DIMS, name='decoder_input', sequence_axis=C.Axis('decoder_seq')) # encoder 차원과 decoer의 차원의 개수는 항상 일치해야 하는가?

    edal_layer = self_attention_layer(TOKEN_DIMS, TOKEN_DIMS, k_ph=True, v_ph=True, as_block=True)
    m = edal_layer(Y, encoder.output, encoder.output)
    print(m.eval({X:v.reshape(1,3,4), Y:y.reshape(1,input_size,4)}))

    msedal_layer = multi_headed_self_attention_layer(TOKEN_DIMS, SA_DIMS, HEAD_DIMS, k_ph=True, v_ph=True, as_block=True)
    m = msedal_layer(Y, encoder.output, encoder.output)
    print(m.eval({X:v, Y:y}))




    de_layer = decoder(TOKEN_DIMS, SA_DIMS, HEAD_DIMS, HIDDEN_DIMS, encoder, as_block=False)
    model = de_layer(Y, encoder, encoder)
    print(model.eval({X:v, Y:y}))
    decoder = model


    #region training test
    # a = np.array([ [1,0,1], [0,0,0], [1,1,1]], np.float32) # for self_attention
    # answer_for_test = C.sequence.input_variable(OUT_DIMS) # for self_attentio

    answer_for_test = C.sequence.input_variable(TOKEN_DIMS, sequence_axis=C.Axis('encoder_seq')) # else

    loss = C.reduce_mean(C.square(encoder-answer_for_test))

    trainer = C.Trainer(encoder, (loss, None), C.adam(encoder.parameters, 0.001, 0.001))
    print(trainer.train_minibatch(dict(zip(loss.arguments,[v,v]))))
    #endregion


    answer_for_test = C.sequence.input_variable(TOKEN_DIMS, sequence_axis=C.Axis('decoder_seq'))

    loss = C.reduce_mean(C.square(decoder-answer_for_test))

    trainer = C.Trainer(decoder, (loss, None), C.adam(decoder.parameters, 0.001, 0.001))
    print(trainer.train_minibatch(dict(zip(loss.arguments,[y,v,y]))))

#region seq-mask test
    q1 = C.sequence.input_variable(5)
    a1 = np.array(range(4*5),np.float32).reshape(1,-1,5)
    r1 = q1*2
    print('================================')
    print(r1.eval({q1:a1}))
    print(C.sequence.is_first(r1).eval({q1:a1}))
    print(C.layers.Fold(C.plus)(r1).eval({q1:a1}))
#endregion

#region suq-mask test2
    x = [[[0],[0],[0]],
         [[0],[0],[0],[0],[0]]]
    @C.Function
    def double_up(s):
        return s*2
    r1 = C.layers.UnfoldFrom(double_up)(C.Constant(1), C.sequence.input_variable(1))
    r2 = C.layers.UnfoldFrom(lambda x: x*2)(C.Constant(1), C.sequence.input_variable(1))
    print(r1.eval({r1.arguments[0]:x}))
    print(r2.eval({r2.arguments[0]:x}))
#endregion

    from IPython import embed;embed(header='end')




# z = np.arange(5*5).reshape(5,5)

# q = C.sequence.input_variable(5)
# w = C.ones_like(q[0])

# e = C.layers.Recurrence(C.plus, return_full_state=True)(w)
# e_ = C.layers.Recurrence(C.plus, go_backwards=True, return_full_state=True)(w)

# r = C.sequence.unpack(e,0,True)
# r_ = C.sequence.unpack(e_,0,True)

# R = C.times_transpose(r,r_)
# R_c = e*e_

# R_seq = C.to_sequence_like(R,R_c)
# R_mat = R_seq - R_c

# M = C.reduce_max(R)

# mat = (R-1)/(M-1)


# mat3 = C.to_sequence(mat)

# mat2 = C.greater_equal(mat, C.Constant(1.0/3+1e-5))


# m = w
# m.eval({q:z})