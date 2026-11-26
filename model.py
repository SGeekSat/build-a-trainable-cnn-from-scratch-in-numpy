"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np

def argmax_rows(matrix):
    matrix = np.array(matrix)
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    matrix = np.array(matrix)
    return np.max(matrix,axis=1,keepdims=True)
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    matrix = np.array(matrix)
    return np.sum(matrix, axis=1, keepdims=True)
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    logits = np.array(logits)
    return np.exp(logits-row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    logits = np.array(logits)
    exp_value = exp_shifted(logits)
    return exp_value/row_sum(exp_value)# TODO: Compute a numerically stable softmax row-wise over (N, C) logits.

# Step 6 - one_hot
def one_hot(labels, num_classes):
    lables = np.array(labels)
    return np.eye(num_classes, dtype=float)[labels] # TODO: convert integer labels into a (N, num_classes) one-hot float matrix

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    probs = np.array(probs)
    labels = np.array(labels)
    
    return probs[np.arange(len(labels)),labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    probs = np.array(probs)
    labels = np.array(labels)
    true_probs = gather_true_class_probs(probs,labels)
    true_probs = np.clip(true_probs,eps,1.0)
    return -np.mean(np.log(true_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    predicted = argmax_rows(logits_or_probs)
    

    return np.mean(predicted==labels)

# Step 10 - he_std
import numpy as np
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return float(np.sqrt(2.0/fan_in))

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std = he_std(fan_in)
    weights = np.random.normal(0, std, shape)
    return weights.astype(np.float64)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    bvector = np.zeros(length)
    return bvector.astype(np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    if pad ==0:
        return images
    return np.pad(images,
    ((0,0),(0,0),(pad, pad),(pad, pad)), 
    mode ='constant')

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return (input_size - kernel + 2 * padding) // stride + 1

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, padding):

    images = np.array(images)

    N, C, H, W = images.shape

    # Apply zero padding
    imagesPadded = pad_2d(images, padding)

    # Calculate output height and width
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    # Create output matrix
    cols = np.zeros(
    (N * out_h * out_w, C * kernel_h * kernel_w),
    dtype=images.dtype
    )

    row = 0

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):

                patch = imagesPadded[
                    n,
                    :,
                    i * stride : i * stride + kernel_h,
                    j * stride : j * stride + kernel_w
                ]

                # Flatten patch and store as one row
                cols[row] = patch.reshape(-1)

                row += 1

    return cols

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N,C,H,W = input_shape
    #output dimensions
    out_h = output_spatial_size(H,kernel_h,stride,padding)
    out_w = output_spatial_size(W,kernel_w,stride,padding)
    #temperory padded image
    result = np.zeros(
        (N, C, H+2*padding, W+2*padding),
        dtype = cols.dtype
    )
    row = 0
    # Reconstructing the patches
    for n in range (N):
        for i in range(out_h):
            for j in range(out_w):

                #converting the current row back into patch
                patch = cols[row].reshape(C,kernel_h,kernel_w)
                #adding the patch into its corresponding image region
                result[
                    n,
                    :,
                    i*stride: i*stride+kernel_h,
                    j*stride:j*stride+kernel_w
                ]+=patch
                row +=1
    if padding >0:
        result = result[
            :,
            :,
            padding:padding+H,
            padding:padding+W
        ]
    return result

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    N,C_in,H,W = x.shape
    C_out,Cin,kernel_h,kernel_w = weights.shape

    out_h = output_spatial_size(H,kernel_h,stride,padding)
    out_w = output_spatial_size(W,kernel_w,stride,padding)
    #converting image patches into columns 
    cols = im2col(x, kernel_h, kernel_w,stride,padding)

    #flattening the filters
    weights_reshaped = weights.reshape(C_out,-1)
    #multiplying patches with filters
    output = cols @ weights_reshaped.T
    #adding bias
    output+=bias
    #reshaping output
    output = output.reshape(N,out_h,out_w,C_out)
    #transpose
    output = output.transpose(0,3,1,2)
    #Values stored for backward pass
    cache={
        "x_shape":x.shape,
        "weights":weights,
        "cols":cols,
        "stride":stride,
        "padding":padding,
        "kernel_h":kernel_h,
        "kernel_w":kernel_w
    }
    return output,cache

# Step 18 - conv2d_grad_input
import numpy as np

def conv2d_grad_input(d_out, cache):
    weights = cache["weights"]
    x_shape = cache["x_shape"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    N, C_out, out_h, out_w = d_out.shape
    C_out, C_in, kernel_h, kernel_w = weights.shape

    # Reshape weights
    weights_reshaped = weights.reshape(C_out, -1)

    # Convert d_out from N, C_out, H, W to
    # N, H, W, C_out
    d_out_reshaped = d_out.transpose(0, 2, 3, 1)

    # Flatten spatial positions
    d_out_cols = d_out_reshaped.reshape(-1, C_out)

    # Calculate gradient for each input patch
    d_cols = d_out_cols @ weights_reshaped

    # Fold patch gradients back into input layout
    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    cols = cache["cols"]
    weights = cache ["weights"]

    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out,C_in, _,_ = weights.shape
    #reshaped d_out to match the im2col row ordering
    d_out_reshaped = d_out.transpose(0,2,3,1)

    d_out_cols = d_out_reshaped.reshape(-1,C_out)
    d_weights = d_out_cols.T @ cols 

    d_weights = d_weights.reshape(
        C_out, C_in, kernel_h, kernel_w

    )
    return d_weights

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    db = np.sum(d_out,axis= (0,2,3))
    return db

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx= conv2d_grad_input(d_out,cache)
    dW = conv2d_grad_weights(d_out,cache)
    db = conv2d_grad_bias(d_out)
    return dx,dW,db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,H,W = x.shape
    out_h = output_spatial_size(H,kernel,stride,0)
    out_w = output_spatial_size(W,kernel,stride,0)

    out = np.zeros((N,C,out_h,out_w))
    argmax = np.zeros((N,C,out_h,out_w),dtype = int)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i*stride
                    w_start = j*stride

                    window = x[
                        n,c,
                        h_start:h_start+kernel,
                        w_start:w_start+kernel

                    ]
                    flat_window = window.flatten()
                    argmax[n,c,i,j]=np.argmax(flat_window)
                    out[n,c,i,j]=np.max(flat_window)

    cache = {
        'x_shape':x.shape,
        'argmax':argmax,
        'kernel':kernel,
        'stride':stride
            }
    return out,cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    grad_window = np.zeros((kernel,kernel))

    row = argmax_index//kernel
    col = argmax_index%kernel

    grad_window[row,col]=grad_value
    return grad_window

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N,C,H,W = x_shape
    dx = np.zeros(x_shape)

    out_h=d_out.shape[2]
    out_w=d_out.shape[3]
    
    for n in range(N):
        for c in range(C):
            for i in range (out_h):
                for j in range(out_w):
                    h_start = i*stride
                    w_start = j*stride

                    grad_value = d_out[n,c,i,j]
                    argmax_index = argmax[n,c,i,j]

                    grad_window = scatter_grad_window(
                        grad_value, argmax_index, kernel
                    )
                    dx[
                        n,c,
                        h_start:h_start+kernel,
                        w_start:w_start+kernel
                    ]+=grad_window
        
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out = np.maximum(0,x)
    cache={
        'x':x
    }
    return out,cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
    dx = d_out*(x>0)
    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    x = np.asarray(x)
    cache = {'x_shape':x.shape}
    out = x.reshape(x.shape[0],-1)
    return out,cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    x_shape = cache['x_shape']
    dx = d_out.reshape(x_shape)
    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    out = x @ weights + bias
    cache = {
        'x':x,
        'weights':weights
    }
    return out,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights = cache['weights']
    dx = d_out@weights.T
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.

    dW = x.T@dout
    return dW

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    db = np.sum(dout,axis=0)
    return db

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    dx = linear_grad_input(dout,cache)
    dW = linear_grad_weights(x,dout)
    db = linear_grad_bias(dout)
    return dx,dW,db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    prob = stable_softmax(logits)
    return np.mean(cross_entropy_loss(prob,y))

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N,C = logits.shape
    probs = stable_softmax(logits)

    targets = one_hot(y,C)
    dlogits = (probs-targets)/N
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    param =param-lr*grad
    return param

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    M = beta_one*(m)+(1-beta_one)*grad
    return M

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    V = beta_two*v + (1-beta_two)*(grad**2)
    return V

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    BM = moment/(1-(beta**t))
    return BM

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    param = np.array(param, dtype = float)
    m_hat = np.array(m_hat, dtype = float)
    v_hat = np.array(v_hat, dtype = float)
    # TODO: apply one Adam parameter update using bias-corrected moments
    P = param - lr*(m_hat/(np.sqrt(v_hat)+eps))
    return P

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    m = adam_update_m(m,grad,beta_one)
    v = adam_update_v(v,grad,beta_two)
    m_hat = adam_bias_correct(m,beta_one,t)
    v_hat = adam_bias_correct(v,beta_two,t)
    param=adam_param_step(param,m_hat,v_hat,lr,eps)
    return param,m,v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    fan_in = in_channels*kernel_size*kernel_size
    W = he_init((out_channels,in_channels,kernel_size,kernel_size),fan_in,seed)
    b = init_zero_bias(out_channels)
    return {
        'W':W,
        'b':b
    }

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    W = he_init((in_features,out_features),in_features,seed)
    b =init_zero_bias(out_features)
    return{
        'W':W,
        'b':b
    }

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    params = {
        'conv1': init_conv_layer(6,in_channels,5,seed=seed),
        'conv2': init_conv_layer(16,6,5,seed=seed+1),
        'fc1': init_linear_layer(256,120,seed=seed+2),
        'fc2': init_linear_layer(120,num_classes,seed=seed+3)
        
    }
    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    conv_out,conv_cache = conv2d_forward(x,W,b,stride,pad)
    relu_out,relu_cache = relu_forward(conv_out)
    pool_out,pool_cache = maxpool2d_forward(relu_out,pool_size,pool_size)
    

    cache =  {
        'conv_cache':conv_cache,
        
        'relu_cache':relu_cache,
        'pool_cache': pool_cache
    }
    return pool_out,cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    flatten,flatten_cache = flatten_forward(x)
    linear1,fc1_cache = linear_forward(flatten,fc1['W'],fc1['b'])
    relu_out, relu_cache = relu_forward(linear1)
    logits, fc2_cache = linear_forward(relu_out,fc2['W'],fc2['b'])

    cache = {
        'flatten_cache':flatten_cache,
        'fc1_cache':fc1_cache,
        'relu_cache':relu_cache,
        'fc2_cache':fc2_cache
    }
    return logits,cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    #Block1
    out1,cache1 = forward_conv_block(
        x,
        params['conv1']['W'],
        params['conv1']['b'],
        pool_size = 2,
        stride = 1,
        pad=0

    )
    #Block2
    out2,cache2 = forward_conv_block(
        out1,
        params['conv2']['W'],
        params['conv2']['b'],
        pool_size=2,
        stride=1,
        pad = 0
    )
    #Classifier  Flatten -> FC1 -> ReLU -> FC2
    logits, classifier_cache = forward_classifier_block(
        out2,
        params['fc1'],
        params['fc2']

    )
    cache = {
        'block1':cache1,
        'block2': cache2,
        'classifier':classifier_cache

    }
    return logits,cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    drelu = maxpool2d_backward(dout,cache['pool_cache'])
    dconv = relu_backward(drelu,cache['relu_cache'])
    dx,dW,db = conv2d_backward(
        dconv,cache['conv_cache']
    )
    return dx,dW,db

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

