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

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

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

