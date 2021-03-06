import numpy
from six import moves

from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


try:
    import cupy
    from cupy import cuda
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn
    _cudnn_version = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    if _cudnn_version >= 4000:
        _bwd_filter_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        _bwd_data_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
except ImportError:
    print("no cuda")

def _kern():
    # return cuda.elementwise(
    #     'T x', 'T y',
    #     'y = x >= 0 ? 1 : -1',
    #     'binarize')
    ret = cupy.ElementwiseKernel(
        'T x', 'T y',
         'y = x >= 0 ? 1 : -1',
         'binarize')
    return ret

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)



def _check_cudnn_acceptable_type(x_dtype, W_dtype):
    return x_dtype == W_dtype and (
        _cudnn_version >= 3000 or x_dtype != numpy.float16)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class BinaryConv2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.use_cudnn = use_cudnn
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )
        #if n_in.eval() == 3:
        if n_in == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[2:]
        self.col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)

        Xb = numpy.where(self.col>0,1,self.col).astype(x.dtype, copy=False)
        Xb = numpy.where(self.col<0,-1,Xb).astype(x.dtype, copy=False)
        Wb = numpy.where(W>=0,1,-1).astype(W.dtype, copy=False)
        y = numpy.tensordot(
            Xb, Wb, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b

        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        out_c, _, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                      cover_all=self.cover_all)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                      cover_all=self.cover_all)

        #y = cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
        y = cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
        if (self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):
            x = cupy.ascontiguousarray(x)
            W = cupy.ascontiguousarray(W)
            if b is not None:
                b = cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx))
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            workspace_size = cuda.get_max_workspace_size()
            workspace = cupy.empty((workspace_size,), dtype='b')
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionForward(
                handle, one.data, x_desc.value, x.data.ptr,
                self.filter_desc.value, W.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                y_desc.value, y.data.ptr)

            # TODO(beam2d): Support unshared bias
            if b is not None:
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)
        else:
            # Implementation using im2col
            Xb = _kern()(x)

            self.col = conv.im2col_gpu(
                Xb, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
            
            W_mat = W.reshape(out_c, -1)
            col_mats = self.col.reshape(n, -1, out_h * out_w)
            Wb_mat = _kern()(W_mat)

            y_mats = y.reshape(n, out_c, -1)
            # TODO(beam2d): Use streams or batch gemm
            for i in moves.range(n):
                y_mats[i] = Wb_mat.dot(col_mats[i])
            # TODO(beam2d): Support unshared bias
            if b is not None:
                y += b[:, None, None]

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = numpy.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)

        Wb = numpy.where(W>=0,1,-1).astype(W.dtype, copy=False)

        gcol = numpy.tensordot(Wb, gy, (0, 1)).astype(x.dtype, copy=False)

        gcol = numpy.rollaxis(gcol, 3)
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = x.shape

        kh, kw = W.shape[2:]

        #gW = cupy.empty_like(W)
        gW = cupy.empty_like(W)
        if (self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):
            x = cupy.ascontiguousarray(x)
            W = cupy.ascontiguousarray(W)
            gy = cupy.ascontiguousarray(gy)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            gy_desc = cudnn.create_tensor_descriptor(gy)
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            gx = cupy.empty_like(x)

            if _cudnn_version >= 4000:
                workspace_size = cuda.get_max_workspace_size()
                workspace = cupy.empty((workspace_size,), dtype='b')

                algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                    handle, x_desc.value, gy_desc.value,
                    self.conv_desc.value, self.filter_desc.value,
                    _bwd_filter_pref, workspace_size)
                libcudnn.convolutionBackwardFilter_v3(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, self.filter_desc.value, gW.data.ptr)

                algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                    handle, self.filter_desc.value, gy_desc.value,
                    self.conv_desc.value, x_desc.value, _bwd_data_pref,
                    workspace_size)
                libcudnn.convolutionBackwardData_v3(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, x_desc.value, gx.data.ptr)
            else:
                libcudnn.convolutionBackwardFilter_v2(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, self.filter_desc.value, gW.data.ptr)
                libcudnn.convolutionBackwardData_v2(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, x_desc.value, gx.data.ptr)

            if b is not None:
                gb = cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
        else:
            gW_mat = gW.reshape(out_c, c * kh * kw)
            col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
            gy_mats = gy.reshape(n, out_c, out_h * out_w)
            # TODO(beam2d): Use streams or batch gemm
            gW_mat[...] = 0
            for i in moves.range(n):
                gW_mat += cupy.dot(gy_mats[i], col_mats[i].T)

            W_mat = W.reshape(out_c, -1)
            Wb_mat = _kern()(W_mat)

            gcol = cupy.empty_like(self.col)
            gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)

            for i in moves.range(n):
                gcol_mats[i] = cupy.dot(Wb_mat.T, gy_mats[i])

            gx = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)

            if b is not None:
                gb = gy.sum(axis=(0, 2, 3))

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


def func_convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False):
    """Two-dimensional convolution function.

    This is an implementation of two-dimensional convolution in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output,
      respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_O, c_I, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available.
        cover_all (bool): If True, all spatial locations are convoluted into
            some output pixels. It may make the output size larger.


    Returns:
        ~chainer.Variable: Output variable.

    The two-dimensional convolution function is defined as follows.
    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(k_H, k_W)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    .. seealso:: :class:`Convolution2D`

    """
    func = BinaryConv2DFunction(stride, pad, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
