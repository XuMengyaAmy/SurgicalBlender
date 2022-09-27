import cv2
from typing import List
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f
# from torch.nn import init
# import functools
# from torch.optim import lr_scheduler


class DCLoss(nn.Module):
    thresh_low = nn.Threshold(0,0)
    thresh_high = nn.Threshold(-255,255) # We use neg multiplication of tensor as an addtional step to cater to Threshold definition of the function

    # Taken from kornia : START
    def erosion(self,
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = 'unfold',
    ) -> torch.Tensor:
        r"""Return the eroded image applying the same kernel in each channel.

        .. image:: _static/img/erosion.png

        The kernel must have 2 dimensions.

        Args:
            tensor: Image with shape :math:`(B, C, H, W)`.
            kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
                the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
                For full structural elements use torch.ones_like(structural_element).
            structuring_element (torch.Tensor, optional): Structuring element used for the grayscale dilation.
                It may be a non-flat structuring element.
            origin: Origin of the structuring element. Default: ``None`` and uses the center of
                the structuring element as origin (rounding towards zero).
            border_type: It determines how the image borders are handled, where ``border_value`` is the value
                when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
                outside the image when applying the operation.
            border_value: Value to fill past edges of input if border_type is ``constant``.
            max_val: The value of the infinite elements in the kernel.
            engine: ``convolution`` is faster and less memory hungry, and ``unfold`` is more stable numerically

        Returns:
            Eroded image with shape :math:`(B, C, H, W)`.

        .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
        morphology_101.html>`__.

        Example:
            >>> tensor = torch.rand(1, 3, 5, 5)
            >>> kernel = torch.ones(5, 5)
            >>> output = erosion(tensor, kernel)
        """

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

        if len(tensor.shape) != 4:
            raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

        if not isinstance(kernel, torch.Tensor):
            raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

        if len(kernel.shape) != 2:
            raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

        # origin
        se_h, se_w = kernel.shape
        
        origin = [se_h // 2, se_w // 2]

        # pad
        pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
        if border_type == 'geodesic':
            border_value = max_val
            border_type = 'constant'
        output = f.pad(tensor, pad_e, mode=border_type, value=border_value)

        device = tensor.get_device()
        device = "cpu" if device == -1 else device
        neighborhood = torch.zeros_like(kernel, device=device)
        neighborhood[kernel == 0] = -max_val


        if engine == 'unfold':
            output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
            output, _ = torch.min(output - neighborhood, 4)
            output, _ = torch.min(output, 4)
        

        return output

    
    def dilation_pytorch(self,image, strel, origin=(0, 0), border_value=0):
        # first pad the image to have correct unfolding; here is where the origins is used
        image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
        # Flatten the structural element since its two dimensions have been flatten when unfolding
        strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
        sums = image_unfold + strel_flatten
        # Take maximum over the neighborhood
        result, _ = sums.max(dim=1)
        # Reshape the image to recover initial shape
        return torch.reshape(result, image.shape)
    
    def normalize_kernel2d(self,input: torch.Tensor) -> torch.Tensor:
        r"""Normalize both derivative and smoothing kernel."""
        if len(input.size()) < 2:
            raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
        norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
        return input / (norm.unsqueeze(-1).unsqueeze(-1))
    def _compute_padding(self, kernel_size: List[int]) -> List[int]:
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]

        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]

            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front

            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear

        return out_padding
    def get_box_kernel2d(self,kernel_size) -> torch.Tensor:
        r"""Utility function that returns a box filter."""
        kx: float = float(kernel_size[0])
        ky: float = float(kernel_size[1])
        scale: torch.Tensor = torch.tensor(1.0) / torch.tensor([kx * ky])
        tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
        return scale.to(tmp_kernel.dtype) * tmp_kernel
    def filter2d(self,
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
    ) -> torch.Tensor:
        r"""Convolve a tensor with a 2d kernel.

        The function applies a given kernel to a tensor. The kernel is applied
        independently at each depth channel of the tensor. Before applying the
        kernel, the function applies padding according to the specified mode so
        that the output remains in the same shape.

        Args:
            input: the input tensor with shape of
            :math:`(B, C, H, W)`.
            kernel: the kernel to be convolved with the input
            tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
            border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``.
            normalized: If True, kernel will be L1 normalized.
            padding: This defines the type of padding.
            2 modes available ``'same'`` or ``'valid'``.

        Return:
            torch.Tensor: the convolved tensor of same size and numbers of channels
            as the input with shape :math:`(B, C, H, W)`.

        Example:
            >>> input = torch.tensor([[[
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 5., 0., 0.],
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 0., 0., 0.],]]])
            >>> kernel = torch.ones(1, 3, 3)
            >>> filter2d(input, kernel, padding='same')
            tensor([[[[0., 0., 0., 0., 0.],
                    [0., 5., 5., 5., 0.],
                    [0., 5., 5., 5., 0.],
                    [0., 5., 5., 5., 0.],
                    [0., 0., 0., 0., 0.]]]])
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

        if not isinstance(kernel, torch.Tensor):
            raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

        if not isinstance(border_type, str):
            raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

        if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
            raise ValueError(
                f"Invalid border type, we expect 'constant', \
            'reflect', 'replicate', 'circular'. Got:{border_type}"
            )

        if not isinstance(padding, str):
            raise TypeError(f"Input padding is not string. Got {type(padding)}")

        if padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

        if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
            raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

        if normalized:
            tmp_kernel = self.normalize_kernel2d(tmp_kernel)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        height, width = tmp_kernel.shape[-2:]

        # pad the input tensor
        if padding == 'same':
            padding_shape: List[int] = self._compute_padding([height, width])
            input = f.pad(input, padding_shape, mode=border_type)

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

        # convolve the tensor with the kernel.
        output = f.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        if padding == 'same':
            out = output.view(b, c, h, w)
        else:
            out = output.view(b, c, h - height + 1, w - width + 1)

        return out
    def box_blur(self,
        input: torch.Tensor, kernel_size, border_type: str = 'reflect', normalized: bool = True
    ) -> torch.Tensor:
        r"""Blur an image using the box filter.

        .. image:: _static/img/box_blur.png

        The function smooths an image using the kernel:

        .. math::
            K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
            \begin{bmatrix}
                1 & 1 & 1 & \cdots & 1 & 1 \\
                1 & 1 & 1 & \cdots & 1 & 1 \\
                \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                1 & 1 & 1 & \cdots & 1 & 1 \\
            \end{bmatrix}

        Args:
            image: the image to blur with shape :math:`(B,C,H,W)`.
            kernel_size: the blurring kernel size.
            border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            normalized: if True, L1 norm of the kernel is set to 1.

        Returns:
            the blurred tensor with shape :math:`(B,C,H,W)`.

        .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
        filtering_operators.html>`__.

        Example:
            >>> input = torch.rand(2, 4, 5, 7)
            >>> output = box_blur(input, (3, 3))  # 2x4x5x7
            >>> output.shape
            torch.Size([2, 4, 5, 7])
        """
        kernel: torch.Tensor = self.get_box_kernel2d(kernel_size)
        if normalized:
            kernel = self.normalize_kernel2d(kernel)
        return self.filter2d(input, kernel, border_type)   
    # Taken from kornia : END
    
    def compute_dc(self, im, sz):
        r = im[0]
        g = im[1]
        b = im[2]
        min_dc = cv2.min(cv2.min(r,g),b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
        dark = cv2.erode(min_dc,kernel)
        return dark, min_dc
    
    def compute_dc_pytorch(self, im, sz):
        r = im[0]
        g = im[1]
        b = im[2]
       
        min_dc = torch.minimum(torch.minimum(r,g),b)
        kernel = torch.ones((sz,sz))
        dark = self.erosion(min_dc.unsqueeze(dim=0).unsqueeze(dim=-1),kernel)
        
        return dark, min_dc
    def apply_guided_filter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
        mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
        mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
        cov_Ip = mean_Ip - mean_I*mean_p;

        mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
        var_I   = mean_II - mean_I*mean_I;

        a = cov_Ip/(var_I + eps);
        b = mean_p - a*mean_I;

        mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
        mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

        q = mean_a*im + mean_b

        return q
    def apply_guided_filter_pytorch(self, im, p, r, eps):
        mean_I = self.box_blur(im,(r,r));
        mean_p = self.box_blur(p,(r,r));
        mean_Ip = self.box_blur(im*p,(r,r));
        cov_Ip = mean_Ip - mean_I*mean_p;

        mean_II = self.box_blur(im*im, (r,r));
        var_I   = mean_II - mean_I*mean_I;

        a = cov_Ip/(var_I + eps);
        b = mean_p - a*mean_I;

        mean_a = self.box_blur(a,(r,r));
        mean_b = self.box_blur(b,(r,r));

        q = mean_a*im + mean_b;

        return q;

    def refine_dc(self, im, dc, min_dc):
        r = 15;
        eps = 0.0001;
        dc_rfd = self.apply_guided_filter(min_dc,dc,r,eps);

        return dc_rfd;
    
    def refine_dc_pytorch(self, im, dc, min_dc):
        r = 15;
        eps = 0.0001;
        dc_rfd = self.apply_guided_filter_pytorch(min_dc,dc,r,eps);

        return dc_rfd;

    def get_refined_dc(self, input_img, refine=False):
        # input_img = cv2.imread(path_image);
        dc, min_dc = self.compute_dc(input_img.cpu().data.numpy(), 15)
        if refine:
            dc = self.refine_dc(input_img,dc,min_dc)*255
        dc[dc < 0] = 0
        dc[dc > 255] = 255

        return np.uint8(dc)
    
    def get_refined_dc_pytorch(self, input_img, refine=False):
        # input_img = cv2.imread(path_image);
        dc, min_dc = self.compute_dc_pytorch(input_img, 15)
        if refine:
            dc = self.refine_dc_pytorch(input_img,dc,min_dc)*255
        dc = self.thresh_low(dc)
        dc*=-1
        dc = self.thresh_high(dc)
        dc*=-1

        return dc

    def __call__(self, hazy_image, refine=False):
        batch_size, channels, height, width = hazy_image.size()
        index = np.random.randint(batch_size)
        input_img = hazy_image[index]
        refined_dc = self.get_refined_dc(input_img,refine)
        loss = refined_dc.mean()*0.05

        return loss
    def call_pytorch_impl(self,hazy_image, refine=False):
        batch_size,_,_,_ = hazy_image.size()
        index = np.random.randint(batch_size)
        input_img = hazy_image[index]
        refined_dc = self.get_refined_dc_pytorch(input_img, refine)
        loss = refined_dc.mean()*0.05

        return loss
