import torch
import lpips
from math import exp
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F
from torch.autograd import Variable


def psnr(img_pred:Float[Tensor, "H W C"], img_gt:Float[Tensor, "H W C"]):
    """
    Compute the PSNR between two images.

    Parameters
    ----------
    img1 : torch.Tensor
        The first image.
    img2 : torch.Tensor
        The second image.
    """
    assert img_pred.shape == img_gt.shape, "The shape of the two images should be the same."
    l2 = l2_loss(img_pred, img_gt, return_mean=False)
    l2_reshape = l2.view(img_pred.shape[0], -1)
    l2_mean = l2_reshape.mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(l2_mean))

def l1_loss(pred:Float[Tensor, "H ..."], gt:Float[Tensor, "H ..."], return_mean:bool=True):
    """
    Compute the L1 loss between the network output and the ground truth.

    Parameters
    ----------
    pred : torch.Tensor
        The network prediction.
    gt : torch.Tensor
        The ground truth.
    
    Returns
    -------
    torch.Tensor
        The L1 loss.
    """
    assert pred.shape == gt.shape, "The shape of the two tensor should be the same."
    if not return_mean:
        return torch.abs((pred - gt))
    return torch.abs((pred - gt)).mean()

def l2_loss(pred:Float[Tensor, "H ..."], gt:Float[Tensor, "H ..."], return_mean:bool=True):
    """
    Compute the L2 loss between the network output and the ground truth.

    Parameters
    ----------
    pred : torch.Tensor
        The network prediction.
    gt : torch.Tensor
        The ground truth.
    
    Returns
    -------
    torch.Tensor
        The L2 loss.
    """
    assert pred.shape == gt.shape, "The shape of the two tensor should be the same."
    if not return_mean:
        return ((pred - gt) ** 2)
    return ((pred - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def ssim(img1:Float[Tensor, "H W C"], img2:Float[Tensor, "H W C"], window_size=11, size_average=True):
    """
    Compute the SSIM between two images, 
    adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blame/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L65

    Parameters
    ----------
    img1 : torch.Tensor
        The first image.
    img2 : torch.Tensor
        The second image.
    window_size : int, optional
        The window size, by default 11
    size_average : bool, optional
        Whether to average the SSIM or not, by default True
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
    
class LPIPS(object):
    def __init__(self, device='cuda', net='vgg'):
        if net == 'vgg':
            self.model = lpips.LPIPS(net='vgg').to(device)
        elif net == 'alex':
            self.model = lpips.LPIPS(net='alex').to(device)
        else:
            assert False, "Could not recognize network type for LPIPS!"

    def __call__(self, img1, img2):
        return self.model(img1, img2)

class LPIPS(object):
    """
    LPIPS loss function.

    Parameters
    ----------
    device : str
        The device to run the loss function.
    net : str
        The network type for LPIPS.
    """
    def __init__(self, device='cuda', net='vgg'):
        if net == 'vgg':
            self.model = lpips.LPIPS(net='vgg').to(device)
        elif net == 'alex':
            self.model = lpips.LPIPS(net='alex').to(device)
        else:
            assert False, "Could not recognize network type for LPIPS!"

    def __call__(self, img1, img2):
        return self.model(img1, img2)
    