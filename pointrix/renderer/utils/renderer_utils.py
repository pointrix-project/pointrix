import torch
from typing import Dict

class RenderFeatures:
    """
    A class for process the rendered features.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments.
    """
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to(self, device: str) -> None:
        """
        Move the features to the device.

        Parameters
        ----------
        device : str
            The device to move the features.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
    
    def combine(self)-> torch.Tensor:
        """
        Combine the features for DPTR rendering

        Returns
        -------
        torch.Tensor
            The combined features for DPTR rendering.
        """
        # Combine the features
        features = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                features.append(v)
        
        render_features = torch.cat(features, dim=-1)
        return render_features
    
    def split(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split the features for loss calculation

        Parameters
        ----------
        feature : torch.Tensor
            The combined features.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            The split features for loss calculation.
        """
        # Split the features
        rendered_features_dict = {}
        start = 0
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                end = start + v.shape[-1]
                rendered_features_dict[k] = feature[start:end, ...]
                start = end
        
        return rendered_features_dict

