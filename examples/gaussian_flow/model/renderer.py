import torch
from pointrix.model.renderer import MsplatRender, RENDERER_REGISTRY

@RENDERER_REGISTRY.register()
class GaussianFlowRender(MsplatRender):
    
    def render_batch(self, render_dict: dict) -> dict:
        """
        Render the batch of point clouds.

        Parameters
        ----------
        render_dict : dict
            The render dictionary.
        batch : List[dict]
            The batch data.

        Returns
        -------
        dict
            The rendered image, the viewspace points, 
            the visibility filter, the radii, the xyz, 
            the color, the rotation, the scales, and the xy.
        """
        rendered_features = {}
        uv_points = []
        visibilitys = []
        radii = []
    
        batched_render_keys = ["extrinsic_matrix", "camera_center", "position", "rotation", "shs"]
        
        for i in range(render_dict['extrinsic_matrix'].shape[0]):
            render_iter_dict = {}
            for key in render_dict.keys():
                if key not in batched_render_keys:
                    render_iter_dict[key] = render_dict[key]
                else:
                    render_iter_dict[key] = render_dict[key][i, ...]

            render_results = self.render_iter(**render_iter_dict)
            for feature_name in render_results["rendered_features_split"].keys():
                if feature_name not in rendered_features:
                    rendered_features[feature_name] = []
                rendered_features[feature_name].append(
                    render_results["rendered_features_split"][feature_name])

            uv_points.append(render_results["uv_points"])
            visibilitys.append(
                render_results["visibility"].unsqueeze(0))
            radii.append(render_results["radii"].unsqueeze(0))

        for feature_name in rendered_features.keys():
            rendered_features[feature_name] = torch.stack(
                rendered_features[feature_name], dim=0)

        return {**rendered_features,
                "uv_points": uv_points,
                "visibility": torch.cat(visibilitys).any(dim=0),
                "radii": torch.cat(radii, 0).max(dim=0).values
                }