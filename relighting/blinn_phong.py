import torch
import torch
import torch.nn as nn
from utils.general_utils import get_expon_lr_func

def blinn_phong_shader(viewPos, positions, normals, light_position, light_color, material_color, shininess, ambient_color, ambient_intensity):

    
    normals = torch.nn.functional.normalize(normals, dim=1)

    # Calculate the direction from the surface points to the light source
    light_directions = torch.nn.functional.normalize(light_position - positions, dim=1)

    # Calculate the direction from the surface points to the camera (assuming camera at the origin)
    view_directions = torch.nn.functional.normalize(viewPos - positions, dim=1)

    # Calculate the halfway vectors between the light directions and the view directions
    halfway_vectors = torch.nn.functional.normalize(light_directions + view_directions, dim=1)

    # Calculate the diffuse terms
    diffuse_terms = torch.max(torch.sum(normals * light_directions, dim=1), torch.tensor(0.0))
    diffuse_terms = diffuse_terms.unsqueeze(1).repeat(1,3)         
    
    # Calculate the specular terms
    specular_sum = torch.sum(normals * halfway_vectors, dim=1)
    specular_max = torch.max(specular_sum, torch.tensor(0.000001))   

    specular_terms = torch.pow(specular_max.unsqueeze(1), shininess.clamp_min(0.0))
    specular_terms = specular_terms.repeat(1,3)

    light_color = torch.sigmoid(light_color)
    #ambient_color = torch.sigmoid(ambient_color)
    # Calculate the ambient term
    ambient_term = torch.sigmoid(ambient_color) * ambient_intensity

    # Calculate the final colors by combining the diffuse, specular, and ambient terms
    #colors = light_color.repeat(positions.shape[0], 1) * (material_color * diffuse_terms + specular_terms) + ambient_term.repeat(positions.shape[0], 1)
    colors = light_color.clamp(0.0,1.0).repeat(positions.shape[0], 1) * (material_color * diffuse_terms + specular_terms) + ambient_term.unsqueeze(0) 
    #colors = torch.clamp(light_color.repeat(positions.shape[0], 1) * (material_color * diffuse_terms) + ambient_term.repeat(positions.shape[0], 1),0.0,1.0)

    return colors

class BlinnPhongModel(nn.Module):
    def __init__(self, training_args = None,
                 light_position = torch.zeros([3], device="cuda"), 
                 light_color = torch.zeros([3], device="cuda"), 
                 ambient_color = torch.zeros([3], device="cuda"), 
                 ambient_intensity = torch.zeros([1], device="cuda")):
        super(BlinnPhongModel, self).__init__()
        self.light_position = nn.Parameter(light_position).requires_grad_(True)
        self.light_color = nn.Parameter(light_color).requires_grad_(True)
        self.ambient_color = nn.Parameter(ambient_color).requires_grad_(True)
        self.ambient_intensity = nn.Parameter(ambient_intensity).requires_grad_(True)        

    def forward(self, viewPos, positions, normals, material_colors, shininess):
        return blinn_phong_shader(viewPos, positions, normals, self.light_position, self.light_color, material_colors, shininess, self.ambient_color, self.ambient_intensity)

