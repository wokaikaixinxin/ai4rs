# Copyright (c) ai4rs. All rights reserved.
import torch
from torch import Tensor
import math

EPS = 1e-10

def rotate_poly(poly: Tensor, rad):
    '''
    Args:
        poly (Tensor): [n, 4, 2]
        rad (list): [n,]
    Returns:
        new_poly (Tensor): [n, 4, 2]
    '''
    n_pts, dd = poly.shape[-2:]
    if n_pts < 3 or dd != 2:
        raise ValueError

    matrix = torch.stack([torch.cos(rad), torch.sin(rad),
                          -torch.sin(rad), torch.cos(rad)], dim=-1).view(-1, 2, 2) # [n, 2, 2]
    new_poly = torch.bmm(poly, matrix)   # [n, 4, 2]
    return new_poly


def min_rect(poly: Tensor):
    '''
    Args:
        poly (Tensor): [n, 4, 2]
    Returns:
        min_rect (Tensor): [n, 4, 2]
        area (Tensor): [n,]
    '''
    _min = torch.min(poly, dim=1).values   # [n, 2]
    x_min, y_min = _min[:, 0], _min[:, 1]   # [n,],[n,]
    _max = torch.max(poly, dim=1).values   # [n,2]
    x_max, y_max = _max[:, 0], _max[:, 1]   # [n,],[n,]

    point1 = torch.stack([x_min, y_min], dim=-1)    # [n, 2]
    point2 = torch.stack([x_max, y_min], dim=-1)    # [n, 2]
    point3 = torch.stack([x_max, y_max], dim=-1)    # [n, 2]
    point4 = torch.stack([x_min, y_max], dim=-1)    # [n, 2]

    min_rect = torch.stack([point1, point2,
                            point3, point4], dim=1) # [n, 4, 2]
    area = (x_max - x_min) * (y_max - y_min)    # [n,]
    return min_rect, area


def get_rad(poly: Tensor):
    '''
    Args:
        poly (Tensor): [n, 4, 2]
    Returns:
        rad (list): {[n,], [n,], ...}
    '''
    n_pts, dd = poly.shape[-2:]
    if n_pts < 3 or dd != 2:
        raise ValueError

    rad = []
    for i in range(n_pts):
        vector = poly[:,i-1] - poly[:,i]    # [n, 2]
        rad.append(torch.atan(vector[:,1] / (vector[:,0]+EPS)))
    return rad


def min_rotate_rect(poly: Tensor):
    '''
    Args:
        poly (Tensor): [n, 4, 2]
    Returns:
        min_rect_r (Tensor): [n,4,2]
        area_min (Tensor): [n,]
        rad_min (Tensor): [n,]
    '''
    rect_min, area_min = min_rect(poly) # [n,4,2], [n,]
    rad_min = torch.zeros_like(area_min)    # [n,]
    rad = get_rad(poly) # {[n,], [n,], ...}
    for r in rad:
        new_poly = rotate_poly(poly, -r)    # [n,4,2]
        rect, area = min_rect(new_poly) # [n,4,2], [n,]
        idx = area < area_min
        rect_min[idx], area_min[idx], rad_min[idx] = rect[idx], area[idx], r[idx]
    min_rect_r = rotate_poly(rect_min, rad_min) # [n,4,2]
    return min_rect_r, area_min, rad_min


def qbox2rbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    poly = boxes.view(-1, 4, 2)
    min_rect_r, area_min, rad_min = min_rotate_rect(poly)   # [n,4,2], [n,], [n,]

    _min = torch.min(min_rect_r, dim=1).values   # [n,2]
    x_min, y_min = _min[:, 0], _min[:, 1]   # [n,], [n,]
    _max = torch.max(min_rect_r, dim=1).values   # [n,2]
    x_max, y_max = _max[:, 0], _max[:, 1]   # [n,], [n,]

    cx = (x_min + x_max) / 2    # [n,]
    cy = (y_min + y_max) / 2    # [n,]
    l1 = torch.sqrt((min_rect_r[:,0,0]-min_rect_r[:,1,0]) ** 2 +
                    (min_rect_r[:,0,1]-min_rect_r[:,1,1]) ** 2)   # [n,]
    l2 = torch.sqrt((min_rect_r[:,1,0]-min_rect_r[:,2,0]) ** 2 +
                    (min_rect_r[:,1,1]-min_rect_r[:,2,1]) ** 2)   # [n,]

    w = torch.max(l1, l2) # [n,]
    h = torch.min(l1, l2) # [n,]

    theta = torch.where((rad_min < -1e-6) | (rad_min > 1e-6),
                        torch.where(l1 >= l2, -rad_min, -rad_min - (math.pi / 2)),
                        torch.where(l1 >= l2, 0., - (math.pi/2)))  # [n,]
    rbox = torch.stack([cx, cy, w, h, theta], dim=-1)
    return rbox


if __name__ == "__main__":
    bbox = torch.tensor([[[2.2,9.9],[-1.8,10.2],[-2.2,-9.8],[1.9,-10.2]],
                         [[10,-0.2],[9,-2],[-10,-0.2],[-9,2]],
                         [[0,0],[1,0],[1,1],[0,1]]])
    rbox=qbox2rbox(bbox)
    print(rbox)
