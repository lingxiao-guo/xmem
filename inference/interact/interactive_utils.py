# Modifed from https://github.com/seoungwugoh/ivs-demo

import numpy as np

import torch
import torch.nn.functional as F
from util.palette import davis_palette
from dataset.range_transform import im_normalization

def image_to_torch(frame: np.ndarray, device='cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device)/255
    frame_norm = im_normalization(frame)
    return frame_norm, frame

def torch_prob_to_numpy_mask(prob):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask

def index_numpy_to_one_hot_torch(mask, num_classes):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()

"""
Some constants fro visualization
"""
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
color_map_np = (color_map_np.astype(np.float32)*1.5).clip(0, 255).astype(np.uint8)
color_map_np[1] = np.array([200, 120, 200], dtype=np.uint8)
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255

grayscale_weights = np.array([[0.3,0.59,0.11]]).astype(np.float32)
grayscale_weights_torch = torch.from_numpy(grayscale_weights).to(device).unsqueeze(0)
boundary_color_np = np.array([140, 80, 140], dtype=np.uint8)
boundary_color_torch = torch.tensor([140, 80, 140], device=device) / 255
boundary_width = 3

def _dilate_mask_numpy(binary_mask, iterations=1):
    out = binary_mask
    for _ in range(iterations):
        up = np.pad(out[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=False)
        down = np.pad(out[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=False)
        left = np.pad(out[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=False)
        right = np.pad(out[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=False)
        up_left = np.pad(out[1:, 1:], ((0, 1), (0, 1)), mode='constant', constant_values=False)
        up_right = np.pad(out[1:, :-1], ((0, 1), (1, 0)), mode='constant', constant_values=False)
        down_left = np.pad(out[:-1, 1:], ((1, 0), (0, 1)), mode='constant', constant_values=False)
        down_right = np.pad(out[:-1, :-1], ((1, 0), (1, 0)), mode='constant', constant_values=False)
        out = out | up | down | left | right | up_left | up_right | down_left | down_right
    return out

def _mask_boundary_numpy(binary_mask, width=1):
    up = np.pad(binary_mask[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=False)
    down = np.pad(binary_mask[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=False)
    left = np.pad(binary_mask[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=False)
    right = np.pad(binary_mask[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=False)
    eroded = binary_mask & up & down & left & right
    boundary = binary_mask & ~eroded
    if width <= 1:
        return boundary
    return _dilate_mask_numpy(boundary, iterations=width-1)

def _dilate_mask_torch(binary_mask, iterations=1):
    out = binary_mask
    for _ in range(iterations):
        up = torch.zeros_like(out)
        down = torch.zeros_like(out)
        left = torch.zeros_like(out)
        right = torch.zeros_like(out)
        up_left = torch.zeros_like(out)
        up_right = torch.zeros_like(out)
        down_left = torch.zeros_like(out)
        down_right = torch.zeros_like(out)
        up[:-1, :] = out[1:, :]
        down[1:, :] = out[:-1, :]
        left[:, :-1] = out[:, 1:]
        right[:, 1:] = out[:, :-1]
        up_left[:-1, :-1] = out[1:, 1:]
        up_right[:-1, 1:] = out[1:, :-1]
        down_left[1:, :-1] = out[:-1, 1:]
        down_right[1:, 1:] = out[:-1, :-1]
        out = out | up | down | left | right | up_left | up_right | down_left | down_right
    return out

def _mask_boundary_torch(binary_mask, width=1):
    up = torch.zeros_like(binary_mask)
    down = torch.zeros_like(binary_mask)
    left = torch.zeros_like(binary_mask)
    right = torch.zeros_like(binary_mask)
    up[:-1, :] = binary_mask[1:, :]
    down[1:, :] = binary_mask[:-1, :]
    left[:, :-1] = binary_mask[:, 1:]
    right[:, 1:] = binary_mask[:, :-1]
    eroded = binary_mask & up & down & left & right
    boundary = binary_mask & ~eroded
    if width <= 1:
        return boundary
    return _dilate_mask_torch(boundary, iterations=width-1)

def get_visualization(mode, image, mask, layer, target_object):
    if mode == 'fade':
        return overlay_davis(image, mask, fade=True)
    elif mode == 'davis':
        return overlay_davis(image, mask, draw_boundary=True)
    elif mode == 'light':
        return overlay_davis(image, mask, 0.9)
    elif mode == 'white':
        return overlay_white(image, mask)
    elif mode == 'green_screen':
        return overlay_green_screen(image, mask)
    elif mode == 'popup':
        return overlay_popup(image, mask, target_object)
    elif mode == 'layered':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis(image, mask)
        else:
            return overlay_layer(image, mask, layer, target_object)
    else:
        raise NotImplementedError

def get_visualization_torch(mode, image, prob, layer, target_object):
    if mode == 'fade':
        return overlay_davis_torch(image, prob, fade=True)
    elif mode == 'davis':
        return overlay_davis_torch(image, prob, draw_boundary=True)
    elif mode == 'light':
        return overlay_davis_torch(image, prob, 0.9)
    elif mode == 'white':
        return overlay_white_torch(image, prob)
    elif mode == 'green_screen':
        return overlay_green_screen_torch(image, prob)
    elif mode == 'popup':
        return overlay_popup_torch(image, prob, target_object)
    elif mode == 'layered':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis_torch(image, prob)
        else:
            return overlay_layer_torch(image, prob, layer, target_object)
    else:
        raise NotImplementedError

def overlay_davis(image, mask, alpha=0.5, fade=False, draw_boundary=False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if draw_boundary:
        boundary_mask = _mask_boundary_numpy(binary_mask, width=boundary_width)
        im_overlay[boundary_mask] = boundary_color_np.astype(image.dtype)
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)

def overlay_white(image, mask):
    """ Overlay segmentation as solid white on top of the RGB image. """
    im_overlay = image.copy()
    binary_mask = (mask > 0)
    im_overlay[binary_mask] = np.array([255, 255, 255], dtype=image.dtype)
    return im_overlay.astype(image.dtype)

def overlay_green_screen(image, mask):
    """ Overlay segmentation as solid green on top of the RGB image. """
    im_overlay = image.copy()
    binary_mask = (mask > 0)
    im_overlay[binary_mask] = np.array([0, 255, 0], dtype=image.dtype)
    return im_overlay.astype(image.dtype)

def overlay_popup(image, mask, target_object):
    # Keep foreground colored. Convert background to grayscale.
    im_overlay = image.copy()

    binary_mask = ~(np.isin(mask, target_object))
    colored_region = (im_overlay[binary_mask]*grayscale_weights).sum(-1, keepdims=-1)
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)

def overlay_layer(image, mask, layer, target_object):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    obj_mask = (np.isin(mask, target_object)).astype(np.float32)[:, :, np.newaxis]
    layer_alpha = layer[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255
    layer_rgb = layer[:, :, :3]
    background_alpha = np.maximum(obj_mask, layer_alpha)
    im_overlay = (image * (1 - background_alpha) + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 255)
    return im_overlay.astype(image.dtype)

def overlay_davis_torch(image, mask, alpha=0.5, fade=False, draw_boundary=False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(mask, dim=0).indices

    colored_mask = color_map_torch[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if draw_boundary:
        boundary_mask = _mask_boundary_torch(binary_mask, width=boundary_width)
        im_overlay[boundary_mask] = boundary_color_torch.to(device=image.device, dtype=image.dtype)
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_white_torch(image, mask):
    """ Overlay segmentation as solid white on top of the RGB image. """
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(mask, dim=0).indices

    binary_mask = (mask > 0)
    im_overlay[binary_mask] = 1.0

    im_overlay = (im_overlay * 255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_popup_torch(image, mask, target_object):
    # Keep foreground colored. Convert background to grayscale.
    image = image.permute(1, 2, 0)
    
    if len(target_object) == 0:
        obj_mask = torch.zeros_like(mask[0]).unsqueeze(2)
    else:
        # I should not need to convert this to numpy.
        # uUsing list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = mask[np.array(target_object,dtype=np.int32)].sum(0).unsqueeze(2)
    gray_image = (image*grayscale_weights_torch).sum(-1, keepdim=True)
    im_overlay = obj_mask*image + (1-obj_mask)*gray_image

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_green_screen_torch(image, mask):
    """ Overlay segmentation as solid green on top of the RGB image. """
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(mask, dim=0).indices

    binary_mask = (mask > 0)
    im_overlay[binary_mask] = torch.tensor([0.0, 1.0, 0.0], device=image.device, dtype=image.dtype)

    im_overlay = (im_overlay * 255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_layer_torch(image, prob, layer, target_object):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)

    if len(target_object) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_object, dtype=np.int32)].sum(0).unsqueeze(2)
    layer_alpha = layer[:, :, 3].unsqueeze(2)
    layer_rgb = layer[:, :, :3]
    background_alpha = torch.maximum(obj_mask, layer_alpha)
    im_overlay = (image * (1 - background_alpha) + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 1)

    im_overlay = (im_overlay * 255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay
