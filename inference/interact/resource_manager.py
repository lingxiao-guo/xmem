import os
from os import path
import shutil
import collections

import cv2
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from util.palette import davis_palette
import progressbar
 

# https://bugs.python.org/issue28178
# ah python ah why
class LRU:
    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize
 
    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()


class ResourceManager:
    def __init__(self, config):
        # determine inputs
        images = config['images']
        video = config['video']
        self.workspace = config['workspace']
        self.mask_dir = config.get('mask_dir')
        self.prefer_input_images = config.get('prefer_input_images', False)
        self.reset_workspace_masks = config.get('reset_workspace_masks', False)
        self.size = config['size']
        self.palette = davis_palette

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError(
                    'Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        direct_image_input = bool(images is not None and self.prefer_input_images and self.size < 0)

        if direct_image_input:
            self.image_dir = images
        else:
            self.image_dir = path.join(self.workspace, 'images')
            if images is not None:
                # explicit --images should override stale workspace/images.
                need_resizing = True
            elif path.exists(path.join(self.workspace, 'images')):
                pass
            elif video is not None:
                # will decode video into frames later
                need_decoding = True
            os.makedirs(self.image_dir, exist_ok=True)

        if self.mask_dir is None:
            self.mask_dir = path.join(self.workspace, 'masks')
        os.makedirs(self.mask_dir, exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=config['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=config['buffer_size'])

        if self.reset_workspace_masks:
            self.clear_all_masks()

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._clear_cached_images()
            self._copy_resize_frames(images)

        self.reload_image_dir(self.image_dir, clear_masks=False)
        self.visualization_init = False

    def _clear_cached_images(self):
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        for name in os.listdir(self.image_dir):
            ext = path.splitext(name)[1].lower()
            if ext not in image_exts:
                continue
            image_path = path.join(self.image_dir, name)
            if path.isfile(image_path):
                os.remove(image_path)

    def _extract_frames(self, video):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            if self.size > 0:
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
            cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        print('Done!')

    def _copy_resize_frames(self, images):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in progressbar.progressbar(image_list):
            if self.size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def save_mask(self, ti, mask):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        mask.save(path.join(self.mask_dir, self.names[ti]+'.png'))
        self.invalidate(ti)

    def save_visualization(self, ti, image):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)
        if not self.visualization_init:
            self.visualization_dir = path.join(self.workspace, 'visualization')
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.visualization_init = True

        image = Image.fromarray(image)
        image.save(path.join(self.visualization_dir, self.names[ti]+'.jpg'))

    def _get_image_unbuffered(self, ti):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.image_files[ti]))
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti]+'.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        else:
            return None

    def read_external_image(self, file_name, size=None):
        image = Image.open(file_name)
        is_mask = image.mode in ['L', 'P']
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), 
                    resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
        image = np.array(image)
        return image

    def invalidate(self, ti):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti,))

    def clear_all_masks(self):
        for name in os.listdir(self.mask_dir):
            if not name.lower().endswith('.png'):
                continue
            mask_path = path.join(self.mask_dir, name)
            if path.isfile(mask_path):
                os.remove(mask_path)
        self.get_mask.clear()

    def reload_image_dir(self, image_dir, clear_masks=True):
        self.image_dir = image_dir
        if clear_masks:
            self.clear_all_masks()

        self.get_image.clear()
        self.get_mask.clear()

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.image_files = sorted(
            f for f in os.listdir(self.image_dir) if path.splitext(f)[1].lower() in image_exts
        )
        self.names = [path.splitext(f)[0] for f in self.image_files]
        self.length = len(self.image_files)

        assert self.length > 0, f'No images found in {self.image_dir}.'
        self.height, self.width = self.get_image(0).shape[:2]
        print(f'{self.length} images loaded from {self.image_dir}.')

    def __len__(self):
        return self.length

    @property
    def h(self):
        return self.height

    @property
    def w(self):
        return self.width
