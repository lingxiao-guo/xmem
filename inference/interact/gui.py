"""
Based on https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN 
(which is based on https://github.com/seoungwugoh/ivs-demo)

This version is much simplified. 
In this repo, we don't have
- local control
- fusion module
- undo
- timers

but with XMem as the backbone and is more memory (for both CPU and GPU) friendly
"""

import functools
import json

import os
import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import numpy as np
import torch
from PIL import Image
try:
    from torch import mps
except:
    print('torch.MPS not available.')

from PySide6.QtWidgets import (QWidget, QApplication, QComboBox, QCheckBox,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QSpinBox, QFileDialog,
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, QRadioButton,
    QMessageBox)

from PySide6.QtGui import QPixmap, QKeySequence, QImage, QTextCursor, QIcon, QShortcut
from PySide6.QtCore import Qt, QTimer

from model.network import XMem

from inference.inference_core import InferenceCore
from .s2m_controller import S2MController
from .fbrs_controller import FBRSController

from .interactive_utils import *
from .interaction import *
from .resource_manager import ResourceManager
from .gui_utils import *


class App(QWidget):
    def __init__(self, net: XMem, 
                resource_manager: ResourceManager, 
                s2m_ctrl:S2MController, 
                fbrs_ctrl:FBRSController, config, device):
        super().__init__()

        self.initialized = False
        self.num_objects = config['num_objects']
        self.s2m_controller = s2m_ctrl
        self.fbrs_controller = fbrs_ctrl
        self.config = config
        self.output_path = os.path.normpath(
            self.config.get('output_path', os.path.join(self.config['workspace'], 'outputs'))
        )
        os.makedirs(self.output_path, exist_ok=True)
        self.scene_mode_message = 'select time segment for reconstructing  scene'
        self.scene_mode_popup_message = (
            'Please first select timesteps for reconstructing the scene and downsample ratio, '
            'then run downsample.'
        )
        self.processor = InferenceCore(net, config)
        self.processor.set_all_labels(list(range(1, self.num_objects+1)))
        self.res_man = resource_manager
        self.device = device

        self.num_frames = len(self.res_man)
        self.height, self.width = self.res_man.h, self.res_man.w

        # set window
        self.setWindowTitle('XMem Demo')
        self.setGeometry(100, 100, self.width, self.height+100)
        self.setWindowIcon(QIcon('docs/icon.png'))

        # some buttons
        self.play_button = QPushButton('Play Video')
        self.play_button.clicked.connect(self.on_play_video)
        self.commit_button = QPushButton('Commit')
        self.commit_button.clicked.connect(self.on_commit)
        self.export_button = QPushButton('Export Overlays as Video')
        self.export_button.clicked.connect(self.on_export_visualization)

        self.forward_run_button = QPushButton('Forward Propagate')
        self.forward_run_button.clicked.connect(self.on_forward_propagation)
        self.forward_run_button.setMinimumWidth(150)

        self.backward_run_button = QPushButton('Backward Propagate')
        self.backward_run_button.clicked.connect(self.on_backward_propagation)
        self.backward_run_button.setMinimumWidth(150)

        self.reset_button = QPushButton('Reset Frame')
        self.reset_button.clicked.connect(self.on_reset_mask)
        self.reset_object_button = QPushButton('Clear Object')
        self.reset_object_button.clicked.connect(self.on_reset_object)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(120)
        self.lcd.setText('{: 4d} / {: 4d}'.format(0, self.num_frames-1))

        # Current Mask LCD
        self.object_dial = QSpinBox()
        self.object_dial.setReadOnly(False)
        self.object_dial.setMaximumHeight(28)
        self.object_dial.setMaximumWidth(56)
        self.object_dial.setMinimum(0)
        self.object_dial.setMaximum(self.num_objects)
        self.object_dial.editingFinished.connect(self.on_object_dial_change)

        # timeline slider
        self.tl_slider = QSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(self.tl_slide)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames-1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)
        
        # brush size slider
        self.brush_label = QLabel()
        self.brush_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brush_label.setMinimumWidth(150)
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.valueChanged.connect(self.brush_slide)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(3)
        self.brush_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.setMinimumWidth(300)

        # reconstruction time controls
        self.recon_segment = QSpinBox()
        self.recon_segment.setMaximumHeight(28)
        self.recon_segment.setMaximumWidth(60)
        self.recon_segment.setMinimum(1)
        self.recon_segment.setMaximum(9999)
        self.recon_segment.valueChanged.connect(self.on_recon_segment_change)

        self.recon_start = QSpinBox()
        self.recon_start.setMaximumHeight(28)
        self.recon_start.setMaximumWidth(80)
        self.recon_start.setMinimum(0)
        self.recon_start.setMaximum(self.num_frames-1)
        self.recon_start.valueChanged.connect(self.on_recon_time_change)
        self.recon_start.editingFinished.connect(self.on_recon_time_commit)

        self.recon_end = QSpinBox()
        self.recon_end.setMaximumHeight(28)
        self.recon_end.setMaximumWidth(80)
        self.recon_end.setMinimum(0)
        self.recon_end.setMaximum(self.num_frames-1)
        self.recon_end.valueChanged.connect(self.on_recon_time_change)
        self.recon_end.editingFinished.connect(self.on_recon_time_commit)

        self.recon_total_label = QLabel('Reconstruction Time:')
        self.recon_total_display = QSpinBox()
        self.recon_total_display.setMaximumHeight(28)
        self.recon_total_display.setMaximumWidth(80)
        self.recon_total_display.setMinimum(0)
        self.recon_total_display.setMaximum(self.num_frames)
        self.recon_total_display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.recon_total_display.setReadOnly(True)
        self.recon_total_display.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.recon_total_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.downsample_ratio_label = QLabel('Downsample ratio:')
        self.downsample_ratio_box = QSpinBox()
        self.downsample_ratio_box.setMaximumHeight(28)
        self.downsample_ratio_box.setMaximumWidth(100)
        self.downsample_ratio_box.setMinimum(1)
        self.downsample_ratio_box.setMaximum(100)
        self.downsample_ratio_box.setSingleStep(1)
        self.downsample_ratio_box.setValue(1)

        self.run_downsample_button = QPushButton('Run downsample')
        self.run_downsample_button.clicked.connect(self.on_run_downsample)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("white")
        self.combo.addItem("green_screen")
        self.combo.addItem("popup")
        self.combo.addItem("layered")
        self.combo.currentTextChanged.connect(self.set_viz_mode)

        self.save_visualization_checkbox = QCheckBox(self)
        self.save_visualization_checkbox.toggled.connect(self.on_save_visualization_toggle)
        self.save_visualization_checkbox.setChecked(False)
        self.save_visualization = False

        # Radio buttons for type of interactions
        self.curr_interaction = 'Click'
        self.interaction_group = QButtonGroup()
        self.radio_fbrs = QRadioButton('Click')
        self.radio_s2m = QRadioButton('Scribble')
        self.radio_free = QRadioButton('Free')
        self.interaction_group.addButton(self.radio_fbrs)
        self.interaction_group.addButton(self.radio_s2m)
        self.interaction_group.addButton(self.radio_free)
        self.radio_fbrs.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggled.connect(self.interaction_radio_clicked)
        self.radio_free.toggled.connect(self.interaction_radio_clicked)
        self.radio_fbrs.toggle()

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True) # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        # Minimap -> Also a QLabel
        self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding)
        self.minimap.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.minimap.setMinimumSize(100, 100)

        # Zoom-in buttons
        self.zoom_p_button = QPushButton('Zoom +')
        self.zoom_p_button.clicked.connect(self.on_zoom_plus)
        self.zoom_m_button = QPushButton('Zoom -')
        self.zoom_m_button.clicked.connect(self.on_zoom_minus)

        # Parameters setting
        self.clear_mem_button = QPushButton('Clear memory')
        self.clear_mem_button.clicked.connect(self.on_clear_memory)

        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge('GPU mem. (all processes, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge('GPU mem. (used by torch, w/o caching)')

        self.update_memory_size()
        self.update_gpu_usage()

        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(1, 100, 'Min. working memory frames', 
                                                        callback=self.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(2, 100, 'Max. working memory frames', 
                                                        callback=self.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(1000, 100000, 
                                                        'Max. long-term memory size', step=1000, callback=self.update_config)
        self.num_prototypes_box, self.num_prototypes_box_layout = create_parameter_box(32, 1280, 
                                                        'Number of prototypes', step=32, callback=self.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(1, 100, 'Memory frame every (r)', 
                                                        callback=self.update_config)

        self.work_mem_min.setValue(self.processor.memory.min_mt_frames)
        self.work_mem_max.setValue(self.processor.memory.max_mt_frames)
        self.long_mem_max.setValue(self.processor.memory.max_long_elements)
        self.num_prototypes_box.setValue(self.processor.memory.num_prototypes)
        self.mem_every_box.setValue(self.processor.mem_every)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(self.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(self.on_import_layer)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # navigator
        navi = QHBoxLayout()

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_recon_box = QHBoxLayout()
        interact_downsample_box = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interact_topbox.addWidget(self.lcd)
        interact_topbox.addWidget(self.play_button)
        interact_topbox.addWidget(self.radio_s2m)
        interact_topbox.addWidget(self.radio_fbrs)
        interact_topbox.addWidget(self.radio_free)
        interact_topbox.addWidget(self.reset_button)
        interact_topbox.addWidget(self.reset_object_button)
        interact_botbox.addWidget(QLabel('Current Object ID:'))
        interact_botbox.addWidget(self.object_dial)
        interact_botbox.addWidget(self.brush_label)
        interact_botbox.addWidget(self.brush_slider)
        interact_recon_box.addWidget(QLabel('Segment:'))
        interact_recon_box.addWidget(self.recon_segment)
        interact_recon_box.addWidget(QLabel('Recon start:'))
        interact_recon_box.addWidget(self.recon_start)
        interact_recon_box.addWidget(QLabel('Recon end:'))
        interact_recon_box.addWidget(self.recon_end)
        interact_recon_box.addWidget(self.recon_total_label)
        interact_recon_box.addWidget(self.recon_total_display)
        interact_recon_box.addWidget(self.downsample_ratio_label)
        interact_recon_box.addWidget(self.downsample_ratio_box)
        interact_downsample_box.addStretch(1)
        interact_downsample_box.addWidget(self.run_downsample_button)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        interact_subbox.addLayout(interact_recon_box)
        interact_subbox.addLayout(interact_downsample_box)
        navi.addLayout(interact_subbox)

        apply_fixed_size_policy = lambda x: x.setSizePolicy(QSizePolicy.Policy.Fixed, 
                                                            QSizePolicy.Policy.Fixed)
        apply_to_all_children_widget(interact_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(interact_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        navi.addStretch(1)
        overlay_subbox = QVBoxLayout()
        overlay_topbox = QHBoxLayout()
        overlay_botbox = QHBoxLayout()
        overlay_botbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        overlay_topbox.addWidget(QLabel('Overlay Mode'))
        overlay_topbox.addWidget(self.combo)
        overlay_topbox.addWidget(QLabel('Save overlay during propagation'))
        overlay_topbox.addWidget(self.save_visualization_checkbox)
        overlay_botbox.addWidget(self.export_button)
        overlay_subbox.addLayout(overlay_topbox)
        overlay_subbox.addLayout(overlay_botbox)
        navi.addLayout(overlay_subbox)
        apply_to_all_children_widget(overlay_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(overlay_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        navi.addWidget(self.commit_button)
        navi.addWidget(self.forward_run_button)
        navi.addWidget(self.backward_run_button)
        
        # Drawing area, main canvas and minimap
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # Minimap area
        minimap_area = QVBoxLayout()
        minimap_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        mini_label = QLabel('Minimap')
        mini_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        minimap_area.addWidget(mini_label)

        # Minimap zooming
        minimap_ctrl = QHBoxLayout()
        minimap_ctrl.setAlignment(Qt.AlignmentFlag.AlignTop)
        minimap_ctrl.addWidget(self.zoom_p_button)
        minimap_ctrl.addWidget(self.zoom_m_button)
        minimap_area.addLayout(minimap_ctrl)
        minimap_area.addWidget(self.minimap)

        # Parameters 
        minimap_area.addLayout(self.work_mem_gauge_layout)
        minimap_area.addLayout(self.long_mem_gauge_layout)
        minimap_area.addLayout(self.gpu_mem_gauge_layout)
        minimap_area.addLayout(self.torch_mem_gauge_layout)
        minimap_area.addWidget(self.clear_mem_button)
        minimap_area.addLayout(self.work_mem_min_layout)
        minimap_area.addLayout(self.work_mem_max_layout)
        minimap_area.addLayout(self.long_mem_max_layout)
        minimap_area.addLayout(self.num_prototypes_box_layout)
        minimap_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        minimap_area.addLayout(import_area)

        # console
        minimap_area.addWidget(self.console)

        draw_area.addLayout(minimap_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(self.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # current frame info
        self.curr_frame_dirty = False
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8) 
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros((self.num_objects, self.height, self.width), dtype=torch.float).to(self.device)

        # initialize visualization
        self.viz_mode = 'davis'
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.cursur = 0
        self.on_showing = None

        # Zoom parameters
        self.zoom_pixels = 150
        
        # initialize action
        self.interaction = None
        self.pressed = False
        self.right_click = False
        self.current_object = 1
        self.object_dial.setValue(self.current_object)
        self.last_ex = self.last_ey = 0

        self.propagating = False

        # Objects shortcuts
        for i in range(1, self.num_objects+1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(self.hit_number_key, i))
            QShortcut(QKeySequence(f"Ctrl+{i}"), self).activated.connect(functools.partial(self.hit_number_key, i))

        # <- and -> shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.on_next_frame)

        self.interacted_prob = None
        self.overlay_layer = None
        self.overlay_layer_torch = None

        # the object id used for popup/layered overlay
        self.vis_target_objects = [1]
        # try to load the default overlay
        self._try_load_layer('./docs/ECCV-logo.png')

        self._init_recon_times()
        self._init_scene_times()
        self._update_scene_downsample_controls()
        self.load_current_image_mask()
        self.show_current_frame()
        self.show()

        self.console_push_text('Initialized.')
        self.initialized = True

    def resizeEvent(self, event):
        self.show_current_frame()

    def console_push_text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text+'\n')

    def interaction_radio_clicked(self, event):
        self.last_interaction = self.curr_interaction
        if self.radio_s2m.isChecked():
            self.curr_interaction = 'Scribble'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_fbrs.isChecked():
            self.curr_interaction = 'Click'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_free.isChecked():
            self.brush_slider.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = 'Free'
        if self.curr_interaction == 'Scribble':
            self.commit_button.setEnabled(True)
        else:
            self.commit_button.setEnabled(False)

    def load_current_image_mask(self, no_mask=False):
        self.current_image = self.res_man.get_image(self.cursur)
        self.current_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.cursur)
            if loaded_mask is None:
                h, w = self.current_image.shape[:2]
                self.current_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                self.current_mask = loaded_mask.copy()
            self.current_prob = None

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)

    def compose_current_im(self):
        self.viz = get_visualization(self.viz_mode, self.current_image, self.current_mask, 
                            self.overlay_layer, self.vis_target_objects)

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        vis_map = self.vis_map
        vis_alpha = self.vis_alpha
        brush_vis_map = self.brush_vis_map
        brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz*(1-vis_alpha) + vis_map*vis_alpha
        self.viz_with_stroke = self.viz_with_stroke*(1-brush_vis_alpha) + brush_vis_map*brush_vis_alpha
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_minimap(self):
        ex, ey = self.last_ex, self.last_ey
        r = self.zoom_pixels//2
        ex = int(round(max(r, min(self.width-r, ex))))
        ey = int(round(max(r, min(self.height-r, ey))))

        patch = self.viz_with_stroke[ey-r:ey+r, ex-r:ex+r, :].astype(np.uint8)

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.minimap.setPixmap(QPixmap(qImg.scaled(self.minimap.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)))

    def update_current_image_fast(self):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        self.viz = get_visualization_torch(self.viz_mode, self.current_image_torch_no_norm, 
                    self.current_prob, self.overlay_layer_torch, self.vis_target_objects)
        if self.save_visualization:
            self.res_man.save_visualization(self.cursur, self.viz)

        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        qImg = QImage(self.viz.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)))

    def show_current_frame(self, fast=False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast()
        else:
            self.compose_current_im()
            self.update_interact_vis()
            self.update_minimap()

        self.lcd.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames-1))
        self.tl_slider.setValue(self.cursur)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh/oh
        w_ratio = nw/ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh/dominate_ratio, nw/dominate_ratio
        x -= (fw-ow)/2
        y -= (fh-oh)/2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = (
            (x < 0) or
            (y < 0) or
            (x > self.width-1) or 
            (y > self.height-1)
        )

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.width-1, x))
        y = max(0, min(self.height-1, y))

        return x, y

    def clear_visualization(self):
        self.vis_map.fill(0)
        self.vis_alpha.fill(0)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        self.interaction = None
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def set_viz_mode(self):
        self.viz_mode = self.combo.currentText()
        self.show_current_frame()

    def _is_scene_mode(self):
        return self.current_object == 0

    def _show_scene_mode_popup(self):
        QMessageBox.information(self, 'Reminder', self.scene_mode_popup_message)

    def _update_scene_downsample_controls(self):
        show = self._is_scene_mode()
        self.downsample_ratio_label.setVisible(show)
        self.downsample_ratio_box.setVisible(show)
        self.run_downsample_button.setVisible(show)

    def _scene_indices_from_segments(self):
        if not self.scene_segments:
            return []

        scene_index_set = set()
        for segment in self.scene_segments.values():
            start = int(segment.get('start', 0))
            end = int(segment.get('end', 0))
            lo = max(0, min(start, end))
            hi = min(self.num_frames - 1, max(start, end))
            scene_index_set.update(range(lo, hi + 1))

        return sorted(scene_index_set)

    def _reload_session_with_image_dir(self, image_dir):
        self.pause_propagation()
        self.res_man.reload_image_dir(image_dir, clear_masks=True)

        self.num_frames = len(self.res_man)
        self.height, self.width = self.res_man.h, self.res_man.w
        self.cursur = 0
        self.curr_frame_dirty = False
        self.interacted_prob = None
        self.interaction = None
        self.pressed = False
        self.right_click = False

        self.processor.clear_memory()
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros((self.num_objects, self.height, self.width), dtype=torch.float).to(self.device)

        self.viz = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.viz_with_stroke = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.last_ex = min(max(self.last_ex, 0), self.width - 1)
        self.last_ey = min(max(self.last_ey, 0), self.height - 1)

        self.tl_slider.blockSignals(True)
        self.tl_slider.setMaximum(self.num_frames - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.blockSignals(False)
        self.lcd.setText('{: 3d} / {: 3d}'.format(0, self.num_frames-1))

        self.recon_start.setMaximum(self.num_frames - 1)
        self.recon_end.setMaximum(self.num_frames - 1)
        self.recon_total_display.setMaximum(self.num_frames)

        self.scene_segments = {1: {'start': 0, 'end': self.num_frames - 1}}
        self.current_scene_segment = 1
        self.recon_segments = {
            obj_id: {1: {'start': 0, 'end': self.num_frames - 1}}
            for obj_id in range(1, self.num_objects+1)
        }
        self.current_segment_per_object = {
            obj_id: 1 for obj_id in range(1, self.num_objects+1)
        }

        # After downsample/reload, go back to object segmentation mode.
        self.current_object = 1 if self.num_objects >= 1 else 0
        self.object_dial.blockSignals(True)
        self.object_dial.setValue(self.current_object)
        self.object_dial.blockSignals(False)
        self._update_scene_downsample_controls()
        self._sync_recon_controls()

        self.load_current_image_mask()
        self.show_current_frame()

    def _reset_state_for_object_switch(self):
        self.res_man.clear_all_masks()
        self.current_mask.fill(0)
        self.current_prob = None
        self.interacted_prob = None
        self.curr_frame_dirty = False
        self.current_image_torch = None
        self.processor.clear_memory()
        self.reset_this_interaction()
        self.load_current_image_mask()

    def _save_object_specific_mask(self):
        if self.current_object <= 0:
            return
        mask_dir = self._object_mask_dir(self.current_object)
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f'{self.res_man.names[self.cursur]}.png')
        object_mask = np.where(self.current_mask == self.current_object, self.current_object, 0).astype(np.uint8)
        Image.fromarray(object_mask).save(mask_path)

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.cursur, self.current_mask)
        self._save_object_specific_mask()

    def tl_slide(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        if not self.propagating:
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.cursur = self.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def brush_slide(self):
        self.brush_size = self.brush_slider.value()
        self.brush_label.setText('Brush size (in free mode): %d' % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_next_frame
            self.backward_run_button.setEnabled(False)
            self.forward_run_button.setText('Pause Propagation')
            self.on_propagation()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_prev_frame
            self.forward_run_button.setEnabled(False)
            self.backward_run_button.setText('Pause Propagation')
            self.on_propagation()

    def on_pause(self):
        self.propagating = False
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_mem_button.setEnabled(True)
        self.forward_run_button.setText('Forward Propagate')
        self.backward_run_button.setText('Backward Propagate')
        self.console_push_text('Propagation stopped.')

    def on_propagation(self):
        # start to propagate
        self.load_current_torch_image_mask()
        self.show_current_frame(fast=True)

        self.console_push_text('Propagation started.')
        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        # clear
        self.interacted_prob = None
        self.reset_this_interaction()

        self.propagating = True
        self.clear_mem_button.setEnabled(False)
        # propagate till the end
        while self.propagating:
            self.propagate_fn()

            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)

            self.current_prob = self.processor.step(self.current_image_torch)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

            self.save_current_mask()
            self.show_current_frame(fast=True)

            self.update_memory_size()
            QApplication.processEvents()

            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
        self.tl_slide()
        QApplication.processEvents()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            return
        self.complete_interaction()
        self.update_interacted_mask()

    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1)
        self.tl_slider.setValue(self.cursur)

    def on_next_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.tl_slider.setValue(self.cursur)

    def on_play_video_timer(self):
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play Video')
        else:
            self.timer.start(1000 // 30)
            self.play_button.setText('Stop Video')

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = f"{self.config['workspace']}/visualization/"
        save_folder = self.config['workspace']
        if os.path.exists(image_folder):
            # Sorted so frames will be in order
            self.console_push_text(f'Exporting visualization to {self.config["workspace"]}/visualization.mp4')
            images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape
            # 10 is the FPS -- change if needed
            video = cv2.VideoWriter(f"{save_folder}/visualization.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))
            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))
            video.release()
            self.console_push_text(f'Visualization exported to {self.config["workspace"]}/visualization.mp4')
        else:
            self.console_push_text(f'No visualization images found in {image_folder}')

    def on_object_dial_change(self):
        object_id = self.object_dial.value()
        self.hit_number_key(object_id)

    def on_reset_mask(self):
        self.current_mask.fill(0)
        if self.current_prob is not None:
            self.current_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        if self.current_object < 1 or self.current_object > self.num_objects:
            return
        self.current_mask[self.current_mask == self.current_object] = 0
        self.current_prob = None
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()
        self.console_push_text(f'Cleared object {self.current_object} on current frame.')

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def set_navi_enable(self, boolean):
        self.zoom_p_button.setEnabled(boolean)
        self.zoom_m_button.setEnabled(boolean)
        self.run_button.setEnabled(boolean)
        self.tl_slider.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.export_button.setEnabled(boolean)
        self.lcd.setEnabled(boolean)

    def hit_number_key(self, number):
        if number == self.current_object:
            return
        self.current_object = number
        self.object_dial.setValue(number)
        self._reset_state_for_object_switch()
        self._sync_recon_controls()
        self._write_current_mode_metadata()
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()
        self.console_push_text(f'Current object changed to {number}.')
        if self._is_scene_mode():
            self.console_push_text(self.scene_mode_message)
            self._show_scene_mode_popup()
        self._update_scene_downsample_controls()
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.show_current_frame()

    def on_run_downsample(self):
        if not self._is_scene_mode():
            self.console_push_text('Downsample is only available when Current Object ID is 0.')
            return

        self.run_downsample_button.setEnabled(False)
        try:
            ratio = int(self.downsample_ratio_box.value())

            scene_indices = self._scene_indices_from_segments()
            if not scene_indices:
                self.console_push_text('No scene timesteps selected.')
                return

            downsampled_indices = scene_indices[::ratio]
            if not downsampled_indices:
                self.console_push_text('No frames selected after timestep downsample.')
                print('Downsample: no frames selected after timestep downsample.', flush=True)
                return

            out_dir = os.path.join(self._scene_dir(), 'downsample')
            os.makedirs(out_dir, exist_ok=True)
            for name in os.listdir(out_dir):
                out_path = os.path.join(out_dir, name)
                if os.path.isfile(out_path):
                    os.remove(out_path)

            total_to_save = len(downsampled_indices)
            print(
                f'Downsample started: ratio={ratio}, input_frames={len(scene_indices)}, '
                f'output_frames={total_to_save}, output={out_dir}',
                flush=True,
            )

            saved = 0
            for idx in downsampled_indices:
                image = self.res_man.get_image(idx)
                out_name = self.res_man.image_files[idx]
                Image.fromarray(image).save(os.path.join(out_dir, out_name))
                saved += 1
                if saved % 20 == 0:
                    QApplication.processEvents()
                if saved % 50 == 0 or saved == total_to_save:
                    print(f'Downsample progress: {saved}/{total_to_save}', flush=True)

            self.console_push_text(
                f'Timestep downsample done. ratio={ratio:g}, input_frames={len(scene_indices)}, '
                f'output_frames={saved}, output={out_dir}'
            )
            print(
                f'Downsample finished: saved {saved}/{total_to_save} frames to {out_dir}',
                flush=True,
            )
            self._reload_session_with_image_dir(out_dir)
            self.console_push_text(f'Reloaded downsampled images for segmentation from {out_dir}')
        except Exception as exc:
            self.console_push_text(f'Downsample failed: {exc}')
            print(f'Downsample failed: {exc}', flush=True)
        finally:
            self.run_downsample_button.setEnabled(True)

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(self.brush_vis_map, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, color_map[self.current_object], thickness=-1)
        self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1)

    def on_mouse_press(self, event):
        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            return

        # mid-click
        if (event.button() == Qt.MouseButton.MiddleButton):
            ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
            target_object = self.current_mask[int(ey),int(ex)]
            if target_object in self.vis_target_objects:
                self.vis_target_objects.remove(target_object)
            else:
                self.vis_target_objects.append(target_object)
            self.console_push_text(f'Target objects for visualization changed to {self.vis_target_objects}')
            self.show_current_frame()
            return

        if self._is_scene_mode():
            self.console_push_text(self.scene_mode_message)
            return

        self.right_click = (event.button() == Qt.MouseButton.RightButton)
        self.pressed = True

        h, w = self.height, self.width

        self.load_current_torch_image_mask()
        image = self.current_image_torch

        last_interaction = self.interaction
        new_interaction = None
        if self.curr_interaction == 'Scribble':
            if last_interaction is None or type(last_interaction) != ScribbleInteraction:
                self.complete_interaction()
                new_interaction = ScribbleInteraction(image, torch.from_numpy(self.current_mask).float().to(self.device), 
                        (h, w), self.s2m_controller, self.num_objects)
        elif self.curr_interaction == 'Free':
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                new_interaction = FreeInteraction(image, self.current_mask, (h, w), 
                        self.num_objects)
                new_interaction.set_size(self.brush_size)
        elif self.curr_interaction == 'Click':
            if (last_interaction is None or type(last_interaction) != ClickInteraction 
                    or last_interaction.tar_obj != self.current_object):
                self.complete_interaction()
                self.fbrs_controller.unanchor()
                new_interaction = ClickInteraction(image, self.current_prob, (h, w), 
                            self.fbrs_controller, self.current_object)

        if new_interaction is not None:
            self.interaction = new_interaction

        # Just motion it as the first step
        self.on_mouse_motion(event)

    def on_mouse_motion(self, event):
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                obj = 0 if self.right_click else self.current_object
                self.vis_map, self.vis_alpha = self.interaction.push_point(
                    ex, ey, obj, (self.vis_map, self.vis_alpha)
                )
        self.update_interact_vis()
        self.update_minimap()

    def update_interacted_mask(self):
        self.current_prob = self.interacted_prob
        self.current_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.show_current_frame()
        self.save_current_mask()
        self.curr_frame_dirty = False

    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interaction = None

    def on_mouse_release(self, event):
        if self._is_scene_mode():
            self.pressed = False
            self.right_click = False
            return

        if not self.pressed:
            # this can happen when the initial press is out-of-bound
            return

        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())

        self.console_push_text('%s interaction at frame %d.' % (self.curr_interaction, self.cursur))
        interaction = self.interaction

        if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
            self.on_mouse_motion(event)
            interaction.end_path()
            if self.curr_interaction == 'Free':
                self.clear_visualization()
        elif self.curr_interaction == 'Click':
            ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
            self.vis_map, self.vis_alpha = interaction.push_point(ex, ey,
                self.right_click, (self.vis_map, self.vis_alpha))

        self.interacted_prob = interaction.predict().to(self.device)
        self.update_interacted_mask()
        self.update_gpu_usage()

        self.pressed = self.right_click = False

    def wheelEvent(self, event):
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        if self.curr_interaction == 'Free':
            self.brush_slider.setValue(self.brush_slider.value() + event.angleDelta().y()//30)
        self.clear_brush()
        self.vis_brush(ex, ey)
        self.update_interact_vis()
        self.update_minimap()

    def update_gpu_usage(self):
        if self.device.type == 'cuda':
            info = torch.cuda.mem_get_info()
        elif self.device.type == 'mps':
            info = (0, mps.current_allocated_memory()) # NOTE: torch.mps does not support accessing free and total memory
        else:
            info = (0, 0)
        global_free, global_total = info
        global_free /= (2**30)
        global_total /= (2**30)
        global_used = global_total - global_free

        self.gpu_mem_gauge.setFormat(f'{global_used:.01f} GB / {global_total:.01f} GB')
        self.gpu_mem_gauge.setValue(round(global_used/global_total*100))

        used_by_torch = torch.cuda.max_memory_allocated() / (2**20)
        self.torch_mem_gauge.setFormat(f'{used_by_torch:.0f} MB / {global_total:.01f} GB')
        self.torch_mem_gauge.setValue(round(used_by_torch/global_total*100/1024))

    def on_gpu_timer(self):
        self.update_gpu_usage()

    def update_memory_size(self):
        try:
            max_work_elements = self.processor.memory.max_work_elements
            max_long_elements = self.processor.memory.max_long_elements

            curr_work_elements = self.processor.memory.work_mem.size
            curr_long_elements = self.processor.memory.long_mem.size

            self.work_mem_gauge.setFormat(f'{curr_work_elements} / {max_work_elements}')
            self.work_mem_gauge.setValue(round(curr_work_elements/max_work_elements*100))

            self.long_mem_gauge.setFormat(f'{curr_long_elements} / {max_long_elements}')
            self.long_mem_gauge.setValue(round(curr_long_elements/max_long_elements*100))

        except AttributeError:
            self.work_mem_gauge.setFormat('Unknown')
            self.long_mem_gauge.setFormat('Unknown')
            self.work_mem_gauge.setValue(0)
            self.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.work_mem_min.setValue(min(self.work_mem_min.value(), self.work_mem_max.value()-1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.work_mem_max.setValue(max(self.work_mem_max.value(), self.work_mem_min.value()+1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            self.config['min_mid_term_frames'] = self.work_mem_min.value()
            self.config['max_mid_term_frames'] = self.work_mem_max.value()
            self.config['max_long_term_elements'] = self.long_mem_max.value()
            self.config['num_prototypes'] = self.num_prototypes_box.value()
            self.config['mem_every'] = self.mem_every_box.value()

            self.processor.update_config(self.config)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            mps.empty_cache()
        self.update_gpu_usage()
        self.update_memory_size()

    def _open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, prompt, "", "Image files (*)", options=options)
        return file_name

    def _init_recon_times(self):
        self.frame_id_width = self._compute_frame_id_width()
        self.recon_segments = {
            obj_id: {
                1: {
                    'start': 0,
                    'end': self.num_frames-1,
                }
            }
            for obj_id in range(1, self.num_objects+1)
        }
        self.current_segment_per_object = {
            obj_id: 1 for obj_id in range(1, self.num_objects+1)
        }
        self._load_recon_times()
        self._sync_recon_controls()

    def _compute_frame_id_width(self):
        if all(name.isdigit() for name in self.res_man.names):
            return max(len(name) for name in self.res_man.names)
        return 6

    def _frame_id_str(self, index):
        return f'{index:0{self.frame_id_width}d}'

    def _scene_dir(self):
        return os.path.join(self.output_path, 'scene')

    def _scene_json_path(self):
        return os.path.join(self._scene_dir(), 'scene.json')

    def _object_dir(self, obj_id):
        return os.path.join(self.output_path, f'object_{obj_id}')

    def _object_json_path(self, obj_id):
        return os.path.join(self._object_dir(obj_id), 'object.json')

    def _object_mask_dir(self, obj_id):
        return os.path.join(self._object_dir(obj_id), 'masks')

    def _legacy_recon_json_path(self):
        return os.path.join(self.config['workspace'], 'object.json')

    def _legacy_scene_json_path(self):
        return os.path.join(self.config['workspace'], 'scene.json')

    def _parse_frame_id(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _read_json_file(self, json_path):
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception as exc:
            self.console_push_text(f'Failed to load {json_path}: {exc}')
            return None

    def _parse_object_entries(self, payload):
        if isinstance(payload, dict) and isinstance(payload.get('objects'), list):
            return payload.get('objects', [])
        if isinstance(payload, list):
            return payload
        return []

    def _parse_object_segments(self, entry):
        segments_data = entry.get('segments')
        segments = {}
        if isinstance(segments_data, list) and segments_data:
            for idx, segment in enumerate(segments_data, start=1):
                seg_id = segment.get('segment id', idx)
                if isinstance(seg_id, str) and seg_id.isdigit():
                    seg_id = int(seg_id)
                if not isinstance(seg_id, int):
                    continue

                start_id = segment.get('start id')
                end_id = segment.get('end id')
                start_index = self._parse_frame_id(start_id)
                end_index = self._parse_frame_id(end_id)
                if start_index is None or end_index is None:
                    continue
                segments[seg_id] = {
                    'start': min(max(start_index, 0), self.num_frames-1),
                    'end': min(max(end_index, 0), self.num_frames-1),
                }
        else:
            start_id = entry.get('start id')
            end_id = entry.get('end id')
            start_index = self._parse_frame_id(start_id)
            end_index = self._parse_frame_id(end_id)
            if start_index is None or end_index is None:
                return {}
            segments[1] = {
                'start': min(max(start_index, 0), self.num_frames-1),
                'end': min(max(end_index, 0), self.num_frames-1),
            }
        return segments

    def _extract_segments_for_object(self, entries, obj_id):
        for entry in entries:
            try:
                entry_obj_id = int(entry.get('object id'))
            except Exception:
                continue
            if entry_obj_id != obj_id:
                continue
            return self._parse_object_segments(entry)
        return {}

    def _load_recon_times(self):
        legacy_payload = None
        legacy_entries = []
        for obj_id in range(1, self.num_objects+1):
            payload = self._read_json_file(self._object_json_path(obj_id))
            if payload is None:
                if legacy_payload is None:
                    legacy_payload = self._read_json_file(self._legacy_recon_json_path())
                    legacy_entries = self._parse_object_entries(legacy_payload)
                entries = legacy_entries
            else:
                entries = self._parse_object_entries(payload)

            segments = self._extract_segments_for_object(entries, obj_id)
            if segments:
                self.recon_segments[obj_id] = segments
                self.current_segment_per_object[obj_id] = sorted(segments.keys())[0]

    def _init_scene_times(self):
        self.scene_segments = {
            1: {
                'start': 0,
                'end': self.num_frames-1,
            }
        }
        self.current_scene_segment = 1
        self._load_scene_times()

    def _load_scene_times(self):
        payload = self._read_json_file(self._scene_json_path())
        if payload is None:
            payload = self._read_json_file(self._legacy_scene_json_path())
        if payload is None:
            return

        segments_data = None
        if isinstance(payload, dict):
            segments_data = payload.get('segments')
        elif isinstance(payload, list):
            segments_data = payload

        segments = {}
        if isinstance(segments_data, list) and segments_data:
            for idx, segment in enumerate(segments_data, start=1):
                seg_id = segment.get('segment id', idx)
                if isinstance(seg_id, str) and seg_id.isdigit():
                    seg_id = int(seg_id)
                if not isinstance(seg_id, int):
                    continue
                start_index = self._parse_frame_id(segment.get('start id'))
                end_index = self._parse_frame_id(segment.get('end id'))
                if start_index is None or end_index is None:
                    continue
                segments[seg_id] = {
                    'start': min(max(start_index, 0), self.num_frames-1),
                    'end': min(max(end_index, 0), self.num_frames-1),
                }

        if segments:
            self.scene_segments = segments
            self.current_scene_segment = sorted(segments.keys())[0]

    def _ensure_scene_segment(self, segment_id):
        if segment_id not in self.scene_segments:
            self.scene_segments[segment_id] = {
                'start': self.cursur,
                'end': self.cursur,
            }

    def _write_scene_times(self):
        segments = []
        for seg_id in sorted(self.scene_segments.keys()):
            recon = self.scene_segments[seg_id]
            segments.append({
                'segment id': seg_id,
                'start id': self._frame_id_str(recon['start']),
                'end id': self._frame_id_str(recon['end']),
            })

        payload = {'segments': segments}
        scene_path = self._scene_json_path()
        os.makedirs(os.path.dirname(scene_path), exist_ok=True)
        try:
            with open(scene_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
            self.console_push_text(f'Scene reconstruction time saved to {scene_path}')
        except Exception as exc:
            self.console_push_text(f'Failed to write {scene_path}: {exc}')

    def _ensure_segment(self, obj_id, segment_id):
        segments = self.recon_segments.setdefault(obj_id, {})
        if segment_id not in segments:
            segments[segment_id] = {
                'start': self.cursur,
                'end': self.cursur,
            }

    def _get_active_segments(self):
        if self._is_scene_mode():
            return self.scene_segments
        return self.recon_segments.get(self.current_object, {})

    def _compute_total_selected_timesteps(self, segments):
        if not segments:
            return 0
        intervals = []
        for segment in segments.values():
            start = int(segment.get('start', 0))
            end = int(segment.get('end', 0))
            intervals.append((min(start, end), max(start, end)))

        intervals.sort()
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return sum(end - start + 1 for start, end in merged)

    def _update_recon_button_text(self):
        total_steps = self._compute_total_selected_timesteps(self._get_active_segments())
        self.recon_total_display.setValue(total_steps)

    def _sync_recon_controls(self):
        if self._is_scene_mode():
            seg_id = self.current_scene_segment
            if seg_id not in self.scene_segments and self.scene_segments:
                seg_id = sorted(self.scene_segments.keys())[0]
                self.current_scene_segment = seg_id
            self._ensure_scene_segment(seg_id)
            recon = self.scene_segments[seg_id]
        else:
            obj_id = self.current_object
            seg_id = self.current_segment_per_object.get(obj_id, 1)
            segments = self.recon_segments.get(obj_id, {})
            if seg_id not in segments and segments:
                seg_id = sorted(segments.keys())[0]
                self.current_segment_per_object[obj_id] = seg_id
            self._ensure_segment(obj_id, seg_id)
            recon = self.recon_segments[obj_id][seg_id]

        self.recon_segment.blockSignals(True)
        self.recon_start.blockSignals(True)
        self.recon_end.blockSignals(True)
        self.recon_segment.setValue(seg_id)
        self.recon_start.setValue(recon['start'])
        self.recon_end.setValue(recon['end'])
        self.recon_segment.blockSignals(False)
        self.recon_start.blockSignals(False)
        self.recon_end.blockSignals(False)
        self._update_recon_button_text()

    def _write_recon_times(self, obj_id):
        segments = self.recon_segments.get(obj_id, {})
        segment_entries = []
        for seg_id in sorted(segments.keys()):
            recon = segments[seg_id]
            segment_entries.append({
                'segment id': seg_id,
                'start id': self._frame_id_str(recon['start']),
                'end id': self._frame_id_str(recon['end']),
            })

        payload = {'objects': [{'object id': obj_id, 'segments': segment_entries}]}
        recon_path = self._object_json_path(obj_id)
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)
        try:
            with open(recon_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
            self.console_push_text(f'Reconstruction time saved to {recon_path}')
        except Exception as exc:
            self.console_push_text(f'Failed to write {recon_path}: {exc}')

    def _write_current_mode_metadata(self):
        if self._is_scene_mode():
            self._write_scene_times()
        else:
            self._write_recon_times(self.current_object)
        self._update_recon_button_text()

    def on_recon_time_change(self, *_):
        if not hasattr(self, 'recon_segments'):
            return
        seg_id = self.recon_segment.value()
        start_index = self.recon_start.value()
        end_index = self.recon_end.value()
        if end_index < start_index:
            if self.recon_end.hasFocus():
                return
            end_index = start_index
            self.recon_end.setValue(end_index)
        if self._is_scene_mode():
            self.current_scene_segment = seg_id
            self._ensure_scene_segment(seg_id)
            self.scene_segments[seg_id] = {
                'start': start_index,
                'end': end_index,
            }
        else:
            obj_id = self.current_object
            self._ensure_segment(obj_id, seg_id)
            self.recon_segments[obj_id][seg_id] = {
                'start': start_index,
                'end': end_index,
            }
        self._write_current_mode_metadata()

    def on_recon_time_save(self):
        self.on_recon_time_commit()

    def on_recon_time_commit(self):
        if not hasattr(self, 'recon_segments'):
            return
        seg_id = self.recon_segment.value()
        start_index = self.recon_start.value()
        end_index = self.recon_end.value()
        if end_index < start_index:
            end_index = start_index
            self.recon_end.setValue(end_index)
        if self._is_scene_mode():
            self.current_scene_segment = seg_id
            self._ensure_scene_segment(seg_id)
            self.scene_segments[seg_id] = {
                'start': start_index,
                'end': end_index,
            }
        else:
            obj_id = self.current_object
            self._ensure_segment(obj_id, seg_id)
            self.recon_segments[obj_id][seg_id] = {
                'start': start_index,
                'end': end_index,
            }
        self._write_current_mode_metadata()

    def on_recon_segment_change(self, *_):
        if not hasattr(self, 'recon_segments'):
            return
        seg_id = self.recon_segment.value()
        if self._is_scene_mode():
            created = seg_id not in self.scene_segments
            self.current_scene_segment = seg_id
            self._ensure_scene_segment(seg_id)
        else:
            obj_id = self.current_object
            self.current_segment_per_object[obj_id] = seg_id
            created = seg_id not in self.recon_segments.get(obj_id, {})
            self._ensure_segment(obj_id, seg_id)
        self._sync_recon_controls()
        if created:
            self._write_current_mode_metadata()

    def on_recon_segment_new(self):
        if not hasattr(self, 'recon_segments'):
            return
        if self._is_scene_mode():
            next_seg_id = max(self.scene_segments.keys(), default=0) + 1
            self.current_scene_segment = next_seg_id
            self.scene_segments[next_seg_id] = {
                'start': self.cursur,
                'end': self.cursur,
            }
        else:
            obj_id = self.current_object
            segments = self.recon_segments.get(obj_id, {})
            next_seg_id = max(segments.keys(), default=0) + 1
            self.current_segment_per_object[obj_id] = next_seg_id
            self.recon_segments[obj_id][next_seg_id] = {
                'start': self.cursur,
                'end': self.cursur,
            }
        self._sync_recon_controls()
        self._write_current_mode_metadata()

    def on_import_mask(self):
        file_name = self._open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.read_external_image(file_name, size=(self.height, self.width))

        shape_condition = (
            (len(mask.shape) == 2) and
            (mask.shape[-1] == self.width) and 
            (mask.shape[-2] == self.height)
        )

        object_condition = (
            mask.max() <= self.num_objects
        )

        if not shape_condition:
            self.console_push_text(f'Expected ({self.height}, {self.width}). Got {mask.shape} instead.')
        elif not object_condition:
            self.console_push_text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.console_push_text(f'Mask file {file_name} loaded.')
            self.current_image_torch = self.current_prob = None
            self.current_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self._open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.read_external_image(file_name, size=(self.height, self.width))

            if layer.shape[-1] == 3:
                layer = np.concatenate([layer, np.ones_like(layer[:,:,0:1])*255], axis=-1)

            condition = (
                (len(layer.shape) == 3) and
                (layer.shape[-1] == 4) and 
                (layer.shape[-2] == self.width) and 
                (layer.shape[-3] == self.height)
            )

            if not condition:
                self.console_push_text(f'Expected ({self.height}, {self.width}, 4). Got {layer.shape}.')
            else:
                self.console_push_text(f'Layer file {file_name} loaded.')
                self.overlay_layer = layer
                self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device)/255
                self.show_current_frame()
        except FileNotFoundError:
            self.console_push_text(f'{file_name} not found.')

    def on_save_visualization_toggle(self):
        self.save_visualization = self.save_visualization_checkbox.isChecked()
