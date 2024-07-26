# coding: utf-8

"""
Pipeline of LivePortrait
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image
from .utils.filter import smooth
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper

from .live_portrait_pipeline import LivePortraitPipeline

def execute_frame(self, frame, source):
    # for convenience
    inf_cfg = self.live_portrait_wrapper.inference_cfg
    device = self.live_portrait_wrapper.device
    crop_cfg = self.cropper.crop_cfg

    flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
    flag_drive_crop = inf_cfg.flag_do_crop

    ######## load source input ########
    img_rgb = resize_to_limit(source, inf_cfg.source_max_dim, inf_cfg.source_division)

    crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
    if crop_info is None:
        raise Exception("No face detected in the source image!")
    source_lmk = crop_info['lmk_crop']
    img_crop_256x256 = crop_info['img_crop_256x256']

    if inf_cfg.flag_do_crop:
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
    else:
        img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

    lip_delta_before_animation = None
    # let lip-open scalar to be 0 at first
    if flag_normalize_lip:
        c_d_lip_before_animation = [0.]
        combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
        if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
            lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

    # if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
    #     mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

    frame = resize_to_limit(frame, inf_cfg.source_max_dim, inf_cfg.source_division)
    frame_crop_info = self.cropper.crop_source_image(frame, crop_cfg)
    if frame_crop_info is None:
        raise Exception("No face detected in the source image!")
    frame_lmk = frame_crop_info['lmk_crop']
    frame_img_crop_256x256 = frame_crop_info['img_crop_256x256']

    if flag_drive_crop:
        I_d_0 = self.live_portrait_wrapper.prepare_source(frame_img_crop_256x256)
    else:
        frame_img_crop_256x256 = cv2.resize(frame, (256, 256))
        I_d_0 = self.live_portrait_wrapper.prepare_source(frame_img_crop_256x256)
    first_frame_info = self.live_portrait_wrapper.get_kp_info(I_d_0)

    return x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info

def generate_frame(self, x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info, last_info, frame):
    # for convenience
    inf_cfg = self.live_portrait_wrapper.inference_cfg
    device = self.live_portrait_wrapper.device
    crop_cfg = self.cropper.crop_cfg

    flag_drive_crop = inf_cfg.flag_do_crop
    flag_pasteback = True

    ######## process image ########
    frame = resize_to_limit(frame, inf_cfg.source_max_dim, inf_cfg.source_division)

    frame_crop_info = self.cropper.crop_source_image(frame, crop_cfg)
    if frame_crop_info is None:
        raise Exception("No face detected in the source image!")
    frame_lmk = frame_crop_info['lmk_crop']
    frame_crop_256x256 = frame_crop_info['img_crop_256x256']

    if flag_drive_crop:
        I_d_i = self.live_portrait_wrapper.prepare_source(frame_crop_256x256)
    else:
        frame_crop_256x256 = cv2.resize(frame, (256, 256))  # force to resize to 256x256
        I_d_i = self.live_portrait_wrapper.prepare_source(frame_crop_256x256)

    x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
    R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

    ### hard coded for testing ###
    x_d_i_info['exp'] = last_info['exp'] * 0.1 + x_d_i_info['exp'] * 0.9
    x_d_i_info['scale'] = last_info['scale'] * 0.1 + x_d_i_info['scale'] * 0.9
    R_new = R_s
    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - first_frame_info['exp']) * 1.3
    # scale_new = x_s_info['scale']
    scale_new = x_s_info['scale'] / (x_d_i_info['scale'] / first_frame_info['scale']) ** 0.3
    t_new = x_s_info['t']
    # t_new = x_s_info['t'] + (x_d_i_info['t'] - first_frame_info['t'])
    ### ---------- ---------- ###

    x_d_i_new = scale_new * (x_s_info['kp'] @ R_new + delta_new) + t_new
    if inf_cfg.flag_stitching:
        x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
    if lip_delta_before_animation is not None:
        x_d_i_new += lip_delta_before_animation

    out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
    I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

    if flag_pasteback:
        mask_ori = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        I_p_i_to_ori_blend = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
        return I_p_i_to_ori_blend, x_d_i_info
    else:
        return I_p_i, x_d_i_info

LivePortraitPipeline.execute_frame = execute_frame
LivePortraitPipeline.generate_frame = generate_frame
