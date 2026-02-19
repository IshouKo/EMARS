#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 15:57
from enum import Enum, unique

# この関数は、ローリングシャッター画像のデブラー（ぼかし除去）推論に使うバッチ（データのまとまり）のテンプレートとなる辞書（dict）を返し
# このテンプレートを使うことで、データの受け渡しや管理を統一的に行うことができる
def get_rs_deblur_inference_batch():
    return {
        "video_name": "NONE",
        "rolling_blur_frame_name": "NONE",
        "events": "NONE",
        "events_for_gs_sharp_frames": "NONE",
        "rolling_blur_frame_color": "NONE",
        "rolling_sharp_frame_color": "NONE",
 
        "rolling_blur_frame_gray": "NONE",
        "rolling_blur_start_time": "NONE",
        "rolling_blur_end_time": "NONE",
        "rolling_blur_exposure_time": "NONE",

        "rs_sharp_pred_frame": "NONE",

        "global_sharp_frames": "NONE",
        # time map
        "rs_sharp_timemap": "NONE", 
        "gs_timemaps": "NONE",   
        # Output: global sharp frame
        "global_sharp_pred_frames": "NONE",
        "global_sharp_pred_frames_differential": "NONE",  # List[] N x B
    }
