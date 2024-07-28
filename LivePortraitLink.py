import io
from PIL import Image
import numpy as np
import time

import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
import src.live_portrait_pipeline_custom
from src.live_portrait_pipeline import LivePortraitPipeline

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)

inference_cfg.flag_do_torch_compile = True

live_portrait_pipeline = LivePortraitPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg
)

source_image_path = "images/test2.jpg"
x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info, last_info = None, None, None, None, None, None, None, None, None

def init(frame):
    global x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info, last_info
      # Set the source image path here
    source_image = Image.open(source_image_path)
    source_image = np.array(source_image)
    x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info = live_portrait_pipeline.execute_frame(frame, source_image)
    last_info = first_frame_info


def inference(frame):
    global last_info
    result, last_info = live_portrait_pipeline.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, first_frame_info, last_info, frame)
    return result

# import torch
# from RobustVideoMatting.model import MattingNetwork
# model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
# model.load_state_dict(torch.load('RobustVideoMatting/rvm_mobilenetv3.pth'))

# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from RobustVideoMatting.inference_utils import VideoReader, VideoWriter

# bgr = torch.tensor([0, 0, 0]).view(3, 1, 1).cuda()  # background.
# rec = [None] * 4                                       # Initial recurrent states.
# downsample_ratio = 0.25                                # Adjust based on your video.

# def RVM(frame):
#     global rec
#     frame = ToTensor()((frame / 255.).astype('float32')).unsqueeze(0).cuda()
#     with torch.no_grad():
#         fgr, pha, *rec = model(frame.cuda(), *rec, downsample_ratio)
#         com = fgr * pha + bgr * (1 - pha)
#         frame = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
#     return frame[0]

from fastapi import WebSocket
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()



@app.post("/init")
async def init_image(file: UploadFile = File(...)):
    img = Image.open(file.file)
    # 在这里进行图像处理
    init(np.array(img))
    return 'success'

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    start = time.time()
    img = Image.open(file.file)
    print('Read time:', time.time() - start)

    # 在这里进行图像处理
    img_array = np.array(img)
    processed_img = inference(img_array)
    # processed_img = img_array
    print('Inference time:', time.time() - start)

    # processed_img = RVM(processed_img)

    img_io = io.BytesIO()
    Image.fromarray(processed_img).save(img_io, 'JPEG')
    img_io.seek(0)
    print('Save time:', time.time() - start)

    return StreamingResponse(img_io, media_type="image/jpeg")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            start = time.time()
            # 在这里进行图像处理
            img_array = np.array(Image.open(io.BytesIO(data)))
            # print(img_array.shape)
            processed_img = inference(img_array)
            img_io = io.BytesIO()
            Image.fromarray(processed_img).save(img_io, 'JPEG')
            img_io.seek(0)
            print('Inference time:', time.time() - start)
            await websocket.send_bytes(img_io.getvalue())
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
