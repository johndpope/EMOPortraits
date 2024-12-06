
import argparse
import os
import cv2
import torch
import numpy as np
from torch import nn
from glob import glob
from PIL import Image
import PIL
from typing import Optional, List, Dict, Union,Tuple
from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm.notebook import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage
from infer import InferenceWrapper
from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor
from logger import logger
import traceback
import time

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1) 
to_512 = lambda x: x.resize((512, 512), Image.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.LANCZOS)



def get_video_frames_as_images(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертация из BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = to_512(Image.fromarray(frame_rgb))
        frames.append(img)

    cap.release()
    return frames

def make_video(source, drivers, out_frames_b, path, fps=30.0):
    videodims = (3*512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, videodims)
    #draw stuff that goes on every frame here
    for i in tqdm(range(0,len(out_frames_b))):
        our_s2 = out_frames_b[i]
        out_img = np.array(np.concatenate([np.array(source.resize((512, 512), Image.LANCZOS))[:, :, :3], np.array(drivers[i].resize((512, 512), Image.LANCZOS)), np.array(our_s2.resize((512, 512), Image.LANCZOS))], axis=1))
        video.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    video.release()

def get_bg(s_img, mdnt = True):
    gt_img_t = to_tensor(s_img)[:3].unsqueeze(dim=0).cuda()
    m = get_mask(gt_img_t) if mdnt else get_mask_fp(gt_img_t)
    kernel_back = np.ones((21, 21), 'uint8')
    mask = (m >= 0.8).float()
    mask = mask[0].permute(1,2,0)
    dilate_mask = cv2.dilate(mask.cpu().numpy(), kernel_back, iterations=2)
    dilate_mask = torch.FloatTensor(dilate_mask).unsqueeze(0).unsqueeze(0).cuda()
    background = lama(gt_img_t.cuda(), dilate_mask.cuda())
    bg_img = to_image(background[0])
    bg = to_tensor(bg_img.resize((512, 512), Image.BICUBIC))
    return bg, bg_img

def get_modnet_mask(img):
    im_transform = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    im = im_transform(img)
    ref_size = 512
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda(), True)

    # resize and save matteget_mask
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

    return matte

@torch.no_grad()
def get_mask(source_img_crop):
    source_img_mask = get_modnet_mask(source_img_crop)
    source_img_mask = source_img_mask
    source_img_mask = source_img_mask.clamp_(max=1, min=0)
    
    return source_img_mask


@torch.no_grad()
def get_mask_fp(source_img_crop):
    face_mask_source, _, _, cloth_s = face_idt.forward(source_img_crop)
    trashhold = 0.6
    face_mask_source = (face_mask_source > trashhold).float()

    source_mask_modnet = get_mask(source_img_crop)

    face_mask_source = (face_mask_source*source_mask_modnet).float()

    return face_mask_source

def make_video_crop(source, drivers, out_frames_b, path, fps=30.0, k=2, size=128):
    videodims = (3*512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, videodims)
    #draw stuff that goes on every frame here
    for i in tqdm(range(0,len(out_frames_b))):
        our_s2 = out_frames_b[i]
        out_img = np.array(np.concatenate([np.array(source.resize((512, 512), Image.LANCZOS))[:, :, :3], np.array(to_512(to_512(drivers[i]).crop((256-size*k, 256-size*k, 256+size*k, 256+size*k))) ), np.array(to_512(to_512(our_s2).crop((256-size*k, 256-size*k, 256+size*k, 256+size*k)))    )], axis=1))
        video.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    video.release()

def connect_img_and_bg(img, bg, mdnt=True):
    pred_img_t = to_tensor(img)[:3].unsqueeze(0).cuda()
    _source_img_mask = get_modnet_mask(pred_img_t) if mdnt else get_mask_fp(pred_img_t)
    mask_sss = torch.where(_source_img_mask>0.3, _source_img_mask, _source_img_mask*0)**8
    out_nn = mask_sss.cpu()*pred_img_t.cpu()+ (1-mask_sss.cpu())*bg.cpu()
    return to_image(out_nn[0])

def drive_image_with_video(source: PIL.Image.Image, 
                          video_path: str = '/path/to/your/xxx.mp4',
                          max_len: Optional[int] = None) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
    """
    Drive source image with video frames to generate animated sequence.
    
    Args:
        source: Source identity image
        video_path: Path to driving video
        max_len: Maximum number of frames to process
    """
    try:
        logger.info("\n=== Starting Video-Driven Animation ===")
        logger.info(f"Configuration - video_path: {video_path}, max_frames: {max_len}")

        # Initialize frame lists
        all_srs = []
        all_bgs = []
        all_img_bg = []

        # Load video frames
        logger.info("Loading video frames...")
        all_curr_d = get_video_frames_as_images(video_path)
        if max_len:
            all_curr_d = all_curr_d[:max_len]
        logger.info(f"Loaded {len(all_curr_d)} frames from video")
        
        # Process first frame
        logger.info("Processing first frame...")
        first_d = all_curr_d[0]
        img = inferer.forward(
            source, first_d, 
            crop=False, 
            smooth_pose=False, 
            target_theta=True, 
            mix=True, 
            mix_old=False, 
            modnet_mask=False
        )
        all_srs.append(source)
        logger.debug("First frame processed successfully")
        
        # Generate background
        logger.info("Generating background...")
        bg, bg_img = get_bg(source, mdnt=False)
        all_bgs.append(bg_img)
        logger.debug(f"Background generated - shape: {bg.shape}")
        
        # Process first frame with background
        logger.info("Compositing first frame with background...")
        img_with_bg = connect_img_and_bg(img[0][0], bg, mdnt=False)
        all_img_bg.append(img_with_bg)
        logger.debug("First frame composited successfully")
        
        # Process remaining frames
        logger.info(f"Processing {len(all_curr_d)-1} remaining frames...")
        start_time = time.time()
        frame_times = []
        
        for i, curr_d in enumerate(all_curr_d[1:], 1):
            frame_start = time.time()
            
            # Generate frame
            img = inferer.forward(
                None, curr_d,
                crop=False,
                smooth_pose=False,
                target_theta=True,
                mix=True,
                mix_old=False,
                modnet_mask=False
            )
            
            # Composite with background
            img_with_bg = connect_img_and_bg(img[0][0], bg, mdnt=False)
            all_img_bg.append(img_with_bg)
            
            # Track timing
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Log progress every 10 frames
            if i % 10 == 0:
                avg_fps = 1.0 / (sum(frame_times[-10:]) / min(10, len(frame_times)))
                eta = (len(all_curr_d) - i) * (sum(frame_times) / len(frame_times))
                logger.info(f"Progress: {i}/{len(all_curr_d)} frames | "
                          f"FPS: {avg_fps:.2f} | "
                          f"ETA: {eta:.2f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = len(all_curr_d) / total_time
        
        logger.info("\n=== Animation Complete ===")
        logger.info(f"Total frames: {len(all_curr_d)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Average time per frame: {1000 * total_time / len(all_curr_d):.2f}ms")

        return all_img_bg, all_curr_d

    except Exception as e:
        logger.error(f"Error in video-driven animation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_custom_crop_first(img, k=1.2, in_s=512, first_frame=True, center=None, size=None):
    mpl = in_s//512
    img_cv2 =  np.asarray(img)*255
    if first_frame:
        _faces = face_detector(img_cv2, rgb=False)
        if _faces.shape[0]==0:
            raise ValueError('Face not found')
        else:
            faces = _faces[0]
            center = (int(faces[0] + (faces[2]-faces[0])/2), int(faces[1]+ (faces[3]-faces[1])/2))
            size = max(int((faces[2]-faces[0])/2), int((faces[3]-faces[1])/2))
            new_img = to_512(img.crop((center[0]-size*k, center[1]-size*k, center[0]+size*k, center[1]+size*k)))
            return new_img, center, size
    else:
        new_img = to_512(img.crop((center[0]-size*k, center[1]-size*k, center[0]+size*k, center[1]+size*k)))
        return new_img


project_dir = os.path.dirname(os.path.abspath(__file__))
args_overwrite = {'l1_vol_rgb':0}
face_idt = FaceParsing(None, 'cuda')

lama = torch.jit.load('repos/jit_lama.pt').cuda()

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load('repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
modnet.eval();

threshold = 0.8
device = 'cuda'
face_detector = RetinaFacePredictor(threshold=threshold, device=device, model=(RetinaFacePredictor.get_model('mobilenet0.25')))

inferer = InferenceWrapper(experiment_name = 'Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1', model_file_name = '328_model.pth',
                           project_dir = project_dir, folder = 'logs', state_dict = None,
                           args_overwrite=args_overwrite, pose_momentum = 0.1, print_model=False, print_params = True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_image_path', type=str, default='data/IMG_1.png', help='Path to source image')
    parser.add_argument('--driven_video_path', type=str, default='data/VID_1.mp4', help='Path to driving video')
    parser.add_argument('--saved_to_path', type=str, default='data/result.mp4', help='Path to save result video')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS of output video')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum number of frames to process')

    args = parser.parse_args()

    source_img = to_512(Image.open(args.source_image_path))
    driving_video_path = args.driven_video_path

    driven_result, drivers = drive_image_with_video(source_img, driving_video_path, max_len=args.max_len)

    save_path = args.saved_to_path

    make_video_crop(source_img, driven_result, drivers, save_path, fps=args.fps)