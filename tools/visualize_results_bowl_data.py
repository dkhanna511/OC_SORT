import pdb
import os 
import cv2 
from yolox.utils import vis 
import numpy as np
import argparse
import sys
import subprocess
'''
    MOT submission format:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
'''



BOWL_VIDEO_LEN = {
    # "BOWL_TITAN": 11692,
    # "BOWL18": 1800,
    "BOWL_DC": 601
}

BOWL_VIDEO_SPLIT = dict()

for video_name in BOWL_VIDEO_LEN:
    num_images = BOWL_VIDEO_LEN[video_name]
    BOWL_VIDEO_SPLIT[video_name] = dict()
    BOWL_VIDEO_SPLIT[video_name]["full"] = [1, num_images]

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def visualize_box(img, text, box, color_index):
    x0, y0, width, height = box 
    x0, y0, width, height = int(x0), int(y0), int(width), int(height)
    color = (_COLORS[color_index%80] * 255).astype(np.uint8).tolist()
    txt_color = (0, 0, 0) if np.mean(_COLORS[color_index%80]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
    cv2.rectangle(img, (x0, y0), (x0+width, y0+height), color, 2)

    txt_bk_color = (_COLORS[color_index%80] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=1)
    return img
    

def visualize_detections(img_dir, out_dir, detections_dir, mode="val_half", path="{}/{}_detections.txt", dataset="mot17", test=False):
    if dataset == "bowl":
        VIDEO_LEN = BOWL_VIDEO_LEN
        VIDEO_SPLIT = BOWL_VIDEO_SPLIT
    else:
        assert 0

    for video_name in VIDEO_LEN:
        print(" VIDEO LEN is ", VIDEO_LEN)
        print(" detection dir is : ", detections_dir)
        # exit(0)
        detection_f = path.format(detections_dir, video_name)
        # detection_f = os.path.join(detections_dir, "{}_detections.txt".format(video_name))
        f = open(detection_f)
        dets = np.loadtxt(f, delimiter=",")
        print(" video split is : ", VIDEO_SPLIT)
        # exit(0)
        frame_range = VIDEO_SPLIT[video_name][mode]
        assert(frame_range[1]-frame_range[0] == dets[:, 0].max()-dets[:,0].min())
        frame_gap = dets[:,0].min() - frame_range[0]
        video_img_dir = os.path.join(img_dir, video_name, "img1")
        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        fake_frame_min = int(dets[:,0].min())
        fake_frame_max = int(dets[:,0].max())
        for frame_ind in range(fake_frame_min, fake_frame_max+1):
            real_frame_ind = frame_ind - frame_gap 
            frame_dets = dets[np.where(dets[:,0]==frame_ind)]
            im_path = os.path.join(video_img_dir, "%06d.png" % real_frame_ind)
            img = cv2.imread(im_path)
            for i in range(frame_dets.shape[0]):
                box = frame_dets[i]
                score = box[6]
                text = '{:.1f}'.format(score * 100)
                img = visualize_box(img, text, box[2:6], i)
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (2, 2), (120, 30), (30,30,30), -1)
            cv2.putText(img, "%06d.png" % real_frame_ind, (10, 20), font, 0.6, (255,255,255), thickness=2)
            cv2.imwrite(os.path.join(video_out_dir, "%06d.png" % real_frame_ind), img)


def visualize_tracks(img_dir, out_dir, tracks_dir, mode, dataset="mot17"):
    if dataset == "bowl":
        VIDEO_LEN = BOWL_VIDEO_LEN
    
    # import pdb; pdb.set_trace()
    for video_name in VIDEO_LEN:
        print("video_name is : ", video_name)
        video_name = video_name.replace(".txt", "")
        print("video name now is : ", video_name)
        track_f = os.path.join(tracks_dir, "{}.txt".format(video_name))
        print(" track_f is : ", track_f)
        f = open(track_f)
        tracks = np.loadtxt(f, delimiter=",")
        if dataset == "bowl":
            frame_range = BOWL_VIDEO_SPLIT[video_name]["full"]
        
        # assert(frame_range[1]-frame_range[0] == tracks[:, 0].max()-tracks[:,0].min())
        frame_gap = tracks[:,0].min() - frame_range[0]
        video_img_dir = os.path.join(img_dir, video_name, "img1")
        print("videoimg dir is : ", video_img_dir)
        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        fake_frame_min = int(tracks[:,0].min())
        fake_frame_max = int(tracks[:,0].max())
        for frame_ind in range(fake_frame_min, fake_frame_max+1):
            real_frame_ind = frame_ind - frame_gap 
            frame_tracks = tracks[np.where(tracks[:,0]==frame_ind)]
            im_path = os.path.join(video_img_dir, "%05d.png" % real_frame_ind)
            img = cv2.imread(im_path)
            for i in range(frame_tracks.shape[0]):
                box = frame_tracks[i]
                obj_id = int(box[1])
                text = '{}'.format(obj_id)
                img = visualize_box(img, text, box[2:6], obj_id)
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (2, 2), (120, 30), (30,30,30), -1)
            cv2.putText(img, "%05d.png" % real_frame_ind, (10, 20), font, 0.6, (255,255,255), thickness=2)
            cv2.imwrite(os.path.join(video_out_dir, "%05d.png" % real_frame_ind), img)
        cmd = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264 -pix_fmt yuv420p {}/{}.mp4".format(video_out_dir, out_dir, video_name)
        subprocess.call(cmd, shell = True)


def visualize_gt(img_dir, out_dir):
    
    for video_name in VIDEO_LEN:
        track_f = os.path.join(img_dir, video_name, "gt/gt.txt")
        f = open(track_f)
        tracks = np.loadtxt(f, delimiter=",")
        video_img_dir = os.path.join(img_dir, video_name, "img1")
        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        fake_frame_min = int(tracks[:,0].min())
        fake_frame_max = int(tracks[:,0].max())
        for frame_ind in range(fake_frame_min, fake_frame_max+1):
            real_frame_ind = frame_ind
            frame_tracks = tracks[np.where(tracks[:,0]==frame_ind)]
            im_path = os.path.join(video_img_dir, "%05d.png" % real_frame_ind)
            img = cv2.imread(im_path)
            for i in range(frame_tracks.shape[0]):
                box = frame_tracks[i]
                obj_id = int(box[1])
                text = '{}'.format(obj_id)
                img = visualize_box(img, text, box[2:6], obj_id)
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (2, 2), (120, 30), (30,30,30), -1)
            cv2.putText(img, "%05d.png" % real_frame_ind, (10, 20), font, 0.6, (255,255,255), thickness=2)
            cv2.imwrite(os.path.join(video_out_dir, "%06d.png" % real_frame_ind), img)
        cmd = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264 -pix_fmt yuv420p {}/{}.mp4".format(video_out_dir, out_dir, video_name)
        # os.popen(cmd)


def merge_visualization(det_dir, track_dir, gt_dir, out_dir):
    os.makedirs(out_dir,  exist_ok=True)
    seqs = os.listdir(track_dir)
    for seq in seqs:
        if "mp4" in seq:
            continue
        seq_track_dir = os.path.join(track_dir, seq)
        seq_det_dir = os.path.join(det_dir, seq)
        seq_gt_dir = os.path.join(gt_dir, seq)
        seq_out_dir = os.path.join(out_dir, seq)
        os.makedirs(seq_out_dir, exist_ok=True)
        frames = sorted(os.listdir(seq_track_dir))
        for frame in frames:
            f_track_path = os.path.join(seq_track_dir, frame)
            f_det_path = os.path.join(seq_det_dir, frame)
            f_gt_path = os.path.join(seq_gt_dir, frame)
            im1 = cv2.imread(f_track_path)
            im2 = cv2.imread(f_det_path)
            im3 = cv2.imread(f_gt_path)
            im_concat = cv2.vconcat([im2, im3, im1])
            f_out_dir = os.path.join(seq_out_dir, frame)
            cv2.imwrite(f_out_dir, im_concat)
        
        cmd = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264 -pix_fmt yuv420p {}/merged_{}.mp4".format(seq_out_dir, out_dir, seq)
        os.popen(cmd)


def make_parser():
    parser = argparse.ArgumentParser("Visualize Results")
    parser.add_argument('--mode', default="val_half", type=str)
    parser.add_argument('--img_dir', default="datasets/mot/train")
    parser.add_argument('--exp_dir', default="yolox_x_ablation", type=str)
    parser.add_argument('--exp_name', default="track_results", type=str)
    parser.add_argument('--vis', default="det", type=str, help="det/track/gt")
    parser.add_argument("--dataset", default="mot17", type=str)
    parser.add_argument("--res", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_parser()
    if args.dataset == "bowl":
        img_dir = "datasets/BOWL/train"
    
    # result_src_dir = "YOLOX_outputs/"
    # result_src_dir = "evaldata/trackers/mot_challenge/MOT17-val/"
    res_dir = args.res
    out_src_dir = "visualizations"

    # exp_dir = args.exp_dir
    exp_name = args.exp_name
    # res_dir = os.path.join(result_src_dir, exp_name, "data")
    out_dir = os.path.join(out_src_dir, args.dataset, exp_name, args.vis)
    os.makedirs(out_dir, exist_ok=True)

    if args.vis == "det":
        if args.dataset =="bowl":
            var = "BOWL"
        res_dir = "datasets/{}/test".format(var)
        out_dir = "visualizations/{}/dets".format(var)
        visualize_detections(img_dir, out_dir, res_dir, mode="full", path="{}/{}/det/det.txt", dataset = args.dataset)
        # visualize_detections(img_dir, out_dir, res_dir, mode=args.mode)
    elif args.vis == "track":
        if args.dataset =="bowl":
            var = "BOWL"
            # res_dir = "datasets/{}/train".format(var)
            tracks_dir = "datasets/BOWL/"
        visualize_tracks(img_dir, out_dir, tracks_dir= tracks_dir, mode=args.mode, dataset=args.dataset)
    elif args.vis == "merge":
        det_dir = "visualizations/yolox_x_ablation/{}/det".format(args.exp_name)
        track_dir = "visualizations/yolox_x_ablation/{}/track".format(args.exp_name)
        gt_dir = "visualizations/GTs"
        out_dir = "visualizations/yolox_x_ablation/{}/merged".format(args.exp_name)
        merge_visualization(det_dir, track_dir, gt_dir, out_dir)
    elif args.vis == "det_fasterrcnn":
        res_dir = "datasets/mot/train"
        out_dir = "visualizations/mot17/fasterrcnn_dets"
        visualize_detections(img_dir, out_dir, res_dir, mode="full", path="{}/{}/det/det.txt")
    elif args.vis == "gt":
        out_dir = os.path.join(out_src_dir, "GTs")
        os.makedirs(out_dir, exist_ok=True)
        visualize_gt(img_dir, out_dir)

