'''
    This script makes tracking over the results of existing
    tracking algorithms. Namely, we run OC-SORT over theirdetections.
    Output in such a way is not strictly accurate because
    building tracks from existing tracking results causes loss
    of detections (usually initializing tracks requires a few
    continuous observations which are not recorded in the output
    tracking results by other methods). But this quick adaptation
    can provide a rough idea about OC-SORT's performance on
    more datasets. For more strict study, we encourage to implement 
    a specific detector on the target dataset and then run OC-SORT 
    over the raw detection results.
    NOTE: this script is not for the reported tracking with public
    detection on MOT17/MOT20 which requires the detection filtering
    following previous practice. See an example from centertrack for
    example: https://github.com/xingyizhou/CenterTrack/blob/d3d52145b71cb9797da2bfb78f0f1e88b286c871/src/lib/utils/tracker.py#L83
'''

from loguru import logger
import time

import sys
sys.path.append('./')
from trackers.ocsort_tracker.ocsort import OCSort
from utils.args import make_parser
import os
import motmetrics as mm
import numpy as np


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(args):
    results_folder = args.out_path
    raw_path = args.raw_results_path
    os.makedirs(results_folder, exist_ok=True)

    dataset = args.dataset

    total_time = 0 
    total_frame = 0 

    if dataset == "headtrack":
        test_seqs = ["HT21-11", "HT21-12", "HT21-13", "HT21-14", "HT21-15"]
        cats = ["head"]
    elif dataset == "bowlparttrack":
        test_seqs = ["BOWL18"]
        cats = ["assembly_part"]
    
    else:
        assert(0)

    cat_ids = {cat: i for i, cat in enumerate(cats)}

    for seq_name in test_seqs:
        print("starting seq {}".format(seq_name))
        tracker = OCSort(args.track_thresh, iou_threshold=args.iou_thresh, delta_t=args.deltat, 
            asso_func=args.asso, inertia=args.inertia)
        if dataset in ["kitti", "bdd"]:
            seq_trks = np.empty((0, 18))
        elif dataset == "headtrack":
            seq_trks = np.empty((0, 10))
        seq_file = os.path.join(raw_path, "{}".format(seq_name))
        seq_file = open(seq_file)
        out_file = os.path.join(results_folder, "{}".format(seq_name))
        out_file = open(out_file, 'w')
        lines = seq_file.readlines()
        line_count = 0 
        for line in lines:
            print("{}/{}".format(line_count,len(lines)))
            line_count+=1
            line = line.strip()
            if dataset in ["kitti", "bdd"]:
                tmps = line.strip().split()
                tmps[2] = cat_ids[tmps[2]]
            elif dataset == "headtrack":
                tmps = line.strip().split(",")
            trk = np.array([float(d) for d in tmps])
            trk = np.expand_dims(trk, axis=0)
            seq_trks = np.concatenate([seq_trks, trk], axis=0)
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            print("{}:{}/{}".format(seq_name, frame_ind, max_frame))
            if dataset in ["kitti", "bdd"]:
                dets = seq_trks[np.where(seq_trksO[:,0]==frame_ind)][:,6:10]
                cates = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2]
                scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,-1]
            elif dataset == "headtrack":
                dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
                cates = np.zeros((dets.shape[0],))
                scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
                dets[:, 2:] += dets[:, :2] # xywh -> xyxy
            else:
                assert(0)
            assert(dets.shape[0] == cates.shape[0])
            t0 = time.time()
            online_targets = tracker.update_public(dets, cates, scores)
            t1 = time.time()
            total_frame += 1
            total_time += t1 - t0
            trk_num = online_targets.shape[0]
            boxes = online_targets[:, :4]
            ids = online_targets[:, 4]
            frame_counts = online_targets[:, 6]
            sorted_frame_counts = np.argsort(frame_counts)
            frame_counts = frame_counts[sorted_frame_counts]
            cates = online_targets[:, 5]
            cates = cates[sorted_frame_counts].tolist()
            cates = [cats[int(catid)] for catid in cates]
            boxes = boxes[sorted_frame_counts]
            ids = ids[sorted_frame_counts]
            for trk in range(trk_num):
                lag_frame = frame_counts[trk]
                if frame_ind < 2*args.min_hits and lag_frame < 0:
                    continue
                """
                    NOTE: here we use the Head Padding (HP) strategy by default, disable the following
                    lines to revert back to the default version of OC-SORT.
                """
                if dataset in ["kitti", "bdd"]:
                    out_line = "{} {} {} -1 -1 -1 {} {} {} {} -1 -1 -1 -1000 -1000 -1000 -10 1\n".format\
                        (int(frame_ind+lag_frame), int(ids[trk]), cates[trk], 
                        boxes[trk][0], boxes[trk][1], boxes[trk][2], boxes[trk][3])
                elif dataset == "headtrack":
                    out_line = "{},{},{},{},{},{},{},-1,-1,-1\n".format(int(frame_ind+lag_frame), int(ids[trk]),
                        boxes[trk][0], boxes[trk][1], 
                        boxes[trk][2]-boxes[trk][0],
                        boxes[trk][3]-boxes[trk][1], 1)
                out_file.write(out_line)

    print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)