# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
from collections import defaultdict

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes, add_center_dist_, filter_eval_boxes_
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_fap_mr, calc_ar, calc_tp, calc_fap, calc_far, calc_aap, calc_aar
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList, DetectionMetricData
from pyquaternion import Quaternion
import pickle
from copy import deepcopy
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
import torch 
from tqdm import tqdm 
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import Polygon

import os
import cv2
from det3d.utils.simplevis import *

import pdb
from itertools import tee 

def box_center(boxes):
    center_box = np.array([box.translation for box in boxes]) 
    return center_box

def box_scores(boxes):
    box = np.array([box.forecast_score for box in boxes]) 
    return box

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def get_time_(nusc, src_token, dst_token):
    # time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    # time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    # time_diff = time_first - time_last
    #time_diff=0.2
    # time_diff = 0.2
    time_diff = 1

    return time_diff 

def distance_matrix(A, B, squared=False):
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def box2d_iou(boxA, boxB): 
    A = Box(center=boxA.translation, size=boxA.size, orientation=Quaternion(boxA.rotation))
    B = Box(center=boxB.translation, size=boxB.size, orientation=Quaternion(boxB.rotation))

    cornersA = view_points(A.corners(), np.eye(4), normalize=False)[:2, :].T
    cornersB = view_points(B.corners(), np.eye(4), normalize=False)[:2, :].T

    polyA = Polygon([(cornersA[0][0], cornersA[0][1]), (cornersA[1][0], cornersA[1][1]), (cornersA[5][0], cornersA[5][1]), (cornersA[4][0], cornersA[4][1])])
    polyB = Polygon([(cornersB[0][0], cornersB[0][1]), (cornersB[1][0], cornersB[1][1]), (cornersB[5][0], cornersB[5][1]), (cornersB[4][0], cornersB[4][1])])

    intersection = polyA.intersection(polyB).area 
    
    if intersection > 0:
        union = polyA.union(polyB).area
        return intersection / union
        
    else:
        return 0 

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles.

    :param quaternion: numpy array of shape (4,), representing quaternion in wxyz order.
    :return: numpy array of shape (3,), representing euler angles in roll-pitch-yaw order, in radians.
    """
    qw, qx, qy, qz = quaternion
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

def trajectory(nusc, box: DetectionBox) -> float:
    target = box.forecast_boxes[-1]
    # time = [get_time_(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]
    time = [get_time(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]

    static_forecast = deepcopy(box.forecast_boxes[0])

    if box2d_iou(target, static_forecast) > 0:
        return "static"

    linear_forecast = deepcopy(box.forecast_boxes[0])
    vel = linear_forecast.velocity[:2]

    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)
    linear_forecast.translation = linear_forecast.translation + disp
    
    if box2d_iou(target, linear_forecast) > 0:
        return "linear"

    return "nonlinear"

def trajectory_(nusc, box: DetectionBox) -> float:
    target = box.forecast_boxes[-1]
    time = [get_time_(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]
    # time = [get_time(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]

    static_forecast = deepcopy(box.forecast_boxes[0])

    if box2d_iou(target, static_forecast) > 0:
        return "static"

    linear_forecast = deepcopy(box.forecast_boxes[0])
    vel = linear_forecast.velocity[:2]

    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)
    linear_forecast.translation = linear_forecast.translation + disp
    
    if box2d_iou(target, linear_forecast) > 0:
        return "linear"

    return "nonlinear"

def trajectory_v2(nusc, box: DetectionBox) -> float:
    target = box.forecast_boxes[-1]
    time = [get_time_(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]
    # time = [get_time(nusc, token[0], token[1]) for token in window([b.sample_token for b in box.forecast_boxes], 2)]

    static_forecast = deepcopy(box.forecast_boxes[0])


    linear_forecast = deepcopy(box.forecast_boxes[0])
    vel = linear_forecast.velocity[:2]

    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)
    linear_forecast.translation = linear_forecast.translation + disp

    if center_distance(target, static_forecast) < max(target.size[0], target.size[1]):
        return "static"
    elif center_distance(target, linear_forecast) < max(target.size[0], target.size[1]):
        return "linear"
    else:
        return "nonlinear"
    
    return "nonlinear"


def serialize_box(box):
    box = DetectionBox(sample_token=box["sample_token"],
                        translation=box["translation"],
                        size=box["size"],
                        rotation=box["rotation"],
                        velocity=box["velocity"])
    return box

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 forecast: int = 7,
                 tp_pct: float = 0.6,
                 static_only: bool = False,
                 cohort_analysis: bool = False,
                 topK: int = 1,
                 root: str = "/ssd0/nperi/nuScenes", 
                 association_oracle=False,
                 nogroup=False):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.forecast = forecast
        self.tp_pct = tp_pct
        self.static_only = static_only
        self.cohort_analysis = cohort_analysis
        self.topK = topK
        self.association_oracle = association_oracle
        self.nogroup = nogroup
        self.cfg.dist_ths = [0.25, 0.5, 1.0, 2.0]

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        if self.eval_set == 'mini_val':
            self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
            self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
            nus_flag=True
        else:
            self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
            self.gt_boxes, self.meta = load_prediction(self.result_path.split("infos_")[0]+"infos_gt_val_10sweeps_withvelo_filter_True.json", self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
            nus_flag=False

        
        sample_tokens = [s["token"] for s in nusc.sample]

        print("Deserializing forecast data")
        for sample_token in tqdm(self.gt_boxes.boxes.keys()):
            for box in self.pred_boxes.boxes[sample_token]:
                box.forecast_boxes = [serialize_box(box) for box in box.forecast_boxes]

            for box in self.gt_boxes.boxes[sample_token]:
                box.forecast_boxes = [serialize_box(box) for box in box.forecast_boxes]
                
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        
        pred_static_count = 0
        pred_linear_count = 0
        pred_nonlinear_count = 0
        gt_static_count = 0
        gt_linear_count = 0
        gt_nonlinear_count = 0

        
        if self.cohort_analysis:
            for sample_token in self.pred_boxes.boxes.keys():
                for box in self.pred_boxes.boxes[sample_token]:
                    if nus_flag:
                        label = trajectory(nusc, box)
                    else:
                        #label = trajectory_(nusc, box)
                        label = trajectory_v2(nusc, box)

                    if label == 'static':
                        pred_static_count +=1
                    elif label == 'linear':
                        pred_linear_count +=1
                    elif label == 'nonlinear':
                        pred_nonlinear_count +=1
                    
                    name = label + "_" + box.detection_name
                    # if label != 'nonlinear':
                    #     print("label : {}".format(label))

                    box.detection_name = name

                    for i in range(self.forecast):
                        box.forecast_boxes[i].detection_name = name

            for sample_token in self.gt_boxes.boxes.keys():
                for box in self.gt_boxes.boxes[sample_token]:
                    if nus_flag:
                        label = trajectory(nusc, box)
                    else:
                        #label = trajectory_(nusc, box)
                        label = trajectory_v2(nusc, box)

                    if label == 'static':
                        gt_static_count +=1
                    elif label == 'linear':
                        gt_linear_count +=1
                    elif label == 'nonlinear':
                        gt_nonlinear_count +=1

                    name = label + "_" + box.detection_name

                    if label != 'nonlinear':
                        print("label : {}".format(label))

                    box.detection_name = name

                    for i in range(self.forecast):
                        box.forecast_boxes[i].detection_name = name
            
        print("pred state :=======\n")
        print("static : {} \n".format(pred_static_count))
        print("linear : {} \n".format(pred_linear_count))
        print("nonlinear : {} \n".format(pred_nonlinear_count))

        print("gt state :=======\n")
        print("static : {} \n".format(gt_static_count))
        print("linear : {} \n".format(gt_linear_count))
        print("nonlinear : {} \n".format(gt_nonlinear_count))

        if self.static_only:
            for sample_token in self.pred_boxes.boxes.keys():
                static_boxes = []
                for boxes in self.pred_boxes.boxes[sample_token]:
                    if trajectory(nusc, boxes) != "static":
                        continue
                    
                    static_boxes.append(boxes)

                self.pred_boxes.boxes[sample_token] = static_boxes
                

        # Add center distances.
        if self.eval_set == 'mini_val':
            self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
            self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        else:
            self.pred_boxes = add_center_dist_(nusc, self.pred_boxes)
            self.gt_boxes = add_center_dist_(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')

        if self.eval_set == 'mini_val':
            self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
            self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        else:
            self.pred_boxes = filter_eval_boxes_(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
            self.gt_boxes = filter_eval_boxes_(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        
        if verbose:
            print('Filtering ground truth annotations')
        
        
        ########################################################################################################
        max_thresh_det = {"car" : 0.5, "pedestrian" : 0.125}
        max_thresh_forecast = {"car" : 1, "pedestrian" : 0.25}

        pred_boxes_topK = {}
        for class_name in ["car", "pedestrian"]:
            taken = set()
            for sample_token in self.gt_boxes.boxes.keys():
                if sample_token not in pred_boxes_topK:
                    pred_boxes_topK[sample_token] = []

                pred_boxes = [box for box in self.pred_boxes.boxes[sample_token] if class_name in box.detection_name]
                pred_confs = [box.detection_score for box in pred_boxes]
                sorted_pred_boxes = [b for _, b in sorted(zip(pred_confs, pred_boxes), key=lambda x: x[0], reverse=True)]

                groups = set([box.forecast_id for box in sorted_pred_boxes])

                for group in groups:
                    boxes = [box for box in pred_boxes if box.forecast_id == group]
                    scores = box_scores(boxes)
                    boxes = [b for _, b in sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)][:topK]

                    test_box = deepcopy(boxes[0])

                    if topK == 1:
                        pred_boxes_topK[sample_token].append(test_box)
                        continue 

                    min_dist = np.inf
                    match_gt_idx = None

                    for gt_idx, gt_box in enumerate(self.gt_boxes.boxes[sample_token]):
                        if class_name not in gt_box.detection_name:
                            continue 

                        this_distance = center_distance(gt_box, test_box)
                        if this_distance < min_dist and (sample_token, gt_idx) not in taken:
                            min_dist = this_distance
                            match_gt_idx = gt_idx

                    if min_dist < max_thresh_det[class_name]:
                        taken.add((sample_token, match_gt_idx))
                    else:
                        pred_boxes_topK[sample_token].append(test_box)
                        continue 

                    min_fde = np.inf 
                    match_box = None
                    if match_gt_idx is not None:
                        for box in boxes:
                            match_gt = self.gt_boxes.boxes[sample_token][match_gt_idx]
                            fde = center_distance(match_gt.forecast_boxes[-1], box.forecast_boxes[-1])
                            if fde < min_fde:
                                min_fde = fde 
                                match_box = deepcopy(box)

                        if min_fde < max_thresh_forecast[class_name]:
                            match_box.detection_score = test_box.detection_score
                            match_box.forecast_score = test_box.forecast_score
                            pred_boxes_topK[sample_token].append(match_box)
                        else:
                            pred_boxes_topK[sample_token].append(test_box)

                    else:
                        pred_boxes_topK[sample_token].append(test_box)
        
        pred_count = 0
        for sample_token in self.gt_boxes.boxes.keys():
            self.pred_boxes.boxes[sample_token] = pred_boxes_topK[sample_token]
            pred_count += len(pred_boxes_topK[sample_token])

        print("Predicted Trajectories @ K={}: {}".format(topK, pred_count))
        
        ########################################################################
        self.sample_tokens = self.gt_boxes.sample_tokens

        for sample_token in self.sample_tokens:
            boxes = [box for box in self.pred_boxes.boxes[sample_token]]
            sorted_boxes = sorted(boxes, key= lambda x : x.detection_score, reverse=True)

            self.pred_boxes.boxes[sample_token] = sorted_boxes[:self.cfg.max_boxes_per_sample]


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        self.cfg.dist_ths = [0.25, 0.5, 1.0, 2.0]
        for class_name in tqdm(self.cfg.class_names):
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.nusc, self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th, self.forecast, self.topK, self.cohort_analysis, self.association_oracle)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                fap_mr = calc_fap_mr(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)

                ar = calc_ar(deepcopy(metric_data))

                fap = calc_fap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                far = calc_far(deepcopy(metric_data))

                aap = calc_aap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                aar = calc_aar(deepcopy(metric_data))

                metrics.add_label_ap(class_name, dist_th, ap)
                metrics.add_label_fap_mr(class_name, dist_th, fap_mr)

                metrics.add_label_ar(class_name, dist_th, ar)
                metrics.add_label_fap(class_name, dist_th, fap)
                metrics.add_label_far(class_name, dist_th, far)
                metrics.add_label_aap(class_name, dist_th, aap)
                metrics.add_label_aar(class_name, dist_th, aar)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name, self.tp_pct)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList, cohort_analysis=False) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'), cohort_analysis=cohort_analysis)

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             cohort_analysis: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            #example_dir = os.path.join(self.output_dir, 'examples')
            #if not os.path.isdir(example_dir):
            #    os.mkdir(example_dir)
            #for sample_token in sample_tokens:
            #    visualize_sample(self.nusc,
            #                     sample_token,
            #                     self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
            #                     # Don't render test GT.
            #                     self.pred_boxes,
            #                     eval_range=max(self.cfg.class_range.values()),
            #                     savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list, cohort_analysis=cohort_analysis)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)


        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        print('mFAP_MR: %.4f' % (metrics_summary['mean_fap_mr']))

        print('mAR: %.4f' % (metrics_summary['mean_ar']))

        print('mFAP: %.4f' % (metrics_summary['mean_fap']))
        print('mFAR: %.4f' % (metrics_summary['mean_far']))

        print('mAAP: %.4f' % (metrics_summary['mean_aap']))
        print('mAAR: %.4f' % (metrics_summary['mean_aar']))

        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE',
            'avg_disp_err' : 'mADE',
            'final_disp_err' : 'mFDE',
            'miss_rate' : 'mMR',
            #'reverse_avg_disp_err' : 'mRADE',
            #'reverse_final_disp_err' : 'mRFDE',
            #'reverse_miss_rate' : 'mRMR',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tFAP_MR\tAR\tFAP\tFAR\tAAP\tAAR\tATE\tASE\tAOE\tAVE\tAAE\tADE\tFDE\tMR')
        class_aps = metrics_summary['mean_dist_aps']
        class_faps_mr = metrics_summary['mean_dist_faps_mr']

        class_ars = metrics_summary['mean_dist_ars']

        class_faps = metrics_summary['mean_dist_faps']
        class_fars = metrics_summary['mean_dist_fars']

        class_aaps = metrics_summary['mean_dist_aaps']
        class_aars = metrics_summary['mean_dist_aars']

        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name], class_faps_mr[class_name], class_ars[class_name], class_faps[class_name], class_fars[class_name], class_aaps[class_name], class_aars[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err'],
                     class_tps[class_name]['avg_disp_err'],
                     class_tps[class_name]['final_disp_err'],
                     class_tps[class_name]['miss_rate'],
                     #class_tps[class_name]['reverse_avg_disp_err'],
                     #class_tps[class_name]['reverse_final_disp_err'],
                     #class_tps[class_name]['reverse_miss_rate'],
                     ))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
