import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
        if cfg.MODEL.POST_PROCESSING.get('NUMBER_OF_POINTS', False):
            metric['tp_vehicle_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['tp_vehicle_points_rcnn_%s' % str(cur_thresh)])
            metric['fp_vehicle_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fp_vehicle_points_rcnn_%s' % str(cur_thresh)])
            metric['gt_vehicle_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['gt_vehicle_points_rcnn_%s' % str(cur_thresh)])
            metric['fn_vehicle_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fn_vehicle_points_rcnn_%s' % str(cur_thresh)])
            metric['tp_pedestrian_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['tp_pedestrian_points_rcnn_%s' % str(cur_thresh)])
            metric['fp_pedestrian_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fp_pedestrian_points_rcnn_%s' % str(cur_thresh)])
            metric['gt_pedestrian_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['gt_pedestrian_points_rcnn_%s' % str(cur_thresh)])
            metric['fn_pedestrian_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fn_pedestrian_points_rcnn_%s' % str(cur_thresh)])
            metric['tp_cyclist_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['tp_cyclist_points_rcnn_%s' % str(cur_thresh)])
            metric['fp_cyclist_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fp_cyclist_points_rcnn_%s' % str(cur_thresh)])
            metric['gt_cyclist_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['gt_cyclist_points_rcnn_%s' % str(cur_thresh)])
            metric['fn_cyclist_points_rcnn_%s' % str(cur_thresh)].append(ret_dict['fn_cyclist_points_rcnn_%s' % str(cur_thresh)])
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def save_number_of_points(cfg, metric):
    nms_thresh = cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH
    datasets = cfg.DATA_CONFIG.DATASET
    if datasets == 'WaymoDataset':
        data_name = 'waymo'
    else:
        data_name = 'kitti'
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        for class_name in ['vehicle', 'pedestrian', 'cyclist']:
            for pred_type in ['tp', 'fp', 'gt', 'fn']:
                save_this = metric['%s_%s_points_rcnn_%s' % (pred_type, class_name, str(cur_thresh))]
                with open('/usrhomes/s1420/tp_fp_number_of_points/%s/nms%s_%s_%s_points_rcnn_%s.pkl' %
                          (data_name, str(nms_thresh), pred_type, class_name, str(cur_thresh)), 'wb') as f:
                    pickle.dump(save_this, f)




def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
        if cfg.MODEL.POST_PROCESSING.get('NUMBER_OF_POINTS', False):
            metric['tp_vehicle_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fp_vehicle_points_rcnn_%s' % str(cur_thresh)] = []
            metric['gt_vehicle_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fn_vehicle_points_rcnn_%s' % str(cur_thresh)] = []
            metric['tp_pedestrian_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fp_pedestrian_points_rcnn_%s' % str(cur_thresh)] = []
            metric['gt_pedestrian_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fn_pedestrian_points_rcnn_%s' % str(cur_thresh)] = []
            metric['tp_cyclist_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fp_cyclist_points_rcnn_%s' % str(cur_thresh)] = []
            metric['gt_cyclist_points_rcnn_%s' % str(cur_thresh)] = []
            metric['fn_cyclist_points_rcnn_%s' % str(cur_thresh)] = []

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.MODEL.POST_PROCESSING.get('NUMBER_OF_POINTS', False):
        save_number_of_points(cfg, metric)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )
    # if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
    #     for class_name in class_names:
    #         result_dict['average_3d_result'] += result_dict['%s_3d/easy_R40' % class_name]
    #         result_dict['average_3d_result'] += result_dict['%s_3d/moderate_R40' % class_name]
    #         result_dict['average_3d_result'] += result_dict['%s_3d/hard_R40' % class_name]


    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
