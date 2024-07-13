import itertools
import logging
import os.path as osp
import tempfile
import time
import datetime
import torch
import mmcv
import numpy as np
#from mmcv.utils import print_log
#from pycocotools.coco import COCO
from mmdet.datasets.api_wrappers import COCO
from .sodad_eval.sodadeval import SODADeval
from terminaltables import AsciiTable
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from ..functional import eval_recalls
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmcv.ops.nms import nms
from typing import Dict, List, Optional, Sequence, Union
from mmengine.logging import MMLogger
from mmengine.fileio import dump, get_local_path, load
from collections import OrderedDict
from mmdet.structures.mask import encode_mask_results

@METRICS.register_module()
class SODADMetric(BaseMetric):

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items


        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    # def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
    #     gt_bboxes = []
    #     for i in range(len(self.img_ids)):
    #         ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
    #         ann_info = self.coco.loadAnns(ann_ids)
    #         if len(ann_info) == 0:
    #             gt_bboxes.append(np.zeros((0, 4)))
    #             continue
    #         bboxes = []
    #         for ann in ann_info:
    #             if ann.get('ignore', False) or ann['iscrowd']:
    #                 continue
    #             x1, y1, w, h = ann['bbox']
    #             bboxes.append([x1, y1, x1 + w, y1 + h])
    #         bboxes = np.array(bboxes, dtype=np.float32)
    #         if bboxes.shape[0] == 0:
    #             bboxes = np.zeros((0, 4))
    #         gt_bboxes.append(bboxes)

    #     recalls = eval_recalls(
    #         gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
    #     ar = recalls.mean(axis=1)
    #     return ar

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    # def translate(self, bboxes, x, y):
    #     dim = bboxes.shape[-1]
    #     translated = bboxes + \
    #                  np.array([x, y] * int(dim / 2), dtype=np.float32)
    #     return translated

    # def merge_dets(self,
    #                        results,
    #                        with_merge=True,
    #                        nms_iou_thr=0.5,
    #                        nproc=10,
    #                        save_dir=None,
    #                        **kwargs):
    #     # if mmcv.is_list_of(results, tuple):
    #     #     dets, segms = results
    #     # else:
    #     #     dets = results
    #     dets = results
    #     # get patch results for evaluating
    #     if not with_merge:
    #         results = [(data_info['id'], result)
    #                    for data_info, result in zip(self.data_infos, results)]
    #         # TODO:
    #         if save_dir is not None:
    #             pass
    #         return results

    #     print('\n>>> Merge detected results of patch for whole image evaluating...')
    #     start_time = time.time()
    #     collector = defaultdict(list)
    #     # ensure data_infos and dets have the same length
    #     for data_info, result in zip(self.data_infos, dets):
    #         x_start, y_start = data_info['start_coord']
    #         new_result = []
    #         for i, res in enumerate(result):
    #             bboxes, scores = res[:, :-1], res[:, [-1]]
    #             bboxes = self.translate(bboxes, x_start, y_start)
    #             labels = np.zeros((bboxes.shape[0], 1)) + i
    #             new_result.append(np.concatenate(
    #                 [labels, bboxes, scores], axis=1
    #             ))

    #         new_result = np.concatenate(new_result, axis=0)
    #         collector[data_info['ori_id']].append(new_result)

    #     merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=nms_iou_thr)
    #     if nproc > 1:
    #         pool = Pool(nproc)
    #         merged_results = pool.map(merge_func, list(collector.items()))
    #         pool.close()
    #     else:
    #         merged_results = list(map(merge_func, list(collector.items())))

    #     # TODO:
    #     if save_dir is not None:
    #         pass

    #     stop_time = time.time()
    #     print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
    #     return merged_results


    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:

        # """ Evaluating detection results on top of COCO API """
        # eval_results = {}
        # cocoGt = self.ori_coco
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # merged_results = self.merge_dets(
        #     results=results,
        #     with_merge=True,
        #     nms_iou_thr=0.5,
        #     nproc=8,
        #     save_dir=None
        # )

        # img_ids = [result[0] for result in merged_results]
        # results = [result[1] for result in merged_results]

        # self.CLASSES = ('people', 'rider', 'bicycle', 'motor', 'vehicle',
        #        'traffic-sign', 'traffic-light', 'traffic-camera',
        #        'warning-cone')
        # # sort the detection results based on the original annotation order which
        # # conform to COCO API.
        # empty_result = [np.empty((0, 5), dtype=np.float32) for cls in self.CLASSES]
        # sort_results = []
        # self.ori_img_ids = self._coco_api.getImgIds()
        # for ori_img_id in self.ori_img_ids:
        #     if ori_img_id in img_ids:
        #         sort_results.append(results[img_ids.index(ori_img_id)])
        #     else:
        #         sort_results.append(empty_result)
        # results = sort_results

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results
        for metric in self.metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            logger.info(msg)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, self.proposal_nums, self.iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                coco_dt = self._coco_api.loadRes(result_files[metric])
            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            SODADEval = SODADeval(self._coco_api, coco_dt, iou_type)
            SODADEval.params.catIds = self.cat_ids  # self.cat_ids
            SODADEval.params.imgIds = self.img_ids
            SODADEval.params.maxDets = list(self.proposal_nums)
            SODADEval.params.iouThrs = self.iou_thrs
            # mapping of cocoEval.stats
            tod_metric_names = {
                'AP': 0,
                'AP_50': 1,
                'AP_75': 2,
                'AP_eS': 3,
                'AP_rS': 4,
                'AP_gS': 5,
                'AP_Normal': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1000': 9,
                'AR_eS@1000': 10,
                'AR_rS@1000': 11,
                'AR_gS@1000': 12,
                'AR_Normal@1000': 13
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in tod_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                SODADEval.params.useCats = 0
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for item in metric_items:
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()

                if True:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = SODADEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = SODADEval.stats[:7]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
                    f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} '
                    f'{ap[6]:.3f} '
                )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

# def _merge_func(info, CLASSES, iou_thr):
#     img_id, label_dets = info
#     label_dets = np.concatenate(label_dets, axis=0)
#     labels, dets = label_dets[:, 0], label_dets[:, 1:]

#     ori_img_results = []
#     for i in range(len(CLASSES)):
#         cls_dets = dets[labels == i]
#         if len(cls_dets) == 0:
#             ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
#             continue
#         bboxes, scores = cls_dets[:, :-1], cls_dets[:, [-1]]
#         bboxes = torch.from_numpy(bboxes).to(torch.float32).contiguous()
#         scores = torch.from_numpy(np.squeeze(scores, 1)).to(torch.float32).contiguous()
#         results, inds = nms(bboxes, scores, iou_thr)
#         # If scores.shape=(N, 1) instead of (N, ), then the results after NMS
#         # will be wrong. Suppose bboxes.shape=[N, 4], the results has shape
#         # of [N**2, 5],
#         results = results.numpy()
#         ori_img_results.append(results)
#     return img_id, ori_img_results