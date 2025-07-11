import torch
import torch.nn as nn
import copy
import collections

from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import ConvModule
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean, build_bbox_coder
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet3d.models import builder

from perceptron.layers.head.det3d.bbox.util import normalize_bbox
from perceptron.utils.e2e_utils.utils import pos2posemb3d


@HEADS.register_module()
class SparseE2EHead(BaseModule):
    def __init__(
        self,
        in_channels,
        modal="Fusion",
        depth_num=64,
        num_query=900,
        init_radar_num_query=0,
        hidden_dim=128,
        grid_size=[1440, 1440, 40],
        norm_bbox=True,
        downsample_scale=8,
        scalar=10,
        noise_scale=1.0,
        noise_trans=0.0,
        dn_weight=1.0,
        split=0.75,
        #  assigner_cfg=None,
        train_cfg=None,
        test_cfg=None,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        transformer=None,
        bbox_coder=None,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, reduction="mean", gamma=2, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(
            type="L1Loss",
            reduction="mean",
            loss_weight=0.25,
        ),
        loss_heatmap=dict(type="GuassianFocalLoss", reduction="mean"),
        separate_head=dict(type="SeparateMlpHead", init_bias=-2.19, final_kernel=3),
        init_cfg=None,
        repaired_timestamp=False,
        alpha=0.2,  # htt: 需要标记注明一下alpha的含义
        use_roi_mask=False,
        with_multiview=False,
        bg_cls_weight=0.1,
        use_dn=False,
        lidar_use_query_mlp=True,
        refine_reg_branch=False,
        **kwargs,
    ):
        self.use_dn = use_dn
        self.bg_cls_weight = bg_cls_weight
        if loss_cls is not None and "bg_cls_weight" in loss_cls:
            assert (
                loss_cls["bg_cls_weight"] == bg_cls_weight
            ), f"bg_cls_weight in loss_cls: {loss_cls['bg_cls_weight']} != bg_cls_weight in {self.__class__}: {bg_cls_weight}, please use bg_cls_weight in {self.__class__}"

        assert init_cfg is None
        assert isinstance(modal, list)
        assert set(modal).issubset(["Camera", "Lidar", "LiDAR", "Radar"])

        super(SparseE2EHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        # self.assigner_cfg = assigner_cfg
        self.train_cfg = train_cfg
        self.grid_size = grid_size
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.init_radar_num_query = init_radar_num_query
        self.in_channels = in_channels
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False
        # timestamp problem
        self.repaired_timestamp = repaired_timestamp
        self.alpha = alpha
        # roi mask
        self.use_roi_mask = use_roi_mask
        # transformer
        self.transformer = build_transformer(transformer)

        if "Radar" in modal:
            self.bev_embedding = nn.Sequential(
                nn.Linear(384, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if "Lidar" in modal:

            self.shared_conv = ConvModule(
                self.hidden_dim * 3 // 2,
                self.hidden_dim,
                kernel_size=3,
                stride=1,  # 下采样x2
                padding=1,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
            self.front_lidar_grid_size = kwargs.get("front_lidar_grid_size", None)

        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim, heads=heads, num_cls=num_cls, groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)

        if refine_reg_branch:  # FIXME: 更细致的reg设计
            self.refine_reg_branches = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim // 2, 3),
            )
        else:
            self.refine_reg_branches = None

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        """Generate de_noising query from gt_boxes
        Args:
            batch_size: int
            reference_points: (bs, n, c or 3)
            img_metas: dict{
                "gt_boxes": tensor, pos labels start from 1. # TODO: need to support LidarBoxes3d.

            }
        """
        if self.training:
            if "gt_boxes" in img_metas:
                gt_boxes = img_metas["gt_boxes"]
                mask = torch.any(gt_boxes[..., :9], dim=2)
                targets = [gt_boxes[i, mask[i, :], :9] for i in range(batch_size)]
                labels = [gt_boxes[i, mask[i, :], 9] - 1 for i in range(batch_size)]
            elif "ff_gt_bboxes_list" in img_metas:
                gt_boxes = img_metas["ff_gt_bboxes_list"]  # 10 dims
                labels = [x - 1 for x in img_metas["ff_gt_labels_list"]]  # cmt里面增加dn得-1
                targets = gt_boxes
            # add noise
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0]
                )
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1]
                )
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2]
                )
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(batch_size, pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device
                )

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True
                else:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True

            mask_dict = {
                "known_indice": torch.as_tensor(
                    known_indice
                ).long(),  # indices of gt bboxes correponding to dn query, shape [batch_size*pad_size]
                "batch_idx": torch.as_tensor(
                    batch_idx
                ).long(),  # batch indices of dn query, shape [batch_size*pad_size/group]
                "map_known_indice": torch.as_tensor(
                    map_known_indice
                ).long(),  # total indices of dn query shape [batch_size*pad_size]
                "known_lbs_bboxes": (
                    known_labels,
                    known_bboxs,
                ),  # target of dn query, known_labels include positive and negative label, shape tuple ([batch_size*pad_size],[batch_size*pad_size,9])
                "known_labels_raw": known_labels_raw,  # raw labels of dn query(corresponding gt bbox) , shape [batch_size*pad_size]
                "know_idx": know_idx,  # lables mask (unpad) ,shape tuple ([sample_0_gt_num],[sample_1_gt_num],...[sample_(batch_size-1)_gt_num])
                "pad_size": pad_size,  # total size of dn query per batch after padding shape [1]
            }

        else:
            padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _get_targets_single(
        self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits, roi_mask, query_bboxes, fov_boardline
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([torch.where(gt_labels_3d == class_name.index(i) + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_roi_mask = []
        task_fov_boardline = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            task_roi_mask.append(roi_mask)
            task_fov_boardline.append(fov_boardline)
            flag2 += len(mask)

        def task_assign(
            bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes, roi_mask, query_bboxes, fov_boardline
        ):
            num_bboxes = bbox_pred.shape[0]
            if self.use_roi_mask:
                assign_results = self.assigner.assign(
                    bbox_pred,
                    logits_pred,
                    gt_bboxes,
                    gt_labels,
                    roi_mask=roi_mask,
                    query_bboxes=query_bboxes,
                    fov_boardline=fov_boardline,
                )
            else:
                assign_results = self.assigner.assign(
                    bbox_pred, logits_pred, gt_bboxes, gt_labels, roi_mask=None, query_bboxes=None, fov_boardline=None
                )
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes,), num_classes, dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.use_roi_mask:
                label_weights = gt_bboxes.new_zeros(num_bboxes)  # num_query
                label_weights[pos_inds] = 1
                label_weights[neg_inds] = 1
            else:
                label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        (
            labels_tasks,
            labels_weights_tasks,
            bbox_targets_tasks,
            bbox_weights_tasks,
            pos_inds_tasks,
            neg_inds_tasks,
        ) = multi_apply(
            task_assign,
            pred_bboxes,
            pred_logits,
            task_boxes,
            task_classes,
            self.num_classes,
            task_roi_mask,
            query_bboxes,
            task_fov_boardline,
        )

        return (
            labels_tasks,
            labels_weights_tasks,
            bbox_targets_tasks,
            bbox_weights_tasks,
            pos_inds_tasks,
            neg_inds_tasks,
        )

    def get_targets(
        self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits, roi_mask, query_bboxes, fov_boardline
    ):
        """ "Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        if fov_boardline is None:
            fov_boardline = [None for _ in range(len(roi_mask))]
        # gt_labels_3d = [ gt - 1 for gt in gt_labels_3d]
        (
            labels_list,
            labels_weight_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            preds_bboxes,
            preds_logits,
            roi_mask,
            query_bboxes,
            fov_boardline,
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))]
            )
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))]
            )
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))]
            )

        return (
            task_labels_list,
            task_labels_weight_list,
            task_bbox_targets_list,
            task_bbox_weights_list,
            num_total_pos_tasks,
            num_total_neg_tasks,
        )

    def _loss_single_task(
        self,
        pred_bboxes,
        pred_logits,
        labels_list,
        labels_weights_list,
        bbox_targets_list,
        bbox_weights_list,
        num_total_pos,
        num_total_neg,
    ):
        """ "Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor)

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single(
        self, pred_bboxes, pred_logits, gt_bboxes_3d, gt_labels_3d, roi_mask, query_bboxes, fov_boardline=None
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[tensor]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        query_bboxes_list = []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
            query_bboxes_list.append([task_query_bbox[idx] for task_query_bbox in query_bboxes])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list, roi_mask, query_bboxes_list, fov_boardline
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self, pred_bboxes, pred_logits, mask_dict):
        if mask_dict is not None:
            known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]
            map_known_indice = mask_dict["map_known_indice"].long()
            known_indice = mask_dict["known_indice"].long()
            batch_idx = mask_dict["batch_idx"].long()
            bid = batch_idx[known_indice.to(batch_idx.device)]
            known_labels_raw = mask_dict["known_labels_raw"]
            # trans openpcdet sytle to mmdet style
            # known_labels -= 1
            pred_logits = pred_logits[(bid, map_known_indice)]  # 1x12x9
            pred_bboxes = pred_bboxes[(bid, map_known_indice)]  # 1x12x10
            num_tgt = known_indice.numel()

            # filter task bbox
            task_mask = known_labels_raw != pred_logits.shape[-1]
            task_mask_sum = task_mask.sum()

            if task_mask_sum > 0:
                # pred_logits = pred_logits[task_mask]
                # known_labels = known_labels[task_mask]
                pred_bboxes = pred_bboxes[task_mask]
                known_bboxs = known_bboxs[task_mask]

            # classification loss
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split

            label_weights = torch.ones_like(known_labels)
            cls_avg_factor = max(cls_avg_factor, 1)

        else:
            # pred_logits 12x9
            # known_labels 12
            # label_weights 12
            # cls_avg_factor num

            cls_avg_factor = 1
            pred_logits = pred_logits[:, :10].flatten(0, 1)
            pred_bboxes = pred_bboxes[:, :10].flatten(0, 1)
            known_labels = torch.ones(len(pred_logits), device="cuda").long() * -1
            known_bboxs = torch.ones_like(pred_bboxes, device="cuda")
            label_weights = torch.zeros_like(known_labels, device="cuda")
            num_tgt = 1
            task_mask_sum = 0
        loss_cls = self.loss_cls(pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        if mask_dict is not None:
            bbox_weights = torch.ones_like(pred_bboxes)
        else:
            bbox_weights = torch.zeros_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        # vel is dummy input for private
        if self.test_cfg["dataset"] == "Private":
            bbox_weights[:, 8:10] = 0

        loss_bbox = self.loss_bbox(
            pred_bboxes[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_tgt,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self, pred_bboxes, pred_logits, dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict)
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _get_embeds(self, radar_points, x_lidar, x_img, x_radar, img_metas, device, query_embedding):
        """计算radar/lidar bev PE与camera PE"""
        radar_bev_pos_embeds = None  # Radar BEV Pos Emd
        rv_pos_embeds = None  # Camera Pos Emd
        lidar_bev_pos_embeds = None
        if x_img is not None:
            if isinstance(x_img, list):
                assert len(x_img) == len(img_metas)
                rv_pos_embeds = list()
                for x_img_per, img_meta_per in zip(x_img, img_metas):
                    rv_pos_embeds.append(self._rv_pe(x_img_per, img_meta_per))
            else:
                rv_pos_embeds = self._rv_pe(x_img, img_metas)

        if x_radar is not None:
            radar_bev_pos_embeds = self.bev_embedding(pos2posemb3d(radar_points))
        if x_lidar is not None:

            lidar_bev_pos_embeds = self.bev_embedding(
                pos2posemb3d(torch.cat((self.coords_bev, torch.ones(len(self.coords_bev), 1) * 0.5), -1).to(device))
            )
        return rv_pos_embeds, radar_bev_pos_embeds, lidar_bev_pos_embeds

    def forward(
        self,
        img_feats,
        img_metas,
        query_feats,
        query_embeds,
        reference_points,
        attn_mask=None,
        mask_dict=None,
        radar_points=None,
        x_radar=None,
        lidar_feats=None,
        query_embedding=None,
        query_padding_masks=None,
    ):
        """
        x: [bs c h w], lidar_feats
        x_img: [bs n c h w], multi_cam img_feats

        return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        batch_size = len(img_metas["ida_mats"])
        if lidar_feats is not None:
            lidar_feats = self.shared_conv(lidar_feats)

        ret_dicts = []
        if reference_points.ndim == 2:
            reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        elif reference_points.ndim != 3:
            raise ValueError(f"unexcepted dim for reference points: {reference_points.shape}")

        # flatten
        if img_feats is not None:
            BN, C, H, W = img_feats.shape
            feat_flatten = img_feats.reshape(BN, C, -1).transpose(1, 2)  # BN, HW, C
            spatial_flatten = torch.tensor([[H, W]], device=feat_flatten.device)
            level_start_index = torch.cat((spatial_flatten.new_zeros((1,)), spatial_flatten.prod(1).cumsum(0)[:-1]))
        else:
            feat_flatten = None
            spatial_flatten = None
            level_start_index = None

        if lidar_feats is not None:
            B, C, H, W = lidar_feats.shape
            lidar_feat_flatten = lidar_feats.reshape(B, C, -1).transpose(1, 2)  # B HW, C
            lidar_spatial_flatten = torch.tensor([[H, W]], device=lidar_feat_flatten.device)
            lidar_level_start_index = torch.cat(
                (lidar_spatial_flatten.new_zeros((1,)), lidar_spatial_flatten.prod(1).cumsum(0)[:-1])
            )
        else:
            lidar_feat_flatten = None
            lidar_spatial_flatten = None
            lidar_level_start_index = None

        pc_range = torch.tensor(self.pc_range, device=query_feats.device)
        outs_dec, outs_reference = self.transformer(  # TODO 添加bev feats
            query=query_feats,
            query_pos=query_embeds,  # query pos embeds
            feat_flatten=feat_flatten,
            img_spatial_flatten=spatial_flatten,
            lidar_feat_flatten=lidar_feat_flatten,
            lidar_spatial_flatten=lidar_spatial_flatten,
            level_start_index=level_start_index,
            lidar_level_start_index=lidar_level_start_index,
            attn_masks=attn_mask,
            reference_points=reference_points,
            pc_range=pc_range,
            img_metas=img_metas,
            reg_branches=self.refine_reg_branches,
            query_embedding=query_embedding,
        )

        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(outs_reference.clone())

        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs["center"] + reference[:, :, :, :2]).sigmoid()
            height = (outs["dim"][..., 1:2] + reference[:, :, :, 2:3]).sigmoid()  # 对应第4:5维
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            last_reference_points = torch.cat((center[-1], height[-1]), dim=-1)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs["center"] = _center
            outs["dim"][..., 1:2] = _height

            _reference = center.new_zeros(center.shape)
            _reference[..., 0:1] = (
                reference[:, :, :, 0:1].sigmoid() * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            _reference[..., 1:2] = (
                reference[:, :, :, 1:2].sigmoid() * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            # roi_mask
            outs["reference"] = _reference

            all_bbox_preds = torch.cat(
                (  # look like cx, cy, cz, w, l, h, rot, but is cx, cy, w, l, cz, h, rot
                    outs["center"],  # preds_dicts[0]: only support task_num = 0
                    outs["height"],
                    outs["dim"],
                    outs["rot"],
                    outs["vel"],
                ),
                dim=-1,
            ).to(outs["center"].dtype)

            if mask_dict and mask_dict["pad_size"] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict["known_lbs_bboxes"][0]
                known_labels_raw = task_mask_dict["known_labels_raw"]
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [torch.where(known_lbs_bboxes_label == class_name.index(i) + flag) for i in class_name]
                task_masks_raw = [torch.where(known_labels_raw == class_name.index(i) + flag) for i in class_name]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict["known_lbs_bboxes"] = (new_lbs_bboxes_label, task_mask_dict["known_lbs_bboxes"][1])
                task_mask_dict["known_labels_raw"] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs["dn_" + key] = outs[key][:, :, : mask_dict["pad_size"], :]
                    outs[key] = outs[key][:, :, mask_dict["pad_size"] :, :]
                outs["dn_mask_dict"] = task_mask_dict

            outs.update(
                {
                    "all_cls_scores": outs["cls_logits"].to(outs["center"].dtype),
                    "all_bbox_preds": all_bbox_preds,
                    "enc_cls_scores": None,
                    "enc_bbox_preds": None,
                    "query_feats": outs_dec[-1],
                    "reference_points": last_reference_points,
                }
            )

            ret_dicts.append(outs)

        return ret_dicts

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, preds_dicts, roi_mask, fov_boardline=None, **kwargs):
        """ "Loss function.
        Args:
            gt_bboxes_3d (dict): (batch_size, num_gts, 10).
                                ff_gt_bboxes_list: gt_boxes
                                ff_gt_labels_list: gt_labels.
                                    same as CMTFusionHead, ff_gt_labels_list starts from 1
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ff_gt_bboxes_list = gt_bboxes_3d["ff_gt_bboxes_list"]
        ff_gt_labels_list = gt_bboxes_3d["ff_gt_labels_list"]
        batch_size = len(ff_gt_bboxes_list)

        gt_boxes_3d_list = [ff_gt_bboxes_list[i] for i in range(batch_size)]
        gt_labels_3d_list = [ff_gt_labels_list[i] - 1 for i in range(batch_size)]

        num_decoder = preds_dicts[0]["center"].shape[0]
        all_pred_bboxes, all_pred_logits, all_query_bboxes = (
            collections.defaultdict(list),
            collections.defaultdict(list),
            collections.defaultdict(list),
        )

        for dec_id in range(num_decoder):
            pred_bbox = torch.cat(
                (  # look like cx, cy, cz, w, l, h, rot, but is cx, cy, w, l, cz, h, rot
                    preds_dicts[0]["center"][dec_id],  # preds_dicts[0]: only support task_num = 0
                    preds_dicts[0]["height"][dec_id],
                    preds_dicts[0]["dim"][dec_id],
                    preds_dicts[0]["rot"][dec_id],
                    preds_dicts[0]["vel"][dec_id],
                ),
                dim=-1,
            )
            all_pred_bboxes[dec_id].append(pred_bbox)
            all_pred_logits[dec_id].append(preds_dicts[0]["cls_logits"][dec_id])
            all_query_bboxes[dec_id].append(preds_dicts[0]["reference"][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]
        all_query_bboxes = [all_query_bboxes[idx] for idx in range(num_decoder)]

        loss_cls, loss_bbox = multi_apply(
            self.loss_single,
            all_pred_bboxes,
            all_pred_logits,
            [gt_boxes_3d_list for _ in range(num_decoder)],
            [gt_labels_3d_list for _ in range(num_decoder)],
            [roi_mask for _ in range(num_decoder)],
            all_query_bboxes,
            [fov_boardline for _ in range(num_decoder)],
        )

        loss_dict = dict()
        loss_dict["loss_cls"] = loss_cls[-1]
        loss_dict["loss_bbox"] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1], loss_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1

        if self.use_dn:  # and "dn_mask_dict" in preds_dicts[0]:
            dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
            dn_mask_dicts = collections.defaultdict(list)
            for dec_id in range(num_decoder):
                if "dn_mask_dict" in preds_dicts[0]:
                    pred_bbox = torch.cat(
                        (
                            preds_dicts[0]["dn_center"][dec_id],
                            preds_dicts[0]["dn_height"][dec_id],
                            preds_dicts[0]["dn_dim"][dec_id],
                            preds_dicts[0]["dn_rot"][dec_id],
                            preds_dicts[0]["dn_vel"][dec_id],
                        ),
                        dim=-1,
                    )
                    dn_pred_bboxes[dec_id].append(pred_bbox)
                    dn_pred_logits[dec_id].append(preds_dicts[0]["dn_cls_logits"][dec_id])

                    dn_mask_dicts[dec_id].append(preds_dicts[0]["dn_mask_dict"])
                else:
                    # 用来传参
                    pred_bbox = torch.cat(
                        (  # look like cx, cy, cz, w, l, h, rot, but is cx, cy, w, l, cz, h, rot
                            preds_dicts[0]["center"][dec_id],  # preds_dicts[0]: only support task_num = 0
                            preds_dicts[0]["height"][dec_id],
                            preds_dicts[0]["dim"][dec_id],
                            preds_dicts[0]["rot"][dec_id],
                            preds_dicts[0]["vel"][dec_id],
                        ),
                        dim=-1,
                    )
                    dn_pred_bboxes[dec_id].append(pred_bbox)
                    dn_pred_logits[dec_id].append(preds_dicts[0]["cls_logits"][dec_id])
                    dn_mask_dicts[dec_id].append(None)

            dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
            dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
            dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]

            dn_loss_cls, dn_loss_bbox = multi_apply(self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts)

            loss_dict["dn_loss_cls"] = dn_loss_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_loss_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1], dn_loss_bbox[:-1]):
                loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
                num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_e2e_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas["box_type_3d"][i](bboxes, bboxes.size(-1))
            scores = preds["scores"]
            labels = preds["labels"]
            obj_idxes = preds["obj_idxes"]
            track_scores = preds["track_scores"]
            forecasting = preds["forecasting"]
            velocity = preds["velocity"]
            if velocity is not None:
                ret_list.append([bboxes, scores, labels, obj_idxes, track_scores, forecasting, velocity])
            else:
                ret_list.append([bboxes, scores, labels, obj_idxes, track_scores, forecasting])
        return ret_list
