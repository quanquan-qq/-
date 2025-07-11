""" Z10 8V1L
Description:  Trained with Z10 labeled
Cmd: DET3D_EXPID=$(cat /proc/sys/kernel/random/uuid) rlaunch --max-wait-duration=72h -P 4  --cpu=48 --gpu=8 --memory=300000 --private-machine yes --preemptible no -- python3 perceptron/exps/end2end/private/object/det/det_private_base_exp_8v1l_y300x32_deformable.py --no-clearml -b 4 -e 80 --amp --sync_bn 4
Author: GXT
Data: 2024-03-02

pretrained_model: s3://gxt-share-qy/det-ckpt/merge/merge_8v1l_weight_80e_80e_0415.pth

ckpt: s3://gxt-share-qy/det-ckpt/det__det_private_base_exp_8v_y300x32_deformable_120m_400q_lr_5e4_refine_18w_fixoptim/2025-04-15T14:18:31/checkpoint_epoch_21.pth

[Rank #0] | 2025-04-21 at 11:51:42 | INFO | Config:cam_l3_cq_sixclass

[Rank #0] | 2025-04-21 at 11:51:42 | INFO | mAP:0.7027511856925344

[Rank #0] | 2025-04-21 at 11:51:42 | INFO | |-----------------|-------|--------------------|--------------------|---------------------|---------------------|--------------------|---------------------|----------------------|---------------------|--------------------|
| merged_category | count |         AP         |     max_recall     |      trans_err      |     trans_err_x     |    trans_err_y     |      scale_err      |      orient_err      |     speed_err_x     |    speed_err_y     |
|-----------------|-------|--------------------|--------------------|---------------------|---------------------|--------------------|---------------------|----------------------|---------------------|--------------------|
|       Bus       |  648  | 0.4925508656627751 | 0.5740000000000001 |  0.8559294356941897 | 0.18614914666265764 | 0.8130576055129535 | 0.20688985364171567 | 0.06885830184795014  |  0.5269367906141427 | 3.9767416389494286 |
|       Car       | 73772 | 0.9478247686580371 | 0.9560000000000001 | 0.47180223531730214 | 0.08452131032070036 | 0.4503154894555146 | 0.09874904958148403 | 0.027975399851315648 |  0.2565900854401694 | 5.3699977100382466 |
|      Cycle      |  3958 | 0.6649518898445643 |        0.76        |  0.6886331061628448 |  0.2467799408779298 | 0.5891018423394812 |  0.2006131828620519 |  0.2858342433992601  |  1.547126702806799  | 2.206865286335579  |
|    Pedestrian   |  2026 | 0.6502185849020632 |       0.803        |  0.7267171617391153 | 0.23193599343480606 | 0.6438915528372596 | 0.23575604604200434 |  0.6496890166487919  |  0.8675066031896509 | 0.475002833427781  |
|     Tricycle    |  461  | 0.7125507840469151 |       0.754        | 0.43047469184080345 | 0.13406533016401728 | 0.3864236570206722 |  0.2027012770865078 | 0.17505497535255635  |  0.5000141784941678 | 1.8607180407240969 |
|      Truck      |  2787 | 0.7484102210408512 |       0.837        |  0.6257198271189667 |  0.1276155442496775 | 0.586879809109245  | 0.17016699541510746 | 0.03612084550143616  | 0.26436786082081265 | 2.5885593473831316 |
|-----------------|-------|--------------------|--------------------|---------------------|---------------------|--------------------|---------------------|----------------------|---------------------|--------------------|

"""
import sys
import refile
import torch
import mmcv
from perceptron.engine.cli import Det3DCli
from perceptron.utils import torch_dist as dist
from perceptron.layers.lr_scheduler import WarmCosineLRScheduler

from perceptron.exps.end2end.private.object.model_cfg.det_model_cfg_8v1l_sparse_y120x32m import MODEL_CFG
from perceptron.exps.base_exp import BaseExp

from perceptron.exps.end2end.private.object.data_cfg.det_annos_hf_200m_32m_8v5r1l_mmL_chengqu_Z10_new_fovrange_120 import (
    base_dataset_cfg as DATA_TRAIN_CFG,
    val_dataset_cfg as DATA_VAL_CFG,
)

from perceptron.data.det3d.private.private_multimodal import PrivateE2EDataset
from perceptron.data.sampler import InfiniteSampler
from torch.utils.data import DistributedSampler
from perceptron.models.end2end.perceptron.perceptron import VisionEncoder

seed = 42  # 你可以使用任何整数作为种子

torch.manual_seed(seed)  # 设置 PyTorch 的种子
torch.cuda.manual_seed(seed)  # 如果你使用 GPU，设置 CUDA 的种子
torch.cuda.manual_seed_all(seed)  # 如果你使用多个 GPU，设置所有 GPU 的种子

SOFT_OCC_THRESHOLD = 0.4


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epoch=20,
        **kwargs,
    ):
        super(Exp, self).__init__(batch_size_per_device, total_devices, max_epoch)
        self.exp_name = "__".join(refile.SmartPath(sys.argv.copy()[0]).parts[-2:])[:-3]
        # 1. Training setting
        self.lr = 2e-4  #
        self.init_scale = 512
        self.print_interval = 50
        self.num_keep_latest_ckpt = 5
        self.dump_interval = 1
        self.grad_clip_value = 35
        print("!" * 10, "lr  changed!!!!!!!!!!!!")

        # 2. Dataset and model configuration

        # -------------------------------------Z10 label-------------------------------------
        self.data_train_cfg_cq_z10 = mmcv.Config(DATA_TRAIN_CFG)
        self.data_train_cfg_cq_z10["loader"]["datasets_names"] = ["z10_label_1230_train", "Z10_label_0207_7w"]
        self.data_train_cfg_cq_z10["annotation"]["box"]["label_key"] = "labels"
        self.data_train_cfg_cq_z10["loader"]["only_key_frame"] = True
        self.data_train_cfg_cq_z10["annotation"]["box"]["occlusion_threshold"] = 1
        self.data_train_cfg_cq_z10["annotation"]["box"]["soft_occ_threshold"] = SOFT_OCC_THRESHOLD

        self.data_train_cfg2 = mmcv.Config(DATA_TRAIN_CFG)
        self.data_train_cfg2["loader"]["datasets_names"] = [
            # "z10_label_0401_bmk01",
            "z10_label_0401_teshucheliang",
            "z10_label_0401_mijivru",
        ]
        self.data_train_cfg2["lidar"]["lidar_names"] = ["rfu_front_2_lidar", "fuser_lidar"]
        self.data_train_cfg2["sensor_names"]["lidar_names"] = ["rfu_front_2_lidar", "fuser_lidar"]
        self.data_train_cfg2["annotation"]["box"]["label_key"] = "labels"
        self.data_train_cfg2["loader"]["only_key_frame"] = True
        self.data_train_cfg2["annotation"]["box"]["occlusion_threshold"] = 1
        self.data_train_cfg2["annotation"]["box"]["soft_occ_threshold"] = SOFT_OCC_THRESHOLD
        self.data_val_cfg = mmcv.Config(DATA_VAL_CFG)
        self.model_cfg = mmcv.Config(MODEL_CFG)

        # 3. other configuration change in this function
        self._change_cfg_params()

        self.eval_name = "Z10_eval_bmk_cloud_all_epoch_71"
        # self.data_val_cfg["loader"]["datasets_names"] = [
        #     "z1_label_1230_bmk_qy",
        #     "z10_label_0401_bmk",
        # ]  # ["bmk_new_withOCC"]
        # self.data_val_cfg["loader"]["datasets_names"] = ["z1_label_1230_bmk_qy", "BMK_02"]
        self.data_val_cfg["loader"]["datasets_names"] = ["BMK_CLOUD_ALL"]
        # self.data_val_cfg["loader"]["datasets_names"] = ["BMK_02"]
        # self.data_val_cfg["loader"]["datasets_names"] = ["BMK_598"]
        # self.data_val_cfg["loader"]["datasets_names"] = ["BMK_highway"]
        # self.data_val_cfg["loader"]["datasets_names"] = [
        #     # "s3://tf-rhea-data-bpp/track_labeled/labeled_data/car_z03/20241209_dp-track_yueying_checked/ppl_bag_20241209_112305_det/v0_241212_181736/0083.json", # Truck
        #     # "s3://tf-rhea-data-bpp/track_labeled/labeled_data/car_z01/20241204_dp-track_yueying_checked/ppl_bag_20241204_085828_det/v0_241207_084800/0117.json", # Truck
        #     # "s3://tf-rhea-data-bpp/track_labeled/labeled_data/car_z03/20241208_dp-track_yueying_checked/ppl_bag_20241208_121535_det/v0_241212_235824/0097.json", # Truck

        #     # "s3://tf-rhea-data-bpp/track_labeled/labeled_data/car_z03/20241208_dp-track_yueying_checked/ppl_bag_20241208_121535_det/v0_241212_235824/0098.json", # Tricycle

        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-11/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-15/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-16/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-17/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-24/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-30/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-32/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-36/prelabel_tracking.json",
        #     # "s3://tanwei-share/od_p1010_to_z08/20250429-37/prelabel_tracking.json",
        #     # "s3://megsim/bmk/dataset/tmp_release/20250508/od/clip_67b54e422c32f9e838a430f9_20250508_130314.json",
        # ]
        # self.data_val_cfg["evaluator"]["extra_eval_cfgs"] = ["e2e_l3_far_32m_cq", "cam_l3_cq"]
        self.data_val_cfg["annotation"]["box"]["label_key"] = "labels"
        self.data_val_cfg["lidar"]["lidar_names"] = ["rfu_front_2_lidar", "fuser_lidar"]
        self.data_val_cfg["sensor_names"]["lidar_names"] = ["rfu_front_2_lidar", "fuser_lidar"]

    def _change_cfg_params(self):
        r"""
        This func is designed to change cfg `optionally`. Such as, `open training with checkpint`, `set print interval` \
        which depend on your requirement. For those should be inherited, should be called in `self.__init__`

        Example:
        ----------
        ```
        >>> class YourExp(BaseExp):
                def __init__(self, *args, **kwargs):
                    self._change_cfg_params()

                def _change_cfg_params(self):
                    self.model_cfg["camera_encoder"]["img_backbone_conf"]["with_cp"] = True  # open checkpoint save gpu mem.
                    self.model_cfg["bev_encoder"]["with_cp"] = True  # open checkpoint save gpu mem.
                    self.model_cfg["det_head"]["dense_head"]["with_cp"] = True  # open checkpoint save gpu mem.
                    self.print_interval = 20 # set print interval.
                    pass
                ....
        ```
        """
        self.model_cfg.det_head.bbox_coder.score_threshold = 0.1
        self.model_cfg.num_query = 0 + 300 + 100
        self.model_cfg["det_head"]["init_radar_num_query"] = 0
        self.model_cfg["det_head"]["num_query"] = 0 + 300 + 100
        self.model_cfg["det_head"]["modal"] = ["Lidar", "Camera"]
        self.model_cfg.num_near_query = 100

        self.model_cfg["radar_encoder"] = None
        self.model_cfg["det_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][1]["with_lidar"] = True
        self.model_cfg["det_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][1]["with_img"] = True
        self.model_cfg["det_head"]["transformer"]["decoder"]["num_layers"] = 3
        self.model_cfg["det_head"]["separate_head"]["final_kernel"] = 1
        self.model_cfg["dn_cfg"] = dict(
            use_dn=True, scalar=6, noise_scale=1.0, noise_trans=0.0, split=0.75, dn_weight=1.0
        )
        self.model_cfg["det_head"]["use_roi_mask"] = True  # gs cq 联合训练必须有roi mask

        self.model_cfg["det_head"]["refine_reg_branch"] = True
        self.use_ray_nms = self.model_cfg.get("use_ray_nms", False)

    def _configure_model(self):
        model = VisionEncoder(
            model_cfg=self.model_cfg,
        )
        return model

    def _configure_train_dataloader(self):
        train_dataset1 = PrivateE2EDataset(**self.data_train_cfg_cq_z10)
        train_dataset2 = PrivateE2EDataset(
            **self.data_train_cfg2,
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        train_dataset.batch_postcollate_fn = train_dataset1.batch_postcollate_fn
        train_dataset.batch_preforward_fn = train_dataset1.batch_preforward_fn

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            drop_last=False,
            shuffle=False,
            collate_fn=PrivateE2EDataset.collate_fn,
            sampler=InfiniteSampler(len(train_dataset), seed=self.seed if self.seed else 0)
            if dist.is_distributed()
            else None,
            pin_memory=True,
            num_workers=10,
        )
        return train_loader

    def _configure_val_dataloader(self):
        val_dataset = PrivateE2EDataset(
            **self.data_val_cfg,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=PrivateE2EDataset.collate_fn,
            num_workers=8,
            sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False, seed=42)
            if dist.is_distributed()
            else None,
            # sampler=None,
            pin_memory=True,
        )
        return val_loader

    def _configure_test_dataloader(self):
        val_dataset = PrivateE2EDataset(
            **self.data_val_cfg,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=PrivateE2EDataset.collate_fn,
            num_workers=0,
            sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False) if dist.is_distributed() else None,
            pin_memory=False,
        )
        return val_loader

    def training_step(self, batch):
        if "roi_mask" not in batch:
            batch["roi_mask"] = None
        ret_dict, loss_dict, _ = self.model(**batch)
        loss = sum(_value for _key, _value in loss_dict.items() if "loss" in _key)

        return loss, loss_dict

    @torch.no_grad()
    def test_step(self, batch):

        pred_dicts, _ = self.model(**batch)
        remap_pred_dicts = []
        for pred_dict in pred_dicts:
            remap_pred_dict = {}
            for k, v in pred_dict.items():
                if k == "bboxes":
                    remap_pred_dict["pred_boxes"] = v
                elif k == "labels":
                    remap_pred_dict["pred_" + k] = v
                else:
                    remap_pred_dict["pred_" + k] = v
            if True:  # nms
                from perceptron.data.det3d.modules.utils.post_process import StandardNMSPostProcess

                boxes3d = remap_pred_dict["pred_boxes"]
                top_scores = remap_pred_dict["pred_scores"]
                if top_scores.shape[0] != 0:
                    if not self.use_ray_nms:
                        selected = StandardNMSPostProcess._nms_gpu_3d(
                            boxes3d[:, :7],
                            top_scores,
                            thresh=0.8,
                            pre_maxsize=300,
                            post_max_size=300,
                        )
                    else:
                        selected = StandardNMSPostProcess._ray_nms(
                            boxes3d[:, :7].cpu().numpy(),
                            top_scores.cpu().numpy(),
                            thresh=5.0,
                            pre_maxsize=300,
                            post_max_size=300,
                        )
                    remap_pred_dict["pred_boxes"] = remap_pred_dict["pred_boxes"][selected]
                    remap_pred_dict["pred_scores"] = remap_pred_dict["pred_scores"][selected]
                    remap_pred_dict["pred_labels"] = remap_pred_dict["pred_labels"][selected]
            remap_pred_dicts.append(remap_pred_dict)
        return dict(pred_dicts=remap_pred_dicts)

    def _configure_optimizer(self):
        from torch.optim import AdamW

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": self.lr * 0.1, "weight_decay": 0.01},
                {"params": other_params, "lr": self.lr, "weight_decay": 0.01},
            ]
        )
        return optimizer

    def _configure_lr_scheduler(self):
        scheduler = WarmCosineLRScheduler(
            optimizer=self.optimizer,
            lr=self.lr,
            iters_per_epoch=len(self.train_dataloader),
            total_epochs=self.max_epoch,
            warmup_epochs=0.5,
            warmup_lr_start=1.0 / 3 * self.lr,
            end_lr=1e-6,  # eta_min
        )
        return scheduler


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    Det3DCli(Exp).run()
