user_root = '/data/zju_YanKH'

# configs/vild/vild_r50_fpn_1x_lvis.py
_base_ = [
    '/data/zju_YanKH' + '/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '/data/zju_YanKH' + '/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/data/zju_YanKH' + '/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/data/zju_YanKH' + '/mmdetection/configs/_base_/default_runtime.py'
]

# classes = ['table', 'nut', 'bolt', 'gear', 'base', 'peg', 'hole']
# classes = ['peg', 'hole']
classes = ['Cylinder-Protrusion', 'Cylinder-Hole', 'Square-Protrusion', 'Square-Hole', 'Groove', 'Tongue']
dataset_type = 'CustomDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadDepthImg', factor_depth=10000),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'target_img_path', 'scale_factor', 'flip', 'flip_direction',
            'rotation', 'translation')
        )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadDepthImg', factor_depth=10000),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'target_img_path', 'rotation', 'translation',
                   'scale_factor'))
]
# # TODO: 自定义数据集Pipeline
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='LoadDepthImg'),
#     dict(type='CustomResize', scale=(224, 224), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='LoadDepthImg'),
#     dict(type='CustomResize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# ----------------------------
# 模型修改：支持 ROI 特征输入（需在 model 中实现）
# ----------------------------
model = dict(
    type='TwoStagePose',
    freeze_modules=['neck', 'rpn_head'],
    # TODO: 合并深度通道
    data_preprocessor=dict(
        type='MergeFeatureDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 0.0],
        std=[58.395, 57.12, 57.375, 1.0],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        in_channels=4,
        norm_eval=True,
        frozen_stages=4,
        init_cfg=dict(
            checkpoint='projects/TwoStagePose/rpn/20251218_135018/epoch_50.pth',
            type='Pretrained',
            prefix='backbone',
        ),
    ),
    # TODO: 合并特征图
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4096],
        out_channels=256,
        num_outs=5,
        init_cfg=dict(
            checkpoint='projects/TwoStagePose/rpn/20251218_135018/epoch_50.pth',
            type='Pretrained',
            prefix='neck',
        ),        
    ),
    rpn_head=dict(
        type='MutipRPNHead',
        num_classes=1,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True),
        init_cfg=dict(
            checkpoint='projects/TwoStagePose/rpn/20251218_135018/epoch_50.pth',
            type='Pretrained',
            prefix='bbox_head',
        ),        
    ),
    roi_head=dict(
        type='PoseHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        attention_head=dict(
            type='AttentionHead',
            hidden_dim=256,
            num_heads=4,
            posed_dim=9,
            topk=4,
            dp=0.0,
            batch_first=True,
            add_sa_module=False,
        ),
        loss_pose=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(num=32),
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(num=32),
        )
    ),
)

# ----------------------------
# 数据集设置
# ----------------------------
data_root = user_root + '/dataset/debug/NOCS'

# 使用你的 Dataset JSON
train_dataloader = dict(
    batch_size=1,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train_crop/train_coco.json',
        data_prefix=dict(img='train_crop/', seg='train_crop/'),
        pipeline=train_pipeline,
        # pipeline=[
        #     dict(type='LoadImageFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        #     # # 自定义：加载 RoI 特征
        #     # dict(
        #     #     type='LoadRoIFeatures',
        #     #     file_path_template='data/lvis_vild/features/train/{img_id}.npy'
        #     # ),
        #     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        #     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        #     dict(type='PackDetInputs',
        #          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
        #                     'scale_factor', 'flip', 'flip_direction',
        #                     'roi_boxes', 'has_roi_features'))  # 确保 roi_boxes 传入
        # ]
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test_crop/test_coco.json',
        data_prefix=dict(img='test_crop/', seg='test_crop/'),
        pipeline=test_pipeline,
        # pipeline=[
        #     dict(type='LoadImageFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        #     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        #     dict(type='PackDetInputs',
        #          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
        #                     'scale_factor', 'flip', 'flip_direction'))
        # ]
    ),
    sampler=dict(shuffle=False, type='DefaultSampler'),
)
test_dataloader = val_dataloader

# ----------------------------
# 评估器：bbox and segm Metric
# ----------------------------
val_evaluator = dict(
    type='Pose6DEvaluator',
    ann_file=data_root + '/test_crop/test_coco.json',
    metric=['proposal'],
    custom_metric=['pose'],
)
test_evaluator = val_evaluator

# ----------------------------
# 训练设置
# ----------------------------
train_cfg = dict(max_epochs=10, val_interval=2)
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, weight_decay=4.0e-5),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'backbone.layer4': dict(lr_mult=0.1)  # 降低骨干网络的学习率（如果未完全冻结）
    #     }
    # )
)
param_scheduler = [
    # NOTE: start_lr = start_factor * base_lr, i.e., 0.01 * 0.32 = 0.0032
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1024),
    dict(type='MultiStepLR', by_epoch=True, begin=0, milestones=[40, 45, 48], gamma=0.1)
]

# log config
default_hooks = dict(logger=dict(interval=64))
# Freeze rpn
# custom_hooks = [
#     dict(type='FreezeModulesHook', modules=['neck', 'rpn_head'])
# ]

env_cfg = dict(
    cudnn_benchmark=True,  # Whether to enable cudnn benchmark
)

work_dir = 'projects/TwoStagePose/attention'