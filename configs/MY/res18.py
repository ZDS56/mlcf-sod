_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/dy_coco.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        #with_cdfa=True,           # 启用CDFA
        #cdfa_num_heads=8,         # 注意力头数
        #cdfa_kernel_size=3,
       #cdfa_stride=1,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint='weights/resnet18-f37072fd.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=512,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaDYHead',
        num_classes=1,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        cls_base=-8,
        reg_base=0,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=200
            ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=500))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

find_unused_parameters = True

# auto_scale_lr = dict(base_batch_size=16)
