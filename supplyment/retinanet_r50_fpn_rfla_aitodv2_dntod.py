# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5),
    neck_agg=None,
    bbox_head=dict(
        type='DN_RetinaHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        device = 'cuda:0',
        w1 = 1.0,
        cls_alpha = 0.5,
        with_CSR = [True, True],
        with_RBR = True,
        save_path = '/data/zhr/DN-TOD/retinanet_save_path/retinanet_second_noise_0.0',
        anchor_generator=dict(
            type='RFGenerator', # Effective Receptive Field as prior
            fpn_layer='p2', # bottom FPN layer P2
            fraction=0.5, # the fraction of ERF to TRF
            strides=[4, 8, 16, 32, 64]),
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
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HieAssigner', # Hierarchical Label Assigner (HLA)
            ignore_iof_thr=-1,
            gpu_assign_thr=512,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='kl', # KLD as RFD for label assignment
            topk=[3,1],
            ratio=0.9), # decay factor
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
