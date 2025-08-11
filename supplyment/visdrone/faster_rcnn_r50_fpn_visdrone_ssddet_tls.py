norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add
debug = False
# model settings
det_loss_weight = 4.0
stage_modes=['CBP', 'PBR']
num_stages = 2
model = dict(
    type='ENoiseBox_TLS',
    w1 = 0.5,
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
        num_outs=5,  # 5
        norm_cfg=norm_cfg
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=det_loss_weight), ),  # 1.0
    roi_head=dict(
        type='EP2BplusHead',
        num_stages=num_stages,
        stage_modes=stage_modes,
        top_k=7,
        with_atten=False,
        cluster_mode='mil_cls',  ###'cluster','upper'
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head1=dict(
            type='Shared2FCBBoxHeadSMS',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            device = 'cuda:0',
            frozen_iter = 140000, 
            samples = 200,
            strategy = 0, # 0代表移除，1代表修改
            threshold = (1, 0), 
            w1 = 1.0,
            cls_alpha = 0.5,
            with_CSR = [True, True],
            with_ALC = False,
            with_RBR = True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        bbox_head=dict(
            type='Shared2FCInstanceMILHeadEPLUS',
            num_stages=num_stages,
            stage_modes=stage_modes,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            num_ref_fcs=0,
            with_reg=True,
            with_sem=False,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            with_loss_pseudo=False,
            loss_p2b_weight=1.0,  # 7/19 dididi
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='binary_cross_entropy'),  # weight
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),  # weight
            loss_bbox_ori=dict(
                type='L1Loss', loss_weight=0.25),
            loss_bbox=dict(
                type='L1Loss', loss_weight=0.25),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_per_scale=200,
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.2],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            gen_num_per_box=10,
            iou_thr=0.3,
            gen_num_neg=200,
        ),
        rpn=dict(
            assigner=dict(
                type='HieAssigner', # Hierarchical Label Assigner (HLA)
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl', # KLD as RFD for label assignment
                topk=[3,1],
                ratio=0.9), # decay factor
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))