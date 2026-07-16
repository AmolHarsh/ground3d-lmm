_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['uniseg3d'])
find_unused_parameters = True

# model settings
num_channels = 32
num_instance_classes = 198
num_semantic_classes = 200

pred_iou = True
use_pseudo_cls_supervise = True
inst_weight = 1.0
sem_weight = 1.0
hype_lambda = 1.0
contra_hype_lambda = 1.0
class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
    'bathtub', 'otherfurniture']

model = dict(
    type='Grounded_LMM_main',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.02,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    query_thr=0.5,
    inst_test_iou = False,
    pano_test_iou = True,
    pred_iou = pred_iou,
    set_query_mask=True,
    set_all_mask=True,
    is_type_embedding=True,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='Grounded_Decoder_Eval_Text',
        num_layers=6,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        sphere_cls = True,
        vocabulary_cls_embedding_path = 'data/scannet/scannet_cls_embedding.pth',
        target_classes = class_names,
        qwen_model_path='Qwen/Qwen3-VL-4B-Instruct',
        lora_type='qkvo_all',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        lora_bias='none',
        base_model='qwen3_vl',
        max_point_tokens=3000,
        save_pred_qa_dir='pred_qa_scanrefer'),
    criterion=dict(
        type='Part3DGlammCriterion_v2',
        lang_seg_criterion=dict(
            type='TextPrompt_Criterion_Glamm',
            fix_dice_loss_weight=True,
            fix_mean_loss=True,
            loss_weight = [1.0, 1.0],
            total_weight = 1.0),
        lang_criterion=dict(
            type='CoTCriterion',
            loss_weight=1.0)),

    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1]))

# dataset settings
dataset_type = 'ScanNetUnifiedSegDataset'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=False,
        with_sp_mask_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='AddSuperPointAnnotations_ScanRefer',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1],
        merge_non_stuff_cls=False),
    dict(type='QA_Generation_ScanRefer',
         reason3d_file='data/scannet/oneformer3d_scanrefer/ScanRefer_filtered_train_by_scene.json',
         num_qa=10),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.02,
        p=0.5),
    dict(
        type='Pack3DDetInputs_Reason3D',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'selected_qa_data',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords','is_novel', 'sp_gt_seg',
            'pts_instance_objextId_shuffle', 'point_prompt_distance_norms', 'scene_name'
        ])
        ]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5]),
            dict(
                type='AddSuperPointAnnotations_ScanRefer',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False),
            dict(type='QA_Generation_ScanRefer_Test',
                reason3d_file='data/scannet/oneformer3d_scanrefer/ScanRefer_filtered_val_by_scene.json'),
        ]),
    dict(
        type='Pack3DDetInputs_Reason3D',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'selected_qa_data',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords','is_novel', 'sp_gt_seg',
            'pts_instance_objextId_shuffle', 'point_prompt_distance_norms', 'scene_name'
        ])
]

data_root = 'data/scannet/oneformer3d_scanrefer'
# run settings
train_dataloader = dict(
    batch_size=1,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        ann_file='scanrefer_infos_train.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='scanrefer_infos_val.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='Part3DGlamm_SegMetric',
    collect_device='gpu',
    min_num_points=1,
    id_offset=2**16)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'backbone': dict(lr_mult=0.01),
            'reason': dict(lr_mult=0.1),
            'decoder': dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)

epoch = 10

param_scheduler = dict(type='PolyLR', begin=0, end=epoch, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

default_hooks = dict(
    checkpoint=dict(interval=1,
                    max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50),)

load_from = 'work_dirs/ground3d_scanrefer_3d/pytorch_model.pth'  # released HF checkpoint; eval also accepts the ckpt as a CLI arg

randomness = dict(
    seed=1,
    diff_rank_seed=True,
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epoch,
    val_interval=2,
    dynamic_intervals=[(1, 2)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

env_cfg = dict(
    dist_cfg=dict(backend='nccl', timeout=7200))

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False
)
