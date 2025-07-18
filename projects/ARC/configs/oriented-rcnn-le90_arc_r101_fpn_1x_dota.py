_base_ = [
    '../../../configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'
]

custom_imports = dict(
    imports=['projects.ARC.arc'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        type='ARCResNet',
        depth=101,
        replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
             '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'],
            ['0', '1', '2']
        ],
        kernel_number = 4,
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/'
                                 'ARC/ARC_ResNet101_xFFF_n4.pth')),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.5)})
)

train_cfg = dict(val_interval=12)

# base_batch_size = (1 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2, enable=False)