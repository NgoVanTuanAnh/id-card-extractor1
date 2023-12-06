import albumentations as A

def get_augumentation_pipelines():    
    augumentations = A.Sequential([
        # A.InvertImg(p=0.2),
        A.OneOf([
            A.ChannelDropout(p=0.4),
            A.ChannelShuffle(p=0.4),
            A.HueSaturationValue(p=0.4),
            A.Equalize(p=0.4),
            A.RandomBrightness(),
            A.RandomBrightnessContrast(),
        ], p=0.2),
        A.OneOf([
            A.OneOf([
                A.GaussianBlur(),
                A.MedianBlur(blur_limit=(3, 5)),
                A.MotionBlur(3),
                A.AdvancedBlur(blur_limit=(3, 5))
            ]),
            A.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5))
        ]),
        A.OneOf([
            A.CLAHE(clip_limit=4, tile_grid_size=(8, 8)),
            A.Emboss(p=0.2),
            A.GaussNoise(var_limit=(50, 150)),
            A.ISONoise(color_shift=(0.1, 0.3), intensity=(0.25, 0.65)),
        ]),
        A.OneOf([
            A.RandomRain(),
            A.RandomFog(),
            A.RandomShadow(),
            A.RandomSunFlare()
        ])
    ])
    
    return augumentations