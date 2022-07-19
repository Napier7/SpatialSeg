cfg = dict(
    train = dict(
        max_iters = 160000,    # total iters
        interval = 4000,       # interval iters to validation or save
        cuda = True,           # wheather use CUDA
        resume = False,        # wheather use checkpoint
        checkpoint_root = r'../autodl-tmp/checkpoints/water/DeeplabV3/20220702',          # checkpoint save root 
        checkpoint = r'../autodl-tmp/checkpoints/green/segformer/202206019/Iter152000-Train_Loss0.5477-Val_Loss0.5582.pth',      # checkpoint loading path 
    ),
    # 模型配置
    model = dict(
        type = 'deeplabv3',     # model type, (you can use other model by config 'model/__init__.py')
        n_class = 2             # output classes
    ),
    # 数据集路径
    dataset = dict(
        im_root=r'../autodl-tmp/water-samples/img',                         # img root path
        label_root=r'../autodl-tmp/water-samples/label',                    # label root path
        train_im_anns=r'../autodl-tmp/water-samples/config/train.txt',     # train set order
        val_im_anns=r'../autodl-tmp/water-samples/config/val.txt',         # val set order
    ),
    # 数据载入配置
    dataloader = dict(
        isShuffle = True,       # whether to shuffle the internal order of the dataset                        
        batch_size = 2,         # The data size of once input                          
        num_worker = 4,         # The data size of preload                      
    ),
    # 优化器配置
    optimizer = dict(
        lr_start= 1e-4,         # learning rate at beginning
        weight_decay = 0.01,    # weight_decay to optimize parameters
        betas = (0.9, 0.999),   # a param of optimizer
    ),
    # 数据增强
    transform = dict(
        scales = [0.75, 2],         # random crop scales
        cropsize = [1024, 1024],    # random crop size
        mean = (0.3293, 0.3269, 0.3062), # the mean of the dataset
        std = (0.1515, 0.1583, 0.1889)  # standard deviation of the dataset
    )


)