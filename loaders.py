import numpy as np

from DataLoader import FlatDirectoryImageDataset, \
    get_transform, get_data_loader, FoldersDistributedDataset


def loaders(args):
    data_source = FlatDirectoryImageDataset if not args.folder_distributed \
        else FoldersDistributedDataset
    # FlatDirectoryImageDataset(Dataset): <用于通用平面目录图像数据集的pyTorch数据集包装器>
    # args.folder_distributed: <“images”目录是否包含文件夹>
    # FoldersDistributedDataset: <用于mnist数据集的pyTorch数据集包装器>

    dataset = data_source(
        args.images_dir,
        # 获取输入数据所需的图像变换
        # new_size -调整后图像的大小   flip_horizontal -是否随机镜像输入图像进入翻译页面
        transform=get_transform((int(np.power(2, args.depth + 1)),  # power(x, y) 函数，计算 x 的 y 次方。
                                 int(np.power(2, args.depth + 1))),
                                flip_horizontal=args.flip_augment))
    # 从给定的数据集生成data_loader
    # dataset - F2T(?)数据集   batch_size - 数据的批处理大小   num_workers - 并行的数量
    data = get_data_loader(dataset, args.batch_size, args.num_workers)
    print("Total number of images in the dataset:", len(dataset))

    return data
