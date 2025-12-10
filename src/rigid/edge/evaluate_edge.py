import argparse
import torch
import os
from tqdm import tqdm
import scipy.io as sio
from dataloaders.datasets.bsds_hd5_dim1 import Mydataset
from torch.utils.data import DataLoader
from modeling.rindnet_edge import *

            
def write_filenames_to_txt(folder_path, output_file):
    file_names = os.listdir(folder_path)
    file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]
    with open(output_file, 'w') as file:
        for file_name in file_names:
            absolute_path = os.path.abspath(os.path.join(folder_path, file_name))
            file.write(absolute_path + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='bsds',
                        choices=['bsds'], help='dataset name (default: pascal)')
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--data_path", type=str, help="path to the training data",
                        default="./")
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--loss-type', type=str, default='attention',
                        choices=['ce', 'focal', 'attention'],
                        help='loss func type (default: ce)')
    # test hyper params
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--evaluate-model-path', type=str, default='./epoch_70_checkpoint.pth.tar')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    write_filenames_to_txt(args.input_path, './test.lst')


    model_dict = torch.load(args.evaluate_model_path, map_location='cpu')
    checkpoint_dict = model_dict['state_dict']
    model = MyNet()
    model.load_state_dict(checkpoint_dict)
    model.cuda()
    model.eval()

    test_dataset = Mydataset(root_path=args.data_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        for batch_index, images in enumerate(tqdm(test_loader)):
            name = test_loader.dataset.images_name[batch_index]
            image = images.cuda()
            with torch.no_grad():
                edge = model(image)

            edge = edge.data.cpu().numpy()
            edge = edge.squeeze()
            from PIL import Image
            edge_image = Image.fromarray((255 * edge).astype('uint8'))
            edge_image.save(os.path.join(args.output_dir, f'{name}.png'))



