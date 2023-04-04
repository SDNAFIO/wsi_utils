import torch
import numpy as np
import os
import re
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from wsi_core.WholeSlideImage import WholeSlideImage
import torchvision.transforms as transforms
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles


class TCGAData(Dataset):
    def __init__(self, dir):
        super().__init__()

        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.coords_file_name = 'patchcoords.pkl'

        self.seg_params = {'seg_level': 3, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [],
                           'exclude_ids': []}
        self.filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
        self.vis_params = {'vis_level': 3, 'line_thickness': 250}  # This is to visualize the mask
        self.patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt',
                             'patch_level': 0, 'patch_size': 256, 'step_size': 256, 'save_path': 'test_out/patches',
                             'custom_downsample': 1}

        self.patch_coords = []
        self.wsis = []

        self.dir = dir
        self.parse_dir(dir)

        self.patch_lens = [len(x) for x in self.patch_coords]
        self.cum_lens = np.cumsum(self.patch_lens)

    def __len__(self):
        return np.sum(self.patch_lens)

    def __getitem__(self, idx):
        slide_idx = np.searchsorted(self.cum_lens, idx, side='right')
        if slide_idx != 0:
            patch_idx = idx - self.cum_lens[slide_idx - 1]
        else:
            patch_idx = idx

        y, x = self.patch_coords[slide_idx][patch_idx]
        patch_PIL = self.wsis[slide_idx].wsi.read_region((x,y), self.patch_params['patch_level'],
                                                         (self.patch_params['patch_size'], self.patch_params['patch_size'])).convert('RGB')
        if self.patch_params['custom_downsample'] > 1:
            patch_PIL = patch_PIL.resize((self.patch_params['target_patch_size'],
                                          self.patch_params['target_patch_size']))

        #patch_info = {'x':x // (self.patch_params['patch_downsample'][0] * self.patch_params['custom_downsample']),
        #                      'y':y // (self.patch_params['patch_downsample'][1] * self.patch_params['custom_downsample']),
                #              }

        tensor_img = self.transform(patch_PIL)
        return tensor_img

    def parse_dir(self, dir):
        dir_content = os.listdir(dir)
        print(dir_content)
        slide_folders = [x for x in dir_content if re.match(r".+-.+-.+-.+", x)]
        slide_folders = sorted(slide_folders)

        for idx, slide_folder in enumerate(slide_folders):
            if idx == 1:
                break

            folder_content = os.listdir(os.path.join(dir, slide_folder))
            svs_path = [x for x in folder_content if ('.svs' in x and not '.partial' in x)]
            if len(svs_path) == 0:
                print(f'Skipping {svs_path} due to missing .svs')
                continue
            svs_path = os.path.join(dir, slide_folder, svs_path[0])

            coord_path = [x for x in folder_content if self.coords_file_name in x]

            wsi_img = WholeSlideImage(svs_path)
            self.wsis.append(wsi_img)

            if len(coord_path) == 0:
                print(f'Prefetching Patch-coords for: {svs_path}...')

                # Execute Segmentation
                wsi_img.segmentTissue(**self.seg_params, filter_params=self.filter_params)

                # Get the mask
                mask = wsi_img.visWSI(**self.vis_params)

                mask.save(os.path.join(dir, slide_folder, 'mask.png'))

                # Visualize the mask
                plt.imshow(mask)
                plt.show()

                # Perform patching
                contours = wsi_img.contours_tissue
                for idx, cont in enumerate(contours):
                    patch_coords = wsi_img._getPatchGenerator(cont, idx, **self.patch_params)
                    pkl.dump(patch_coords, open(os.path.join(dir, slide_folder, self.coords_file_name), 'wb'))
                    self.patch_coords.append(patch_coords)
            else:
                print(f'Patch-coords already extracted for: {svs_path}')
                self.patch_coords.append(pkl.load(open(os.path.join(dir, slide_folder, self.coords_file_name), 'rb')))


if __name__ == '__main__':
    dir = '/mnt/storage/wsi/hissl/resources/data/tcga_bc'
    data = TCGAData(dir)
    loader = DataLoader(data, batch_size=4, shuffle=True)

    for batch in loader:
        plt.imshow(batch[0].permute(1,2,0))
        plt.show()