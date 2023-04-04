import matplotlib.pyplot as plt
import numpy as np
from wsi_core.WholeSlideImage import WholeSlideImage

# Create WSI object
path = 'test_data/TCGA-WN-AB4C-01Z-00-DX1.9A983740-01DA-42EB-BC00-B10A581C519F/3a14c446-3ec2-44aa-8125-6c5609531852/TCGA-WN-AB4C-01Z-00-DX1.9A983740-01DA-42EB-BC00-B10A581C519F.svs'
wsi_img = WholeSlideImage(path)
img = wsi_img.visWSI(top_left=(800,100), bot_right=(900,200))

# Set parameters for segmentation and patching
seg_params = {'seg_level': 3, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [], 'exclude_ids': []}
filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
vis_params = {'vis_level': 3, 'line_thickness': 250}  # This is to visualize the mask
patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt', 'patch_level': 0, 'patch_size': 256, 'step_size': 256, 'save_path': 'test_out/patches', 'custom_downsample': 1}

# Execute Segmentation
wsi_img.segmentTissue(**seg_params, filter_params=filter_params)

# Get the mask
mask = wsi_img.visWSI(**vis_params)

# Visualize the mask
plt.imshow(mask)
plt.show()

# Perform patching
contours = wsi_img.contours_tissue
for idx, cont in enumerate(contours):
    patch_coords = wsi_img._getPatchGenerator(cont, idx, **patch_params)

    for coords in patch_coords:
        x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
        img_patch = np.array(img_patch)[np.newaxis,...]
        img_shape = img_patch.shape

        plt.imshow(img_patch[0])
        plt.show()

        print('')
