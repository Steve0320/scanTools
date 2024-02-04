from torch.utils.data import Dataset
import torchvision.io as tvio


# Optimized dataloader to U2Net inference. Expects to be initialized with a
# list of image file names.
class U2Dataset(Dataset):

    def __init__(self, image_name_list, transform=None):
        self.image_name_list = image_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # Read in file as tensor (tensor is 3xHxW)
        image = tvio.read_image(self.image_name_list[idx], tvio.ImageReadMode.RGB)

        # We need to hang on to some additional context info, so package into a structure
        # return {'filename': self.image_name_list[idx], 'image': image, 'original_image': original_image}
        return {'filename': self.image_name_list[idx], 'image': image}
