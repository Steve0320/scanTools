import os
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Normalize, ToPILImage
import torch.nn.functional as F
from torch.multiprocessing import Pool
import itertools
from PIL import Image

from data_loader import U2Dataset
from model import U2NET


# Write out the given tensor data to an image
def write_image(filename, data, out_dir, extension):
    image_name = os.path.splitext(os.path.basename(filename))[0]
    path = os.path.join(out_dir, image_name) + os.extsep + extension
    ToPILImage()(data).save(path)


def batch_remove_background(input_dir=os.path.join(os.getcwd(), 'inputs'),
                            output_dir=os.path.join(os.getcwd(), 'outputs'),
                            model_path=os.path.join(os.getcwd(), 'models', 'u2net.pth'),
                            image_size=None,
                            write_concurrency=torch.multiprocessing.cpu_count(),
                            dataloader_workers=4,
                            batch_size=5,
                            pin_memory=False,
                            background_fill=None,
                            output_format=None):

    if background_fill is None:
        background_fill = [0, 0, 0]

    # Enable transparency support if alpha channel provided
    enable_transparency = len(background_fill) == 4

    if output_format is None:
        output_format = 'png' if enable_transparency else 'jpg'

    image_names = glob.glob(input_dir + os.sep + '*')
    print(f"Running inference on {len(image_names)} images...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if image_size is None:
        img = Image.open(image_names[0])
        w, h = img.size
        image_size = (h, w)

    # Load model
    print("Loading model...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path))
    net.cuda().eval()

    # Prepare data loader
    dataset = U2Dataset(image_name_list=image_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=pin_memory)

    # Process loaded data. Dimensions BxCxHxW
    with tqdm(total=len(dataset)) as pbar:
        for batch_sample in dataloader:

            filenames = batch_sample['filename']
            o_images = batch_sample['image']

            # Move full image to GPU for processing. We don't expand to float until we get to the GPU to
            # save on some host memory. We also keep a copy of the original image in host memory so that we
            # don't have to reload it later.
            images = o_images.to('cuda').float()
            del batch_sample['image']

            # Downsample to the expected 320x320 model input. This also discards the original image reference
            # since it's very large and we no longer need it.
            images = F.interpolate(images, size=(320, 320), mode='bilinear')

            # Squash values to 0-1 and apply standard data normalization using ImageNet values
            images = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images / images.max())

            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = net(images)

            del images, d2, d3, d4, d5, d6, d7

            # Normalize (again) from 0-1, expand to full image size, and shuttle back to host memory
            d_min, d_max = d1.aminmax()
            mask = F.interpolate((d1 - d_min) / (d_max - d_min), size=image_size, mode='bilinear').round().byte().to('cpu')
            del d1

            # Repeat the given background color into a form that we can combine with the original image
            rgb = torch.tensor(background_fill, dtype=torch.uint8).expand(image_size[0], image_size[1], len(background_fill)).permute(2, 0, 1)

            if enable_transparency:
                o_images = F.pad(o_images, (0, 0, 0, 0, 0, 1), 'constant', 255)

            # Expand calculated mask to full image size and shuttle it back to host memory. We do the masking step
            # on the CPU since it's a quick operation, and so we don't have to keep the full image loaded into the GPU
            # during the rest of the calculations.
            results = torch.where(mask > 0, o_images, rgb)
            del o_images, mask, rgb

            # Image writing with PIL can be slow, so parallelize the operations.
            with Pool(write_concurrency) as p:
                p.starmap(write_image, zip(filenames, results, itertools.repeat(output_dir), itertools.repeat(output_format)))

            del results

            pbar.update(dataloader.batch_size)


if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    batch_remove_background()
    # torch.cuda.memory._dump_snapshot('testsnapshot6')
    # torch.cuda.memory._record_memory_history(enabled=None)
