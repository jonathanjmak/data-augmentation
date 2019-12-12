import os
import pickle
import argparse
import shutil
import torch
import numpy as np
import vgg

from torchvision.utils import save_image
from torch.autograd import Variable

from tqdm import tqdm, trange
from main import Generator, Discriminator


def sample_image(n_per_class, batches_done, generator, threshold):
    if threshold is not None:
        discriminator = vgg.__dict__["vgg19"](n_classes=2).eval()
        discriminator.features = torch.nn.DataParallel(discriminator.features)
        checkpoint = torch.load("../vggnet/run_logs/12_02_03_01_55__threshold/checkpoint_24.tar")
        discriminator.load_state_dict(checkpoint['state_dict'])
        discriminator.cuda()

        dir = f'thresholded_datasets/thresh_{threshold:.2f}'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        print("Not thresholding")
        if not os.path.exists('new_dataset'):
            os.makedirs("new_dataset")

    # Get labels ranging from 0 to n_classes for n rows
    for i in range(n_classes):
        labels = (torch.ones(n_per_class).cuda() * i).long()
        if threshold is not None:
            thresholded_gen_imgs = None
            with tqdm(total=n_per_class) as pbar:
                while thresholded_gen_imgs is None or len(thresholded_gen_imgs) < n_per_class:
                    z = Variable(FloatTensor(np.random.normal(0, 1, (n_per_class, latent_dim))))
                    gen_imgs = generator(z, labels).detach()
                    realness = discriminator(gen_imgs).detach().squeeze()
                    keep = realness >= threshold
                    mask = torch.nonzero(keep)
                    keep_imgs = gen_imgs[mask].squeeze(dim=1)

                    if thresholded_gen_imgs is None:
                        thresholded_gen_imgs = keep_imgs
                    else:
                        thresholded_gen_imgs = torch.cat((thresholded_gen_imgs, keep_imgs))
                    pbar.update(len(keep_imgs))
                    pbar.set_description(f"threshold: {threshold}, class: {i}, num_imgs: {min(len(thresholded_gen_imgs), n_per_class)}")

            # Save sample
            save_image(thresholded_gen_imgs[:10].data, f"{dir}/sample_{i}.png", nrow=10, normalize=True)

            thresholded_gen_imgs = thresholded_gen_imgs[:n_per_class]
            assert len(thresholded_gen_imgs) == n_per_class

            with open(f"{dir}/data_{i}.pkl", "wb") as f:
                pickle.dump(thresholded_gen_imgs, f)
        else:
            with open(f"new_dataset/data_{i}.pkl", "wb") as f:
                pickle.dump(gen_imgs, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--threshold", default=None, type=float)
    args = parser.parse_args()
    print(args)

    latent_dim = 100
    n_classes = 10
    img_shape = (3, 32 ,32)

    generator = Generator(10,100, img_shape)
    generator.load_state_dict(torch.load("models/generator_epoch_199.pth"))
    generator = generator.cuda().eval()


    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    sample_image(5000, 0, generator, args.threshold)
