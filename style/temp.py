from torchvision import datasets
from torchvision.utils import save_image
ds = datasets.CIFAR10(
    "./data",
    train=True,
    download=True,
)

seen = set()
for im, l in ds:
  if l not in seen:
    seen.add(l)
    print(im)
    print(l)
    im.save(f"{l}.png")
