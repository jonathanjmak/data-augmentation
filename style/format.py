import os
import torch
import pickle

from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":
  for dir in glob("custom-output-images*"):
    for path in sorted(list(glob(dir + "/*"))):
      parent, fn = path.split("/")
      idx = int(fn.split("_")[0])
      new_fn = f"{idx:05}_{'_'.join(fn.split('_')[1:])}"
      if fn != new_fn:
        os.rename(parent + "/" + fn, parent + "/" + new_fn)

  label_to_num = {k: i for i, k in enumerate(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])}
  to_tensor = transforms.ToTensor()
  for dir in glob("custom-output-images*"):
    for label, num in label_to_num.items():
      data = None
      for path in tqdm(sorted(list(glob(dir + f"/*{label}*.jpg")))):
        if label in path:
          img = to_tensor(Image.open(path))[None,:,:,:]
          # label = path.replace("-","_").split("/")[-1].split("_")[1]
          # num = label_to_num[label]
          if data is None:
            data = img
          else:
            data = torch.cat((data, img))
      print(data.size())
      with open(f"datasets/{dir.split('-')[-1]}_data_{num}.pkl", "wb") as f:
        pickle.dump(data, f)
