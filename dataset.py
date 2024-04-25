import os
import numpy as np
from PIL import Image
import config
from torch.utils.data import Dataset, DataLoader

class MyImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        image = Image.open(img_path)
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image

def test():
    transform = config.both_transforms  # Example using a transform from config
    dataset = MyImageFolder(root_dir="new_data/", transform=transform)
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for image in loader:
        print(image.shape)

if __name__ == "__main__":
    test()
