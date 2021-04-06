import torch
import pickle
import os
from torchvision.transforms import functional as F
from torchvision import transforms


img_transforms = transforms.Compose([
#     transforms.Resize((800, 1066)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_data_transforms(train):
    tfms = []
    # functional transforms that should run every time
    # (move data to tensor, resize, etc)
    # tfms.append()
    if train:
        # functional transforms that should run only during training
        # ie data aug
        pass
    return Compose(tfms)    

class SunRGBD:
    def __init__(self, root, img_transforms, data_transforms=None, bbox_format='pascal_voc'):
        self.root = root
        self.data = list(sorted(os.listdir(os.path.join(root, "sunrgbd_train_test_data"))))
        self.bbox_format = bbox_format
        # transforms related to loading and normalizing images
        self.img_transforms = img_transforms
        # transforms related to cropping/rotation/data aug that affect
        # image label data
        self.data_transforms = data_transforms
    
    def __getitem__(self, idx):
        
        sample_path = os.path.join(self.root, "sunrgbd_train_test_data", self.data[idx])
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        img = sample['rgb_img']
        
        #sunrgbd bboxes are in coco format:
        # [x_min, y_min, width, height]
        boxes = sample['boxes']['bdb2D_pos']
        if self.bbox_format == 'pascal_voc':
            # change to pascal bboxes
            # [x_min, y_min, x_max, y_max]
            boxes[:,2] = boxes[:,0] + boxes[:,2]
            boxes[:,3] = boxes[:,1] + boxes[:,3]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(sample['boxes']['size_cls'], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
#         target["masks"] = torch.tensor([])
        
        img = self.img_transforms(img)

        if self.data_transforms is not None:
            img, target = self.data_transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.data)