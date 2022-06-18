import PIL.Image
import torch
import torchvision
from torchvision import ops
import torchvision.transforms.functional as TF
from pycocotools import mask as maskUtils
from torchvision import transforms
from util import box_ops
import utils

colormap =  {
    1: (36,179,83),
    2: (131,224,112),
    3: (51,221,255),
    4: (250,50,83),
    5: (255,96,55),
    6: (184,61,245),
    7: (50,183,250),
    8: (255,204,51),
    9: (89,134,179),
    10: (66,201,18),
    11: (240,120,240),
    12: (42,125,209),
    13: (170,240,209),
    14: (255,0,204),
    15: (61,61,245),
    16: (255,0,0),
    17: (0,0,0),
}

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


        
def resize(image,  boxes, dim=300):
    dims = (int(dim*1.41), dim*1)
    new_image = TF.resize(image, dims)
    #new_mask = TF.resize(mask, dims, interpolation=PIL.Image.NEAREST)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_dims = torch.FloatTensor([new_image.width, new_image.height, new_image.width, new_image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims
    new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def segToMask(h,w, segm):
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return maskUtils.decode(rle)

class TexBiGDataset(torchvision.datasets.coco.CocoDetection):

    def __init__(self, root, ann_file, remove_images_without_annotations=True, transforms=None):
        super(TexBiGDataset, self).__init__(root, root + ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = [] 
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        
    def collate_fn(self, batch):
        #masks = list()
        labels = list()
        boxes = list()
        
        data = list()
        for b in batch:
            boxes.append(torch.Tensor(b[1][0]["boxes"]))
            #masks.append(b[1][0]["masks"])
            labels.append(torch.LongTensor(b[1][0]["labels"]))
            data.append(torch.Tensor(b[0]))
        return data, [{
                'boxes' : boxes,
                #'masks' : masks,
                'labels' : labels,
            }]
        pass
    
    def normBoxes(self,img,x1, y1,w1,h1):    #accepts box coordinates in KITTI SHape (X1,Y1,X1,Y2)
      
     

        #im = Image.open(img)
        
        width = img.size()[2] # should be 705,500
        height =img.size()[1]
        
      
        dw=1./width
        dh=1./height
      
       
        x=x1*dw
        w=w1*dw
        y=y1*dh
        h=h1*dh
        return (x,y,w,h)
        
        
    def renormBox(self,image,x,y,w,h):
        
          W, H = img.size()
          x1 = (x - w / 2) * W
          y1 = (y - h / 2) * H
          x2 = (x + w / 2) * W
          y2 = (y + h / 2) * H
                    
          return x1,y1,x2,y2
        
        
    def __getitem__(self, idx):
        img, anno = super(TexBiGDataset, self).__getitem__(idx)
        
        target = {}
        
        transformation =transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
        
        
        # filter crowd annotations
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
        
        boxes = ops.box_convert(boxes, "xywh", "xyxy")
        #boxes = ops.clip_boxes_to_image(boxes, img.size)
        
        
        target["boxes"] = boxes
    
        
        
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target["labels"] = classes

        height, width = img.size
  
        #rles = [obj["segmentation"] for obj in anno]
        
        
        #masks = [segToMask(height, width, rle) for rle in rles]
        #masks = torch.as_tensor(masks)
        #target["masks"] = masks
        
        img,  target["boxes"] = resize(img, target["boxes"], dim=500)
        
        #target["image_id"] = torch.tensor(self.id_to_img_map[idx])
        #target["area"] = torch.as_tensor([obj["area"] for obj in anno], dtype=torch.float32)
        #target["iscrowd"] = torch.as_tensor([obj["iscrowd"] for obj in anno],  dtype=torch.int64)
        
        #blow up ground truth to 70 
        #length=len(target["labels"])
        #paddLbl=torch.nn.ZeroPad2d((0,100-length))
        #paddBox=torch.nn.ZeroPad2d((0,0,0,100-length))
            
        #target["labels"]=paddLbl(target["labels"])
        #target["boxes"]=paddBox(target["boxes"])
        img=transformation(img)
        img1=img
    
        
        
        x, y, x1, y1 = target["boxes"].unbind(1)
        h1=y1-y
        w1=x1-x
        
        x0,y0,x1,y1= self.normBoxes(img1, x,y,w1,h1)
        box = [x0, y0, x1, y1]
        box=torch.stack(box, dim=1)
        target["boxes"]=box
       
        
      
        
        return img, [{  'labels': target['labels'], 'boxes':target['boxes']}]#box_ops.box_xyxy_to_cxcywh(target['boxes'] )}]

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    