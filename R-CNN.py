import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def parse_added_objects(file_path):
    added_objects = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            img_name, obj_id = line.strip().split()
            obj_id = int(obj_id)
            if img_name not in added_objects:
                added_objects[img_name] = []
            added_objects[img_name].append(obj_id)
    return added_objects

def bounding_box_from_mask(mask, obj_id):
    pos = np.where(mask == obj_id)
    if pos[0].size == 0 or pos[1].size == 0:
        return None
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]

def parse_annotation_txt(txt_file):
    boxes = []
    labels = []
    with open(txt_file, "r") as f:
        for line in f:
            if "Bounding box for object" in line:
                # (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)
                coords = line.split(":")[-1].strip()
                xmin, ymin = coords.split(") - (")[0].replace("(", "").split(",")
                xmax, ymax = coords.split(") - (")[1].replace(")", "").split(",")
                boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                labels.append(1)  # pedestrian class = 1
    return boxes, labels

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root #root directory
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "Annotation"))))

        # Parse added object list
        added_file = os.path.join(root, "added-object-list.txt")
        self.added_objects = parse_added_objects(added_file)

    def __getitem__(self, idx):
        #get bounding boxes from annotations then get bounding boxes from mask for added objects 
        # Load image
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load mask (for added objects only)
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        mask = np.array(Image.open(mask_path))

        # Load annotation txt
        annot_path = os.path.join(self.root, "Annotation", self.annots[idx])
        boxes, labels = parse_annotation_txt(annot_path)

        # Add objects from added-object-list.txt
        img_name = self.imgs[idx]
        if img_name in self.added_objects:
            for obj_id in self.added_objects[img_name]:
                box = bounding_box_from_mask(mask, obj_id)
                if box:
                    boxes.append(box)
                    labels.append(1)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),#[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64) # checks if more than one object in a bounding box - This is 0 for all bounding boxes for PennFudanDataset
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")

    # Replace classifier head for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # coco softmax has 91 class we only need 2 so we replace the box predictor
    return model

def compute_iou(box1, box2):
    #Compute IoU between two boxes: [xmin, ymin, xmax, ymax].
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_true = []
    all_scores = []

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                matched = set()

                for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                    is_tp = 0
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if i in matched:
                            continue
                        if pl == gl and compute_iou(pb, gb) >= iou_threshold:
                            is_tp = 1
                            matched.add(i)
                            break
                    
                    all_true.append(is_tp)     # 1 = matched GT, 0 = false positive
                    all_scores.append(ps)      # confidence score

    # Convert to numpy
    all_true = np.array(all_true)
    all_scores = np.array(all_scores)

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(all_true, all_scores)
    ap = average_precision_score(all_true, all_scores)

    print(f"Average Precision (AP): {ap:.4f}")

    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.show()

    return precision, recall, ap

def visualize_prediction(model, dataset, device, idx=0, score_thresh=0.5):
    model.eval()
    img, _ = dataset[idx]
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    img_vis = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img_vis)
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score >= score_thresh:
            draw.rectangle(list(box), outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

    plt.imshow(img_vis)
    plt.axis("off")
    plt.show()
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PennFudanDataset(os.path.join("Data","dataset"), transforms=transform)

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42) #cant pass dataset directly to train_test_split()

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset  = torch.utils.data.Subset(dataset, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders for train and test
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    model = get_model(num_classes=2)
    model.to(device)

    # Training setup
    params = [p for p in model.parameters() if p.requires_grad] #to skip frozen layers, no frozen layers in current code
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch {epoch}, Loss: {losses.item():.4f}")

    # Evaluate on test split
    evaluate_model(model, test_loader, device)
    visualize_prediction(model, test_dataset, device, idx=5)