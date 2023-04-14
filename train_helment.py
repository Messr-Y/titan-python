import copy
import os
import time
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 保证程序能够正确识别显卡设备并在其上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 指定使用的GPU编号
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # 使CUDA在启动内核计算前等待所有前面的计算完成，然后再开始新的计算，以确保程序能够正确运行

CLASSES = ['background', 'hat', 'person']
root = 'C:/Users/HP/pythonProject/VOC2028/'
annopath = os.path.join(root, "Annotations", "%s.xml")
imgpath = os.path.join(root, "JPEGImages", "%s.jpg")
imgsetpath = os.path.join(root, "ImageSets", "Main", "%s.txt")

# 获取训练集，验证集，测试集数据
with open(imgsetpath % 'train') as f:
    img_ids_train = f.readlines()
n = len(img_ids_train)  # 获取文件列表的总长度
img_ids_train = img_ids_train[:4050]
img_ids_train = [x.strip() for x in img_ids_train]  # ['000009', '000052']
img_rub = []
for name in img_ids_train:
    i = name.find("PartA_")
    if i >= 0:
        img_rub.append(name)
for i in range(len(img_rub)):
    for j in range(len(img_ids_train)):
        if img_ids_train[j] == img_rub[i]:
            img_ids_train.remove(img_rub[i])
            break

with open(imgsetpath % 'val') as ff:
    img_ids_val = ff.readlines()
m = len(img_ids_val)  # 获取文件列表的总长度
print(m)
img_ids_val = img_ids_val[:463]  # 只获取前四分之一的数据
img_ids_val = [x.strip() for x in img_ids_val]
img_rub = []
for name in img_ids_val:
    i = name.find("PartA_")
    if i >= 0:
        img_rub.append(name)
for i in range(len(img_rub)):
    for j in range(len(img_ids_val)):
        if img_ids_val[j] == img_rub[i]:
            img_ids_val.remove(img_rub[i])
            break
print(len(img_ids_val))




with open(imgsetpath % 'test') as fff:
    img_ids_test = fff.readlines()
img_ids_test = [x.strip() for x in img_ids_test]


# 设定图片固定宽高
w = 320
h = 320

num_person = 0
num_hat = 0

# 定义VOC数据集的处理
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, transforms=None):
        global num_person, num_hat, flag
        super(VOCDataset, self).__init__()
        self.transforms = transforms
        self.images = [] # 存储每一张照片
        self.annotations = [] # 存储每张照片对应的目标的坐标
        # 读取VOC数据集中的图片和标注信息
        i = 0
        for image_file in img_ids:
            i += 1
            img = imgpath % str(image_file)
            self.images.append(img)
            annotation_file = ET.parse(annopath % str(image_file)).getroot()
            boxes = [] # 存储一张照片的所有目标坐标
            labels = [] # 存储该照片的每个目标坐标对应的标签
            for obj in annotation_file.findall("object"):
                width = float(annotation_file.find('size').find('width').text)
                height = float(annotation_file.find('size').find('height').text)
                label = obj.find("name").text
                if label == 'person' and flag:
                    num_person += 1
                elif label == 'hat' and flag:
                    num_hat += 1

                bbox = obj.find("bndbox")
                # 将坐标大小转为图像预处理后的大小，并除去无效边界框
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                if xmin > 0 and xmax > 0 and ymax > 0 and ymin > 0:
                    # xmin = float(bbox.find("xmin").text)
                    # ymin = float(bbox.find("ymin").text)
                    # xmax = float(bbox.find("xmax").text)
                    # ymax = float(bbox.find("ymax").text)


                    xmin = float((float(bbox.find("xmin").text)) / width * w )
                    ymin = float((float(bbox.find("ymin").text)) / height * h)
                    xmax = float((float(bbox.find("xmax").text)) / width * w)
                    ymax = float((float(bbox.find("ymax").text)) / height * h)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(CLASSES.index(label))
            self.annotations.append({"boxes": boxes, "labels": labels})


            # if i % 20 == 0 and flag:
            #     image = Image.open(imgpath % str(image_file))
            #     image = image.resize((320, 320))
            #     draw = ImageDraw.Draw(image)
            #     for j, box in enumerate(boxes):
            #         if labels[j] == 0:
            #             draw.rectangle(box, outline='red')
            #         else:
            #             draw.rectangle(box, outline='green')
            #     image.show()



    def __getitem__(self, index):
        # 将数据集转为张量
        image = Image.open(self.images[index]).convert("RGB")
        boxes = self.annotations[index]["boxes"]
        labels = self.annotations[index]["labels"]
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, boxes, labels



    def __len__(self):
        return len(self.images)



# 定义训练参数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 2 # 一般设为32以上，我的电脑只能支持到2
num_epochs = 50 # 训练次数
lr = 1e-4 # 学习率

# 定义数据预处理
transforms_train = transforms.Compose([transforms.Resize((w, h)),
                                       # transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率
                                       # transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transforms_val = transforms.Compose([transforms.Resize((w, h)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# 定义模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 3  # VOC数据集中的类别数
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # 以上改变模型分离器的输入与输出
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)) # 设定瞄框大小和宽高比
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2) # 在特征图上对RoIs进行池化操作
model.rpn_anchor_generator = anchor_generator
model.roi_heads.box_roi_pool = roi_pooler
model.to(device)

# checkpoint = torch.load('checkHelmet.pth')
# model.load_state_dict(checkpoint['state_dict'])

# 定义优化器和学习率调度器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

# optimizer.load_state_dict(checkpoint['optimizer'])

# 定义损失函数
def new_criterion(outputs, targets):
    losses = {}

    # 分类损失
    losses["loss_classifier"] = outputs["loss_classifier"]
    # 目标检测损失
    losses["loss_objectness"] = outputs["loss_objectness"]
    # 边界框回归损失
    losses["loss_box_reg"] = outputs["loss_box_reg"]
    # RPN 网络的边界框回归损失
    losses["loss_rpn_box_reg"] = outputs['loss_rpn_box_reg']

    # 计算总损失
    loss = sum(losses.values())
    return loss



# 定义数据加载器
flag = 1
train_dataset = VOCDataset(img_ids_train, transforms_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: tuple(zip(*batch)))
flag = 0
val_dataset = VOCDataset(img_ids_val, transforms_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: tuple(zip(*batch)))
flag = 0
test_dataset = VOCDataset(img_ids_test, transforms_val)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: tuple(zip(*batch)))

print("num_person={}".format(num_person))
print("num_hat={}".format(num_hat))

# 计算两个边缘框之间的IoU的函数
def box_iou(box1, box2):
    """
    计算两个边界框之间的 IoU 值
    :param box1: 边界框 1，形状为 [N, 4]
    :param box2: 边界框 2，形状为 [M, 4]
    :return: IoU 值，形状为 [N, M]
    """
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # 左上角坐标的最大值
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # 右下角坐标的最小值
    wh = (rb - lt).clamp(min=0)  # 相交区域的宽度和高度，注意要使用 clamp 函数将负数部分截断
    inter = wh[:, :, 0] * wh[:, :, 1]  # 相交区域的面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # 边界框 1 的面积
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # 边界框 2 的面积
    iou = inter / (area1[:, None] + area2 - inter)  # 计算 IoU 值
    return iou


# 定义训练和测试函数
def train(model, loader, optimizer, criterion, device):
    model.train()
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)
    person = 0
    hat = 0
    running_loss = 0.0
    for images, boxes, labels in loader:
        # # 验证数据的正确性
        # each_boxes = [{"boxes": box} for box in zip(boxes)]
        # for i, img in enumerate(images):
        #     print(each_boxes[i]['boxes'][0])
        #     plt.imshow(img.permute(1, 2, 0))
        #     for box in each_boxes[i]['boxes'][0]:
        #         plt.plot([box[0], box[0], box[2], box[2], box[0]],
        #                  [box[1], box[3], box[3], box[1], box[1]],
        #                  color='r')
        #     plt.show()
        images = [image.to(device) for image in images]
        targets = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(boxes, labels)]

        for i, target in enumerate(targets):
            for j in range(len(target["labels"])):
                if target['labels'][j].item() == 2:
                    person += 1
                elif target['labels'][j].item() == 1:
                    hat += 1

        optimizer.zero_grad()
        outputs = model(images, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("train person:{}".format(person))
    print("train hat:{}".format(hat))
    return running_loss / len(loader)



def test(model, loader, device, score_threshold=0.7, iou_threshold=0.1,max_num_targets=100):
    model.eval()
    results = []
    selected_boxes = []
    selected_labels = []
    total_boxes = 0
    select_boxes = 0
    all_true_boxes = 0
    tp = 0  # 正确检测的目标数量
    fp = 0  # 误检测的目标数量
    hat = 0
    person = 0
    with torch.no_grad():
        for images, boxes, labels in loader:
            images = [image.to(device) for image in images]
            targets = [{"boxes": box.to(device), "labels": label.to(device)} for box, label in zip(boxes, labels)]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes  = output['boxes']
                labels = output['labels']
                scores = output['scores']
                true_boxes = targets[i]['boxes']
                true_labels = targets[i]['labels']
                n = len(boxes)
                m = len(true_boxes)
                total_boxes += n
                all_true_boxes += m

                # 进行非极大抑制，筛选出一定区域内属于同一种类得分最大的框
                selected_indices = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
                boxes = boxes[selected_indices]
                scores = scores[selected_indices]
                labels = labels[selected_indices]
                select_boxes += len(boxes)

                result = {"boxes": boxes, "labels": labels}
                results.append(result)


                for k, score in enumerate(scores):
                    if score.item() > score_threshold:
                        selected_boxes.append(boxes[k])
                        selected_labels.append(labels[k])



                # print("boxes:{}".format(len(boxes)))
                # print("true:{}".format(len(true_boxes)))


                for i in range(boxes.shape[0]):
                    iou_max = torch.zeros(len(true_boxes))
                    for j in range(len(true_boxes)):
                        box_gt = true_boxes[j]
                        iou = box_iou(boxes[i], box_gt)
                        iou_max[j], _ = torch.max(iou, dim=1)
                    iou_max, indices = torch.max(iou_max, dim=0)
                    if iou_max >= iou_threshold:
                        label_gt = true_labels[indices]
                        if labels[i] == label_gt:
                            tp += 1
                            if labels[i] == 2:
                                person += 1
                            elif labels[i] == 1:
                                hat += 1
                        else:
                            fp += 1
                    else:
                        fp += 1
    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("预测框个数:{}".format(total_boxes))
    print("筛选后的预测框个数:{}".format(select_boxes))
    print("分数高的预测框个数:{}".format(len(selected_boxes)))
    print("正确框个数:{}".format(tp))
    print("真实框个数:{}".format(all_true_boxes))
    print("安全帽个数:{}".format(hat))
    print("人的个数:{}".format(person))

    # 返回预测结果，准确率，召回率
    return results, tp/total_boxes, tp/all_true_boxes






# 记录当前时间
since = time.time()
# 定义模型保存路径和保存间隔
save_path = "models/"
# 初始化最高准确率和对应的epoch
best_accuracy = 0.0
best_epoch = -1
# 记录每次学习率的值
LRs = [optimizer.param_groups[0]['lr']]

if __name__=="__main__":
    # 训练模型
    writer = SummaryWriter("logs")
    # 模型保存
    filename = 'checkHelmet.pth'
    # 模型保存
    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())

    # 最大容忍，迭代patience后模型效果不再上升，则停止训练
    patience = 50
    count = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, new_criterion, device)
        results, test_accuracy, test_back_accuracy = test(model, val_loader, device)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_results", test_accuracy, epoch)
        lr_scheduler.step()
        print("Epoch: {}, Train Loss: {:.4f}, Test Results: {}, Recall Result: {}".format(epoch+1, train_loss, test_accuracy, test_back_accuracy))
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

        # 如果准确率提高了，则保存模型
        if test_accuracy > best_accuracy:
            count = 0
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

            # 保存模型
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, filename)
        else:
            count += 1
            if count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 保留最后一次训练模型
    best_model_wts = copy.deepcopy(model.state_dict())
    state = {
        'state_dict': model.state_dict(),
        'best_acc': test_accuracy,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, "lastModel.pth")

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    # 打印最高准确率和对应的epoch
    print("Best Test Accuracy: {:.4f} at Epoch {}".format(best_accuracy, best_epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



    # 测试模型
    results, test_results, recall = test(model, test_loader, device)
    for i in range(len(results)):
        image = np.array(test_dataset[i][0].permute(1, 2, 0))
        boxes = results[i]["boxes"]
        labels = results[i]["labels"]
        scores = results[i]["scores"]
        for j in range(len(boxes)):
            if scores[j] > 0.5:
                xmin, ymin, xmax, ymax = boxes[j].tolist().astype(np.int32)
                label = labels[j]
                score = scores[j]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, "{}: {:.2f}".format(label, score), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
