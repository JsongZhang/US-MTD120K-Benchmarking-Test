import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision import models
from tqdm import tqdm
import torchvision.models
from repvgg import repvgg_model_convert, create_RepVGG_B1g2
from repvggplus import repvgg_model_convert, create_RepVGGplus_L2pse
from ConfusionMatrix import ConfusionMatrix
from PIL import Image
import os
from alter_mix_nn import dnn_50
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os

dnn50 = True # or Fasle
REPvgg = False # or True

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


Best_ACC = 0.0
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "supervised_work", "clc")  # UMTD set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    UMTD_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in UMTD_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    if dnn50:
        net = dnn_50(num_classes=28)
        net.to(device)

    if REPvgg:
        net = create_RepVGGplus_L2pse(deploy=False)
        net.load_state_dict(torch.load("./RepVGGplus-L2pse-train256-acc84.06.pth")) #imagenet transfer
        in_channel = net.linear.in_features
        net.linear = nn.Linear(in_channel, 28)

        net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()


    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
    # optimizer = optim.AdamW(params, lr=0.0005, weight_decay=5E-2)
    epochs = 300
    best_acc = 0.0
    save_path = ' ' #set your save path
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            # logits = torch.tensor(logits)
            if type(logits) is dict:
                loss = 0.0
                for name, pred in logits.items():
                    if 'aux' in name:
                        loss += 0.1 * loss_function(pred, labels.to(device))
                    else:
                        loss += loss_function(pred, labels.to(device))
            else:
                loss = loss_function(logits, labels.to(device))
            # print(type(logits))
            # loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0 # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                outputs = torch.softmax(outputs, dim=1)

                predict_y = torch.argmax(outputs, dim=1)

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        global Best_ACC

        if val_accurate > best_acc:
            best_acc = val_accurate
            Best_ACC = best_acc

            torch.save(net.state_dict(), save_path)


    print('Finished Training')



if __name__ == '__main__':
    main()
    print("Over of Training and Val")
    print(Best_ACC)
