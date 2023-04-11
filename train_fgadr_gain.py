import copy
import time
from datetime import datetime, timedelta

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import utils
from data_manager.dataset import dataset_Aptos, dataset_eyepacs, dataset_idrid, dataset_fgadr
from loss.MultiClassMetrics import *
from models.FinetuneVTmodels_MOD import MIL_VT_FineTune
from utils import *
from loss.segmentation_losses import *
from models.GAIN import GAIN

torch.multiprocessing.set_sharing_strategy('file_system')

####################################


def generate_explantion(model, attentions, prediction, threshold=0.5):
    ''' non_probabilistic = prediction.argmax(dim=1)
    for i, j in zip(prediction, non_probabilistic):
        i[j.item()].backward(retain_graph=True)'''
    # gradients = []
    # binarymap = torch.zeros((activations.shape[0], self.input_size[0], self.input_size[1]))
    # print(len(activation_outputs))
    # for prediction in activation_outputs:
    # print(prediction.shape)

    # from here
    # prediction.backward()
    # # gradients.append( model.get_activations_gradient())
    # # pull the gradients out of the model
    # gradients = model.get_activations_gradient()
    # # for i in range(len(gradients)):
    # if torch.isnan(gradients.sum()):
    #     gradients[gradients != gradients] = 1
    # if torch.isnan(gradients.sum()):
    #     print('There is a nan value after gradients')
    # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # if torch.isnan(activations.sum()):
    #     activations[activations != activations] = 1
    # if torch.isnan(activations.sum()):
    #     print('There is a nan value in activation')
    # for j in range(512):
    #     activations[:, j, :, :] *= pooled_gradients[j]

    nh = attentions.shape[1]  # number of head
    w, h = 512 - 512 % 16, 512 - 512 % 16


    w_featmap = 512 // 16
    h_featmap = 512 // 16
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    # print(attentions.shape)
    attentions_max = torch.max(attentions)
    attentions = attentions / attentions_max
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[
        0].cpu()
    attentions = torch.mean(attentions, 0)
    attentions_binary = torch.where(attentions > 0.5, 1.0, 0.0)
    # print(attentions.shape, attentions.max(), attentions.min(), attentions_binary.max(), attentions_binary.min())

    return attentions, attentions_binary


    heatmap = torch.mean(activations, dim=1)
    heatmap = nn.ReLU(heatmap)
    # heatmap = torch.abs(heatmap)
    max_val = torch.max(heatmap)
    if max_val == 0.0:
        max_val = max_val + 0.00001  # adding smoth
    if torch.isnan(max_val):
        print('max val is nan')

    heatmap = heatmap / max_val
    if torch.isnan(heatmap.sum()):
        print('There is a nan value in heatmap')

    upsample = torch.nn.Upsample(size=(heatmap.shape[0], 512, 512), mode='nearest')
    heatmap = upsample(heatmap.unsqueeze_(0).unsqueeze_(0))
    heatmap = heatmap.squeeze(0).squeeze(0)
    thershold = nn.Threshold(0.4, 0)
    binarymap = nn.thershold(heatmap)
    if torch.isnan(binarymap.sum()):
        print('There is a nan value in binarymap')
    return binarymap, heatmap.squeeze()

def main():

    """Basic Setting"""
    data_path = r'/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Original_Images'
    full_seg_path = r'/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Full_Segmentation'

    csv_path = r'/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/'

    save_model_path = r'/mnt/sda/haal02-data/results/MIL-VT_results/fgadr_new_resnet50'
    csvName = csv_path + 'DR_Seg_Grading_Label_Combined_Renamed.csv'  ##the csv file store the path of image and corresponding label

    gpu_ids = [0]
    start_epoch = 0
    max_epoch = 80
    save_fraq = 10

    batch_size = 8
    img_size = 512
    initialLR = 2e-5
    n_classes = 5
    step_update = 3000

    balanceFlag = True  #balanceFlag is set to True to balance the sampling of different classes
    debugFlag = False  #Debug flag is set to True to train on a small dataset

    base_model = 'MIL_VT_small_patch16_'+str(img_size)  #nominate the MIL-VT model to be used
    MODEL_PATH_finetune = 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'

    dateTag = datetime.today().strftime('%Y%m%d')
    prefix = base_model + '_' + dateTag
    model_save_dir = os.path.join(save_model_path,  prefix)
    tbFileName = os.path.join(model_save_dir, 'runs/' + prefix)
    savemodelPrefix = prefix + '_ep'

    ##resume training with an interrupted model
    resumeFlag = False # True of False
    resumeEpoch = 3
    resumeModel = '/mnt/sda/haal02-data/results/MIL-VT_results/MIL_VT_small_patch16_512_20220222/MIL_VT_small_patch16_512_20220222_ep_bestmodel.pth.tar'

    print('####################################################')
    print('Save model Path', model_save_dir)
    print('Save training record Path', tbFileName)
    print('####################################################')

    #################################################
    sys.stdout = Logger(os.path.join(model_save_dir,
                     savemodelPrefix[:-3] + 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))
    tbWriter = SummaryWriter(tbFileName)

    torch.cuda.set_device(gpu_ids[0])
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(gpu_ids[0]), torch.cuda.get_device_capability(gpu_ids[0]))

    #################################################
    """Set up the model, loss function and optimizer"""

    ## set the model and assign the corresponding pretrain weight
    # model = MIL_VT_FineTune(base_model, MODEL_PATH_finetune, num_classes=n_classes)
    # model = model.cuda()

    # model = GAIN(grad_layer='_features.7', num_classes=5).cuda()
    model = GAIN(grad_layer='layer4', num_classes=5).cuda()
    if len(gpu_ids) >= 2:
        model = DataParallel(model, device_ids=gpu_ids)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    multiLayers = list()
    for name, layer in model._modules.items():
        if name.__contains__('MIL_'):
            multiLayers.append({'params': layer.parameters(), 'lr': 5*initialLR})
        else:
            multiLayers.append({'params': layer.parameters()})
    optimizer = torch.optim.Adam(multiLayers, lr = initialLR, eps=1e-8, weight_decay=1e-5)
    class_weights = torch.tensor([0.169, 0.238, 0.168, 0.205, 0.220])
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    if resumeFlag:
        print(" Loading checkpoint from epoch '%s'" % (
             resumeEpoch))
        checkpoint = torch.load(resumeModel)
        initialLR = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Model weight loaded')

    #################################################
    """Load the CSV as DF and split train / valid set"""
    DF0= pd.read_csv(csvName, encoding='UTF')

    if debugFlag == True:
        indexes = np.arange(len(DF0))
        np.random.seed(0)
        np.random.shuffle(indexes)
        DF0 = DF0.iloc[indexes[:600], :]
        DF0 = DF0.reset_index(drop=True)

    indexes = np.arange(len(DF0))

    np.random.seed(0)
    np.random.shuffle(indexes)
    trainNum = np.int(len(indexes)*.8)
    valNum = np.int(len(indexes)*1.0)
    # update for full train
    DF_train = DF0.loc[indexes[:trainNum]]

    # 10000 val and rest test
    DF_val = DF0.loc[indexes[trainNum:]]
    DF_test = DF0.loc[indexes[trainNum:]]

    DF_train = DF_train.reset_index(drop=True)
    DF_val = DF_val.reset_index(drop=True)
    DF_test = DF_test.reset_index(drop=True)

    print('Train: ', len(DF_train), 'Val: ', len(DF_val), 'Test: ', len(DF_test))
    for tempLabel in [0,1,2,3,4]:
        print(tempLabel, np.sum(DF_train['level']==tempLabel),\
                        np.sum(DF_val['level']==tempLabel),
                        np.sum(DF_test['level']==tempLabel))

    #################################################

    transform_train = transforms.Compose([
        transforms.Resize((img_size+40, img_size+40)),
        transforms.RandomCrop((img_size, img_size)),  #padding=10
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.434, 0.210, 0.070], [0.309, 0.165, 0.0844]),
    ])

    transform_train_simple = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.434, 0.210, 0.070], [0.309, 0.165, 0.0844]),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # transforms.Normalize([0.434, 0.210, 0.070], [0.309, 0.165, 0.0844]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.434, 0.210, 0.070], [0.309, 0.165, 0.0844]),
    ])

    # dataset_train = dataset_Aptos(data_path, DF_train, transform = transform_train)
    # dataset_valid = dataset_Aptos(data_path, DF_val, transform = transform_test)
    # dataset_test = dataset_Aptos(data_path, DF_test, transform=transform_test)
    #
    dataset_train = dataset_fgadr(data_path, DF_train, full_seg_path=full_seg_path, transform = transform_train_simple, transform_mask=transform_mask)
    dataset_valid = dataset_fgadr(data_path, DF_val, full_seg_path=full_seg_path, transform = transform_test, transform_mask=transform_mask)
    dataset_test = dataset_fgadr(data_path, DF_test, full_seg_path=full_seg_path, transform=transform_test, transform_mask=transform_mask)

    """assign sample weight to deal with the unblanced classes"""
    weights = make_weights_for_balanced_classes(DF_train, n_classes)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    if balanceFlag == True:
        train_loader = DataLoader(dataset_train, batch_size,
                              sampler = sampler,
                              num_workers=4,  drop_last=True, shuffle=False) #shuffle=False when using the balance sampler,
    else:
        train_loader = DataLoader(dataset_train, batch_size, num_workers=8,  drop_last=True, shuffle=True) #shuffle=True,
    valid_loader = DataLoader(dataset_valid, batch_size, num_workers=8, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size, num_workers=8,  drop_last=False)

    #################################################

    """The training procedure"""

    start_time = time.time()
    train_time = 0
    best_perform = 0
    for epoch in range(start_epoch, max_epoch + 1):
        start_train_time = time.time()
        currentLR = 0

        for param_group in optimizer.param_groups:
            currentLR = param_group['lr']
        print('lr:', currentLR)

        train(epoch, model, criterion, optimizer, train_loader, max_epoch, tbWriter)
        train_time += np.round(time.time() - start_train_time)

        AUC_val, wF1_val = \
            val(epoch, model, criterion, valid_loader, max_epoch, tbWriter)

        if wF1_val > best_perform and debugFlag == False:
            best_perform = wF1_val
            state_dict = model_without_ddp.state_dict()
            saveCheckPointName = savemodelPrefix + '_bestmodel.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            # best_model = copy.deepcopy(model) # not saving best model for now
            print('Checkpoint saved, ', saveCheckPointName)

        if epoch>0 and (epoch) % save_fraq == 0 and debugFlag == False:

            state_dict = model_without_ddp.state_dict()
            wF1_val = round(wF1_val, 3)
            saveCheckPointName = savemodelPrefix + str(epoch) + '_' + str(wF1_val) + '.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            print('Checkpoint saved, ', saveCheckPointName)


    elapsed = np.round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    train_time = str(timedelta(seconds=train_time))


    print('###################################################')
    print('Performance on Test Set with last model')
    test(epoch, model, criterion, test_loader, tbWriter)

    print('###################################################')
    print('Performance on Test Set with best model')
    test(epoch, best_model, criterion, test_loader, tbWriter)


    tbWriter.close()
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))



def train(epoch, model, criterion, optimizer, train_loader, max_epoch,  tbWriter):
    start_time = time.time()
    model.train()
    losses = utils.AverageMeter()
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    losses3 = utils.AverageMeter()

    ground_truths_multiclass = []
    ground_truths_multilabel = []

    predictions_class = []
    total = 0
    step_update = 3000
    for batch_idx, (inputs, labels_multiclass, labels_onehot, features) in enumerate(train_loader):
        if epoch == 0 and batch_idx % 40 == 0:
            print('iter batch: ', batch_idx)
        inputs = inputs.cuda()
        targets_class = labels_multiclass.cuda()
        # # print(targets_class)
        features_true = features[:,2,:,:]
        features_true = features_true > 0.5
        # features_true = features_true.type(torch.LongTensor)
        features_true = features_true.unsqueeze(1).cuda()

        logits, logits_am, heatmap = model(inputs, targets_class)
        # print(logits.shape, logits_am.shape, heatmap.shape)
        # print(logits)
        # print(logits_am)
        # # print(heatmap)
        # print(heatmap.shape, features_true.shape, type(features_true), type(heatmap),  heatmap.max(), heatmap.min())
        loss1 = criterion(logits, targets_class)
        loss2 = criterion(logits_am, targets_class)

        # loss2 = jaccard_loss(features_true, heatmap)
        # print(loss1, loss2)
        # exit()

        # outputs_class, outputs_MIL = model(inputs)
        # loss1 = criterion(outputs_class, targets_class)
        # loss2 = criterion(outputs_MIL, targets_class)
        loss = 0.5 * loss1 + 0.5 * loss2

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
        outputs_class = logits_am # changed to attention mapped output
        losses.update(loss.data.cpu().numpy())
        losses1.update(loss1.data.cpu().numpy())
        losses2.update(loss2.data.cpu().numpy())

        ###Update learning rate
        steps = len(train_loader) * epoch + batch_idx
        if steps > 0 and steps % step_update == 0:
            print('steps: ', steps)
            adjust_learning_rate(optimizer, 0.5)

        total += targets_class.size(0)
        _, predicted_class = torch.max(outputs_class.data, 1)

        # """Save the losses to tensorboard"""
        tbWriter.add_scalar('AllLoss/train', loss, steps)

        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)

    """Mesure the prediction performance on train set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/train', accuracy, epoch)
    tbWriter.add_scalar('AUC/train', AUC, epoch)
    tbWriter.add_scalar('wF1/train', wF1, epoch)
    tbWriter.add_scalar('wKapa/train', wKappa, epoch)
    tbWriter.add_scalar('Loss/train', losses.avg, epoch)
    tbWriter.add_scalar('Loss1/train', losses1.avg, epoch)
    tbWriter.add_scalar('Loss2/train', losses2.avg, epoch)

    end_time = time.time()
    print('\t Epoch {}/{}'.format(epoch, max_epoch))
    print(
        '\t Train:,  AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, loss1 %.6f, loss2 %.6f, time %3.2f'
        % ( AUC, wKappa, wF1, losses1.avg, losses2.avg, end_time - start_time))
    return 0

    # # end
    #     if epoch == 0 and batch_idx % 40 == 0:
    #         print('iter batch: ', batch_idx)
    #     inputs = inputs.cuda()
    #     targets_class = labels_multiclass.cuda()
    #     features = features.cuda()
    #     # # print(targets_class)
    #     attention_model = copy.deepcopy(model)
    #     outputs_class, outputs_MIL = model(inputs)
    #
    #     ### edit
    #     attentions = attention_model.get_last_selfattention(inputs)
    #     # print(attentions.shape, type(attentions))
    #
    #     attention, attention_binary = generate_explantion(attention_model, attentions, outputs_class, threshold=0.5)
    #     # print(features[0,2,:,:].max(), features[0,2,:,:].min(), features.shape, type(features))
    #     num_features = features[0,2,:,:].data.cpu()
    #     num_features = num_features > 0.5
    #     attention = torch.unsqueeze(attention, 0)
    #     attention = torch.unsqueeze(attention, 0)
    #
    #     attention_binary = torch.unsqueeze(attention_binary, 0)
    #     attention_binary = torch.unsqueeze(attention_binary, 0)
    #
    #     num_features = torch.unsqueeze(num_features, 0)
    #     num_features = torch.unsqueeze(num_features, 0)
    #
    #     # print(type(num_features), num_features.shape)
    #     # print(attention.shape)
    #
    #
    #     # feature_loss = jaccard_score()
    #     attn_loss = jaccard_loss(num_features, attention)
    #     loss3 = Variable(attn_loss, requires_grad=True).cuda()
    #     # attn_loss = jaccard_score(num_features, attention_binary, average='micro')
    #     print(loss3)
    #
    #     # exit()
    #     ## end edit
    #
    #
    #
    #     loss1 = criterion(outputs_class, targets_class)
    #     loss2 = criterion(outputs_MIL, targets_class)
    #     loss = 0.35*loss1 + 0.35*loss2
    #     print(loss1)
    #     print(loss2)
    #     # exit()
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     nn.utils.clip_grad_value_(model.parameters(), 0.1)
    #     optimizer.step()
    #
    #     losses.update(loss.data.cpu().numpy())
    #     losses1.update(loss1.data.cpu().numpy())
    #     losses2.update(loss2.data.cpu().numpy())
    #     losses3.update(loss3.data.cpu().numpy())
    #
    #     ###Update learning rate
    #     steps = len(train_loader)*epoch + batch_idx
    #     if steps>0 and steps % step_update == 0:
    #         print('steps: ', steps)
    #         adjust_learning_rate(optimizer, 0.5)
    #
    #     total += targets_class.size(0)
    #     _, predicted_class = torch.max(outputs_class.data, 1)
    #
    #     #"""Save the losses to tensorboard"""
    #     tbWriter.add_scalar('AllLoss/train', loss, steps)
    #
    #     outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
    #     ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
    #     ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
    #     predictions_class.extend(outputs_class)
    #
    # """Mesure the prediction performance on train set"""
    # gts = np.asarray(ground_truths_multiclass)
    # probs = np.asarray(predictions_class)
    # preds = np.argmax(probs, axis=1)
    # accuracy = metrics.accuracy_score(gts, preds)
    #
    # gts2 = np.asarray(ground_truths_multilabel)
    # trues = np.asarray(gts2).flatten()
    # probs2 = np.asarray(probs).flatten()
    # AUC = metrics.roc_auc_score(trues, probs2)
    #
    # wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    # wF1 = metrics.f1_score(gts, preds, average='weighted')
    #
    # """Save the performance to tensorboard"""
    # tbWriter.add_scalar('accuraccy/train', accuracy, epoch)
    # tbWriter.add_scalar('AUC/train', AUC, epoch)
    # tbWriter.add_scalar('wF1/train', wF1, epoch)
    # tbWriter.add_scalar('wKapa/train', wKappa, epoch)
    # tbWriter.add_scalar('Loss/train', losses.avg, epoch)
    #
    #
    # end_time = time.time()
    # print('\t Epoch {}/{}'.format(epoch, max_epoch))
    # print('\t Train:    Acc %0.3f,  AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, loss1 %.6f, loss2 %.6f, loss3 %.6f, time %3.2f'
    #     % (accuracy, AUC, wKappa, wF1, losses1.avg, losses2.avg, losses3.avg, end_time - start_time))
    # return  0


def val(epoch, model, criterion, val_loader, max_epoch, tbWriter):
    start_time = time.time()
    model.eval()
    losses = utils.AverageMeter()
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()


    ground_truths_multiclass = []
    ground_truths_multilabel = []
    predictions_class = []
    scores = []
    total = 0

    for batch_idx, (inputs,  labels_multiclass, labels_onehot, features) in enumerate(val_loader):
        inputs = Variable(inputs.cuda())
        targets_class = Variable(labels_multiclass.cuda())

        features_true = features[:, 2, :, :]
        features_true = features_true > 0.5
        # features_true = features_true.type(torch.LongTensor)
        features_true = features_true.unsqueeze(1).cuda()

        logits, logits_am, heatmap = model(inputs, targets_class)
        # print(logits.shape, logits_am.shape, heatmap.shape)
        # print(logits)
        # print(logits_am)
        # # print(heatmap)
        # print(heatmap.shape, features_true.shape, type(features_true), type(heatmap), heatmap.max(), heatmap.min())
        loss1 = criterion(logits, targets_class)
        loss2 = criterion(logits_am, targets_class)

        # loss2 = jaccard_loss(features_true, heatmap)

        #
        # outputs_class = model(inputs)
        # loss = criterion(outputs_class, targets_class) #targets_class

        losses.update(loss1.data.cpu().numpy())
        losses1.update(loss1.data.cpu().numpy())
        losses2.update(loss2.data.cpu().numpy())
        # print(loss1, loss2)

        outputs_class = logits
        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)

        total += targets_class.size(0)

    """Mesure the prediction performance on valid set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/valid', accuracy, epoch)
    tbWriter.add_scalar('AUC/valid', AUC, epoch)
    tbWriter.add_scalar('wF1/valid', wF1, epoch)
    tbWriter.add_scalar('wKapa/valid', wKappa, epoch)
    tbWriter.add_scalar('Loss/valid', losses.avg, epoch)
    tbWriter.add_scalar('Loss1/valid', losses1.avg, epoch)
    tbWriter.add_scalar('Loss2/valid', losses2.avg, epoch)


    end_time = time.time()
    print('-------- Epoch {}/{}'.format(epoch, max_epoch))
    print('-------- Val:   AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, loss %.6f, loss1 %.6f, loss2 %.6f, time %3.2f'
        % ( AUC, wKappa, wF1, losses.avg, losses1.avg, losses2.avg, end_time - start_time))
    print('=============================================================')
    return AUC, wF1



def test(epoch, model, criterion, test_loader,  tbWriter):
    start_time = time.time()
    model.eval()
    losses = utils.AverageMeter()


    ground_truths_multiclass = []
    ground_truths_multilabel = []
    predictions_class = []
    total = 0

    for batch_idx, (inputs, labels_multiclass, labels_onehot) in enumerate(test_loader):
        inputs = Variable(inputs.cuda())
        targets_class = Variable(labels_multiclass.cuda())

        outputs_class = model(inputs)
        loss = criterion(outputs_class, targets_class) #targets_class

        losses.update(loss.data.cpu().numpy())

        outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
        ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
        ground_truths_multilabel.extend(labels_onehot.data.cpu().numpy())
        predictions_class.extend(outputs_class)
        total += targets_class.size(0)


    """Mesure the prediction performance on test set"""
    gts = np.asarray(ground_truths_multiclass)
    probs = np.asarray(predictions_class)
    preds = np.argmax(probs, axis=1)
    accuracy = metrics.accuracy_score(gts, preds)

    gts2 = np.asarray(ground_truths_multilabel)
    trues = np.asarray(gts2).flatten()
    probs2 = np.asarray(probs).flatten()
    AUC = metrics.roc_auc_score(trues, probs2)

    wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    wF1 = metrics.f1_score(gts, preds, average='weighted')

    """Save the performance to tensorboard"""
    tbWriter.add_scalar('accuraccy/test', accuracy, epoch)
    tbWriter.add_scalar('AUC/test', AUC, epoch)
    tbWriter.add_scalar('wF1/test', wF1, epoch)
    tbWriter.add_scalar('wKapa/test', wKappa, epoch)
    tbWriter.add_scalar('Loss/test', losses.avg, epoch)

    end_time = time.time()
    print( 'TestSet:   AUC %0.3f, weightedKappa %0.3f, weightedF1 %0.3f, time %3.2f'
        % ( AUC, wKappa, wF1,  end_time - start_time))
    print('=============================================================')
    return AUC, wF1



if __name__ == '__main__':
    main()

