import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
from torchvision.models import resnet50, vgg16

class AttentionBlock(nn.Module):

    def __init__(self):
        super(AttentionBlock, self).__init__()

        self.attention_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.attention_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.context_conv = nn.Conv2d(2048, 128, kernel_size=(1, 1))


    def forward(self, x, f_low, f_high=None):

        x = self.attention_conv1(x)
        # x = self.attention_block1(x)
        # f_low = self.attention_block1(f_low)
        x = torch.cat((x, f_low), dim=1)
        y = F.adaptive_avg_pool2d(f_high, (1, 1))
        y = self.context_conv(y)
        print('y shape: ', y.shape)
        x = torch.mul(x, y).sum(dim=1, keepdim=True)

        return x


class GAIN(nn.Module):
    def __init__(self, grad_layer, num_classes):
        super(GAIN, self).__init__()
        # self.model = make_model(
        #     model_name='resnet50',
        #     pretrained=True,
        #     num_classes=num_classes
        # )
        self.model = resnet50(pretrained=False, num_classes=num_classes)
        # print(self.model)
        # self.model = vgg16(pretrained=False, num_classes=num_classes)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # print(self.model)
        weights = torch.load('/mnt/sda/haal02-data/results/classification_results/resnet50_full_fgadr_attention_ce/saves/best_validation_weights.pt')
        self.model.load_state_dict(weights, strict=False)

        self.attention_model = AttentionBlock()

        # self.grad_layer = 'conv1'
        self.grad_layer = ['conv1', 'layer4']
        print(self.grad_layer)
        self.grad_layer2 = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = []
        # Backward features
        self.backward_features = []

        # Register hooks
        self._register_hooks(self.grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100
        self.thershold = nn.Threshold(0.4, 0, inplace=False)
        self.mask_relu = nn.ReLU()


    def _register_hooks(self, grad_layer):
        print(grad_layer)
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features.append(grad_output)

        def backward_hook(module, grad_input, grad_output):
            self.backward_features.append(grad_output[0])

        hooks = {}
        back_hooks = {}
        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx in grad_layer:
                hooks[idx] = m.register_forward_hook(forward_hook)
                back_hooks[idx] = m.register_backward_hook(backward_hook)
                print("Register forward hook !", idx)
                print("Register backward hook !", idx)
                gradient_layer_found = True
                # break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels, features):

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training



        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.model.train(True)
            logits = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits).argmax(dim=1)
                labels_ohe = self._to_ohe(pred).cuda()
            else:
                labels_ohe = self._to_ohe(labels).cuda()

            gradient = logits * labels_ohe
            # print(gradient.shape, labels_ohe.shape)
            grad_logits = (logits * labels_ohe).sum()  # BS x num_classes
            grad_logits.backward(retain_graph=True)
            self.model.zero_grad()

        if is_train:
            self.model.train(True)
        else:
            self.model.train(False)
            self.model.eval()
            logits = self.model(images)

        backward_features = self.backward_features[1]  # BS x C x H x W
        # bs, c, h, w = backward_features.size()
        # wc = F.avg_pool2d(backward_features, (h, w), 1)  # BS x C x 1 x 1

        """
        The wc shows how important of the features map
        """

        # Eq 2
        print(len(self.feed_forward_features))

        fl = self.feed_forward_features[0]  # BS x C x H x W
        fh = self.feed_forward_features[1]
        print(fl.shape)
        print(fh.shape)
        out = self.attention_model(features, fl, fh)
        print(out.shape)
        Ac = torch.mul(fl, out).sum(dim=1, keepdim=True)
        print(Ac.shape)
        exit()
        # weights = F.adaptive_avg_pool2d(fh, 1)
        weights = F.adaptive_avg_pool2d(fh, (1, 1))
        print(weights.shape)

        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Bc = torch.mul(fl, Ac)

        print(weights.shape, Ac.shape, Bc.shape)
        exit()
        # bs, c, h, w = fl.size()
        # fl = fl.view(1, bs * c, h, w)

        """
        fl is the feature maps during feed-forward
        """

        """
        We do 2d convolution to find the Attention maps. We consider wc as a filter matrix.
        """

        # Ac = F.relu(F.conv2d(fl, wc, groups=bs))
        # # Resize to be as same as of image size
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        # Ac = Ac.permute((1, 0, 2, 3))
        # heatmap = Ac

        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        # print(weights.shape, Ac.shape, fl.shape, backward_features.shape)
        # exit()
        max_val = torch.max(Ac)
        if max_val == 0.0:
            max_val = max_val + 0.00001  # adding smoth
        if torch.isnan(max_val):
            print('max val is nan')

        Ac = Ac / max_val
        if torch.isnan(Ac.sum()):
            print('There is a nan value in heatmap')

        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])

        heatmap = Ac
        # heatmap = self.thershold(heatmap)

        """
        Generate the soft-mask
        """

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask

        logits_am = self.model(masked_image)

        return logits, logits_am, heatmap