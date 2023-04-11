# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models import create_model
import torch.nn.functional as F


def rename_pretrain_weight(checkpoint):
    state_dict_old = checkpoint['state_dict']
    state_dict_new = OrderedDict()
    for key, value in state_dict_old.items():
        state_dict_new[key[len('module.'):]] = value
    return state_dict_new


# LoadTongrenPretrainedWeight_NoDistillation
def MIL_VT_FineTune(base_model='MIL_VT_small_patch16_384', \
                    MODEL_PATH_finetune='weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar', \
                    num_classes=5):
    """Load pretrain weight from distillation model, to train a plain model"""

    model = create_model(model_name=base_model,
                         pretrained=False,
                         num_classes=num_classes,
                         drop_rate=0,
                         drop_path_rate=0.1,
                         drop_block_rate=None,
                         )

    checkpoint0 = torch.load(MODEL_PATH_finetune, map_location='cpu')
    checkpoint_model = rename_pretrain_weight(checkpoint0)

    state_dict = model.state_dict()
    checkpoint_keys = list(checkpoint_model.keys())
    for tempKey in list(state_dict.keys()):
        if tempKey not in checkpoint_keys:
            print('Missing Key not in pretrain model: ', tempKey)

    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model:  # and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    num_extra_tokens_chechpoint = 2
    print('pos_embed: ', embedding_size, num_patches, num_extra_tokens)

    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_chechpoint) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_chechpoint:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

    return model


##
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


class MILVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.size2 = int(np.sqrt(num_patches))
        self.dim = self.embed_dim
        self.L = 256  # self.dim//3
        self.D = 128  # self.dim//5
        self.K = 1  # self.num_classes*1
        self.MIL_Prep = torch.nn.Sequential(
            torch.nn.Linear(self.dim, self.L),
            # torch.nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.L),
            torch.nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.MIL_attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            # nn.Tanh(),
            # nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.D),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.D, self.K)

            # nn.Linear(self.L, self.K)
        )

        self.MIL_classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

        trunc_normal_(self.pos_embed, std=.02)
        # self.head_dist.apply(self._init_weights)
        self.MIL_Prep[0].apply(self._init_weights)
        self.MIL_Prep[1].apply(self._init_weights)
        self.MIL_attention[0].apply(self._init_weights)
        self.MIL_attention[1].apply(self._init_weights)
        self.MIL_attention[4].apply(self._init_weights)
        self.MIL_classifier[0].apply(self._init_weights)

        self.feed_forward_features = None
        self.backward_features = None
        self.grad_layer = 'blocks.11.attn'


        # Register hooks
        self._register_hooks(self.grad_layer)


    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)


    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe

    def grad_rollout(self, attentions, gradients, discard_ratio=.7):
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(attentions, gradients):
                weights = grad
                attention_heads_fused = (attention * weights).mean(axis=1)
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
                # indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0, 1:]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape, "here")
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            # print("blocks: ", x.shape)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x, labels):

        x, x_patches = self.forward_features(x)
        # print("patches: ", x_patches.shape, x.shape)
        vt_out = self.head(x)
        # print('v out shape: ', vt_out.shape)

        labels_ohe = self._to_ohe(labels).cuda()
        gradient = vt_out * labels_ohe
        # print(gradient.shape, labels_ohe.shape)
        grad_logits = (vt_out * labels_ohe).sum()  # BS x num_classes
        grad_logits.backward(retain_graph=True)
        self.zero_grad()

        # vt_out.backward(retain_graph=True)
        fl = self.feed_forward_features
        backward_features = self.backward_features
        # mask = self.grad_rollout(fl.detach().data().cpu(), backward_features.detach().data().cpu(), 0.8)
        # print(mask.shape)
        # print("Features shape: ", fl.shape)

        # weights = F.adaptive_avg_pool2d(backward_features, 1)

        Ac = torch.mul(fl[:, 1:, :], backward_features[:, 1:, :]).sum(dim=2, keepdim=True)
        Ac = Ac.transpose( 2, 1)
        # print("Ac 1: ", Ac.shape)
        Ac = F.relu(Ac)
        Ac = Ac.view(Ac.shape[0], 1, 32, -1)

        max_val = torch.max(Ac)
        if max_val == 0.0:
            max_val = max_val + 0.00001  # adding smoth
        if torch.isnan(max_val):
            print('max val is nan')

        Ac = Ac / max_val
        if torch.isnan(Ac.sum()):
            print('There is a nan value in heatmap')

        Ac = F.upsample_bilinear(Ac, size=(512, 512))
        # print(fl.shape, self.backward_features.shape)
        # print("Ac shape: ", Ac.shape)

        # patch_size =16
        # w, h = x.shape[1] - x.shape[1] % 16, x.shape[2] - x.shape[2] % 16 # patch size 16
        #
        #
        # w_featmap = x.shape[-2] // patch_size
        # h_featmap = x.shape[-1] // patch_size
        #
        # print("attention shape: ", attentions.shape)
        #
        # nh = attentions.shape[1]  # number of head
        #
        # # keep only the output patch attention
        # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        #
        # attentions = attentions.reshape(nh, w_featmap, h_featmap)
        # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        #     0].cpu().numpy()

        """MIL operations for the """
        H = self.MIL_Prep(x_patches)  # B*N*D -->  B*N*L

        A = self.MIL_attention(H)  # B*N*K
        # A = torch.transpose(A, 1, 0)  # KxN
        A = A.permute((0, 2, 1))  # B*K*N
        A = nn.functional.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, H)  # B*K*N X B*N*L --> B*K*L
        M = M.view(-1, M.size(1) * M.size(2))

        mil_out = self.MIL_classifier(M)

        # return vt_out, mil_out, x_patches
        if self.training:
            return vt_out, mil_out, Ac
        else:
            # during inference, return the average of both classifier predictions
            return (vt_out + mil_out) / 2


###################
class MILVisionTransformer_Distil(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.size2 = int(np.sqrt(num_patches))
        self.dim = self.embed_dim
        self.L = 256  # self.dim//3
        self.D = 128  # self.dim//5
        self.K = 1  # self.num_classes*1
        self.MIL_Prep = torch.nn.Sequential(
            torch.nn.Linear(self.dim, self.L),
            # torch.nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.L),
            torch.nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.MIL_attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            # nn.Tanh(),
            # nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.D),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.D, self.K)

            # nn.Linear(self.L, self.K)
        )

        self.MIL_classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
            # nn.Sigmoid()
        )

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.MIL_Prep[0].apply(self._init_weights)
        self.MIL_Prep[1].apply(self._init_weights)
        self.MIL_attention[0].apply(self._init_weights)
        self.MIL_attention[1].apply(self._init_weights)
        self.MIL_attention[4].apply(self._init_weights)
        self.MIL_classifier[0].apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1], x[:, 2:]

    def forward(self, x):
        x, x_dist, x_patches = self.forward_features(x)
        vt_out = self.head(x)
        dist_out = self.head_dist(x_dist)

        """MIL operations for the """
        """MIL operations for the """
        H = self.MIL_Prep(x_patches)  # B*N*D -->  B*N*L

        A = self.MIL_attention(H)  # B*N*K
        # A = torch.transpose(A, 1, 0)  # KxN
        A = A.permute((0, 2, 1))  # B*K*N
        A = nn.functional.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, H)  # B*K*N X B*N*L --> B*K*L
        M = M.view(-1, M.size(1) * M.size(2))

        mil_out = self.MIL_classifier(M)

        # return vt_out, mil_out, x_patches
        if self.training:
            return (vt_out, dist_out), mil_out
        else:
            # during inference, return the average of both classifier predictions
            return (vt_out + dist_out + mil_out) / 3


#######################


###############################


@register_model
def MIL_VT_small_patch16_384(pretrained=False, **kwargs):
    model = MILVisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def MIL_VT_small_patch16_512(pretrained=False, **kwargs):
    model = MILVisionTransformer(
        img_size=512, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


