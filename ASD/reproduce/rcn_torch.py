# -*- coding: utf-8 -*-
"""RCN（Recursive Cascaded Networks, Zhao et al., ICCV 2019）原作程式的 PyTorch 翻譯。

原作：github.com/microsoft/Recursive-Cascaded-Networks @0709490（TensorFlow 1.4 + tflearn）
Copyright (c) Microsoft Corporation，MIT License（全文見同資料夾 LICENSE_RCN.txt）。
每個類別、函式都照原作同名的那一段逐行翻，註解寫出原作的檔名與函式名：
  network/base_networks.py          → VTN、VoxelMorph、VTNAffineStem、affine_flow、det3x3
  network/recursive_cascaded_networks.py → RecursiveCascadedNetworks（串接、合成位移、損失）
  network/spatial_transformer.py    → warp（Dense3DSpatialTransformer）
  network/transform.py              → sample_power、free_form_fields（訓練時對 moving 加的隨機形變）
  network/framework.py              → augment（FrameworkUnsupervised 裡的那段資料增強）

只改逼不得已的（手冊 §31）：
  1. 框架 TF → PyTorch：資料排法 [B, X, Y, Z, C] → [B, C, X, Y, Z]。
     卷積的 'SAME' 補零、tflearn 的權重初始化都照 TF 的定義另外寫（PyTorch 預設的補零位置、初始化不一樣）。
  2. 影像大小 128³ → 192×224×192：
     - VTNAffineStem 最後一層 conv7 的 kernel = 最底層特徵圖大小，原作 2×2×2，我們 3×4×3（同一個寫法，自動算）
     - free_form_fields：原作最後的 reshape 只在 x、y 一樣大時成立（他們都是 128³）。這裡照它的數學定義寫
       （三個軸各自做三次 B-spline），128³ 時結果的統計分佈跟原作一樣（verify_rcn_port.py 實際比對過）
沒有翻的：Fast3DTransformer（原作要另外裝自己編的 TF 1.4，預設不用）、landmark／Dice 那些記錄用的輸出。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TF 的 'SAME' 補零與 tflearn 的初始化 ──────────────────────────────────

def same_pads(n, k, s):
    """TF padding='SAME'：輸出 ceil(n / s)；總補零 max((輸出 − 1)·s + k − n, 0)，前面 floor、後面補剩下的。
    （PyTorch 的 padding=1 是前後各一；stride 2 遇到偶數大小時 TF 只補後面，取樣位置差半格。）"""
    out = -(-n // s)
    tot = max((out - 1) * s + k - n, 0)
    return tot // 2, tot - tot // 2


def uniform_scaling_(w, fan):
    """tflearn 'uniform_scaling'（= tf.uniform_unit_scaling_initializer，factor 1）：
    U(−√(3/fan), √(3/fan))，fan = TF 權重 shape 除了最後一維的乘積。"""
    lim = math.sqrt(3.0 / fan)
    with torch.no_grad():
        w.uniform_(-lim, lim)


class Conv(nn.Module):
    """base_networks.py convolve()：tflearn.layers.conv_3d(padding='same', activation='linear', bias=True)。
    TF 權重 [k, k, k, cin, cout] → fan = k³·cin；bias 初始 0。init=('normal', std) 對應 tflearn normal(stddev)。"""

    def __init__(self, cin, cout, k, s, init='uniform_scaling'):
        super().__init__()
        self.k, self.s = k, s
        self.conv = nn.Conv3d(cin, cout, k, s, padding=0, bias=True)
        if init == 'uniform_scaling':
            uniform_scaling_(self.conv.weight, k ** 3 * cin)
        else:
            nn.init.normal_(self.conv.weight, 0.0, init[1])
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        pads = []
        for n in reversed(x.shape[2:]):         # F.pad 從最後一維開始給
            pads += same_pads(n, self.k, self.s)
        return self.conv(F.pad(x, pads))


class UpConv(nn.Module):
    """base_networks.py upconvolve()：tflearn conv_3d_transpose(padding='same', activation='linear', bias=False)，
    輸出大小由 targetShape 指定（= 編碼器那一層的大小，可以是奇數）。
    TF 的轉置卷積 = 「輸入大小為 targetShape、SAME 補零的卷積」的梯度：先不補零做完整的轉置卷積，再裁掉前面補的那幾格。
    （PyTorch 的 ConvTranspose3d 只能前後對稱，遇到 224 這條軸的 7 → 4 會對不上。）
    TF 權重 [k, k, k, cout, cin] → fan = k³·cout。"""

    def __init__(self, cin, cout, k=4, s=2):
        super().__init__()
        self.k, self.s = k, s
        self.conv = nn.ConvTranspose3d(cin, cout, k, s, padding=0, bias=False)
        uniform_scaling_(self.conv.weight, k ** 3 * cout)

    def forward(self, x, size):
        y = self.conv(x)
        sl = [slice(None), slice(None)]
        for n_in, n in zip(x.shape[2:], size):
            assert -(-n // self.s) == n_in, (n_in, n)
            before = same_pads(n, self.k, self.s)[0]
            sl.append(slice(before, before + n))
        return y[tuple(sl)]


def lrelu(x):
    """base_networks.py convolveLeakyReLU → utils.py LeakyReLU：tflearn leaky_relu，alpha 0.1。"""
    return F.leaky_relu(x, 0.1)


# ── base_networks.py ─────────────────────────────────────────────────────

class VTN(nn.Module):
    """base_networks.py VTN（channels 預設 16）。img1 = fixed、img2 = moving，原作 tf.concat([img1, img2], 4)。
    輸出 {'flow': pred0 × 20 × flow_multiplier}，flow 第 i 個通道 = 沿第 i 軸的位移（voxel）。"""

    def __init__(self, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        c = channels
        self.conv1 = Conv(2, c, 3, 2)
        self.conv2 = Conv(c, c * 2, 3, 2)
        self.conv3 = Conv(c * 2, c * 4, 3, 2)
        self.conv3_1 = Conv(c * 4, c * 4, 3, 1)
        self.conv4 = Conv(c * 4, c * 8, 3, 2)
        self.conv4_1 = Conv(c * 8, c * 8, 3, 1)
        self.conv5 = Conv(c * 8, c * 16, 3, 2)
        self.conv5_1 = Conv(c * 16, c * 16, 3, 1)
        self.conv6 = Conv(c * 16, c * 32, 3, 2)
        self.conv6_1 = Conv(c * 32, c * 32, 3, 1)
        self.pred6 = Conv(c * 32, 3, 3, 1)
        self.upsamp6to5 = UpConv(3, 3)
        self.deconv5 = UpConv(c * 32, c * 16)
        self.pred5 = Conv(c * 32 + 3, 3, 3, 1)          # concat5 = conv5_1、deconv5、upsamp6to5
        self.upsamp5to4 = UpConv(3, 3)
        self.deconv4 = UpConv(c * 32 + 3, c * 8)
        self.pred4 = Conv(c * 16 + 3, 3, 3, 1)
        self.upsamp4to3 = UpConv(3, 3)
        self.deconv3 = UpConv(c * 16 + 3, c * 4)
        self.pred3 = Conv(c * 8 + 3, 3, 3, 1)
        self.upsamp3to2 = UpConv(3, 3)
        self.deconv2 = UpConv(c * 8 + 3, c * 2)
        self.pred2 = Conv(c * 4 + 3, 3, 3, 1)
        self.upsamp2to1 = UpConv(3, 3)
        self.deconv1 = UpConv(c * 4 + 3, c)
        self.upsamp1to0 = UpConv(c * 2 + 3, 3)

    def forward(self, img1, img2):
        concat_imgs = torch.cat([img1, img2], 1)
        conv1 = lrelu(self.conv1(concat_imgs))
        conv2 = lrelu(self.conv2(conv1))
        conv3 = lrelu(self.conv3(conv2))
        conv3_1 = lrelu(self.conv3_1(conv3))
        conv4 = lrelu(self.conv4(conv3_1))
        conv4_1 = lrelu(self.conv4_1(conv4))
        conv5 = lrelu(self.conv5(conv4_1))
        conv5_1 = lrelu(self.conv5_1(conv5))
        conv6 = lrelu(self.conv6(conv5_1))
        conv6_1 = lrelu(self.conv6_1(conv6))
        shape0, shape1, shape2, shape3, shape4, shape5 = (
            t.shape[2:] for t in (concat_imgs, conv1, conv2, conv3, conv4, conv5))

        pred6 = self.pred6(conv6_1)
        concat5 = torch.cat([conv5_1, lrelu(self.deconv5(conv6_1, shape5)), self.upsamp6to5(pred6, shape5)], 1)
        pred5 = self.pred5(concat5)
        concat4 = torch.cat([conv4_1, lrelu(self.deconv4(concat5, shape4)), self.upsamp5to4(pred5, shape4)], 1)
        pred4 = self.pred4(concat4)
        concat3 = torch.cat([conv3_1, lrelu(self.deconv3(concat4, shape3)), self.upsamp4to3(pred4, shape3)], 1)
        pred3 = self.pred3(concat3)
        concat2 = torch.cat([conv2, lrelu(self.deconv2(concat3, shape2)), self.upsamp3to2(pred3, shape2)], 1)
        pred2 = self.pred2(concat2)
        concat1 = torch.cat([conv1, lrelu(self.deconv1(concat2, shape1)), self.upsamp2to1(pred2, shape1)], 1)
        pred0 = self.upsamp1to0(concat1, shape0)
        return {'flow': pred0 * 20 * self.flow_multiplier}


class VoxelMorph(nn.Module):
    """base_networks.py VoxelMorph（RCN 自己寫的 TF 版，等於 VoxelMorph-2：編碼 16-32-32-32、解碼 32-32-32-32-32-16-16）。
    位移場版（不積分）；最後一層 flow 用 normal(stddev=1e-5) 初始化；UpSampling3D = 最近鄰 ×2。"""

    def __init__(self, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        enc = [m * channels for m in [1, 2, 2, 2]]
        dec = [m * channels for m in [2, 2, 2, 2, 2, 1, 1]] + [3]
        self.conv1 = Conv(2, enc[0], 3, 2)
        self.conv2 = Conv(enc[0], enc[1], 3, 2)
        self.conv3 = Conv(enc[1], enc[2], 3, 2)
        self.conv4 = Conv(enc[2], enc[3], 3, 2)
        self.decode4 = Conv(enc[3], dec[0], 3, 1)
        self.decode3 = Conv(dec[0] + enc[2], dec[1], 3, 1)
        self.decode2 = Conv(dec[1] + enc[1], dec[2], 3, 1)
        self.decode1 = Conv(dec[2] + enc[0], dec[3], 3, 1)
        self.decode1_1 = Conv(dec[3], dec[4], 3, 1)
        self.decode0 = Conv(dec[4] + 2, dec[5], 3, 1)
        self.decode0_1 = Conv(dec[5], dec[6], 3, 1)     # 原作 len(self.decoders) == 8 才有這層（預設就是 8）
        self.flow = Conv(dec[6], dec[7], 3, 1, init=('normal', 1e-5))

    def forward(self, img1, img2):
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')     # noqa: E731  keras UpSampling3D
        concat_imgs = torch.cat([img1, img2], 1)
        conv1 = lrelu(self.conv1(concat_imgs))
        conv2 = lrelu(self.conv2(conv1))
        conv3 = lrelu(self.conv3(conv2))
        conv4 = lrelu(self.conv4(conv3))
        net = lrelu(self.decode4(conv4))
        net = lrelu(self.decode3(torch.cat([up(net), conv3], 1)))
        net = lrelu(self.decode2(torch.cat([up(net), conv2], 1)))
        net = lrelu(self.decode1(torch.cat([up(net), conv1], 1)))
        net = lrelu(self.decode1_1(net))
        net = lrelu(self.decode0(torch.cat([up(net), concat_imgs], 1)))
        net = lrelu(self.decode0_1(net))
        return {'flow': self.flow(net) * self.flow_multiplier}


def det3x3(M):
    """base_networks.py det3x3：M[:, i, j]。"""
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) \
        - (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])


def elem_sym_polys_of_eigen_values(M):
    """base_networks.py VTNAffineStem 裡的同名函式：σ1 = 跡、σ2 = 主二階子式和、σ3 = 行列式。"""
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    sigma1 = M[0][0] + M[1][1] + M[2][2]
    sigma2 = (M[0][0] * M[1][1] + M[1][1] * M[2][2] + M[2][2] * M[0][0]) \
        - (M[0][1] * M[1][0] + M[1][2] * M[2][1] + M[2][0] * M[0][2])
    sigma3 = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) \
        - (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])
    return sigma1, sigma2, sigma3


class VTNAffineStem(nn.Module):
    """base_networks.py VTNAffineStem：編碼器 16～512（固定，不跟 channels）＋conv7 輸出 A − I（9 個）與 b（3 個）。
    conv7 是 'valid'、kernel = 最底層特徵圖大小（6 次 stride 2 的 SAME，每次 ceil(n/2)；128 → 2，192 → 3，224 → 4）。
    flow(x) = (A − I)·x + b，x 是以影像中心為原點的座標（affine_flow）；另回傳 det_loss、ortho_loss。"""

    def __init__(self, image_size, flow_multiplier=1.):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        ch = [2, 16, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        st = [2, 2, 2, 1, 2, 1, 2, 1, 2, 1]
        names = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6', 'conv6_1']
        self.convs = nn.ModuleDict({n: Conv(ch[i], ch[i + 1], 3, st[i]) for i, n in enumerate(names)})
        ks = list(image_size)
        for _ in range(6):
            ks = [-(-n // 2) for n in ks]
        self.conv7_W = nn.Conv3d(512, 9, ks, bias=False)
        self.conv7_b = nn.Conv3d(512, 3, ks, bias=False)
        for c in (self.conv7_W, self.conv7_b):
            uniform_scaling_(c.weight, int(np.prod(ks)) * 512)
        # affine_flow 的座標：每一軸 −(n−1)/2 … (n−1)/2（tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0)）
        r = [torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0 for n in image_size]
        self.register_buffer('grid', torch.stack(torch.meshgrid(*r, indexing='ij'), 0), persistent=False)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)
        for conv in self.convs.values():
            x = lrelu(conv(x))
        I = torch.eye(3, device=x.device, dtype=x.dtype)[None]
        W = self.conv7_W(x).reshape(-1, 3, 3) * self.flow_multiplier
        b = self.conv7_b(x).reshape(-1, 3) * self.flow_multiplier
        A = W + I
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x；the model learns W = A - I.（原作註解）
        flow = torch.einsum('bij,jxyz->bixyz', W, self.grid) + b[:, :, None, None, None]
        det = det3x3(A)
        det_loss = 0.5 * ((det - 1.0) ** 2).sum()                      # tf.nn.l2_loss(det - 1.0)
        eps = 1e-5
        C = torch.matmul(A.transpose(1, 2), A) + eps * I               # tf.matmul(A, A, True) + epsI
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = (s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)).sum()
        return {'flow': flow, 'W': W, 'b': b, 'det_loss': det_loss, 'ortho_loss': ortho_loss}


# ── spatial_transformer.py ───────────────────────────────────────────────

_GRIDS = {}


def warp(im, flow):
    """spatial_transformer.py Dense3DSpatialTransformer（原作預設 padding=False）：輸出(x) = im(x + flow(x))，三線性。
    原作把超出範圍的取樣點夾到邊界（tf.clip_by_value；權重用夾過的 x1 算，兩個點都落在邊界時合起來 = 邊界值），
    等於 grid_sample 的 padding_mode='border'、align_corners=True；梯度在範圍外也一樣是 0。"""
    size = flow.shape[2:]
    key = (tuple(size), flow.device, flow.dtype)
    if key not in _GRIDS:
        r = [torch.arange(n, device=flow.device, dtype=flow.dtype) for n in size]
        _GRIDS[key] = torch.stack(torch.meshgrid(*r, indexing='ij'), 0)[None]
    loc = _GRIDS[key] + flow
    loc = torch.stack([2 * loc[:, i] / (n - 1) - 1 for i, n in enumerate(size)], -1)
    return F.grid_sample(im, loc.flip(-1), mode='bilinear', padding_mode='border', align_corners=True)


# ── recursive_cascaded_networks.py ───────────────────────────────────────

def similarity_loss(img1, warped_img2):
    """recursive_cascaded_networks.py similarity_loss：1 − 全域相關係數（整張影像，var 各加 1e-6），batch 加總。"""
    B = img1.shape[0]
    f1 = img1.reshape(B, -1)
    f2 = warped_img2.reshape(B, -1)
    m1 = f1.mean(-1, keepdim=True)
    m2 = f2.mean(-1, keepdim=True)
    var1 = ((f1 - m1) ** 2).mean(-1)
    var2 = ((f2 - m2) ** 2).mean(-1)
    cov12 = ((f1 - m1) * (f2 - m2)).mean(-1)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    return (1 - pearson_r).sum()


def regularize_loss(flow):
    """recursive_cascaded_networks.py regularize_loss：三個方向相鄰差的 tf.nn.l2_loss（= 平方和 / 2）
    除以每一筆的大小（X·Y·Z·3），batch 加總。"""
    l2 = lambda t: 0.5 * (t ** 2).sum()     # noqa: E731
    return (l2(flow[:, :, 1:] - flow[:, :, :-1]) + l2(flow[:, :, :, 1:] - flow[:, :, :, :-1])
            + l2(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])) / float(np.prod(flow.shape[1:]))


class RecursiveCascadedNetworks(nn.Module):
    """recursive_cascaded_networks.py RecursiveCascadedNetworks。
    stems = [VTNAffineStem] + 每一顆 base network 重複 rep 次（同一個物件 = 共用權重），每顆 flow_multiplier = 1/n_cascades；
    相似度只算最後一顆（raw_weight），平滑項每一顆 base 都算自己的 flow（reg_weight），affine 那顆算 det／ortho。
    第 k 顆的輸入 = (fixed, 上一顆搬好的 moving)；合成位移：接在 affine 後面用 (W + I)·flow + affine flow（精確），
    其餘 warp(上一顆的合成位移, flow) + flow；每一顆搬的都是原始的 moving（img2）。"""
    default_params = {'weight': 1, 'raw_weight': 1, 'reg_weight': 1}

    def __init__(self, base_network, n_cascades, image_size, rep=1,
                 det_factor=0.1, ortho_factor=0.1, reg_factor=1.0, warp_gradient=True):
        super().__init__()
        self.det_factor = det_factor
        self.ortho_factor = ortho_factor
        self.reg_factor = reg_factor
        self.warp_gradient = warp_gradient
        base = {'VTN': VTN, 'VoxelMorph': VoxelMorph}[base_network]
        nets = {'affine_stem': VTNAffineStem(image_size)}
        self.stems = [('affine_stem', {'raw_weight': 0, 'reg_weight': 0})]
        for i in range(n_cascades):
            nets['deform_stem_%d' % i] = base(flow_multiplier=1.0 / n_cascades)
            params = {'raw_weight': 0}
            self.stems += [('deform_stem_%d' % i, params)] * rep         # 原作 [(...)] * rep：同一個 dict
        self.stems[-1][1]['raw_weight'] = 1
        for _, param in self.stems:
            for k, v in self.default_params.items():
                if k not in param:
                    param[k] = v
        self.nets = nn.ModuleDict(nets)

    def forward(self, img1, img2):
        """img1 = fixed（我們：atlas）、img2 = moving（我們：受試者）。回傳 loss、最後的合成位移 flow、搬好的 warped、每一顆的結果。"""
        stem_results = []
        stem_result = self.nets['affine_stem'](img1, img2)
        stem_result['warped'] = warp(img2, stem_result['flow'])
        stem_result['agg_flow'] = stem_result['flow']
        stem_results.append(stem_result)

        for name, params in self.stems[1:]:
            prev = stem_results[-1]
            stem_result = self.nets[name](img1, prev['warped'] if self.warp_gradient else prev['warped'].detach())
            if len(stem_results) == 1 and 'W' in prev:
                I = torch.eye(3, device=img1.device, dtype=img1.dtype)[None]
                stem_result['agg_flow'] = torch.einsum('bij,bjxyz->bixyz', prev['W'] + I, stem_result['flow']) + prev['flow']
            else:
                stem_result['agg_flow'] = warp(prev['agg_flow'], stem_result['flow']) + stem_result['flow']
            stem_result['warped'] = warp(img2, stem_result['agg_flow'])
            stem_results.append(stem_result)

        # unsupervised learning with simlarity loss and regularization loss（原作註解）
        for stem_result, (name, params) in zip(stem_results, self.stems):
            if 'W' in stem_result:
                stem_result['loss'] = stem_result['det_loss'] * self.det_factor + stem_result['ortho_loss'] * self.ortho_factor
                if params['raw_weight'] > 0:
                    stem_result['raw_loss'] = similarity_loss(img1, stem_result['warped'])
                    stem_result['loss'] = stem_result['loss'] + stem_result['raw_loss'] * params['raw_weight']
            else:
                if params['raw_weight'] > 0:
                    stem_result['raw_loss'] = similarity_loss(img1, stem_result['warped'])
                if params['reg_weight'] > 0:
                    stem_result['reg_loss'] = regularize_loss(stem_result['flow']) * self.reg_factor
                stem_result['loss'] = sum(stem_result[k] * params[k.replace('loss', 'weight')]
                                          for k in list(stem_result) if k.endswith('loss'))

        loss = sum(r['loss'] * params['weight'] for r, (name, params) in zip(stem_results, self.stems))
        return {'loss': loss, 'flow': stem_results[-1]['agg_flow'], 'warped': stem_results[-1]['warped'],
                'stems': stem_results}


# ── transform.py ＋ framework.py（訓練時的資料增強）──────────────────────

def get_coef(u):
    """transform.py get_coef：三次 B-spline 的四個基底，u ∈ [0, 1)。回傳 (len(u), 4)。"""
    return torch.stack([((1 - u) ** 3) / 6, (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6,
                        (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6, (u ** 3) / 6], 1)


def sample_power(lo, hi, k, size, device=None, generator=None):
    """transform.py sample_power：U(−r, r)、r = ((hi − lo)/2)^(1/k)，再取 |·|^k 保留正負號 → 範圍 [lo, hi]、集中在中間。"""
    r = ((hi - lo) / 2) ** (1 / k)
    center = (hi + lo) / 2
    points = (torch.rand(size, device=device, generator=generator) - 0.5) * 2 * r
    points = points.abs() ** k * points.sign()
    return points + center


def bspline_matrix(N, n, device=None):
    """free_form_fields 單一軸的內插矩陣 (N, n + 2)：控制點補一圈 0 後共 n + 2 個；N 分成 n − 1 段、每段 s = N/(n − 1) 格，
    第 i 段第 q 格用補過的第 i～i+3 個控制點，係數 get_coef((q + 0.5)/s)。"""
    assert N % (n - 1) == 0, (N, n)
    s = N // (n - 1)
    coef = get_coef((torch.arange(s, dtype=torch.float32, device=device) + 0.5) / s)     # (s, 4)
    M = torch.zeros(N, n + 2, device=device)
    for i in range(n - 1):
        M[i * s:(i + 1) * s, i:i + 4] = coef
    return M


def free_form_fields(shape, control_fields):
    """transform.py free_form_fields（padding='same'）：控制點 (B, n, m, t, 3) 補一圈 0，三個軸各自三次 B-spline 內插到 shape。
    回傳 (B, 3, X, Y, Z)。原作是把三個軸一個一個 matmul、中間 reshape／transpose（最後一步只在 X = Y 時成立）；
    這裡用同樣的三個內插矩陣一次乘完，數學上相同。"""
    B, n, m, t, _ = control_fields.shape
    cf = F.pad(control_fields.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1))     # pad_3d(control_fields, 1)
    Mx, My, Mz = (bspline_matrix(N, c, control_fields.device) for N, c in zip(shape, (n, m, t)))
    # 一個軸一個軸乘（三個一起丟給 einsum 的話，沒有 opt_einsum 時中間結果會有 X·7·Y·7·Z·7 個元素）
    f = torch.einsum('zk,bcijk->bcijz', Mz, cf)
    f = torch.einsum('yj,bcijz->bciyz', My, f)
    return torch.einsum('xi,bciyz->bcxyz', Mx, f)


def augment(img2, generator=None):
    """framework.py FrameworkUnsupervised 的資料增強（net_args 沒有 'augmentation' 時，原作預設就是這個）：
    控制點 5×5×5×3 ~ sample_power(−0.4, 0.4, 3) × (影像大小 // 4)，B-spline 內插成隨機形變，只搬 moving（img2）。
    只在訓練時用（原作 tf.cond(tflearn.get_training_mode(), ...)）。"""
    imgs = list(img2.shape[2:])
    control_fields = sample_power(-0.4, 0.4, 3, (img2.shape[0], 5, 5, 5, 3), img2.device, generator) \
        * torch.tensor(np.array(imgs) // 4, dtype=torch.float32, device=img2.device)
    aug_flow = free_form_fields(imgs, control_fields)
    return warp(img2, aug_flow)
