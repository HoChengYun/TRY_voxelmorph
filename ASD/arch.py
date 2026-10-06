# -*- coding: utf-8 -*-
"""改架構用的網路（2026-10-06 起；CLAUDE.md 待辦 5、文獻/對位模型文獻筆記.md §7）。

原則：全部是「加上去的」—— voxelmorph-code 裡的 VxmDense 一行都不改，舊的 .pt 照舊能讀。
老師如果不希望改架構：這個檔、ASD/train_arch.py、run_train.py 的 --arch 拿掉即可；
評估腳本用的 load_model() 遇到舊模型就是原本的 VxmDense.load，留著也沒有影響。

VxmCascade：把 n 顆 VoxelMorph 串起來（RCN：Zhao et al., ICCV 2019）
---------------------------------------------------------------------
    第 1 顆：受試者 → 位移 U1
    第 k 顆：受試者照 U_{k-1} 搬好（每次都從原圖重新內插，不會越搬越糊）→ 修正 u_k
             U_k(x) = u_k(x) + U_{k-1}(x + u_k(x))
             （SpatialTransformer 的慣例是 out(x) = in(x + u(x))，所以「搬好的再搬一次」的總位移就是這條）
    每一顆都是完整的 VxmDense（各自的權重；速度場版的話各自積分，每一段本身都不會翻）。
    跟 VxmDense 一樣：model(受試者, atlas, registration=True) -> (搬好的受試者, 一個全尺寸的總位移)
    → test_dice.py、check_folding.py、visualize_*.py 算法都不用改，只要改用 load_model() 讀檔。
    訓練時 model(受試者, atlas) -> (搬好的受試者, [每一顆積分前的形變場])，平滑項每一顆都要算（見 train_arch.py）。

這個接法跟 ASD/test_multipass.py（第 0 步：同一顆模型連跑幾次）完全一樣；
cascade_from_single() 把同一顆的權重放進每一顆，結果要跟 test_multipass.py 一致（試跑時用來對答案）。

VxmPyramid：由粗到細（第 2 步；Mamba 那篇的 dual stream + pyramid + warping、Dual-PRNet、LapIRN 同一類）
---------------------------------------------------------------------
    VoxelMorph 原文的 U-Net 特徵有粗有細，但形變只在最後輸出一次（TMI 2019 Fig. 3）。這裡改成：
    1. 兩張影像**各自**用同一個編碼器抽特徵（共用權重），不再一開始就疊在一起
       —— 才能只把「移動影像的特徵」照目前的形變拉過去（由粗到細一定要這樣，兩件事綁在一起）
    2. 解碼器從 1/16 開始，**每一層都出一個速度場**、積分成這一層的小修正 u_k，接到前面的總位移上，再放大到下一層：
             U_k(x) = u_k(x) + U↑(x + u_k(x))      （U↑ = 上一層的總位移放大 2 倍、向量也乘 2）
       下一層先用 U↑ 把移動影像的特徵拉過去，看到的是已經對好大半的樣子，只要修細節。
    其他照 VxmDense：層數與通道數（預設 enc 16 32 32 32、dec 32 32 32 32 32 16 16，原文 Fig. 3）、
    3×3×3 卷積＋LeakyReLU(0.2)、步長 2 縮小、最近鄰放大、跳接、速度場積分 7 次（每一層都積分，每一段本身不會翻）。
    參數約 41 萬（VxmDense 30 萬）：解碼器每層多吃一份「固定影像的特徵」。
    訓練時 model(受試者, atlas) -> (搬好的受試者, [每一層積分前的速度場])，平滑項每一層都算。
    （粗的層用的是粗格子的單位；粗格子上的差分剛好就是實際形變的梯度，所以同一個 λ 對每一層的意思一樣。）

load_model(path, device)：看存檔裡的 config['arch'] 決定蓋哪一種網路；沒有 arch 就是原本的 VxmDense。
"""
import os
import sys

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, 'voxelmorph-code') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))
os.environ.setdefault('VXM_BACKEND', 'pytorch')
os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
import voxelmorph as vxm                                                  # noqa: E402
from voxelmorph.torch.modelio import LoadableModel, store_config_args     # noqa: E402


def stage_flows(m, source, target):
    """跟 VxmDense.forward（networks.py:180-205）同一套計算，但同時回傳兩個形變場：
    積分完、全尺寸的位移（拿來搬東西、往下接），以及積分前的形變場（給平滑項，跟 train.py 一樣）。"""
    x = m.unet_model(torch.cat([source, target], dim=1))
    pos = m.flow(x)
    if m.resize:
        pos = m.resize(pos)
    pre = pos
    if m.integrate:
        pos = m.integrate(pos)
        if m.fullsize:
            pos = m.fullsize(pos)
    return pos, pre


class VxmCascade(LoadableModel):
    """n 顆 VxmDense 串起來，最後接成一個總位移。參數名稱跟 VxmDense 一樣，多一個 n_cascades。"""

    @store_config_args
    def __init__(self, inshape, nb_unet_features=None, int_steps=7, int_downsize=2, n_cascades=2,
                 arch='cascade'):
        super().__init__()
        assert arch == 'cascade', arch
        assert n_cascades >= 1, n_cascades
        self.stages = nn.ModuleList([
            vxm.networks.VxmDense(inshape, nb_unet_features=nb_unet_features,
                                  int_steps=int_steps, int_downsize=int_downsize)
            for _ in range(n_cascades)])
        self.transformer = vxm.torch.layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        U, pres = None, []
        for m in self.stages:
            src = source if U is None else self.transformer(source, U)
            u, pre = stage_flows(m, src, target)
            pres.append(pre)
            U = u if U is None else u + self.transformer(U, u)
        moved = self.transformer(source, U)
        if registration:
            return moved, U
        return moved, pres


class VxmPyramid(LoadableModel):
    """由粗到細：兩張影像各自抽特徵，解碼器每一層都出形變、先把移動影像的特徵拉過去再修（見檔頭）。"""

    @store_config_args
    def __init__(self, inshape, nb_unet_features=None, int_steps=7, int_downsize=1, arch='pyramid'):
        super().__init__()
        assert arch == 'pyramid', arch
        assert int_downsize == 1, '由粗到細只做全尺寸積分（--int-downsize 1），粗的層本來就是縮小過的'
        ndims = len(inshape)
        enc_nf, dec_nf = nb_unet_features or ([16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16])
        n = len(enc_nf)                                   # 縮小幾次（預設 4：1/2、1/4、1/8、1/16）
        assert len(dec_nf) > n, '解碼器的長度要比編碼器多（VoxelMorph 的慣例是「層數 + 3」）'
        assert all(s % 2 ** n == 0 for s in inshape), '影像每邊要能被 %d 整除：%s' % (2 ** n, inshape)
        self.n = n
        shapes = [tuple(s // 2 ** k for s in inshape) for k in range(n + 1)]   # 第 k 層 = 原尺寸 / 2^k
        Conv = getattr(nn, 'Conv%dd' % ndims)
        Block = vxm.networks.ConvBlock                    # 3×3×3 卷積 + LeakyReLU(0.2)，跟 VxmDense 同一個

        # 編碼器（兩張影像共用）：輸入 1 個通道，每次步長 2
        self.enc = nn.ModuleList()
        prev = 1
        for nf in enc_nf:
            self.enc.append(Block(ndims, prev, nf, stride=2))
            prev = nf

        # 解碼器：第 k 層吃「搬過的移動影像特徵 + 固定影像特徵 + 上一層放大的解碼特徵」
        self.blocks = nn.ModuleList()
        out_ch = []
        for k in range(n + 1):
            if k == 0:                                    # 原尺寸：跟 VxmDense 的 extras 一樣（影像本身 1+1 個通道）
                prev, chain = 2 + dec_nf[n - 1], nn.ModuleList()
                for nf in dec_nf[n:]:
                    chain.append(Block(ndims, prev, nf))
                    prev = nf
            else:                                         # 第 k 層：一個卷積，通道數照 VxmDense 的 dec_nf
                cin = 2 * enc_nf[k - 1] + (dec_nf[n - k - 1] if k < n else 0)
                prev = dec_nf[n - k]
                chain = nn.ModuleList([Block(ndims, cin, prev)])
            self.blocks.append(chain)
            out_ch.append(prev)

        # 每一層一個「出速度場」的卷積，初始權重很小（同 VxmDense），一開始每一層都幾乎不動
        self.flows = nn.ModuleList()
        for c in out_ch:
            f = Conv(c, ndims, kernel_size=3, padding=1)
            f.weight = nn.Parameter(Normal(0, 1e-5).sample(f.weight.shape))
            f.bias = nn.Parameter(torch.zeros(f.bias.shape))
            self.flows.append(f)

        self.integrate = (nn.ModuleList([vxm.torch.layers.VecInt(s, int_steps) for s in shapes])
                          if int_steps > 0 else None)
        self.warp = nn.ModuleList([vxm.torch.layers.SpatialTransformer(s) for s in shapes])
        self.up_flow = vxm.torch.layers.ResizeTransform(0.5, ndims)     # 放大 2 倍、向量也乘 2
        self.up_feat = nn.Upsample(scale_factor=2, mode='nearest')     # 跟 VxmDense 的解碼器一樣

    def encode(self, x):
        feats = [x]
        for blk in self.enc:
            feats.append(blk(feats[-1]))
        return feats                                      # feats[k] = 第 k 層（0 是影像本身）

    def forward(self, source, target, registration=False):
        fm, ff = self.encode(source), self.encode(target)
        U, h, pres = None, None, []
        for k in range(self.n, -1, -1):                   # 從最粗（1/16）到原尺寸
            if U is not None:
                U, h = self.up_flow(U), self.up_feat(h)
            m_k = fm[k] if U is None else self.warp[k](fm[k], U)   # 移動影像的特徵先照目前的形變拉過去
            x = torch.cat([m_k, ff[k]] + ([h] if h is not None else []), dim=1)
            for blk in self.blocks[k]:
                x = blk(x)
            h = x
            v = self.flows[k](h)
            pres.append(v)
            u = self.integrate[k](v) if self.integrate is not None else v
            U = u if U is None else u + self.warp[k](U, u)          # U_k(x) = u_k(x) + U↑(x + u_k(x))
        moved = self.warp[0](source, U)
        if registration:
            return moved, U
        return moved, pres


ARCHS = {'cascade': VxmCascade, 'pyramid': VxmPyramid}


def load_model(path, device):
    """讀 .pt：新架構看 config['arch'] 蓋對應的網路；沒有 arch 的（以前所有模型）就是 VxmDense.load。"""
    ck = torch.load(path, map_location=torch.device(device))
    arch = ck['config'].get('arch')
    if arch is None:
        return vxm.networks.VxmDense.load(path, device)
    if arch not in ARCHS:
        raise ValueError('不認得的架構 %r（%s）' % (arch, path))
    model = ARCHS[arch](**ck['config'])
    # 存檔時拿掉了 .grid（SpatialTransformer 的座標格子，建模型時會重算），其他權重一個都不能少
    res = model.load_state_dict(ck['model_state'], strict=False)
    bad = [k for k in res.missing_keys if not k.endswith('.grid')] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError('%s 的權重對不上（%d 個），例如 %s' % (path, len(bad), bad[:3]))
    return model


def cascade_from_single(path, n_cascades, device):
    """把一顆 VxmDense 的權重放進每一顆，得到「同一顆連跑 n 次」的串接網路（試跑時對答案用）。"""
    single = vxm.networks.VxmDense.load(path, device)
    c = single.config
    assert not c.get('bidir') and c.get('nb_unet_levels') is None and c.get('unet_feat_mult', 1) == 1, c
    model = VxmCascade(c['inshape'], nb_unet_features=c['nb_unet_features'], int_steps=c['int_steps'],
                       int_downsize=c['int_downsize'], n_cascades=n_cascades)
    for m in model.stages:
        m.load_state_dict(single.state_dict())
    return model
