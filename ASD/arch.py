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

load_model(path, device)：看存檔裡的 config['arch'] 決定蓋哪一種網路；沒有 arch 就是原本的 VxmDense。
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


ARCHS = {'cascade': VxmCascade}


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
