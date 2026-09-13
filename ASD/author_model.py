"""
把 VoxelMorph 作者釋出的 Keras 模型（.h5）搬進 PyTorch。

為什麼要自己搬
--------------
作者的預訓練模型 models/vxm_dense_brain_T1_3D_mse.h5 是 TensorFlow/Keras 存的。
這台的 neurite 跟 TF 2.21 內建的 Keras 3 不相容（import 就失敗），載不起來。
repo 的 PyTorch VxmDense 也不能直接吃：這顆用了 unet_half_res=True
（decoder 少做最後一次放大，形變場直接在半解析度輸出），PyTorch 版沒有這個選項。

所以照 voxelmorph/tf/networks.py 的 Unet / VxmDense 在 PyTorch 重蓋一次，權重逐層照名字搬。
卷積核從 Keras 的 (kx, ky, kz, in, out) 轉成 PyTorch 的 (out, in, kx, ky, kz)。

跟原版可能的差異（都在形變場的內插，不在網路本身）
  - 半解析度形變場放大 2 倍：TF 用 neurite Resize，這裡用 PyTorch ResizeTransform
    （align_corners=True），取樣位置差不到半格
  - 積分與搬影像碰到邊界外：TF 夾到邊界值，PyTorch grid_sample 補 0

⚠️ 這顆不是論文 Table I 的模型：它是微分同胚版（int_steps=7），Table I 主結果是 int_steps=0。

用法
    from author_model import load_author_h5
    model = load_author_h5('models/vxm_dense_brain_T1_3D_mse.h5', 'cuda')
    moved, flow = model(source, target, registration=True)   # 跟 VxmDense 一樣
"""
import json

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import voxelmorph as vxm

# 作者模型的設定（讀 h5 裡的 model_config 核對，不符就不載，避免安靜地搬錯）
_EXPECT = dict(unet_half_res=True, int_downsize=2, nb_unet_conv_per_level=1,
               nb_unet_features=None, bidir=False, use_probs=False, src_feats=1, trg_feats=1)
_ENC, _DEC, _FINAL = [16, 32, 32, 32], [32, 32, 32, 32], [32, 16, 16]   # repo 預設的 unet 特徵數


class _ConvAct(nn.Module):
    """3×3×3 卷積 + LeakyReLU(0.2)，對應 tf/networks.py 的 _conv_block。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)

    def forward(self, x):
        return nnf.leaky_relu(self.conv(x), 0.2)


class AuthorVxmDense(nn.Module):
    def __init__(self, inshape, int_steps):
        super().__init__()
        self.enc = nn.ModuleList()
        c = 2                                   # [source, target] 兩個 channel 疊起來
        for nf in _ENC:
            self.enc.append(_ConvAct(c, nf))
            c = nf
        # decoder：每層卷積完放大 2 倍、接上同尺度的 encoder 輸出；half_res → 最後一層不放大
        skip_nf = _ENC[::-1]
        self.dec = nn.ModuleList()
        for lvl, nf in enumerate(_DEC):
            self.dec.append(_ConvAct(c, nf))
            c = nf + skip_nf[lvl] if lvl < len(_DEC) - 1 else nf
        self.final = nn.ModuleList()
        for nf in _FINAL:
            self.final.append(_ConvAct(c, nf))
            c = nf
        self.flow = nn.Conv3d(c, 3, 3, padding=1)
        half = [s // 2 for s in inshape]
        self.integrate = vxm.torch.layers.VecInt(half, int_steps)
        self.fullsize = vxm.torch.layers.ResizeTransform(1 / 2, 3)
        self.transformer = vxm.torch.layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        x = torch.cat([source, target], dim=1)
        skips = []
        for conv in self.enc:
            x = conv(x)
            skips.append(x)
            x = nnf.max_pool3d(x, 2)
        for lvl, conv in enumerate(self.dec):
            x = conv(x)
            if lvl < len(self.dec) - 1:
                x = nnf.interpolate(x, scale_factor=2, mode='nearest')   # Keras UpSampling3D
                x = torch.cat([x, skips.pop()], dim=1)
        for conv in self.final:
            x = conv(x)
        vel = self.flow(x)                          # 半解析度的速度場
        pos_flow = self.fullsize(self.integrate(vel))
        moved = self.transformer(source, pos_flow)
        return (moved, pos_flow) if registration else (moved, vel)


def load_author_h5(path, device='cpu'):
    with h5py.File(path, 'r') as f:
        cfg = json.loads(f.attrs['model_config'])['config']
        bad = {k: cfg.get(k) for k, v in _EXPECT.items() if cfg.get(k) != v}
        if bad or cfg.get('int_steps', 0) < 1:
            raise ValueError('這個 h5 的設定跟移植的架構不同，不能用：%s int_steps=%s'
                             % (bad, cfg.get('int_steps')))
        model = AuthorVxmDense(tuple(cfg['inshape']), cfg['int_steps'])
        g = f['model_weights']

        def put(mod, name):
            k = g[name][name]['kernel:0'][()]
            b = g[name][name]['bias:0'][()]
            w = torch.from_numpy(np.ascontiguousarray(k.transpose(4, 3, 0, 1, 2)))
            if w.shape != mod.weight.shape:
                raise ValueError('%s 形狀不符：h5 %s vs 模組 %s' % (name, tuple(w.shape), tuple(mod.weight.shape)))
            mod.weight.data.copy_(w)
            mod.bias.data.copy_(torch.from_numpy(b))

        for i, blk in enumerate(model.enc):
            put(blk.conv, 'vxm_dense_unet_enc_conv_%d_0' % i)
        for lvl, blk in enumerate(model.dec):
            put(blk.conv, 'vxm_dense_unet_dec_conv_%d_0' % (len(model.dec) - 1 - lvl))
        for i, blk in enumerate(model.final):
            put(blk.conv, 'vxm_dense_unet_dec_final_conv_%d' % i)
        put(model.flow, 'vxm_dense_flow')
    return model.to(device).eval()
