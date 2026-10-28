# -*- coding: utf-8 -*-
"""驗證 rcn_torch.py 跟 RCN 原作（TensorFlow）逐項相同：直接執行原作的程式碼，給同一組權重、同一組輸入，比輸出。

原作的檔案原封不動讀進來執行（exec），只把它 import 的外部套件換掉：
  tflearn 的 conv_3d／conv_3d_transpose → 用 tf.nn.conv3d／conv3d_transpose（TF 自己的 'SAME'），權重從 PyTorch 模型搬過去
  tflearn.activations.leaky_relu → tf.nn.leaky_relu；keras UpSampling3D 用真的 keras；Network 只留「記下現在是哪一顆」
  （原作是 TF 1.4 的 graph 模式，這台是 TF 2.21 eager：TF1 才有的函式自動改叫 tf.compat.v1 的同名函式）
要 vxm_env（有 TF 2.21 和 torch）。原作 repo 的位置用 --orig-repo 給。

比對項目
  1. warp（Dense3DSpatialTransformer）：輸出與梯度，含超出範圍的取樣點
  2. VTN、VoxelMorph、VTNAffineStem 的 forward（影像大小故意用奇數會出現的 96×112×96：112 → … → 7 → 4 → 2）
  3. 整個串接 RecursiveCascadedNetworks.build：總 loss、各顆的 loss、最後的合成位移（VTN 串 2 顆、VoxelMorph 串 2 顆）
  4. free_form_fields（訓練時的隨機形變）：同一組控制點，原作只能跑 X = Y 的大小 → 用 32³ 比
  5. tflearn 初始化的範圍（uniform_scaling：|w| ≤ √(3/fan)、標準差 ≈ √(1/fan)）

用法
  python ASD\\reproduce\\verify_rcn_port.py --orig-repo D:\\MyHome\\MRI\\orig_repos\\Recursive-Cascaded-Networks
"""
import os
import sys
import argparse
import types

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--orig-repo', default=r'D:\MyHome\MRI\orig_repos\Recursive-Cascaded-Networks')
args = ap.parse_args()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf2                       # noqa: E402
import keras                                   # noqa: E402
import torch                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcn_torch as R                          # noqa: E402

for _stream in (sys.stdout, sys.stderr):       # Windows 主控台預設 cp950，印 ✅ 會中斷
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)


class TF:
    """TF 2 有的用 TF 2，只有 TF 1 才有的（random_uniform 等）用 tf.compat.v1。"""
    def __getattr__(self, k):
        return getattr(tf2, k) if hasattr(tf2, k) else getattr(tf2.compat.v1, k)


tf = TF()
CUR = {'weights': None}            # 現在這一顆網路的權重：{scope: (W_tf, b)}


def conv_3d(incoming, nb_filter, filter_size, strides=1, padding='same', activation='linear', bias=True,
            scope=None, reuse=False, weights_init=None, **kw):
    W, b = CUR['weights'][scope]
    y = tf2.nn.conv3d(incoming, W, strides=[1, strides, strides, strides, 1], padding=padding.upper())
    return y + b if bias else y


def conv_3d_transpose(incoming, nb_filter, filter_size, output_shape, strides=1, padding='same', activation='linear',
                      bias=True, scope=None, reuse=False, weights_init=None, **kw):
    W, _ = CUR['weights'][scope]
    out = tf2.stack([tf2.shape(incoming)[0]] + list(output_shape) + [nb_filter])
    return tf2.nn.conv3d_transpose(incoming, W, out, strides=[1, strides, strides, strides, 1], padding=padding.upper())


tflearn = types.SimpleNamespace(
    layers=types.SimpleNamespace(conv_3d=conv_3d, conv=types.SimpleNamespace(conv_3d_transpose=conv_3d_transpose)),
    activations=types.SimpleNamespace(leaky_relu=lambda x, alpha=0.1, name=None: tf2.nn.leaky_relu(x, alpha),
                                      relu=tf2.nn.relu),
    initializations=types.SimpleNamespace(normal=lambda **kw: None))


class Network:
    """原作 network/utils.py 的 Network：這裡只記下名字；__call__ 時把那一顆的權重設成「現在」。"""
    registry = {}

    def __init__(self, name, trainable=True, reuse=None, **kw):
        self._name = name

    def __call__(self, *a, **k):
        CUR['weights'] = Network.registry[self._name]
        return self.build(*a, **k)


def run_original(fname, replace, extra):
    """讀原作的檔案、拿掉它的 import 行，在給定的名稱空間執行。"""
    src = open(os.path.join(args.orig_repo, 'network', fname), encoding='utf-8').read()
    for a in replace:
        assert a in src, (fname, a)
        src = src.replace(a, '')
    ns = {'tf': tf, 'np': np, 'tflearn': tflearn, **extra}
    exec(compile(src, fname, 'exec'), ns)
    return ns


class Layer:
    """keras.layers.core.Layer 的替身：只要 __call__ → call。"""
    def __init__(self, **kw):
        pass

    def __call__(self, inputs):
        return self.call(inputs)


ST = run_original('spatial_transformer.py', ['from keras.layers.core import Layer', 'import tensorflow as tf'],
                  {'Layer': Layer})
BN = run_original('base_networks.py', ['import tensorflow as tf', 'import tflearn',
                                       'from keras.layers.convolutional import UpSampling3D',
                                       'from tflearn.initializations import normal',
                                       'from .utils import Network, ReLU, LeakyReLU'],
                  {'UpSampling3D': keras.layers.UpSampling3D, 'normal': lambda **kw: None, 'Network': Network,
                   'ReLU': lambda t, name=None: tf2.nn.relu(t),
                   'LeakyReLU': lambda t, alpha=0.1, name=None: tf2.nn.leaky_relu(t, alpha)})
TFM = run_original('transform.py', ['from __future__ import division', 'import tensorflow as tf', 'import tflearn',
                                    'import numpy as np'], {})


class TrilinearSampler:
    def __call__(self, inputs):
        return tf2.zeros_like(inputs[1])


RC = run_original('recursive_cascaded_networks.py',
                  ['import tensorflow as tf', 'import tflearn', 'import numpy as np', 'from .utils import Network',
                   'from .base_networks import VTN, VoxelMorph, VTNAffineStem',
                   'from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer',
                   'from .trilinear_sampler import TrilinearSampler'],
                  {'Network': Network, 'VTN': BN['VTN'], 'VoxelMorph': BN['VoxelMorph'], 'VTNAffineStem': BN['VTNAffineStem'],
                   'Dense3DSpatialTransformer': ST['Dense3DSpatialTransformer'], 'Fast3DTransformer': None,
                   'TrilinearSampler': TrilinearSampler})


def to_tf(x):
    """torch [B, C, X, Y, Z] → TF [B, X, Y, Z, C]"""
    return tf2.constant(x.detach().permute(0, 2, 3, 4, 1).numpy())


def to_torch(y):
    return torch.from_numpy(np.asarray(y)).permute(0, 4, 1, 2, 3)


def tf_weights(net):
    """PyTorch 模型 → {scope: (TF 權重, bias)}；scope = 原作的層名（conv1、deconv5、flow…）。"""
    w = {}
    for name, m in net.named_modules():
        scope = name.split('.')[-2] if name.endswith('.conv') else name.split('.')[-1]
        if isinstance(m, torch.nn.ConvTranspose3d):        # torch (cin, cout, k…) → TF [k, k, k, cout, cin]
            w[scope] = (tf2.constant(m.weight.detach().permute(2, 3, 4, 1, 0).numpy()), None)
        elif isinstance(m, torch.nn.Conv3d):               # torch (cout, cin, k…) → TF [k, k, k, cin, cout]
            b = None if m.bias is None else tf2.constant(m.bias.detach().numpy())
            w[scope] = (tf2.constant(m.weight.detach().permute(2, 3, 4, 1, 0).numpy()), b)
    return w


def report(tag, a, b, tol):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    err = float(np.abs(a - b).max())
    scale = float(np.abs(b).max()) + 1e-12
    ok = err <= tol * max(scale, 1.0)
    print('  %-4s %-58s 最大差 %.2e（數值大小 %.2e）' % ('✅' if ok else '❌', tag, err, scale))
    return ok


OK = []
# 1. warp ------------------------------------------------------------------------------
print('1. warp（Dense3DSpatialTransformer）')
im = torch.rand(2, 1, 9, 11, 10)
flow = (torch.rand(2, 3, 9, 11, 10) - 0.5) * 8            # ±4 格，很多點會超出範圍
tr = ST['Dense3DSpatialTransformer']()
with tf2.GradientTape() as tape:
    ti, tfl = to_tf(im), to_tf(flow)
    tape.watch([ti, tfl])
    out_tf = tr([ti, tfl])
    s_tf = tf2.reduce_sum(out_tf * tf2.constant(np.arange(out_tf.shape.num_elements(), dtype=np.float32).reshape(out_tf.shape) % 7))
g_im_tf, g_fl_tf = tape.gradient(s_tf, [ti, tfl])
im_t, fl_t = im.clone().requires_grad_(), flow.clone().requires_grad_()
out_t = R.warp(im_t, fl_t)
s_t = (out_t * torch.from_numpy(np.arange(out_t.numel(), dtype=np.float32).reshape(out_tf.shape) % 7).permute(0, 4, 1, 2, 3)).sum()
s_t.backward()
OK.append(report('輸出', out_t.detach(), to_torch(out_tf), 1e-5))
OK.append(report('對影像的梯度', im_t.grad, to_torch(g_im_tf), 1e-4))
OK.append(report('對位移的梯度', fl_t.grad, to_torch(g_fl_tf), 1e-4))

# 2. 三種網路 ---------------------------------------------------------------------------
SIZE = (96, 112, 96)
print('2. 三種網路的 forward（影像 %s）' % (SIZE,))
img1, img2 = torch.rand(1, 1, *SIZE), torch.rand(1, 1, *SIZE)
for name, net_t, net_tf in [('VTN', R.VTN(flow_multiplier=0.5), BN['VTN']('vtn', flow_multiplier=0.5)),
                            ('VoxelMorph', R.VoxelMorph(flow_multiplier=0.5), BN['VoxelMorph']('vxm', flow_multiplier=0.5)),
                            ('VTNAffineStem', R.VTNAffineStem(SIZE), BN['VTNAffineStem']('aff'))]:
    if name == 'VoxelMorph':                                # 原作初始化 std 1e-5，輸出太小、比不出差別 → 換成大一點的權重
        torch.nn.init.normal_(net_t.flow.conv.weight, 0, 0.05)
    if name == 'VTNAffineStem':
        for c in (net_t.conv7_W, net_t.conv7_b):
            torch.nn.init.normal_(c.weight, 0, 1e-4)
    Network.registry[net_tf._name] = tf_weights(net_t)
    with torch.no_grad():
        r_t = net_t(img1, img2)
    r_tf = net_tf(to_tf(img1), to_tf(img2))
    OK.append(report('%s flow' % name, r_t['flow'], to_torch(r_tf['flow']), 1e-4))
    for k in ('W', 'b', 'det_loss', 'ortho_loss'):
        if k in r_t:
            OK.append(report('%s %s' % (name, k), r_t[k], np.asarray(r_tf[k]), 1e-4))

# 3. 整個串接 ---------------------------------------------------------------------------
print('3. RecursiveCascadedNetworks（affine ＋ 串 2 顆，相似度、平滑、det、ortho、合成位移）')
for base in ('VTN', 'VoxelMorph'):
    torch.manual_seed(1)
    rcn_t = R.RecursiveCascadedNetworks(base, 2, SIZE)
    with torch.no_grad():                                   # 初始化的形變太小 → 放大，合成位移、搬移才比得出差別
        for c in (rcn_t.nets['affine_stem'].conv7_W, rcn_t.nets['affine_stem'].conv7_b):
            torch.nn.init.normal_(c.weight, 0, 2e-4)
        if base == 'VoxelMorph':
            for i in range(2):
                torch.nn.init.normal_(rcn_t.nets['deform_stem_%d' % i].flow.conv.weight, 0, 0.05)
    fw = types.SimpleNamespace(segmentation_class_value=None)
    rcn_tf = RC['RecursiveCascadedNetworks']('framework', fw, base, 2)
    for n in rcn_t.nets:
        Network.registry[n] = tf_weights(rcn_t.nets[n])
    # 影像用平滑的亂數（從 12×14×12 放大）：純雜訊的話，位移的 float32 誤差會被影像梯度放大，比不出真正的差異
    smooth = lambda: torch.nn.functional.interpolate(torch.rand(1, 1, 12, 14, 12), size=SIZE, mode='trilinear')  # noqa: E731
    fixed, moving = smooth(), smooth()
    with torch.no_grad():
        r_t = rcn_t(fixed, moving)
    seg = tf2.zeros([1, *SIZE, 1])
    pt = -tf2.ones([1, 6, 3])
    r_tf = rcn_tf.build(to_tf(fixed), to_tf(moving), seg, seg, pt, pt)
    OK.append(report('%s 總 loss' % base, r_t['loss'], np.asarray(r_tf['loss'])[0], 1e-4))
    for i, st in enumerate(r_t['stems']):
        for k in ('loss', 'raw_loss', 'reg_loss', 'det_loss', 'ortho_loss'):
            if k in st:
                OK.append(report('%s 第 %d 顆 %s' % (base, i, k), st[k], np.asarray(r_tf['%d_%s' % (i, k)]), 1e-4))
    OK.append(report('%s 最後的合成位移' % base, r_t['flow'], to_torch(r_tf['real_flow']), 1e-4))
    # 位移本身有 float32 的誤差（VTN ×20、位移 20 多格，最大差約 1e-3 格）→ 搬好的影像容許 1e-3
    OK.append(report('%s 搬好的 moving（容許 1e-3）' % base, r_t['warped'], to_torch(r_tf['image_reconstruct']), 1e-3))

# 4. 隨機形變 ---------------------------------------------------------------------------
print('4. free_form_fields（同一組控制點；原作最後的 reshape 只在 X = Y 時成立 → 32³）')
cf = (np.random.rand(2, 5, 5, 5, 3).astype(np.float32) - 0.5) * 10
f_tf = TFM['free_form_fields']([32, 32, 32], tf2.constant(cf))
f_t = R.free_form_fields([32, 32, 32], torch.from_numpy(cf))
err_same = float((f_t - to_torch(f_tf)).abs().max())
OK.append(report('原作的輸出 = 我們的輸出把 x、y 兩軸對調', f_t.transpose(2, 3), to_torch(f_tf), 1e-4))
print('       （不對調時差 %.2f：原作最後的 reshape 把前兩個空間軸跟控制點的軸對調了，只有 X = Y 時跑得動。'
      '控制點是 i.i.d. 亂數，對調不改變隨機形變的分佈；我們照數學定義寫，192×224×192 才跑得動）' % err_same)
u = tf2.constant(np.linspace(0, 1, 11, dtype=np.float32))
OK.append(report('get_coef（B-spline 基底）', R.get_coef(torch.linspace(0, 1, 11)), np.asarray(TFM['get_coef'](u)), 1e-6))
fixed_u = np.random.rand(4, 5, 5, 5, 3).astype(np.float32)
TFM['tf'] = types.SimpleNamespace(random_uniform=lambda size, dtype=None: tf2.constant(fixed_u),
                                  abs=tf2.abs, sign=tf2.sign, float32=tf2.float32)
sp_tf = TFM['sample_power'](-0.4, 0.4, 3, fixed_u.shape)
orig_rand = torch.rand
torch.rand = lambda *a, **k: torch.from_numpy(fixed_u)
sp_t = R.sample_power(-0.4, 0.4, 3, fixed_u.shape)
torch.rand = orig_rand
OK.append(report('sample_power（同一組均勻亂數）', sp_t, np.asarray(sp_tf), 1e-6))

# 5. 初始化 -----------------------------------------------------------------------------
print('5. tflearn uniform_scaling 初始化')
c = R.Conv(64, 128, 3, 1)
fan = 27 * 64
w = c.conv.weight.detach()
print('  %-4s Conv 64→128：|w| 最大 %.4f ≤ √(3/fan) %.4f；標準差 %.4f（理論 √(1/fan) %.4f）'
      % ('✅' if w.abs().max() <= np.sqrt(3 / fan) else '❌', w.abs().max(), np.sqrt(3 / fan), w.std(), np.sqrt(1 / fan)))
OK.append(bool(w.abs().max() <= np.sqrt(3 / fan)) and abs(float(w.std()) - np.sqrt(1 / fan)) < 0.05 * np.sqrt(1 / fan))
u = R.UpConv(64, 32)
fan = 64 * 32
w = u.conv.weight.detach()
print('  %-4s UpConv 64→32（fan = k³·cout）：|w| 最大 %.4f ≤ %.4f；標準差 %.4f（理論 %.4f）'
      % ('✅' if w.abs().max() <= np.sqrt(3 / fan) else '❌', w.abs().max(), np.sqrt(3 / fan), w.std(), np.sqrt(1 / fan)))
OK.append(bool(w.abs().max() <= np.sqrt(3 / fan)) and abs(float(w.std()) - np.sqrt(1 / fan)) < 0.05 * np.sqrt(1 / fan))

print()
print('全部通過' if all(OK) else '❌ 有 %d 項沒過' % (len(OK) - sum(OK)))
sys.exit(0 if all(OK) else 1)
