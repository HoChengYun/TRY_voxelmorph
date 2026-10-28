# -*- coding: utf-8 -*-
"""簡報用的三個架構修改模型：讀同一夾的 deck_models.json（換模型只改那個檔，見 換模型.md）。

  XC, EC：cascade 的實驗名、epoch
  XP, EP：coarse-to-fine
  XM, EM：兩者合併
  LEGACY：10-06 版的三顆＋mix_pyramid_500。延長訓練、腦外 folding 兩頁（build.js 已關）講的是它們，換模型後照樣讀得到
"""
import os
import json

_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deck_models.json'), encoding='utf-8'))
ARCH = {k: (_cfg[k]['exp'], _cfg[k]['epoch']) for k in ('cascade', 'c2f', 'merged')}
(XC, EC), (XP, EP), (XM, EM) = ARCH['cascade'], ARCH['c2f'], ARCH['merged']
LEGACY = [('mix_cascade', '0170'), ('mix_pyramid', '0245'), ('mix_cascade_pyramid', '0140'), ('mix_pyramid_500', '0335')]
