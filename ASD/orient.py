"""
依 atlas 的 FreeSurfer 標籤判斷影像方向，轉成 RAS 再畫圖。

為什麼要這支
------------
視覺化腳本的切面名稱（sagittal = 軸 0 …）和「矢狀面偏離中線 28 格」都假設陣列是
[左→右, 後→前, 下→上]（RAS）。我們的 MNI152 atlas 本來就是 RAS，一直沒出事；
但作者的 atlas（repo 的 atlas.npz，neurite-oasis 空間）是 LIA，照畫的話三個切面名稱全錯、影像轉 90 度。

npz 沒有檔頭（affine），所以改用解剖結構本身判斷，兩種 atlas 都能用：
    左右：右大腦白質（41）減左大腦白質（2）       —— 分得最開的那一軸
    上下：側腦室（4, 43）減腦幹（16）             —— 剩下兩軸裡分得較開的
    前後：尾狀核（11, 50）減小腦皮質（8, 47）     —— 最後一軸
實測：MNI152 判成 RAS、作者 atlas 判成 LIA，都跟 NIfTI 檔頭一致。

只影響畫圖。Dice、NCC 等數字都在轉正之前算，存 NIfTI 也要用轉正前的陣列（配原本的 affine）。
"""
import numpy as np

_NEED = (2, 41, 4, 43, 16, 11, 50, 8, 47)


def canonical_axes(seg):
    """回傳 (perm, flip)：transpose(perm) 再依 flip 翻轉，就變成 RAS。缺結構時回傳不轉。"""
    seg = np.asarray(seg)
    have = set(np.unique(seg).astype(int).tolist())
    if not set(_NEED) <= have:
        return (0, 1, 2), (False, False, False)

    def c(labs):
        return np.argwhere(np.isin(seg, labs)).mean(axis=0)

    r = c([41]) - c([2])
    s = c([4, 43]) - c([16])
    a = c([11, 50]) - c([8, 47])
    lr = int(np.argmax(np.abs(r)))
    rest = [k for k in range(3) if k != lr]
    si = rest[int(np.argmax(np.abs(s[rest])))]
    ap = [k for k in rest if k != si][0]
    return (lr, ap, si), (bool(r[lr] < 0), bool(a[ap] < 0), bool(s[si] < 0))


def axcode(perm, flip):
    """原本每一軸指向哪裡，例如 'LIA'。"""
    letters = (('R', 'L'), ('A', 'P'), ('S', 'I'))
    code = [''] * 3
    for k, ax in enumerate(perm):
        code[ax] = letters[k][1 if flip[k] else 0]
    return ''.join(code)


def to_ras(vol, perm, flip):
    """純量影像（影像、標籤、Jacobian）轉成 RAS。"""
    v = np.transpose(vol, perm)
    for ax, f in enumerate(flip):
        if f:
            v = np.flip(v, axis=ax)
    return np.ascontiguousarray(v)


def flow_to_ras(flow, perm, flip):
    """形變場 (3, X, Y, Z)：位置跟著轉，分量也要換順序；翻轉的軸位移要變號。"""
    return np.stack([(-1.0 if flip[k] else 1.0) * to_ras(flow[perm[k]], perm, flip)
                     for k in range(3)])
