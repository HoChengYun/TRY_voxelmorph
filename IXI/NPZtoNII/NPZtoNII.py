import os
import numpy as np
import nibabel as nib
import argparse

def convert_npz_to_nii(input_path, output_dir=None, affine_template=None):
    """
    將 .npz 檔案中的陣列轉換為 .nii.gz 格式。
    
    :param input_path: .npz 檔案路徑
    :param output_dir: 輸出資料夾路徑，若為 None 則存於原處
    :param affine_template: 參考用的 .nii 檔路徑（用來獲取正確的座標系/Affine）
    """
    
    # 1. 檢查檔案是否存在
    if not os.path.exists(input_path):
        print(f"錯誤：找不到檔案 {input_path}")
        return

    # 2. 設定輸出路徑
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 載入資料
    print(f"正在讀取：{input_path}")
    data = np.load(input_path)
    
    # 4. 處理 Affine 矩陣 (座標系統)
    # 如果有提供模板，就用模板的；沒有的話預設為單位矩陣 (Identity Matrix)
    if affine_template and os.path.exists(affine_template):
        template_img = nib.load(affine_template)
        affine = template_img.affine
        print(f"使用模板 Affine：{affine_template}")
    else:
        affine = np.eye(4)
        print("未提供模板，使用單位矩陣 (Identity Matrix) 作為 Affine。")

    # 5. 遍歷 .npz 內所有陣列
    file_base = os.path.basename(input_path).replace('.npz', '')
    
    for key in data.files:
        array = data[key]
        
        # 過濾掉非數值型態或空陣列
        if not isinstance(array, np.ndarray) or array.size == 0:
            continue
            
        # 建立 NIfTI 物件
        nii_img = nib.Nifti1Image(array, affine)
        
        # 檔名範例：my_data_vol.nii.gz
        output_filename = f"{file_base}_{key}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        
        # 儲存
        nib.save(nii_img, output_path)
        print(f"成功存檔：{output_path} (維度: {array.shape})")

if __name__ == "__main__":
    # --- 你可以在這裡直接修改參數 ---
    INPUT_FILE = "IXI/NPZtoNII/IXI108-Guys-0865-T1.npz"  # 你的 npz 路徑
    OUTPUT_FOLDER = "IXI/NPZtoNII/output_results"      # 輸出資料夾
    TEMPLATE_NII = None                  # 若有原始的 .nii 檔，放路徑可以保留空間座標
    
    convert_npz_to_nii(INPUT_FILE, OUTPUT_FOLDER, TEMPLATE_NII)