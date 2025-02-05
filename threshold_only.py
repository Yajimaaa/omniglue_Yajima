#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image
import pandas as pd

# 特徴点の数でマッチングする関数
def matching_by_number_of_feature(number_of_feature, index, top_number, top_index):
  if top_number < number_of_feature:
    top_number = number_of_feature
    top_index = index
    return number_of_feature, top_index

# 特徴点取得の画像を保存する関数
def save_image(file_name, viz):
    # ディレクトリを作成（存在しない場合のみ）
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # 画像を保存
    plt.imsave(file_name, viz)


def main(argv) -> None:
  """ if len(argv) != 3:
    raise ValueError("Incorrect command line usage - usage: python demo.py <img1_fp> <img2_fp>")
  image0_fp = argv[1]
  image1_fp = argv[2]
  for im_fp in [image0_fp, image1_fp]:
    if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
      raise ValueError(f"Image filepath '{im_fp}' doesn't exist or is not a file.") """

  peg_folder = './object_images/edge/Toy_puzzle/peg'
  hole_folder = './object_images/edge/Toy_puzzle/hole'
  peg_images = sorted(os.listdir(peg_folder))
  hole_images = sorted(os.listdir(hole_folder))
  
  df = pd.DataFrame(columns=["peg_label"] + ["match_hole"])

    
  for i, peg_image in enumerate(peg_images):
    # Load images.
    peg_image = os.path.join(peg_folder, peg_image)
    image0 = np.array(Image.open(peg_image).convert("RGB"))
    object_label_peg = '_'.join(os.path.basename(peg_image).split('_')[:-1])
    
    # 特徴点の数を保存するためのリスト
    matches_list = []
    for j, hole_image in enumerate(hole_images):
      hole_image = os.path.join(hole_folder, hole_image)
      image1 = np.array(Image.open(hole_image).convert("RGB"))
      object_label_hole = '_'.join(os.path.basename(hole_image).split('_')[:-1])
      print("> Loading images...")

      # Load models.
      print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
      start = time.time()
      og = omniglue.OmniGlue(
          og_export="./models/og_export",
          sp_export="./models/sp_v6",
          dino_export="./models/dinov2_vitb14_pretrain.pth",
      )
      print(f"> \tTook {time.time() - start} seconds.")

      # Perform inference.
      print("> Finding matches...")
      start = time.time()
      match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
      num_matches = match_kp0.shape[0]
      print(f"> \tFound {num_matches} matches.")
      print(f"> \tTook {time.time() - start} seconds.")
      

      # Filter by confidence (0.02).
      print("> Filtering matches...")
      match_threshold = 0.02  # Choose any value [0.0, 1.0). 
      keep_idx = []
      for i in range(match_kp0.shape[0]):
        if match_confidences[i] > match_threshold:
          keep_idx.append(i)
      num_filtered_matches = len(keep_idx)
      match_kp0 = match_kp0[keep_idx]
      match_kp1 = match_kp1[keep_idx]
      match_confidences = match_confidences[keep_idx]
      print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")
      
      # 特徴点の数を保存
      matches_list.append(num_filtered_matches)
      # Visualize.
      print("> Visualizing matches...")
      viz = utils.visualize_matches(
          image0,
          image1,
          match_kp0,
          match_kp1,
          np.eye(num_filtered_matches),
          show_keypoints=True,
          highlight_unmatched=True,
          title=f"{num_filtered_matches} matches",
          line_width=2,
      )
      plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
      plt.axis("off")
      save_dir_threshold = f"./result_images/edge/Toy_puzzle/{object_label_peg}_peg/{object_label_hole}_threshold.png"
      save_image(save_dir_threshold, viz)
      
    max_matches = max(matches_list)
  
    max_indices = [k for k, v in enumerate(matches_list) if v == max_matches]
    
    match_hole = [os.path.splitext(hole_images[l])[0] for l in max_indices]
    
    # 1つのセルに縦に並べる
    match_hole_str = [", ".join(match_hole)]
    
    # DataFrame に追加
    new_row = pd.DataFrame([[object_label_peg] + match_hole_str], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
  
    print(f"{object_label_peg}のマッチング終了")
  
  # 画像のサイズを設定
  fig, ax = plt.subplots(figsize=(8, 4))

  # 表を描画
  ax.axis('tight')  # 表の余白を調整
  ax.axis('off')    # 軸を非表示
  table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

  # 画像として保存
  plt.savefig("table_image.png", bbox_inches='tight', dpi=300)
        
        
  # from IPython import embed;embed()

if __name__ == "__main__":
  main(sys.argv)
