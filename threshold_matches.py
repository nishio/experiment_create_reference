"""
閾値に基づくマッチング要素の選択と表示

このスクリプトは以下の機能を提供します：
1. 実験2の結果CSVを読み込む
2. Yの各文に対して0〜2個のXが対応するような閾値を選ぶ
3. 結果を指定された形式で表示する

形式例:
```
y1[i11][i12]
y2[i21]

ref:
i11: X[i11]
...
```
"""

import pandas as pd
import numpy as np
import os

def load_comments(csv_path):
    """
    CSVファイルからコメントを読み込む
    
    Parameters:
        csv_path (str): CSVファイルのパス
        
    Returns:
        list: コメントのリスト
    """
    print(f"Loading comments from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    comments = df['comment'].tolist()
    
    comments = [comment.strip('"') for comment in comments]
    
    print(f"Loaded {len(comments)} comments.")
    return comments

def find_threshold_for_matches(distances, target_count=2):
    """
    指定された数のマッチを得るための閾値を見つける
    
    Parameters:
        distances (list): 距離のリスト
        target_count (int): 目標とするマッチの数（0〜2）
        
    Returns:
        tuple: (閾値, マッチしたインデックスのリスト)
    """
    sorted_indices = np.argsort(distances)
    sorted_distances = [distances[i] for i in sorted_indices]
    
    if target_count is None:
        target_count = np.random.randint(0, 3)
    
    if target_count == 0:
        return -0.1, []
    
    if target_count <= len(sorted_distances):
        threshold = sorted_distances[target_count - 1] + 0.001
        
        matched_indices = [sorted_indices[i] for i in range(target_count)]
        
        return threshold, matched_indices
    
    return 1.0, list(sorted_indices)

def format_results(y_sentences, matched_indices_list, x_comments):
    """
    結果を指定された形式でフォーマットする
    
    Parameters:
        y_sentences (list): Y文のリスト
        matched_indices_list (list): 各Y文に対するマッチしたXのインデックスのリスト
        x_comments (list): Xコメントのリスト
        
    Returns:
        str: フォーマットされた結果
    """
    output = []
    references = []
    
    for i, (sentence, indices) in enumerate(zip(y_sentences, matched_indices_list)):
        full_sentence = sentence
        
        matches = "".join([f"[{idx}]" for idx in indices])
        output.append(f"{full_sentence}{matches}")
        
        for idx in indices:
            comment = x_comments[idx]
            short_comment = comment[:50] + "..." if len(comment) > 50 else comment
            references.append(f"{idx}: {short_comment}")
    
    result = "\n".join(output)
    
    if references:
        result += "\n\nref:\n" + "\n".join(references)
    
    return result

def main():
    """
    メイン関数
    1. 実験2の結果CSVを読み込む
    2. Xコメントを読み込む
    3. 各Y文に対して0〜2個のマッチを選択
    4. 結果を指定された形式で表示
    """
    results_path = "results/experiment2/similarity_results_with_cluster.csv"
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    
    results_df = pd.read_csv(results_path)
    
    x_comments = load_comments("/home/ubuntu/attachments/62c52442-647d-4d4f-8e33-ecb50f16b23d/sample_comments+4.csv")
    
    y_sentences = results_df['sentence'].tolist()
    
    all_matched_indices = []
    
    threshold = 0.14
    print(f"Using threshold: {threshold}")
    
    for i, row in results_df.iterrows():
        distances = [row[f'd{j+1}'] for j in range(5)]
        indices = [int(row[f'i{j+1}']) for j in range(5)]
        
        matched_indices = []
        for j, (idx, dist) in enumerate(zip(indices, distances)):
            if dist <= threshold and len(matched_indices) < 2:
                matched_indices.append(idx)
        
        all_matched_indices.append(matched_indices)
    
    formatted_results = format_results(y_sentences, all_matched_indices, x_comments)
    
    print("\n" + "="*50)
    print("THRESHOLD MATCHING RESULTS")
    print("="*50)
    print(formatted_results)
    print("="*50)
    
    output_path = "results/experiment2/threshold_matches.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_results)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
