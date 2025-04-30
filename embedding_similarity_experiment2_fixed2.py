"""
文字列リストのembedding、近い要素の検索、CSVの出力、ヒストグラム作成 - 実験2

このスクリプトは以下の機能を提供します：
1. 文字列リストXとYをembeddingする（Yは文章ごとに分割）
2. Yの各要素yについて距離の近い順に5件のXの要素xiを計算
3. [i1, i2, i3, i4, i5]とそれぞれの距離[d1, d2, d3, d4, d5]をhstackして縦|Y| * 横10のCSVを出力
4. d1の分布をヒストグラムにする
5. 適当な閾値ごとにその閾値以下のものを選んだらd1 ~ d5のうちの何件が選ばれるかのヒストグラムを作成

参考: https://github.com/nishio/experiment_embed_aggregation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import re

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def create_embeddings(texts, model_name, batch_size=32):
    """
    テキストリストの埋め込みベクトルを生成する
    
    Parameters:
        texts (list): 埋め込みを生成するテキストのリスト
        model_name (str): 使用するSentenceTransformerモデル名
        batch_size (int): バッチサイズ
        
    Returns:
        numpy.ndarray: 埋め込みベクトルの配列
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    return np.array(embeddings)

def find_closest_elements(x_embeddings, y_embeddings, top_k=5):
    """
    各y要素に対して、最も近いx要素のインデックスと距離を計算する
    
    Parameters:
        x_embeddings (numpy.ndarray): X要素の埋め込みベクトル
        y_embeddings (numpy.ndarray): Y要素の埋め込みベクトル
        top_k (int): 各y要素に対して取得する最近傍要素の数
        
    Returns:
        tuple: (closest_indices, closest_distances)
            - closest_indices: 各yに対する最近傍要素のインデックスのリスト
            - closest_distances: 各yに対する最近傍要素との距離のリスト
    """
    print(f"Finding {top_k} closest elements for each y...")
    closest_indices = []
    closest_distances = []
    
    for y_embedding in tqdm(y_embeddings):
        distances = [cosine(y_embedding, x_embedding) for x_embedding in x_embeddings]
        
        indices = np.argsort(distances)[:top_k]
        
        sorted_distances = [distances[i] for i in indices]
        
        closest_indices.append(indices)
        closest_distances.append(sorted_distances)
    
    return closest_indices, closest_distances

def save_results_to_csv(closest_indices, closest_distances, output_path, y_sentences=None):
    """
    最近傍要素のインデックスと距離をCSVファイルに保存する
    
    Parameters:
        closest_indices (list): 各yに対する最近傍要素のインデックスのリスト
        closest_distances (list): 各yに対する最近傍要素との距離のリスト
        output_path (str): 出力CSVファイルのパス
        y_sentences (list): Y文の元のテキスト（オプション）
        
    Returns:
        pandas.DataFrame: 結果のデータフレーム
    """
    print(f"Saving results to {output_path}...")
    
    results = []
    for indices, distances in zip(closest_indices, closest_distances):
        row = list(indices) + list(distances)
        results.append(row)
    
    index_cols = [f'i{i+1}' for i in range(len(closest_indices[0]))]
    distance_cols = [f'd{i+1}' for i in range(len(closest_distances[0]))]
    columns = index_cols + distance_cols
    
    df = pd.DataFrame(results, columns=columns)
    
    if y_sentences is not None:
        df['sentence'] = y_sentences
    
    df.to_csv(output_path, index=False)
    
    return df

def plot_d1_histogram(closest_distances, output_path):
    """
    最近傍距離(d1)の分布ヒストグラムを作成する
    
    Parameters:
        closest_distances (list): 各yに対する最近傍要素との距離のリスト
        output_path (str): 出力画像ファイルのパス
    """
    print(f"Creating d1 histogram at {output_path}...")
    
    d1_values = [distances[0] for distances in closest_distances]
    
    plt.figure(figsize=(10, 6))
    plt.hist(d1_values, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Distance (d1)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Closest Distance (d1) - Experiment 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_threshold_histograms(closest_distances, output_path_template):
    """
    閾値分析のヒストグラムを作成する
    各閾値に対して、その閾値以下の距離の数の分布を表示
    
    Parameters:
        closest_distances (list): 各yに対する最近傍要素との距離のリスト
        output_path_template (str): 出力画像ファイルのパスのテンプレート
    """
    print("Creating threshold analysis histograms...")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for threshold in thresholds:
        counts = []
        for distances in closest_distances:
            count = sum(1 for d in distances if d <= threshold)
            counts.append(count)
        
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=range(7), alpha=0.7, color='green', rwidth=0.8)
        plt.xlabel('Number of distances below threshold')
        plt.ylabel('Frequency')
        plt.title(f'Number of distances (d1-d5) below threshold {threshold} - Experiment 2')
        plt.xticks(range(6))
        plt.grid(True, alpha=0.3)
        
        output_path = output_path_template.format(threshold=threshold)
        plt.savefig(output_path)
        plt.close()

def load_comments_from_csv(csv_path):
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

def simple_split_japanese_sentences(text):
    """
    日本語テキストを文単位に簡易的に分割する
    
    Parameters:
        text (str): 分割するテキスト
        
    Returns:
        list: 文のリスト
    """
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ["。", "．", "！", "？"]:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences

def load_y_data_and_split_into_sentences(file_path):
    """
    Yデータを読み込み、各クラスタの説明文を文単位に分割する
    
    Parameters:
        file_path (str): Yデータファイルのパス
        
    Returns:
        tuple: (all_sentences, cluster_info)
            - all_sentences: 分割された文のリスト
            - cluster_info: 各文が属するクラスタのタイトルのリスト
    """
    print(f"Loading Y data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    clusters = []
    current_cluster = []
    
    for line in content.split('\n'):
        if line.strip():
            current_cluster.append(line)
        elif current_cluster:  # 空行かつ現在のクラスタが空でない
            clusters.append('\n'.join(current_cluster))
            current_cluster = []
    
    if current_cluster:
        clusters.append('\n'.join(current_cluster))
    
    all_sentences = []
    cluster_info = []
    
    for cluster in clusters:
        if not cluster.strip():
            continue
        
        lines = cluster.strip().split('\n')
        
        title = lines[0] if lines else ""
        print(f"Processing cluster: {title}")
        
        description_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if i == 0:  # タイトル行はスキップ
                continue
            if skip_next:
                skip_next = False
                continue
            if re.match(r'^\d+件$', line.strip()):
                skip_next = True  # 件数行の次の空行もスキップ
                continue
            description_lines.append(line)
        
        description = ' '.join(description_lines)
        
        if description:
            sentences = simple_split_japanese_sentences(description)
            
            print(f"Found {len(sentences)} sentences in cluster '{title}'")
            if sentences:
                print(f"First sentence: {sentences[0][:50]}...")
            
            for sentence in sentences:
                all_sentences.append(sentence)
                cluster_info.append(title)
    
    print(f"Split Y data into {len(all_sentences)} sentences total.")
    return all_sentences, cluster_info

def main():
    """
    メイン関数 - 実験2
    1. XとYのデータを読み込む（Yは文章ごとに分割）
    2. 埋め込みベクトルを生成
    3. 最近傍要素を検索
    4. 結果をCSVに保存
    5. ヒストグラムを作成
    """
    os.makedirs("results/experiment2", exist_ok=True)
    
    X = load_comments_from_csv("/home/ubuntu/attachments/62c52442-647d-4d4f-8e33-ecb50f16b23d/sample_comments+4.csv")
    
    Y_sentences, Y_cluster_info = load_y_data_and_split_into_sentences("/home/ubuntu/y_data.txt")
    
    if not Y_sentences:
        print("Error: No sentences extracted from Y data. Please check the input file and sentence splitting logic.")
        return
    
    x_embeddings = create_embeddings(X, MODEL_NAME)
    y_embeddings = create_embeddings(Y_sentences, MODEL_NAME)
    
    closest_indices, closest_distances = find_closest_elements(x_embeddings, y_embeddings)
    
    df = save_results_to_csv(
        closest_indices, 
        closest_distances, 
        "results/experiment2/similarity_results.csv",
        Y_sentences
    )
    
    df['cluster'] = Y_cluster_info
    df.to_csv("results/experiment2/similarity_results_with_cluster.csv", index=False)
    
    print("CSV output (first 10 rows):")
    print(df.head(10))
    
    plot_d1_histogram(closest_distances, "results/experiment2/d1_histogram.png")
    
    plot_threshold_histograms(closest_distances, "results/experiment2/threshold_{threshold}_histogram.png")
    
    print("Experiment 2 completed successfully!")

if __name__ == "__main__":
    main()
