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

def extract_sentences_from_text(text):
    """
    テキストから日本語の文を抽出する
    
    Parameters:
        text (str): 抽出元のテキスト
        
    Returns:
        list: 抽出された文のリスト
    """
    text = text.strip()
    
    pattern = r'[^。．！？]+[。．！？]'
    
    sentences = re.findall(pattern, text)
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    last_part = re.sub(r'.*[。．！？]', '', text).strip()
    if last_part:
        sentences.append(last_part)
    
    return sentences

def parse_y_data(file_path):
    """
    Yデータファイルを解析し、クラスタとその説明文を抽出する
    
    Parameters:
        file_path (str): Yデータファイルのパス
        
    Returns:
        list: クラスタのリスト。各クラスタは(タイトル, 説明文)のタプル
    """
    print(f"Parsing Y data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = re.split(r'\n\s*\n', content)
    
    clusters = []
    current_title = None
    current_description = None
    
    for i, block in enumerate(blocks):
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        
        if len(lines) == 1 and re.match(r'^\d+件$', lines[0]):
            if i > 0 and i+1 < len(blocks):
                title = blocks[i-1].strip()
                description = blocks[i+1].strip()
                clusters.append((title, description))
    
    print(f"Found {len(clusters)} clusters in Y data.")
    return clusters

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
    
    clusters = [
        ("AI技術による環境保護と災害対策の革新", 
         "このクラスタは、AI技術が環境保護や持続可能な開発、さらには自然災害の予測と対応において果たす重要な役割に関する意見を集約しています。参加者は、AIがエネルギー管理の最適化や環境モニタリングの精度向上、災害時の迅速な対応を可能にすることで、持続可能な社会の構築や災害リスクの軽減に寄与することを期待しています。特に、AIの技術革新が環境問題解決に向けた新たな手法を提供することへの前向きな姿勢が示されています。"),
        
        ("AI技術による業務革新とデータ活用の最前線", 
         "このクラスタは、AI技術が企業の業務効率化、サービス向上、マーケティング戦略の最適化、そしてデータ解析を通じた意思決定支援において果たす重要な役割を集約しています。参加者は、AIの導入が業務プロセスの最適化やパーソナライズドサービスの提供を実現し、ビッグデータ解析や金融市場におけるリスク管理の革新を促進することに期待を寄せています。また、AIの進化が企業の競争戦略や国際ビジネスに与える影響についても言及されており、AIが新たなビジネスモデルの形成に寄与する可能性が示唆されています。"),
        
        ("AI技術の倫理、透明性と社会的責任の統合", 
         "このクラスタは、AI技術の発展に伴う倫理的な問題、透明性の確保、社会的責任に関する意見を集約しています。参加者は、AIの判断基準の透明性や不正利用への対策、法整備とプライバシー保護の重要性を強調し、AIの利用がもたらす効率化と新たな課題についても考慮しています。また、AIアルゴリズムの社会的公平性や外部監査の必要性、倫理的な教育や啓蒙活動の重要性が指摘されており、AIが社会問題解決に向けた革新的なアプローチを提供することへの期待も表明されています。"),
        
        ("AI技術による医療と日常生活の革新", 
         "このクラスタは、AI技術が医療分野における革新や予防医学の推進、さらには日常生活における利便性の向上に寄与する意見を集約しています。特に、AIが健康管理のパーソナライズ化やリモート医療の発展を促進し、医療診断の精度向上や介護支援において重要な役割を果たすことが期待されています。また、AIは日常生活においても音声認識技術を通じて便利さを提供し、リモートワーク環境の効率化や生産性向上にも寄与することが強調されています。"),
        
        ("AI技術による都市生活の安全性と効率性向上", 
         "このクラスタは、AI技術が都市生活の質を向上させるための多様な役割に関する意見を集約しています。自動運転技術や公共交通システムの効率化、スマートシティの実現に向けたAIの重要性が強調されており、AIが交通安全や公共サービスの質向上、防犯対策に寄与することへの期待が表れています。参加者は、AI技術が都市のインフラや生活環境を革新し、より安全で効率的な社会を実現するための鍵となると認識しています。"),
        
        ("AIによる創造的な学びと研究の革新", 
         "このクラスタは、AI技術が教育、科学研究、創造性の向上において果たす重要な役割を集約しています。AIは個別最適化学習を実現し、教育現場での学習効果を向上させるだけでなく、研究開発のプロセスを加速し、新たな発見を促進する力を持っています。また、AIはエンターテイメントや文化・芸術の分野においても革新をもたらし、個別化された体験や新しい表現手法を提供することが期待されています。"),
        
        ("AI技術による産業革新と効率化の未来", 
         "このクラスタは、AI技術が製造業、物流、農業などの産業界において効率化や生産性向上を促進し、ビジネスプロセスの変革をもたらす可能性についての意見を集約しています。参加者は、AIの導入が業界全体の競争力を高め、デジタルトランスフォーメーションを加速させることを期待しており、AI技術が新たな市場や雇用を創出する力を持つと認識しています。また、教育やスキルの再構築が必要であるとの意見もあり、AI技術の進展が社会全体に与える影響についての多角的な視点が示されています。")
    ]
    
    all_sentences = []
    cluster_info = []
    
    for title, description in clusters:
        print(f"Processing cluster: {title}")
        sentences = extract_sentences_from_text(description)
        print(f"Found {len(sentences)} sentences")
        
        if sentences:
            print(f"First sentence: {sentences[0][:50]}...")
        
        for sentence in sentences:
            all_sentences.append(sentence)
            cluster_info.append(title)
    
    print(f"Split Y data into {len(all_sentences)} sentences total.")
    
    if not all_sentences:
        print("Error: No sentences extracted from Y data. Please check the input file and sentence splitting logic.")
        return
    
    x_embeddings = create_embeddings(X, MODEL_NAME)
    y_embeddings = create_embeddings(all_sentences, MODEL_NAME)
    
    closest_indices, closest_distances = find_closest_elements(x_embeddings, y_embeddings)
    
    df = save_results_to_csv(
        closest_indices, 
        closest_distances, 
        "results/experiment2/similarity_results.csv",
        all_sentences
    )
    
    df['cluster'] = cluster_info
    df.to_csv("results/experiment2/similarity_results_with_cluster.csv", index=False)
    
    print("CSV output (first 10 rows):")
    print(df.head(10))
    
    plot_d1_histogram(closest_distances, "results/experiment2/d1_histogram.png")
    
    plot_threshold_histograms(closest_distances, "results/experiment2/threshold_{threshold}_histogram.png")
    
    print("Experiment 2 completed successfully!")

if __name__ == "__main__":
    main()
