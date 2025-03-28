import pandas as pd


def calculate_composite_scores(input_csv, target_class, alpha=0.5, output_csv="composite_scores.csv"):

    # 读取数据
    df = pd.read_csv(input_csv)

    # 验证目标类别存在
    if target_class not in df['Class'].unique():
        raise ValueError(f"Target class '{target_class}' not found in dataset classes")

    # 准备结果存储
    results = []

    # 按层和通道分组处理
    for (layer, channel), group in df.groupby(['Layer', 'Channel']):
        # 获取目标类别相似度
        target_sim = group[group['Class'] == target_class]['Similarity']

        # 处理未找到目标类别的情况
        if len(target_sim) == 0:
            print(f"Warning: No target class data for Layer={layer}, Channel={channel}")
            continue

        target_score = target_sim.values[0]

        # 计算其他类别平均相似度
        other_sim = group[group['Class'] != target_class]['Similarity']
        other_avg = other_sim.mean() if len(other_sim) > 0 else 0.0

        # 计算综合得分
        composite_score = target_score - alpha * other_avg

        # 存储结果
        results.append({
            'Layer': layer,
            'Channel': channel,
            'Target_Class': target_class,
            'Target_Similarity': target_score,
            'Other_Avg_Similarity': other_avg,
            'Composite_Score': composite_score
        })

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 按综合得分排序
    result_df = result_df.sort_values(['Layer', 'Composite_Score'], ascending=[True, False])

    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return result_df


if __name__ == "__main__":
    # 使用示例
    calculate_composite_scores(
        input_csv="channel_similarities.csv",
        target_class="cup",  # 替换为你的目标类别
        alpha=0.5,  # 调节平衡参数
        output_csv="channel_composite_scores.csv"
    )