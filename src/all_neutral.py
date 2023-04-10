import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="data/session_test.csv")

    args = parser.parse_args([])

    dataset = pd.read_csv(args.test_path)
    dataset.reset_index(inplace=True)

    label_dict = {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6}
    dataset['labels'] = dataset['labels'].apply(lambda x: label_dict[x])

    labels = dataset['labels']
    predict = [1 for _ in range(len(labels))]

    print(f"All-Neutral Macro F1 Score: {f1_score(labels, predict, average='macro'):.4f}")
    print(f"All-Neutral Weighted F1 Score: {f1_score(labels, predict, average='weighted'):.4f}")

    neutral_idx = np.where(labels != 1)
    micro_labels = np.array(labels)[neutral_idx]
    micro_predict = np.array(predict)[neutral_idx]
    print(f"All-Neutral Micro F1 Score: {f1_score(micro_labels, micro_predict, average='micro'):.4f}")
    print(f"All-Neutral Accuracy Score: {accuracy_score(labels, predict):.4f}")