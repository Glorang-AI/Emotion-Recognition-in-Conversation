import random
import pandas as pd

if __name__ == "__main__":
    random.seed(0)

    df = pd.read_csv('data/data.csv')
    df = df[~df['labels'].str.contains(';')] # Label이 중복 감정 표기일 경우 삭제

    # 각각의 Label을 8:2로 나룰 수 있는 Session을 선택하도록 함.
    # 100번마다 upper_bound, lower_bound를 늘려가면서, 모든 Label의 개수가 그 사이에 들어있을 경우를 선택
    upper_ratio = 0.801
    lower_ratio = 0.799

    idx = 0
    while True:
        
        # Session Shuffle
        session_list = list(df.session.unique())
        random.shuffle(session_list)

        # 앞의 32개의 Session을 Train Dataset Session으로 선택
        train_session = session_list[:32]

        fear_ratio = df[(df.session.isin(train_session)) & (df.labels=="fear")].labels.count() / len(df[df.labels=="fear"])
        disgust_ratio = df[(df.session.isin(train_session)) & (df.labels=="disqust")].labels.count() / len(df[df.labels=="disqust"])
        angry_ratio = df[(df.session.isin(train_session)) & (df.labels=="angry")].labels.count() / len(df[df.labels=="angry"])
        sad_ratio = df[(df.session.isin(train_session)) & (df.labels=="sad")].labels.count() / len(df[df.labels=="sad"])
        surprise_ratio = df[(df.session.isin(train_session)) & (df.labels=="surprise")].labels.count() / len(df[df.labels=="surprise"])
        happy_ratio = df[(df.session.isin(train_session)) & (df.labels=="happy")].labels.count() / len(df[df.labels=="happy"])

        if upper_ratio >= fear_ratio >= lower_ratio and upper_ratio >= disgust_ratio >= lower_ratio and \
              upper_ratio >= angry_ratio >= lower_ratio and upper_ratio >= sad_ratio >= lower_ratio and \
                upper_ratio >= surprise_ratio >= lower_ratio and upper_ratio >= happy_ratio >= lower_ratio:
            break

        idx += 1
        if idx % 100 == 0:
            upper_ratio += 0.001
            lower_ratio -= 0.001


    print("전체 대비 Train Dataset에서 각 Label별 비율")
    for label in df['labels'].unique():
        print(f"{label}의 비율: {df[(df.session.isin(train_session)) & (df.labels==label)].labels.count() / len(df[df.labels==label]): .4f}")

    print()
    print("Train Dataset Label 비율")
    for label in df['labels'].unique():
        print(f"{label}의 비율: {df[(df.session.isin(train_session)) & (df.labels==label)].labels.count() / len(df[df.session.isin(train_session)]): .4f}")

    print()
    print("Test Dataset Label 비율")
    for label in df['labels'].unique():
        print(f"{label}의 비율: {df[(~df.session.isin(train_session)) & (df.labels==label)].labels.count() / len(df[~df.session.isin(train_session)]): .4f}")

    train = df[df.session.isin(train_session)]
    test = df[~df.session.isin(train_session)]

    train.reset_index()[['audio', 'text', 'labels', 'session']].to_csv("data/train.csv")
    test.reset_index()[['audio', 'text', 'labels', 'session']].to_csv("data/test.csv")

    print()
    print("Train Dataset의 개수:", len(train))
    print("Test Dataset의 개수:", len(test))