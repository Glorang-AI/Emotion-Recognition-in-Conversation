import os
import glob
import re
from tqdm import tqdm
import pandas as pd

def clena_text(txt):
    txt = re.sub(r'[cnNulb*+/]/', '', txt) # 특수기호 제거
    txt = re.sub(r'\([^)]*\)', '', txt) # 괄호 속 제거 
    txt= re.sub(' +', ' ', txt).strip() # 연속 공백 제거
    return txt 

def read_text(path):
    with open (path, 'r', encoding='cp949') as text:
        string = text.readline()
    return string.replace('\n', '')


def annotation_to_csv(path):
    """
        annotation 폴더에서 data.csv로 각 정보를 종합하여 생성

        Return
            save 'data.csv'
            | audio | text | labels | session |
            | wav/Session01/Sess01_script01_User001F_001.wav | 아 친구들도? | neutral | Sess01
            ...
    """
    ann = os.path.join(path, 'annotation/*.csv')
    ann_list = glob.glob(ann)

    evaluation_list = []
    segment_list = []
    for ann_path in ann_list:
        ann_df = pd.read_csv(ann_path, encoding = 'utf-8')
        evaluation_list.append(ann_df['Total Evaluation'].to_list()[1:])
        segment_list.append(ann_df['Segment ID'].to_list()[1:])

    evaluation_list = sum(evaluation_list, [])
    segment_list = sum(segment_list, [])

    wav_list = []
    text_list = []
    session_list = []
    for segment in tqdm(segment_list):
        session = segment.split('_')[0].replace('Sess', 'Session')
        text = read_text(os.path.join(path,'wav', session, segment+'.txt')) # text 
        text = clena_text(text)
        wav = os.path.join('wav', session, segment+'.wav') # wav file
        
        session_list.append(segment.split('_')[0])
        text_list.append(text)
        wav_list.append(wav)

    data = {
        'audio':wav_list,
        'text':text_list,
        'labels': evaluation_list,
        'session':session_list
    }
    data = pd.DataFrame(data)
    data = data.sort_values(by='audio').reset_index()[['audio','text','labels','session']]

    data.to_csv(os.path.join(path, "data.csv"))

if __name__ == "__main__":
    annotation_to_csv('./data')
    print('done..!')