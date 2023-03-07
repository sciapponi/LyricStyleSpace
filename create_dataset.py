import re
import os 
import glob
import pandas as pd

def clean_and_join(path1,path2):
    path2_cleaned = re.sub('[*?^:\|\\\/ ]','-', path2)
    return os.path.join(path1,path2_cleaned) 
    
skip = ['(Scriptonite).csv','Booba.csv','BTS.csv','Damso.csv','JuL.csv','Nekfeu.csv','Genius English Translations.csv','Genius Romanizations.csv','Oxxxymiron.csv'] 
data = []
for filename in glob.glob('*.csv'):
    if filename not in skip: 
        df = pd.read_csv(filename,dtype={'title':str,'lyrics':str})[['title','lyrics']].dropna()
        for i,row in df.iterrows():
            data.append((filename[:-4],row['lyrics']))
pd.DataFrame(data,columns=['artist','lyrics']).to_json('data.json',orient='records')