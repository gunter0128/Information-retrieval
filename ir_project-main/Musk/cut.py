import pandas as pd

# 讀取elon.csv檔案
df = pd.read_csv('elon.csv')

# 設定輸出的資料夾，與elon.csv在同一層
output_dir = './'  # 當前資料夾

# 設定檔案編號
file_counter = 2

# 按照title分組並將text合併
for title, group in df.groupby('title'):
    # 合併同一title的text
    combined_text = "\n".join(group['text'].values)
    
    # 將合併後的text輸出到txt檔案
    with open(f'{output_dir}{file_counter}.txt', 'w') as file:
        file.write(combined_text)
    
    file_counter += 1

print('檔案已成功創建。')
