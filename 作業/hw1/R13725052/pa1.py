with open('./1.txt', 'r') as file:
    document = file.read()

# 補上標點符號判斷
punctuations = '.,!?;:\'"'

# 手動移除標點符號
document_no_punct = ''.join([char for char in document if char not in punctuations])

# Tokenization（將句子分割為詞語）
tokens = document_no_punct.split()

# 轉換為小寫字母
tokens = [token.lower() for token in tokens]

# 簡易stopword清單
stopwords = {'the', 'is', 'in', 'on', 'a', 
             'to', 'and', 'of', 'for', 'it', 
             'this', 'that', 'but',
             'with', 'will', 'were', 'are', 'be' 
             ',', '.', '?', '!','\''} #本來要在這裡分辨stopwords 但後來發現應該在split前判斷 否則標點符號會被和單字視為一體

# 去除停用詞
filtered_tokens = [word for word in tokens if word not in stopwords]

# stemming實作
def simple_stemmer(word):
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ied'):
        return word[:-3]+'y' # 這個判斷式要放在下一個前方
    elif word.endswith('ed'):
        return word[:-1]
    elif word.endswith('ies'):
        return word[:-3]+'y'
    elif word.endswith('s') and len(word) > 1:
        return word[:-1]
    elif word.endswith('ly'):
        return word[:-2]
    elif word.endswith('er'):
        return word[:-2]
    elif word.endswith('ness'):
        return word[:-4]
    return word

# 使用集合來保證唯一性 (符合term的定義)
stemmed_tokens = {simple_stemmer(token) for token in filtered_tokens}  

# 將詞語按照字母順序排序
sorted_terms = sorted(stemmed_tokens)

# 按照首字母進行分組
grouped_terms = {}
for term in sorted_terms:
    first_letter = term[0]  # 取得首字母
    if first_letter not in grouped_terms:
        grouped_terms[first_letter] = []
    grouped_terms[first_letter].append(term)

# 輸出結果到 txt 檔案
with open('result.txt', 'w') as result_file:
    for letter in sorted(grouped_terms):  # 按照字母順序輸出
        terms = ', '.join(grouped_terms[letter])
        result_file.write(f"{letter}: {terms}\n")

print("儲存成功")