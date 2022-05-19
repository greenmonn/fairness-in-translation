import os
import json
import urllib
from konlpy.tag import Mecab
from googletrans import Translator
import fasttext


def find_korean_words(text):
    mecab = Mecab()
    words = mecab.pos(text)
    ans = {'NNG': [], 'VA': [], 'NR': []}

    for word in words:
        for word_type in ['NNG', 'VA', 'NR']:
            if word[1] == word_type:
                ans[word_type].append(word[0])

    return ans


def get_fasttext_model():
    model = fasttext.load_model(os.path.join(os.path.dirname(__file__), 'cc.ko.50.bin'))
    return model


def get_similar_words(w, model):
    candidates = model.get_nearest_neighbors(w)
    result = []
    for prob, word in candidates:
        if prob > 0.8:
            result.append(word)

    return result

def translate(target, module, src='ko', dest='en'):
    if module == "Google":
        translator = Translator()
        return translator.translate(target, dest=dest).text

    if module == "Papago":
        client_id = 'U8lk_RxPtD4Lh39Xzy4f'
        client_secret = '7_7tmGaEWi'

        encText = urllib.parse.quote(target)
        data = f"source={src}&target={dest}&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        res_code = response.getcode()

        if res_code == 200:
            result = json.loads(response.read())
            return result['message']['result']['translatedText']
        else:
            return "Error Code:" + res_code

    return "Error"


def create_mutations(text, model=get_fasttext_model(), mecab=Mecab()):
    words_dict = find_korean_words(text)
    mut_dict = {}
    for t, w_list in words_dict.items():
        if t == 'NNG' or t == 'NR':
            for w in w_list:
                similar_words = get_similar_words(w, model)
                # discard non-NNG words
                mut_dict[w] = []
                for w_c in similar_words:
                    words = mecab.pos(w_c)
                    if len(words) != 1:
                        continue

                    if words[0][1] != t:
                        continue

                    mut_dict[w].append(w_c)

    mutated_sentences = []
    for w, w_m_list in mut_dict.items():
        for w_m in w_m_list:
            mutated_sentences.append(text.replace(w, w_m))
    
    return mutated_sentences


if __name__ == '__main__':
    # text = "소녀의 흰 얼굴이, 분홍 스웨터가, 남색 스커트가, 안고 있는 꽃과 함께 범벅이 된다. 모두가 하나의 큰 꽃묶음 같다."
    text = "마지막으로 그는 자신이 아끼는 엽서에 할머니를 위해 그림을 그려줘요."
    print(translate(text, "Google"))
    print(translate(text, "Papago"))
    words_dict = find_korean_words(text)

    model = get_fasttext_model()
    mut_dict = {}
    mecab = Mecab()

    print(create_mutations(text, model=model, mecab=mecab))

    # for t, w_list in words_dict.items():
    #     if t == 'NNG' or t == 'NR':
    #         for w in w_list:
    #             similar_words = get_similar_words(w, model)
    #             # discard non-NNG words
    #             mut_dict[w] = []
    #             for w_c in similar_words:
    #                 words = mecab.pos(w_c)
    #                 if len(words) != 1:
    #                     continue

    #                 if words[0][1] != t:
    #                     continue

    #                 mut_dict[w].append(w_c)

    # print(mut_dict)
    # mutated_sentences = []
    # for w, w_m_list in mut_dict.items():
    #     for w_m in w_m_list:
    #         mutated_sentences.append(text.replace(w, w_m))

    # print(mutated_sentences)




