import os
import json
import urllib
from konlpy.tag import Mecab
from googletrans import Translator
import fasttext
import difflib


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


def similarity_score(original, mutant, metric):
    if metric == 'LCS':
        seq_matcher = difflib.SequenceMatcher()
        seq_matcher.set_seqs(original, mutant)
        return seq_matcher.find_longest_match().size / max(len(original), len(mutant))

    if metric == 'ED':
        if len(original) > len(mutant):
            original, mutant = mutant, original

        distances = range(len(original) + 1)
        for i2, c2 in enumerate(mutant):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(original):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_

        return 1 - distances[-1] / max(len(original), len(mutant))

    if metric == 'tf-idf':
        pass
    if metric == 'BLEU':
        pass
    return 0


def consistency_test(original, mutant):
    original_word = original.split(' ')
    mutant_word = mutant.split(' ')

    sequence_matcher = difflib.SequenceMatcher()
    sequence_matcher.set_seqs(original_word, mutant_word)

    matching_blocks = sequence_matcher.get_matching_blocks()

    original_target = []
    mutant_target = []
    original_index = 0
    mutant_index = 0
    for block in matching_blocks:
        if block[2] == 0:
            break

        if block[0] > original_index:
            for i in range(original_index, block[0]):
                target_sentence = original_word[:i] + original_word[i+1:]
                original_target.append(' '.join(target_sentence))

        if block[1] > mutant_index:
            for i in range(mutant_index, block[1]):
                target_sentence = mutant_word[:i] + mutant_word[i+1:]
                mutant_target.append(' '.join(target_sentence))

    max_sim_score = -1
    for original_sentence in original_target:
        for mutant_sentence in mutant_target:
            max_sim_score = max(max_sim_score, similarity_score(original_sentence, mutant_sentence, 'ED'))

    return max_sim_score


if __name__ == '__main__':
    print(consistency_test('A B C D F', 'B B C G H F'))
    # text = "소녀의 흰 얼굴이, 분홍 스웨터가, 남색 스커트가, 안고 있는 꽃과 함께 범벅이 된다. 모두가 하나의 큰 꽃묶음 같다."
    # text = "마지막으로 그는 자신이 아끼는 엽서에 할머니를 위해 그림을 그려줘요."
    # print(translate(text, "Google"))
    # print(translate(text, "Papago"))
    # words_dict = find_korean_words(text)
    #
    # model = get_fasttext_model()
    # mut_dict = {}
    # mecab = Mecab()
    #
    # print(create_mutations(text, model=model, mecab=mecab))

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
