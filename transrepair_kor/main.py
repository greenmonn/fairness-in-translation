import os
import json
import urllib
from konlpy.tag import Mecab
from googletrans import Translator
import fasttext
import difflib


class Sentence:
    def __init__(self, sentence, src='ko', dest='en'):
        self.mutable = None
        self.sentence = sentence
        self.src = src
        self.dest = dest
        self.translated = self.translate("Google")

    def translate(self, module):
        if module == "Google":
            translator = Translator()
            return translator.translate(self.sentence, dest=self.dest).text

        if module == "Papago":
            client_id = 'U8lk_RxPtD4Lh39Xzy4f'
            client_secret = '7_7tmGaEWi'

            encText = urllib.parse.quote(self.sentence)
            data = f"source={self.src}&target={self.dest}&text=" + encText
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

    def create_mutations(self, model=fasttext.load_model(os.path.join(os.path.dirname(__file__), 'cc.ko.50.bin')),
                         mecab=Mecab()):
        def find_mutant_words():
            words = mecab.pos(self.sentence)
            self.mutable = {'NNG': [], 'VA': [], 'NR': []}

            for word in words:
                for word_type in ['NNG', 'VA', 'NR']:
                    if word[1] == word_type:
                        self.mutable[word_type].append(word[0])

        def get_similar_words(w):
            candidates = model.get_nearest_neighbors(w)
            result = []
            for prob, word in candidates:
                if prob > 0.8:
                    result.append(word)

            return result

        find_mutant_words()
        mutant_dict = {}
        for word_type, word_list in self.mutable.items():
            if word_type == 'NNG' or word_type == 'NR':
                for original_word in word_list:
                    similar_words = get_similar_words(original_word)
                    # discard non-NNG words
                    mutant_dict[original_word][0] = []
                    mutant_dict[original_word][1] = word_type
                    for w_c in similar_words:
                        words = mecab.pos(w_c)
                        if len(words) != 1:
                            continue

                        if words[0][1] != word_type:
                            continue

                        mutant_dict[original_word].append(w_c)

        mutants = []
        for original_word, word_mutants in mutant_dict.items():
            for word_mutant in word_mutants[0]:
                mutants.append(
                    MutantSentence(self.sentence.replace(original_word, word_mutant), self.sentence, original_word,
                                   word_mutant, word_mutants[1]))

        return mutants


class MutantSentence(Sentence):
    def __init__(self, sentence, original, mutant_src, mutant_dest, word_type):
        super().__init__(sentence)
        self.original = original
        self.mutant_src = mutant_src
        self.mutant_dest = mutant_dest
        self.word_type = word_type


def similarity_score(original, mutant, metric):
    original = original.sentence
    mutant = mutant.sentence

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


def consistency_test(original, mutants, threshold):
    def _consistency_test(original, mutant):
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
                    target_sentence = original_word[:i] + original_word[i + 1:]
                    original_target.append(' '.join(target_sentence))

            if block[1] > mutant_index:
                for i in range(mutant_index, block[1]):
                    target_sentence = mutant_word[:i] + mutant_word[i + 1:]
                    mutant_target.append(' '.join(target_sentence))

        max_sim_score = -1
        for original_sentence in original_target:
            for mutant_sentence in mutant_target:
                max_sim_score = max(max_sim_score, similarity_score(original_sentence, mutant_sentence, 'ED'))

        return max_sim_score

    for mutant in mutants:
        score = _consistency_test(original.translated, mutant.translated)

        if score < threshold:
            return False

    return True


def translation_ranking(sentences):
    score = [0] * len(sentences)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_score = similarity_score(sentences[i].translated, sentences[j].translated, 'ED')
                score[i] += sim_score
                score[j] += sim_score

    sentences = [sentence for _, sentence in sorted(zip(score, sentences))]

    return sentences


def repair(original, mutants):
    target_list = translation_ranking([original] + mutants)

    ans = original
    for sentence in target_list:
        if ans == sentence:
            break

        translated_original = None  # word_alignment(original, sentence.mutant_src)
        translated_mutant = None  # word_alignment(sentence, sentence.mutant_dest)

        new_original = Sentence(original.sentence)
        new_original.translated = original.translated.replace(translated_original, translated_mutant)

        if consistency_test(new_original, mutants, 0.7):
            return new_original


if __name__ == '__main__':
    text = Sentence("소녀의 흰 얼굴이, 분홍 스웨터가, 남색 스커트가, 안고 있는 꽃과 함께 범벅이 된다. 모두가 하나의 큰 꽃묶음 같다.")
    # text = Sentence("마지막으로 그는 자신이 아끼는 엽서에 할머니를 위해 그림을 그려줘요.")
    mutants = text.create_mutations()
    print(text.translated)
    if consistency_test(text, mutants, 0.7):
        print("Consistency Error")
        print(repair(text, mutants))
