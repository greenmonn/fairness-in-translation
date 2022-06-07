import difflib
import json
import os
import time
import urllib
import mmh3
import traceback
import editdistance

import fasttext
from googletrans import Translator
# from konlpy.tag import Mecab
from konlpy.tag import Kkma
from simalign import SentenceAligner
from httpx import HTTPError

from functools import cached_property

# SETTINGS
FASTTEXT_MODEL = fasttext.load_model(os.path.join(os.path.dirname(__file__), 'cc.ko.50.bin'))
# KONLPY_MODEL = Mecab()
KONLPY_MODEL = Kkma()
WORD_ALIGNER = SentenceAligner()
TRANSLATION_MODEL = "Google"
SIMILARITY_METRIC = "ED"
SIMILARITY_TEST_THRESHOLD = 0.7
SEQUENCE_MATCHER = difflib.SequenceMatcher()


def print_debug(log, debug=False):
    if debug:
        print(log)


class RemoteTranslator:
    translator = None
    def __init__(self, provider):
        self.provider = provider
        if provider == "Google":
            if RemoteTranslator.translator is None:
                RemoteTranslator.translator = Translator()
            self.translator = RemoteTranslator.translator

        elif provider == "Papago":
            self.client_id = 'U8lk_RxPtD4Lh39Xzy4f'
            self.client_secret = '7_7tmGaEWi'
            self.url = "https://openapi.naver.com/v1/papago/n2mt"

    def translate(self, sentence, src, dest, max_try=5):
        if self.provider == "Google":
            for _ in range(max_try):
                try:
                    return self.translator.translate(sentence, dest=dest).text
                except Exception as exc:
                    traceback.print_exc()
                    time.sleep(5)

            traceback.print_exc()
            raise Exception(f'[Google Translate Error]')

        elif self.provider == "Papago":
            enc_text = urllib.parse.quote(sentence)
            data = f"source={src}&target={dest}&text=" + enc_text
            request = urllib.request.Request(self.url)
            request.add_header("X-Naver-Client-Id", self.client_id)
            request.add_header("X-Naver-Client-Secret", self.client_secret)
            response = urllib.request.urlopen(request, data=data.encode("utf-8"))
            res_code = response.getcode()

            if res_code == 200:
                result = json.loads(response.read())
                return result['message']['result']['translatedText']
            else:
                return "Error Code:" + res_code


class Sentence:
    def __init__(self, sentence, src='ko', dest='en', translator=RemoteTranslator('Google'), debug=False):
        self.sentence = sentence
        self.is_repaired = False
        start_time = time.time()
        self.translated = translator.translate(sentence, src, dest)
        after_translate_time = time.time()
        print_debug(f'Translation time: {after_translate_time - start_time}', debug=debug)
        self.words = KONLPY_MODEL.pos(self.sentence)
        konlpy_time = time.time()
        print_debug(f'Konlpy time: {konlpy_time - after_translate_time}', debug=debug)
        tokens = [t[0] for t in self.words]
        self.aligned = WORD_ALIGNER.get_word_aligns(' '.join(tokens), self.translated)['mwmf']
        print_debug(f'simalign time: {time.time() - konlpy_time}', debug=debug)

    def __str__(self):
        return f'{self.sentence} -> {self.translated}'

    def __repr__(self):
        return self.__str__()

    @cached_property
    def hash(self):
        return mmh3.hash(self.sentence)

    @cached_property
    def mutable_words(self):
        mutable_words = {'NNG': [], 'VA': [], 'NR': []}

        for i, word in enumerate(self.words):
            for word_type in ['NNG', 'VA', 'NR']:
                if word[1] == word_type:
                    mutable_words[word_type].append((word[0], i))

        return mutable_words

    def repair(self, mutant):
        mutant_translation, src_aligned, dest_aligned = mutant.translated, mutant.mutant_dest_alignment, mutant.mutant_src_alignment

        self.is_repaired = True
        self.original_translated = self.translated 
        self.applied_mutant = mutant
        self.mutant_translated = mutant_translation
        
        repaired_translation = mutant_translation.replace(src_aligned, dest_aligned)
        self.translated = repaired_translation
        self.src_aligned = src_aligned
        self.dest_aligned = dest_aligned

    def create_mutations(self, word_embedding_model=FASTTEXT_MODEL, konlpy_model=KONLPY_MODEL, debug=False):
        mutant_dict = {}
        for word_type, word_list in self.mutable_words.items():
            if word_type == 'NNG' or word_type == 'NR':
                for original_word, original_word_index in word_list:
                    similar_words = get_similar_words(original_word, word_embedding_model=word_embedding_model)
                    # discard non-NNG words
                    print_debug(f'Candidates: {original_word} -> {similar_words}', debug=debug)
                    mutant_dict[original_word] = ([], word_type, original_word_index)
                    for candidate_word in similar_words:
                        words = konlpy_model.pos(candidate_word)
                        if len(words) != 1:
                            continue

                        if words[0][1] != word_type:
                            continue

                        mutant_dict[original_word][0].append(candidate_word)

        mutants = []
        for original_word, word_mutants in mutant_dict.items():
            for word_mutant in word_mutants[0]:
                mutants.append(
                    MutantSentence(self.sentence.replace(original_word, word_mutant), self, original_word,
                                   word_mutants[2], word_mutant, word_mutants[1]))

        return mutants


class MutantSentence(Sentence):
    def __init__(self, sentence, original, mutant_src, mutant_index, mutant_dest, word_type):
        super().__init__(sentence)
        self.original = original
        self.mutant_src = mutant_src
        self.mutant_index = mutant_index
        self.mutant_dest = mutant_dest
        self.word_type = word_type

        self.mutant_src_alignment = ''
        for item in self.original.aligned:
            if item[0] == self.mutant_index:
                if self.mutant_src_alignment != '':
                    self.mutant_src_alignment += ' '
                self.mutant_src_alignment += self.original.translated.split(' ')[item[1]]

        self.mutant_dest_alignment = ''
        for item in self.aligned:
            if item[0] == self.mutant_index:
                if self.mutant_dest_alignment != '':
                    self.mutant_dest_alignment += ' '
                self.mutant_dest_alignment += self.translated.split(' ')[item[1]]


def get_similar_words(w, word_embedding_model=FASTTEXT_MODEL, max_k=5):
    candidates = word_embedding_model.get_nearest_neighbors(w)
    result = []
    for prob, word in candidates:
        if prob > 0.8:
            result.append(word)

    if len(result) > max_k:
        result = result[:max_k]

    return result


def similarity_score(original, mutant, metric):
    if metric == 'LCS':
        SEQUENCE_MATCHER.set_seqs(original, mutant)
        return SEQUENCE_MATCHER.find_longest_match().size / max(len(original), len(mutant))

    if metric == 'ED':
        max_len = max(len(original), len(mutant))
        if max_len == 0:
            return 0
        return 1 - (editdistance.eval(original, mutant) / max(len(original), len(mutant)))

    if metric == 'tf-idf':
        pass

    if metric == 'BLEU':
        pass

    return 0


def calc_consistency_score(original, mutant):
    original_word = original.split(' ')
    mutant_word = mutant.split(' ')

    SEQUENCE_MATCHER.set_seqs(original_word, mutant_word)

    matching_blocks = SEQUENCE_MATCHER.get_matching_blocks()

    subsequences_original = []
    subsequences_mutant = []
    original_index = 0
    mutant_index = 0
    for block in matching_blocks:
        if block[2] == 0:
            break

        if block[0] > original_index:
            subsequences_original.append(original_word[:original_index] + original_word[block[0]:])

        if block[1] > mutant_index:
            subsequences_mutant.append(mutant_word[:mutant_index] + mutant_word[block[1]:])

        original_index = block[0] + block[2]
        mutant_index = block[1] + block[2]

    # edge case: when the last block is not matched one
    if original_index < len(original_word):
        subsequences_original.append(original_word[:original_index])

    if mutant_index < len(mutant_word):
        subsequences_mutant.append(mutant_word[:mutant_index])

    if len(subsequences_mutant) == 0 or len(subsequences_original) == 0:
        return 1

    max_sim_score = -1
    for original_sentence in subsequences_original:
        for mutant_sentence in subsequences_mutant:
            max_sim_score = max(max_sim_score,
                                similarity_score(original_sentence, mutant_sentence, SIMILARITY_METRIC))

    return max_sim_score


def test_consistency(original, mutants, threshold=SIMILARITY_TEST_THRESHOLD, debug=False):
    for mutant in mutants:
        score = calc_consistency_score(original.translated, mutant.translated)

        if score < threshold:
            if debug:
                print_debug(f'Inconsistent mutant: {mutant} [{score}]', debug=debug)
            return False, (mutant, score)

    return True, (None, -1.0)


def translation_ranking(sentences):
    score = [0] * len(sentences)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_score = similarity_score(sentences[i].translated, sentences[j].translated, SIMILARITY_METRIC)
                score[i] += sim_score
                score[j] += sim_score

    sentences = [sentence for _, sentence in sorted(zip(score, sentences), key=lambda x: x[0])]

    return sentences


def repair(original, mutants, debug=False):
    target_list = translation_ranking([original] + mutants)

    ans = original
    for sentence in target_list:
        if ans == sentence:
            break

        if debug:
            print_debug(
                f'Mutant Candidate: {sentence}, ({sentence.mutant_src_alignment} -> {sentence.mutant_dest_alignment})',
                debug=debug)

        # if len(translated_original) == len(translated_mutant):

        new_original = Sentence(original.sentence)
        new_original.repair(sentence)

        if test_consistency(new_original, mutants, debug=debug):
            return new_original

    return ans


if __name__ == '__main__':
    targets = ["6년 전 저는 사랑하는 남자와 4년 연애 후 결혼을 했습니다.",
               "괴물은 자신의 틀에 딸을 가두려는 마녀로부터 딸을 꺼내 주는 역할입니다.",
               "그 남자가 와서는 안 됩니다.",
               "그 남자는 부상을 면했어요.",
               "그 남자는 알겠다고 했습니다.",
               "그 남자는 카메라를 목에 걸고 있고 손에는 지도를 들고 있어요.",
               "그 사람도 참석할 수 있어요.",
               "그 여자가 오늘 당신에게 전화했나요?"]

    for target in targets:
        text = Sentence(target)
        mutants = text.create_mutations()
        print(text.translated)
        if not test_consistency(text, mutants, debug=True):
            print("Consistency Error")
            print(repair(text, mutants).translated)
