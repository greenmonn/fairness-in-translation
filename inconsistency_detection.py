from transrepair_kor.main import RemoteTranslator, calc_consistency_score
from tqdm import tqdm
import pandas as pd
import json
import mmh3


def trans_sentence_by_sa(s, sensitive_attribute='race'):
    target_word = None
    for sa_keyword in group_keywords[sensitive_attribute]:
        if sa_keyword in s:
            target_word = sa_keyword
            break
    if target_word is None:
        return []

    return [s.replace(sa_keyword, other_group) for other_group in sa_mutation_map[sa_keyword]]

def print_debug(log, debug=False):
    if debug:
        print(log)

def test_consistency(original, mutants, threshold=0.8):
    result = []
    min_score = 1.0
    for mutant in mutants:
        score = calc_consistency_score(original, mutant)
        result.append(score)

        if score < min_score:
            min_score = score

    return min_score > threshold, result

if __name__ == "__main__":
    keyword_groups = {}
    keyword_groups['gender'] = [["he", "she"], ["him","her"], ["his", "her"], ["male", "female"], ["men", "women"], ["man", "woman"], ["boy", "girl"]]
    keyword_groups['race'] = [["asian", "african", "american", "caucasian", "chinese", "europian", "indian", "korean", "japanese"]]
    keyword_groups['religion'] = [["confucianism", "taoism", "buddhism", "hinduism", "islam", "islamism", "christianity", "catholicism", "judaism"], ["confucianist", "taoist", "buddhist", "hinduist", "islamist", "islamic", "christian", "catholic", "jewish", "jews"]]

    group_keywords = {}
    for sensitive_attr in keyword_groups:
        group_keywords[sensitive_attr] = [keyword for sublist in keyword_groups[sensitive_attr] for keyword in sublist]

    sa_mutation_map = {}

    for group, target_keywords in group_keywords.items():
        for keyword in target_keywords:
            for keyword_subgroups in keyword_groups[group]:
                if keyword in keyword_subgroups:
                    sa_mutation_map[keyword] = [w for w in keyword_subgroups if w != keyword]

    dataset = 'parallel'
    sensitive_attr = 'religion'

    grouped_sentences = {}
    texts = []
    mutants = []

    if dataset == 'parallel':
        df = pd.read_csv(f'./data/groups_kor_eng/{sensitive_attr}.csv')
        
        for i, row in tqdm(df.iterrows()):
            # kor_sentence = row["원문"]
            eng_sentence = ' '.join(row["번역문"].lower().split())
            
            texts.append(eng_sentence)
            mutants.append(trans_sentence_by_sa(eng_sentence, sensitive_attribute=sensitive_attr))
        
    elif dataset == 'cnn':
        df = pd.read_csv(f'./data/CNN/{sensitive_attr}-cnn.csv')
        
        for i, row in tqdm(df.iterrows()):
            eng_sentence = ' '.join(row["article"].lower().split())
            
            texts.append(eng_sentence)
            mutants.append(trans_sentence_by_sa(eng_sentence))
        
    print(texts[0])
    print(mutants[0])

    translator = RemoteTranslator("Google")

    inconsistency_result = {}

    for text, mutant_list in tqdm(list(zip(texts, mutants))[:3000]):
        text_hash = mmh3.hash(text)
        if text_hash in inconsistency_result:
            continue
        
        text_translated = translator.translate(text, 'en', 'ko')
        mutant_translated = [translator.translate(s_m, 'en', 'ko') for s_m in mutant_list]

        is_consistent, scores = test_consistency(text_translated, mutant_translated)

        # if not is_consistent:
        inconsistency_result[text_hash] =  {
            'original_sentence': (text, text_translated),
            'mutants': [(m, t, s) for m, t, s in zip(mutant_list, mutant_translated, scores)],
        }


    with open(f'result/inconsistency_result/{sensitive_attr}_{dataset}.json', 'w') as f:
        json.dump(inconsistency_result, f, ensure_ascii=False, indent=2)
