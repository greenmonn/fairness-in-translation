from transrepair_kor.main import Sentence, test_consistency, repair, calc_consistency_score, Sentence
from tqdm import tqdm
import pandas as pd
import pickle
import json

if __name__ == "__main__":
    dataset = 'parallel'

    kor_sentences = []

    if dataset == 'parallel':
        df = pd.read_csv('./data/groups_kor_eng/gender.csv')
        for i, row in df.iterrows():
            kor_sentences.append(row["원문"])
    elif dataset.startswith('hani'):
        df = pd.read_csv(f'./data/hani/{dataset}.csv')
        for i, row in df.iterrows():
            kor_sentences.append(row["article"])

    print(len(kor_sentences))

    sentences = {}
    for s in tqdm(kor_sentences[:1000]):
        text = Sentence(s)
        sentences[text.hash] = text

    mutant_sentences = {}
    for hash, text in tqdm(sentences.items()):
        if hash in mutant_sentences:
            continue
        mutants = text.create_mutations()
        mutant_sentences[hash] = mutants

    
    with open(f'{dataset}_sentences.pkl', 'wb') as f:
        pickle.dump(sentences, f)

    with open(f'{dataset}_mutant_sentences.pkl', 'wb') as f:
        pickle.dump(mutant_sentences, f)

    repair_results = []
    inconsistency_results = []

    for hash, text in tqdm(sentences.items()):
        mutants = mutant_sentences[hash]
        is_consistent, (inconsistent_mutant, score) = test_consistency(text, mutants)
        if not is_consistent:
            repaired_sentence = repair(text, mutants)
            inconsistency_results.append((inconsistent_mutant, score))
            # repair_results.append((text, mutants, repaired_sentence))
            repair_results.append({
                'original': (text.sentence, text.translated),
                'repaired': repaired_sentence.translated,
                'applied_mutant': (repaired_sentence.applied_mutant, repaired_sentence.mutant_translated),
                'replaced_word': (repaired_sentence.src_aligned, repaired_sentence.dest_aligned)
            })

    with open(f'data/repair_result/{dataset}.json', 'w') as f:
        json.dump(repair_results, f)