import argparse
import json
#from metrics import f1_metric
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
import nltk
import evaluate
#from parlai.core.metrics import RougeMetric, BleuMetric

NO_PASSAGE_USED = "no_passages_used"
KNOWLEDGE_SEP = "__knowledge__"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file")
    parser.add_argument("--ref_file")
    parser.add_argument("--eval_metric", default="f1", choices=["f1", "rouge_l", "bleu"])
    args = parser.parse_args()

    pred_list = []  
    ref_list = []  
    knowledge_list = []   
    with open(args.ref_file, 'r', encoding = 'utf-8') as rf:
        lines = rf.readlines()
        for line in lines:
            ref_list.append(line)
    with open(args.pred_file, mode="r", encoding="utf-8") as pf:
        lines = pf.readlines()
        for line in lines:
            extracted_text = line.split('<|endoftext|>')[1].split('[SPECIAL_END]')[0].strip()
            pred_list.append(extracted_text)

    assert len(pred_list) == len(ref_list)
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=pred_list, references=ref_list, tokenizer=word_tokenize)
    rouge = evaluate.load('rouge')
    results_rouge = rouge.compute(predictions=pred_list, references=ref_list)
    print(results)
    print(results_rouge)
    if args.eval_metric == "f1":
        #print( f1_score(ref_list, pred_list, average='macro'))
        print(f"F1: {f1_metric(pred_list, ref_list)}")
    """ elif args.eval_metric == "rouge_l":
        rl = sum([RougeMetric.compute_many(hyp, [ref])[2].value() for hyp, ref in zip(pred_list, ref_list)]) / len(pred_list)
        print(f"rouge-l: {rl}")
    elif args.eval_metric == "bleu":
        b1 = sum([BleuMetric.compute(hyp, [ref], k=1).value() for hyp, ref in zip(pred_list, ref_list)]) / len(pred_list)
        b2 = sum([BleuMetric.compute(hyp, [ref], k=2).value() for hyp, ref in zip(pred_list, ref_list)]) / len(pred_list)
        b3 = sum([BleuMetric.compute(hyp, [ref], k=3).value() for hyp, ref in zip(pred_list, ref_list)]) / len(pred_list)
        b4 = sum([BleuMetric.compute(hyp, [ref], k=4).value() for hyp, ref in zip(pred_list, ref_list)]) / len(pred_list)
        print(f"Bleu: {b1, b2, b3, b4}") """

    """ else:
        assert False, "Wrong Choice" """


if __name__ == "__main__":
    main()