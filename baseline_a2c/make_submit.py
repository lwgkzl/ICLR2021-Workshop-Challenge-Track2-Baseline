import copy
import pickle
submit_ans = []
one_dic = {"symptom": [], "disease":""}
with open('save.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if "symptom" in line:
            one_dic['symptom'].append(line.split('@@@')[1].strip())
        if "disease" in line:
            one_dic['disease'] = line.split('@@@')[1].strip()
            save_dic = copy.deepcopy(one_dic)
            submit_ans.append(save_dic)
            one_dic = {"symptom": [], "disease": ""}

submit_ans2 = [submit_ans[-1]] + submit_ans[:-1]
print(len(submit_ans), len(submit_ans2))
print(submit_ans2[0])

with open('ans.pk','wb') as f:
    pickle.dump(submit_ans2, f)
