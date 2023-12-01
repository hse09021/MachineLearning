import sys
import numpy as np
from sklearn.metrics import auc
from collections import Counter

testResult_path = sys.argv[1]
label_path = sys.argv[2]

# pred에 해당하는 testResult.txt 파일 읽어오는 부분입니다.
with open(testResult_path, 'r') as file1:
    preds = file1.readlines()

# 정답에 해당하는 label.txt 파일 읽어오는 부분입니다.
with open(label_path, 'r') as file2:
    labels = file2.readlines()
    

# pred와 label의 클래스값만 리스트로 변환하는 부분입니다.
p = np.array([pred.strip().split()[1] for pred in preds])
l = np.array([label.strip().split()[1] for label in labels])

# pred의 클래스 개수를 count하는 부분입니다.
predict_label_count_dict = Counter(p)


## mAP 계산하는 부분입니다.
AP = []
num_class = 10

# 모든 클래스에 대해 반복
for c, freq in predict_label_count_dict.items() :
    TP = 0
    FN = 0

    temp_precision = []
    temp_recall = []
    
    for i in range(len(p)):
        # TP, FN 계산
        if l[i] == c and p[i] == c :
            TP += 1
        elif l[i] != c and p[i] == c :
            FN += 1
        
        # preciisonm recall 계산            
        if TP+FN != 0: 
            temp_precision.append(TP/(TP+FN))
            temp_recall.append(TP/freq)

    # AP 배열에 클래스 각각의 AP value 저장
    # auc : preciison-recall curve의 면적 구해줌
    AP.append(auc(temp_recall, temp_precision))
    
mAP = sum(AP) / num_class

# 각각의 클래스에 대한 AP와 mAP의 Table 출력 부분입니다.
class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
table = "| {:<13} | {:<13} |\n".format("Class", "AP") + "|---------------|---------------|\n"

for c_name, ap in zip(class_name, AP):
    table += "| {:<13} | {:<13.2f} |\n".format(c_name, ap)

table += "| {:<13} | {:<13.2f} |\n".format("mAP", mAP)

print(table)
