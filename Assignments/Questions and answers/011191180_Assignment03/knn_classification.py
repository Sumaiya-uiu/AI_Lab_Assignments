from google.colab import drive
import numpy
from numpy import genfromtxt
import random
import operator
import itertools
from collections import Counter 

def knn_cal(k,train_Set,some_set):
    l_d = dict()
    c_v = 0
    for s in some_set:
        for indi,v in enumerate(train_set):
            sx = numpy.array(s[:len(s)-1])
            vx = numpy.array(v[:len(v)-1])
            ed = numpy.linalg.norm(sx - vx)
            l_d[indi] = ed

        l_d = {g: a for g, a in sorted(l_d.items(), key=lambda item: item[1])}
        k_l_d = dict(itertools.islice(l_d.items(), k))
        found_class_all, count = Counter(k_l_d.values()).most_common(1)[0]
        found_class_list = [id for id,dt in k_l_d.items() if dt == found_class_all]
        found_class_indi = found_class_list[0]
        found_class = train_set[found_class_indi][-1] 
        if found_class == s[-1]:
            c_v += 1
    c_v_a = (c_v / len(some_set))*100
    return c_v_a
    

drive.mount('/content/gdrive')
data_path = '/content/gdrive/MyDrive/DataSet/Iris/iris.csv'
my_data = genfromtxt(data_path, delimiter=',')
data = my_data.tolist()
random.shuffle(data)
train_set,val_set,test_set = [],[],[]
for s in data:
    r = random.uniform(0, 1)
    if r >= 0 and r <= 0.7:
        train_set.append(s)
    elif r > 0.7 and r <= 0.85:
        val_set.append(s)
    else:
        test_set.append(s)
k = [1,3,5,10,15]
f_set = 0
last_k = 0
print("Value(K) ............... Accuracy")
for i in k:
    rs = knn_cal(int(i),train_set,val_set)
    print('{} ............... {}'.format(i,rs))
    if rs >= f_set:
        f_set = rs
        last_k = int(i)
t_set = knn_cal(last_k,train_set,test_set)
print('test_acc :{}'.format(t_set))
