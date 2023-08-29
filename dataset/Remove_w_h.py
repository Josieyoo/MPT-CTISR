import os
from PIL import Image
# 输出所有文件和文件夹
DirList = [r'F:\DataSets\baidu\train_images']

with open(r"F:\DataSets\baidu\trainlabel.txt", "r",encoding='utf-8') as f:    #读取文本
    data = f.readlines()
    f.close()

dirtyid = []
new_data = []
a=0
for path in DirList:
    print(path)
    index = 0
    imglist = os.listdir(path)
    imglist.sort(key=lambda x: int(x.replace("img_", "").split('.')[0]))
    for filename in imglist:
        # print(filename)
        fullName = os.path.join(path, filename)
        fp = open(fullName, 'rb')
        img = Image.open(fp)
        fp.close()
        if img.size[0] < 80 or img.size[0] > 700:
            os.remove(fullName)
            a = a + 1
            dirtyid.append(index)
        index = index + 1
print(a)
#
for x in range(len(data)):
    if x not in dirtyid:
        new_data.append(data[x])  # 将不删的数据赋给另一个列表

with open(r'F:\DataSets\baidu\datanew.txt', 'a+',encoding='utf-8') as f:
        f.writelines(new_data)
        f.close
