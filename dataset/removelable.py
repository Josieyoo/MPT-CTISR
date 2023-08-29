import os
#定义来源文件夹
path_src = r'F:\DataSets\baidu\train_images'

needimg_label = []
needimg_no = []


with open(r"F:\DataSets\baidu\labelall.txt", "r", encoding='utf-8') as f:  # 读取文本
    data = f.readlines()
    f.close()
imglist = os.listdir(path_src)
for i in imglist:
    #获取需要的图片序号
    needimg_no.append((i.replace(".jpg", "")))
# print(len(needimg_no))

for index in range(41133):
    # print(index + 1)
    if str(index+1).zfill(5) in needimg_no:
        needimg_label.append(data[index])  # 将不删的数据赋给另一个列表
# print(needimg_label)

with open(r'F:\DataSets\baidu\datanew2.txt', 'a+', encoding='utf-8') as f:
    f.writelines(needimg_label)
    f.close
