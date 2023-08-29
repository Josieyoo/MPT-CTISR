import os
#定义来源文件夹
path_src = r'F:\DataSets\baidu\my_test\hr'
#定义目标文件夹
path_dst = r'F:\DataSets\baidu\my_test\hr'
#自定义格式，例如“报告-第X份”，第一个{}用于放序号，第二个{}用于放后缀
rename_format = 'test{}{}'
#初始序号
begin_num = 1
def doc_rename(path_src,path_dst,begin_num):
    imglist = os.listdir(path_src)
    imglist.sort(key=lambda x: int(x.replace("eval", "").split('.')[0]))
    for i in imglist:
        #获取原始文件名
        doc_src = os.path.join(path_src, i)
        #重命名
        doc_name = rename_format.format(str(begin_num).zfill(4),os.path.splitext(i)[-1])
        #确定目标路径
        doc_dst = os.path.join(path_dst, doc_name)
        begin_num += 1
        os.rename(doc_src,doc_dst)
#运行函数
doc_rename(path_src,path_dst,begin_num)