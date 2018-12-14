一、快速使用：  
打开model文件夹  
1、训练  
1)设置  
--- load_data.py中  
is_training = True  
debug = False  
2)运行  
直接运行2ch_deep_3264.py即可  
  
2、测试  
1)设置  
--- load_data.py中  
is_training = False  
2)运行  
直接运行test_deep2ch_3264.py即可  
  
(其他模型训练和测试同理)  
  
3、载入参数  
若需要载入参数请将deep2ch_3264_parameters中要载入的参数文件改为paras.pkl  
  
（***小数据Debug请将load_data.py中的debug改为True）  
 (**若想关闭实时画图显示，请将load_data.py中的is_ploting改为False)  
  
  
二、数据预处理（由于已处理好数据，可直接跳过这一步）  
preprocess中为数据预处理程序，包括  
1、del_damaged_img.py 删除imgs中损坏的图片  
  
2、del_rep&non-img.py 删除label.split.txt中重复的行以及没有对应图片的行  
  
3、w_replace.py 对latex进行相似替换的预处理，处理如下  

'''  
letter -> a     eg: x y z -> a a a  
digital -> 0    eg: 1 2 3 -> 0  
sqrt(digital) -> 0      eg: sqrt(1 4/2 5) -> 0  
digital/digital -> 0    eg: sqrt(1 4)/2 5 -> 0  
digital*letter -> a     eg: 4 3 x y z -> a a a  
letter/digital -> a     eg: 4 3 x y z / 5 3 -> a a a  
'''  

4、split_dataset.py 将所有行shuffle后，划分为训练集和测试集(最后10000行)  
5、get_testpair.py 得到测试对(1 or 0)数据集  
6、add_noise.py 生成带噪声公式测试图片    
