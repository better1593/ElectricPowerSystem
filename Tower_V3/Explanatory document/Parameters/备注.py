""" 模型参数的保存与读取
1.假设我们的模型数据都是保存在字典中，使用Python的pickle模块，它可以用于序列化和反序列化对象。
    # 用.pkl的方式
    import pickle

    # 假设我们已经有一个类的实例
    towermodel = {'WireP': 42, 'Node': [1, 2, 3], 'Bran': {'a': 'hello', 'b': 'world'}}

    # 保存实例到文件
    with open('towermodel.pkl', 'wb') as file:
        pickle.dump(towermodel, file)

    # 从文件加载实例
    with open('towermodel.pkl', 'rb') as file:
        loaded_towermodel = pickle.load(file)

    # 打印加载的实例，完成后续的操作
    print(loaded_towermodel)

    需要保证两台电脑的环境是相同的

    CST     https://space.mit.edu/RADIO/CST_online/Python/main.html
    Comsol  https://mph.readthedocs.io/en/stable/tutorial.html
"""

""" 变量命名规范
1.变量：小写字母，单词之间使用下划线“_”分隔。
示例：my_variable, count_of_items

2.函数和方法参数：小写字母，单词之间使用下划线“_”分隔。
示例：calculate_average(price, quantity), get_user_data()

3.常量：全部大写字母，单词之间使用下划线“_”分隔。
示例：MAX_VALUE

4.类：使用驼峰命名法（CamelCase），每个单词的首字母大写，不使用下划线。
示例：TowerModel, CableModel

5.模块和包：小写字母，单词之间使用下划线“_”分隔。
示例：my_module, my_package

6.私有变量和方法：在变量或方法名称前加一个下划线“_”，表示它是私有的，不应该被直接访问。
示例：_internal_variable, _private_method()

7.字典键：尽量使用描述性的键，小写字母，单词之间使用下划线“_”分隔。
示例：user_data = {'first_name': 'John', 'last_name': 'Doe'}
"""

''' 空的操作
Python中对于空的操作与判断主要根据不同的数据类型有所区别
1. 整数、浮点数、复数：
    num = 0
    if num == 0:
        print("变量为0")

2.字符串：
    my_str = ""
    if not my_str:
        print("字符串为空")

    if not my_str.strip():
        print("字符串为空")

3.布尔值：
    my_bool = False
    if not my_bool:
        print("布尔值为空")

4.列表、元祖、集合：
    my_list = []
    if not my_list:
        print("列表为空")

    my_tuple = ()
    if not my_tuple:
        print("元组为空")

    my_set = set()
    if not my_set:
        print("集合为空")
        
5.字典：
    my_dict = {}
    if not my_dict:
        print("字典为空")

6.文件：
    file_path = "example.txt"
    
    if os.path.getsize(file_path) == 0:
        print("文件为空！")
    else:
        print("文件不为空。")

'''

''' 常数的定义
Python中有一些约定俗成的常量，通常以全大写字母表示，如下所示：
1.True 和 False：两个是布尔类型的常量

2.None：表示空值或缺失值

3.数学和科学常量：在 math 模块中提供了一些数学和科学常量，如 math.pi 表示圆周率。
    import math
    
    print(math.pi)  # 输出圆周率π的近似值

4.无穷大和非数字：float 类型的 inf 表示正无穷，-inf 表示负无穷，nan 表示非数字
'''