import numpy as np
import time
import os
import math
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
# defaultdict用于计数
from operator import itemgetter
import pymysql
from tqdm import tqdm
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QStackedWidget, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap
import pymysql
import webbrowser
#from PyQt5.QtCore import QObject, pyqtSignal#实现类内变量与外部函数通信的功能

def LoadMovieLensData( train_rate):
    global old
    old=round(time.time())  #old是系统上一次训练的时间
    db = pymysql.connect(host='localhost',
                     user='root',
                     password='456123zhr',
                     database='recommend',
                     charset='utf8')
####获取ratings表
    cursor = db.cursor()
    sql = "select * from ratings where timestamp < 976636800"
#    sql = "select * from ratings where timestamp < %d"%old   #######重新训练时，所有当前的数据作为训练和验证集传入
    cursor.execute(sql)
    result = cursor.fetchall()
    
    cursor.close()
    # 关闭连接
    db.close()
    
    train = []
    test = []
    random.seed(3)
    for i in range(len(result)):
        if random.random() < train_rate:
            train.append([result[i][0],result[i][1],result[i][2],result[i][3]])
        else:
            test.append([result[i][0],result[i][1],result[i][2],result[i][3]])
    
                              #一种新的测试训练集分类的方法
    return PreProcessData(train), PreProcessData(test)


#数据的预处理
def LoadMovieLensData1():
    global old
    db = pymysql.connect(host='localhost',
                     user='root',
                     password='456123zhr',
                     database='recommend',
                     charset='utf8')

    cursor = db.cursor()
    sql = "select * from ratings where timestamp>=976636800"
#     sql = "select * from ratings where timestamp>=%d"%a
    cursor.execute(sql)
    result = cursor.fetchall() 
     
    test = []
    for i in range(len(result)):
            test.append([result[i][0],result[i][1],result[i][2],result[i][3]]) 
    return PreProcessData(test)


# 数据的预处理
def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": {MovieID1, MoveID2, MoveID3,...}
         "User2": {MovieID12, MoveID5, MoveID8,...}
         ...
        }
    """
    trainData = dict()
    for user, item, rating, timestamp in originData:
        trainData.setdefault(user, {})
        trainData[user][item] = rating
    return trainData

class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""

    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict()  # 物品相似度矩阵

    def similarity(self):
        N = defaultdict(int)  # 记录每个物品的喜爱人数
        for user, items in self._trainData.items():  #
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1

                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)

                    if self._similarity == "cosine":
                        # 余弦相似性用于计算物品相似度矩阵
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
                        # 也可以使用IUF相似度

        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i] * N[j])

        self.list_user = list(train.keys())  # 得到user的列表
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                # 对字典进行归一化操作之后返回新的字典
                self._itemSimMatrix[i] = {k: v / max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        """
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最喜欢的商品个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """ 
        self.list_user = list(train.keys())  #得到user的列表
        if user in self.list_user:
            recommends = dict()
            list1 = list(pop.keys())
            list2 = list(map(int, list1))
            # 先获取user的喜爱物品列表
            items = self._trainData.get(user,{})
            for item,score in items.items():
                # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
                for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                    if i in items:
                        continue  # 如果与user喜爱的物品重复了，则直接跳过
                    recommends.setdefault(i, 0.)
#                     list1 = list(pop.keys())
#                     list2 = list(map(int, list1))
                    if i in list2:
                        recommends[i] += sim * score/pop[i]
                    else:
                        recommends[i] += sim * score
                    #获得每个物品的相似度
            # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
            return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])
        else:
            user_preferences = get_user_preferences()
            print("用户偏好信息：",user_preferences)
            recommendations = recommend_movies(user_preferences,N)
            #转换为数组
            data_array = np.array(recommendations)
            # 然后转化为list形式
            data_list =data_array.tolist()
            data_list1=[]
            for i in range(len(data_list)):
                data_list1.append(data_list[i][0])
            return dict(zip(data_list1[0:N],data_list1[0:N]))    #统一返回值都是字典，方便后序操作

    def all_user_id(self):
        return self.list_user
    
    def train(self):
        self.similarity()


# 计算出算法的反馈指标，F值
def judge(test):
    global list_user
    recommend_items = set()
    all_items = set()
    recall_score=[]
    precision_score=[]
    a=set(test.keys())
    b=set(list_user)
    for user in tqdm(a&b):
        a=set(test[user].keys())   ###某用户test中的电影id集合
        for item in test[user].keys():
            all_items.add(item)    ###得到test里有的所有movieid
        rank1 = itemCF.recommend(user, 5,80)##耗时
        b=set(rank1.keys())        ###某用户推荐出的电影id集合
        c= a & b                   ###交集
        recall_score.append(len(c)/len(a))
        precision_score.append(len(c)/len(b))
        for item in rank1:
            recommend_items.add(item)
    recall=sum(recall_score)/len(recall_score)
    precision=sum(precision_score)/len(precision_score)
    F=recall*precision*2/(recall+precision)
    coverage=len(recommend_items) / (len(all_items) * 1.0)
    return recall,precision,F,coverage


# 动态管理，隔一段时间检测反馈指标是否达标，不达标进重新训练
def feedback():
    global old,interval,hours, minutes, seconds,train,test,recall,F,precision,coverage
    now = round(time.time())
    interval = now-old
    hours, minutes, seconds=convert_seconds(interval)
    if interval > 24*60*60*4*day:                  
        retrain()
    if interval > 24*60*60*day:                 
        test1 = LoadMovieLensData1()
        recall,precision,F,coverage=judge(test1) #受test1大小影响
        if (F<0.01 or coverage<0.05):           
            retrain()
            print("开始重新训练")
        else:
            print("指标符合要求")
    else:
        print("未到训练时间")

def retrain():
    global train,test,pop,recall, precision, F, coverage
    print("开始重新训练")
    train, test = LoadMovieLensData(0.8)
    pop=popularity(test)
    # print("train data size: %d, test data size: %d" % (len(train), len(test)))
    itemCF = ItemCF(train, similarity='iuf', norm=True)#实例化
    itemCF.train()
    recall, precision, F, coverage = judge(test)  ###先算一遍judge指标
    
def popularity(test):
    # 计算物品的流行度
    pop={}
    item_pop = {}
    all_items = set()  ##测试集所有电影的id集合
    for user in test.keys():
        for item in test[user].keys():
            all_items.add(item)
    for user in test:
        for item in test[user]:
            if item not in item_pop:
                item_pop[item] = 0
            item_pop[item] += 1
    for item in all_items:
        # 取对数，防止因长尾问题带来的被流行物品所主导
        pop[item] = math.log(1 + item_pop[item])
    return pop

def get_userid(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("输入无效，请输入一个合法id！")


def get_movie():
    while True:
        input_str = input("请输入喜欢的电影和评分（用空格分隔）: ")
        # 尝试将输入拆分为多个值
        values = input_str.split()

        if len(values) == 1:
            # 只有一个值
            try:
                number = int(values[0])
                if number != 0:
                    print("输入不全，请再输入您的评分")
                else:
                    print("很抱歉未能发现您喜欢的电影")
                    return 0
            except ValueError:
                print("输入无效，请输入一个整数！")
        elif len(values) == 2:
            # 有两个值
            try:
                number1 = int(values[0])  # 电影序号
                number2 = int(values[1])  # 评分
                if number1 < 1 or number1 > 5 or number2 < 1 or number2 > 5:
                    print("输入无效，电影序号介于1和5之间，评分在1和5之间")
                else:
                    return number1, number2
            except ValueError:
                print("输入无效，请输入两个整数！")
        else:
            print("输入无效，请输入一个数或两个数！")

#将此函数放到前端里
def get_user_preferences():
#此为前端需要展示的内容
    genres = movies_data['genres'].str.split('|', expand=True).stack().unique()
    user_preferences = []
    result = user_prefer #接收前端输入的值
    list_ = result.split()  # 使用空格分割输入的字符串为列表
    for i in range(len(list_)):  ######
        user_preferences.append(genres[int(list_[i]) - 1])
    return user_preferences


# 根据用户喜好进行电影推荐
def recommend_movies(user_preferences, N):
    # 根据用户偏好筛选电影
    filtered_movies = movies_data[movies_data['genres'].str.contains('|'.join(user_preferences))]

    # 从筛选后的电影中随机选择推荐
    recommended_movies = filtered_movies.sample(N)

    return recommended_movies[['movieId', 'title', 'genres']]

def get_user_data(userid): 
    global itemCF
    movie_data=[]
#     try:
    #userid = int(userid)
    result = itemCF.recommend(userid, 5, 80)   ##若改N,get_movie函数也需要改
    movie=list(result.keys())                  ##电影id的列表
    for i in range(len(movie)):
        db = pymysql.connect(host='localhost',
             user='root',
             password='456123zhr',
             database='recommend',
             charset='utf8')
        cursor = db.cursor()
        sql = "select movieid,title,genres from movies where movieid=%d"%movie[i]
        cursor.execute(sql)
        result1 = cursor.fetchall()
        cursor.close()
        my_dict = {}
        my_dict['movieid'] = result1[0][0]
        my_dict['title'] = result1[0][1]
        my_dict['genres'] = result1[0][2]
        image_name = "%d.jpg"%result1[0][0]
        directory = "E:/poster/"
        image_path = directory + image_name
        if os.path.exists(image_path):
            pass
        else:
            image_path = "E:/poster/0.jpg"
        my_dict['poster'] = image_path
        movie_data.append(my_dict)
    print(movie_data)
    return movie_data
    #     directory = "D:/poster/poster/"
    #     image_path = directory + image_name
    #     my_dict['poster'] = image_path
    #     movie_data.append(my_dict)
    # return movie_data     




class LoginWindow(QWidget):
    #登录界面
    def __init__(self, stacked_widget):
        super().__init__()
        self.setWindowTitle("Login Window")
        self.layout = QVBoxLayout()

        self.label1 = QLabel("Welcome to Moviematch!")
        self.layout.addWidget(self.label1)

        self.label = QLabel("请输入您的ID，我们会尽可能为您推荐合适的电影")
        self.layout.addWidget(self.label)

        self.id_input = QLineEdit()
        self.layout.addWidget(self.id_input)

        self.login_button = QPushButton("登录")
        #给登录按钮绑定事件，实现页面跳转
        self.login_button.clicked.connect(lambda: self.login(stacked_widget))
        self.layout.addWidget(self.login_button)

        self.setLayout(self.layout)

    def login(self, stacked_widget):
        #登录按钮的事件逻辑
        #try # def get_userid(prompt):
#     while True:
#         try:
#             value = int(input(prompt))
#             return value
#         except ValueError:
#             print("输入无效，请输入一个合法id！")
        feedback()
        user_input = self.id_input.text()
        if not user_input.isdigit():
            QMessageBox.warning(self, "错误", "请您输入正确的数字ID")
            self.id_input.clear()
        else:
            user_id = int(user_input)
        #user_id_int=int(user_id)
#加一个判断条件，根据新老用户以及管理员跳转
            if user_id == manager_id:
                movie_manager_window = MovieManagerWindow(stacked_widget)
                stacked_widget.addWidget(movie_manager_window)
                stacked_widget.setCurrentWidget(movie_manager_window)
            else:
                if user_id in itemCF.all_user_id() :
                    movie_ratings_window = MovieRatingsWindow(stacked_widget, user_id)
                    stacked_widget.addWidget(movie_ratings_window)
                    stacked_widget.setCurrentWidget(movie_ratings_window)
                else:
                    movie_preferences_window = MoviePreferencesWindow(stacked_widget,user_id)
                    stacked_widget.addWidget(movie_preferences_window)
                    stacked_widget.setCurrentWidget(movie_preferences_window)

class MovieManagerWindow(QWidget):
   def __init__(self, stacked_widget):
        super().__init__()

        self.setWindowTitle("电影管理系统")
        self.setGeometry(100, 100, 400, 300)

        # 创建垂直布局
        layout = QVBoxLayout(self)

        # # 创建垂直布局
        # layout = QVBoxLayout(container)

        # 创建标签
        label1 = QLabel(f"精准度:{precision}", self)
        label1.setStyleSheet("font-size: 20px; color: #333333; margin: 10px;")
        layout.addWidget(label1)

        label2 = QLabel(f"召回率:{recall}", self)
        label2.setStyleSheet("font-size: 20px; color: #333333; margin: 10px;")
        layout.addWidget(label2)

        label3 = QLabel(f"覆盖率:{coverage}", self)
        label3.setStyleSheet("font-size: 20px; color: #333333; margin: 10px;")
        layout.addWidget(label3)

        label4 = QLabel(f"F值:{F}", self)
        label4.setStyleSheet("font-size: 20px; color: #333333; margin: 10px;")
        layout.addWidget(label4)

        label5 = QLabel(f"距离上次训练时间:{hours}时{minutes}分{seconds}秒", self)
        label5.setStyleSheet("font-size: 20px; color: #333333; margin: 10px;")
        layout.addWidget(label5)
        # 创建按钮
        button1 = QPushButton("重新训练", self)
        button1.setStyleSheet("font-size: 20px; padding: 10px 20px; background-color: #4CAF50; color: #FFFFFF;")
        button1.clicked.connect(lambda:retrain())  # 绑定按钮1的点击事件
        layout.addWidget(button1)

        # 创建按钮2
        button2 = QPushButton("退出", self)
        button2.setStyleSheet("font-size: 20px; padding: 10px 20px; background-color: #FF0000; color: #FFFFFF;")
        button2.clicked.connect(lambda: self.logout(stacked_widget))  # 绑定按钮2的点击事件
        layout.addWidget(button2)

   # def button1_clicked(self):
   #      # 按钮1点击事件的函数逻辑
   #      pass

   def logout(self, stacked_widget):
        stacked_widget.setCurrentIndex(0)
    
class MovieRatingsWindow(QWidget):
    def __init__(self, stacked_widget, user_id):
        super().__init__()
        self.setWindowTitle("Movie Ratings")
        #水平布局
        self.layout = QHBoxLayout()
        self.user_id = user_id
        print("传到新页面的user_id",self.user_id)
        self.movie_data = get_user_data(user_id)  
         # 调用另一个类的函数获取电影数据
        self.movie_ratings = {}
        self.movie_layouts = []
        #movie_layout = QVBoxLayout()
        for movie in self.movie_data:
            movie_layout = QVBoxLayout()
            movie_label = QLabel(f"{movie['movieid']}:{movie['title']}")
            #暂时只显示电影ID
            #print("显示ID",movie)
            movie_layout.addWidget(movie_label)
            genres_label = QLabel(f"{movie['genres']}")
            movie_layout.addWidget(genres_label)
            poster_label = QLabel()
            pixmap = QPixmap(f"{movie['poster']}")
            poster_label.setPixmap(pixmap)
            movie_layout.addWidget(poster_label)
            link_button = QPushButton("播放")
            #link_button.clicked.connect(get_movie_link())
            link_button.clicked.connect(lambda _, movie_id=movie["movieid"]: self.get_movie_link(movie_id))
            movie_layout.addWidget(link_button)

            self.movie_layouts.append(movie_layout)
            self.layout.addLayout(movie_layout)    
            
        #完整语句：QHBoxLayout().addWidget.(QLabel("你选择的评分是："))
        select_layout = QVBoxLayout()
        select_layout.addWidget(QLabel("您选择的电影是："))
        
        self.movie_selector = QComboBox()
        self.movie_selector.addItems([f"{movie['movieid']}" for movie in self.movie_data])
        select_layout.addWidget(self.movie_selector)
        self.layout.addLayout(select_layout)
        
        rating_layout = QVBoxLayout()
        rating_label = QLabel("您选择的评分是：")
        rating_layout.addWidget(rating_label)
        self.rating_selector = QComboBox()
        self.rating_selector.addItems(["1", "2", "3", "4", "5"])
        rating_layout.addWidget(self.rating_selector)
        self.layout.addLayout(rating_layout)
        
        #提交评分按钮
        self.submit_button = QPushButton("提交评分")
        self.submit_button.clicked.connect(lambda: self.confirm_submit(stacked_widget))
        self.layout.addWidget(self.submit_button)
        #退出登录按钮
        self.logout_button = QPushButton("退出登录")
        self.logout_button.clicked.connect(lambda: self.logout(stacked_widget))
        self.layout.addWidget(self.logout_button)
        self.setLayout(self.layout)
    
    # def get_movie_link(self,movieid):
        
    #     print("获取电影链接：",movieid)
    
    def get_movie_link(self,movieid):
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='456123zhr',
                             database='recommend',
                             charset='utf8')
    
        cursor = db.cursor()
        sql = "select * from link where movieid=%d"%movieid
        cursor.execute(sql)
        result = cursor.fetchall()
        imdbid = result[0][1]
        # 定义要打开的网页链接
        url = 'https://www.imdb.com/title/tt0%d/'%imdbid
    
        # 发送GET请求，获取网页内容
        try:
            webbrowser.open(url)
        except webbrowser.Error as e:
            print("打开链接发生错误:", e)
    
    def confirm_submit(self, stacked_widget):
        reply = QMessageBox.question(self, "小提示", "您确定要给这部电影这个评分吗?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.submit_ratings(stacked_widget)
            print("确认评分")
    def update_rating(self, movie_id, rating):
        self.movie_ratings[movie_id] = rating
    def submit_ratings(self, stacked_widget):
        print("进入提交函数")
        selected_movie = self.movie_selector.currentText()
        rating = self.rating_selector.currentText()
        print(selected_movie)
        print(rating)
        db = pymysql.connect(
             host='localhost',
             user='root',
             password='456123zhr',
             database='recommend',
             charset='utf8'
        )
        cursor = db.cursor()
        print("数据库连接成功！")
        current_time = round(time.time())
        sql = "INSERT INTO ratings (userID, movieID, point, timestamp) VALUES (%s, %s, %s, %s)"
        values = (self.user_id, selected_movie, rating, current_time)
        cursor.execute(sql, values)
        print("成功插入数据库！")
        db.commit()
        db.close()
        self.disable_rating_inputs()
    def disable_rating_inputs(self):
        self.rating_selector.setEnabled(False)
        self.movie_selector.setEnabled(False)
        self.submit_button.setEnabled(False)      
    def logout(self, stacked_widget):
        stacked_widget.setCurrentIndex(0)


class MoviePreferencesWindow(QWidget):
#新用户选择电影类型的页面
#value_input = pyqtSignal(int)  # 定义一个信号，参数类型为int
    def __init__(self, stacked_widget,user_id):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("Movie Preferences")
        self.layout = QVBoxLayout()
        self.label1 = QLabel("检测到您是新用户，请从下面选择您喜欢的电影类型，我们会据此给您进行推荐：")
        self.layout.addWidget(self.label1)
        self.label2 = QLabel("请选择您喜欢的电影类型：\n1. Animation\n2. Children's\n3. Comedy\n4. Adventure\n5. Fantasy\n6. Romance\n7. Drama\n8. Action\n9. Crime\n10. Thriller\n11. Horror\n12. Sci-Fi\n13. Documentary\n14. War\n15. Musical\n16. Mystery\n17. Film-Noir\n18. Western\n 请麻烦输入您喜欢的类型（用空格隔开）")
        self.layout.addWidget(self.label2)
        self.label3 = QLabel("如：我喜欢喜剧和爱情，就输入3 6，可以多选哦！")
        self.layout.addWidget(self.label3)
        self.input_text = QLineEdit()
        self.layout.addWidget(self.input_text)
        self.submit_button = QPushButton("就喜欢它们啦！")
        self.submit_button.clicked.connect(self.confirm_submit)
        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)            
    def confirm_submit(self):
        reply = QMessageBox.question(self, "小提示", "请确定您的喜欢，我们将基于此为您推荐合适的电影！", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.execute_logic()
            movie_ratings_window = MovieRatingsWindow(stacked_widget, self.user_id)
            stacked_widget.addWidget(movie_ratings_window)
            stacked_widget.setCurrentWidget(movie_ratings_window)
    def execute_logic(self):
            global user_prefer
            user_prefer=self.input_text.text()
        
        #回调函数，将类内用户输入的值传到类外
def convert_seconds(seconds):
    hours = seconds // 3600  # 计算小时数
    minutes = (seconds % 3600) // 60  # 计算分钟数
    seconds = seconds % 60  # 计算剩余的秒数

    # 返回转换后的时分秒
    return hours, minutes, seconds
        # 执行一些逻辑操作 
if __name__ == "__main__":
    train, test = LoadMovieLensData(0.8)
    list_user = list(train.keys())
    pop=popularity(test)
    # print("train data size: %d, test data size: %d" % (len(train), len(test)))
    itemCF = ItemCF(train, similarity='iuf', norm=True)#实例化
    itemCF.train()
##############################################################系统管理参数初始化设置
day=7
old=round(time.time())
now=round(time.time())
interval = now-old   
N=5                                                             #推荐个数
K=80 
recall=0
precision=0
F=0
coverage=0
list_user = list(train.keys())
hours=0  
minutes=0 
seconds=0
#recall, precision, F, coverage = judge(test)  ###先算一遍judge指标
###############################################################数据库连接
db = pymysql.connect(host='localhost',
                     user='root',
                     password='456123zhr',
                     database='recommend',
                     charset='utf8')
############################################################################从数据库获取movies表并变成dataframe结构
cursor = db.cursor()
sql = "select * from movies "
cursor.execute(sql)
result2 = cursor.fetchall()
cursor.close()
value1 = []
value2 = []
value3 = []
for i in range(len(result2)):
    value1.append(result2[i][0])
    value2.append(result2[i][1])
    value3.append(result2[i][2])
my_dict = {}
my_dict['movieid'] = value1
my_dict['title'] = value2
my_dict['genres'] = value3
movies_data = pd.DataFrame(my_dict)
######
# movies_data = pd.read_csv('E:/movies.dat', sep='::', header=None, engine='python',encoding='ISO-8859-1')
movies_data.columns = ['movieId', 'title', 'genres']

###############################################################################获取用户信息并推荐
print("训练成功")
        

app = QApplication(sys.argv)
user_prefer="1" #全局变量，接收用户输入
manager_id =2021210113
# train , test = LoadMovieLensData(0.8)
# itemCF = ItemCF(train, similarity='iuf', norm=True)  # 实例化

stacked_widget = QStackedWidget()
login_window = LoginWindow(stacked_widget)
stacked_widget.addWidget(login_window)

stacked_widget.setCurrentIndex(0)
stacked_widget.show()

sys.exit(app.exec_())