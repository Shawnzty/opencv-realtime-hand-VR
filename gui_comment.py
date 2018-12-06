# 建议先看main.py，再看cvImg.py，最后看这个
# 这个部分主要是gui界面设计，包括套接字处理和字符串处理。和cvImg.py相比是比较容易的

# 首先还是导入一大堆库
from tkinter import scrolledtext
# ttk这个库里的控件比tkinter的好看，好看很重要哦
from tkinter import ttk
from tkinter import *
from cvImg import *
from socket import *
import threading
import cv2

# 定义一些全局变量，其实有些也可以放在类里面成为成员变量。因为数量也不多，所以不太考虑内存占用的问题。
global sockobj, interval, Img, show, angle, deepth

# 新定义一个类gui
class gui:
    # 这个类的构造函数就比较多了，其实是把整个界面上显示出来的东西全部放在构造函数里面定义了
    def __init__(self, root):
        # 首先把传入的参数root变成自己的东西
        self.root = root
        # 定义窗口名字
        self.root.title("Opencv 控制台")
        # 定义窗口尺寸，数字中间那个是英文字母小写x
        self.root.geometry('870x500')

        # 定义scltxt是可滚动的文本框，后面再对这个文本框进行操作就可以直接用"scltxt."的形式，tkinter中的大多数控件都是如此
        # 第一个参数是目标窗口，第二个是宽度，第三个是高度，单位应该是一个英文字母的宽度和高度。
        # 第四个是换行方式，这里选的是不断开单词
        self.scltxt = scrolledtext.ScrolledText(root, width=60, height=38, wrap=WORD)
        # 使用scltxt中的place函数进行布局操作，文本框左上角放在窗口左上角
        self.scltxt.place(x=0, y=0)

        # 按键的定义的都差不多，我仔细讲解一个，后面都一样的。
        # 下面这一行相当于两行:
        # self.initial = ttk.Button(root, text ="初始化", command =self.init)
        # self.initial.place(x=450, y=30)
        # 我这边就是把".place"直接放在定义Button后面了，看起来简洁一些
        # Button函数的参数：第一个是目标窗口，第二个是按钮上显示的字，第三个是按这个按钮会触发的函数
        # place函数中，第一个是左上角的x坐标，第二个是左上角的y坐标
        ttk.Button(root, text="初始化", command=self.init).place(x=450, y=30)
        ttk.Button(root, text="打开监视器", command=self.monitor).place(x=630, y=30)
        ttk.Button(root, text="心电开", command=self.ECG).place(x=760, y=30)

        # Label是直接在界面上显示字符， font是指定字体字号，我这边就这是了一个字号
        ttk.Label(root, text="手动设置", font=('16')).place(x=450, y=70)
        # 输入旋转
        ttk.Label(root, text="转角：（顺时针为正）").place(x=450, y=100)
        # Entry是输入框，我们可以读取输入框中的数值来使用
        self.angle = ttk.Entry(root, width=12)
        self.angle.place(x=630, y=100)

        # 输入放缩量
        ttk.Label(root, text="左右放缩：").place(x=450, y=125)
        self.scal_x = ttk.Entry(root, width=12)
        self.scal_x.place(x=630, y=125)

        ttk.Label(root, text="前后放缩：").place(x=450, y=150)
        self.scal_y = ttk.Entry(root, width=12)
        self.scal_y.place(x=630, y=150)

        # 输入模糊量
        ttk.Label(root, text="模糊度：（1最清晰，<60）").place(x=450, y=175)
        self.dim = ttk.Entry(root, width=12)
        self.dim.place(x=630, y=175)


        # 输入前后平移量
        ttk.Label(root, text="平移：（前正后负）").place(x=450, y=200)
        self.deepth = ttk.Entry(root, width=12)
        self.deepth.place(x=630, y=200)

        # 输入左右平移量
        ttk.Label(root, text="平移：（右正左负）").place(x=450, y=225)
        self.lateral = ttk.Entry(root, width=12)
        self.lateral.place(x=630, y=225)

        # 选择手or木头
        # 这里手和木头使用了一个单项选择,首先定义一个变量hand，相当于cvImg.py中的wood
        self.hand = IntVar()
        self.hand.set(0)  # 默认0，即是手
        ttk.Label(root, text="显示：").place(x=450, y=275)
        # Button控件的定义方式相似，区别是第三个参数variable和第四个参数value，就是如果选择了某一项，就要给一个变量赋一个值
        # 之后就是根据赋的值不同来作出不同处理
        self.hand_slt = Radiobutton(root, text='手', variable=self.hand, value=0)  # , command=handset)
        self.hand_slt.place(x=560, y=274)
        self.wood_slt = Radiobutton(root, text='木头', variable=self.hand, value=1)  # , command=handset)
        self.wood_slt.place(x=609, y=274)
        # 这边上下我用的变量名一样，其实也没什么关系，不过最好还是改成不一样的吧
        self.wood_slt = Radiobutton(root, text='空白', variable=self.hand, value=2)  # , command=handset)
        self.wood_slt.place(x=670, y=274)

        # 设置延时时间
        ttk.Label(root, text="延时：（毫秒）").place(x=450, y=250)
        self.delay = ttk.Entry(root, width=12)
        self.delay.place(x=630, y=250)

        # 确认，把修改值传递到cvImg
        set = ttk.Button(master=root, text="确认", width=10, command=self.setting)
        # 这里第三第四个参数定义了按键的尺寸，采用相对值定义法，单位是窗口尺寸，0.39就是“0.39*窗口高度”
        set.place(x=760, y=99, relheight=0.39, relwidth=0.1)

        # 猴子设定
        ttk.Label(root, text="被试特征", font='16').place(x=450, y=325)

        # 这个参数是修补的时候用的，现在不修补就不用管了。默认80，数字本身无物理意义，越小越慢，越大越快
        ttk.Label(root, text="行动速度：").place(x=450, y=350)
        self.speed = ttk.Entry(root, width=12)
        self.speed.place(x=630, y=350)

        # 默认40，代表毛发灰度值，越小颜色越浅，越大颜色越深
        ttk.Label(root, text="表面颜色：").place(x=450, y=375)
        self.color = ttk.Entry(root, width=12)
        self.color.place(x=630, y=375)

        set = ttk.Button(master=root, text="确认", width=10, command=self.character)
        set.place(x=760, y=349, relheight=0.1, relwidth=0.1)

        # MATLAB监听
        ttk.Label(root, text="用MATLAB控制", font='16').place(x=450, y=425)

        ttk.Button(root, text="开始监听", command=self.listen).place(x=450, y=455)
        ttk.Button(root, text="清屏", command=self.clear).place(x=630, y=455)


        ttk.Button(master=root, text='退出', command=self._quit).place(x=760, y=455)


    # 初始化函数
    def init(self):
        global Img
        self.scltxt.delete(1.0, END)  # 清空文本框
        # 后面大量用到的"Img.",就是在这里进行实例化
        Img = cvImg()  # 类实例化
        # 下面这个addString函数会经常用到，功能是在文本框里写东西
        self.addString("cvImg已实例化")
        return 0

    # 在文本框里写东西
    def addString(self, string):
        self.scltxt.insert(INSERT, string + '\n')


    # 这是我给心电留的接口
    def ECG(self):
        # 按“心电开”之后，把按键变成“心电关”
        ttk.Button(self.root, text="心电关", command=self.noECG).place(x=760, y=30)
        self.addString("心电已启用")
        # 新开一个线程，用来监听和处理心电数据
        t_ECG = threading.Thread(target=self.connectECG)
        t_ECG.setDaemon(True)
        t_ECG.start()


    # 真正处理心电的函数
    def connectECG(self):
        # 首先定义主机，空就代表所有
        myHost2 = ''
        # 定义服务器端口，matlab用的是6000，那这边就选个不一样的
        myPort2 = 6001  # 定义一个和matlab不同的接口
        # 定义套接字
        sockobj2 = socket(AF_INET, SOCK_STREAM)
        # 绑定服务器主机和端口
        sockobj2.bind((myHost2, myPort2))
        # 设置最多挂起五条指令，一般都设置这个数
        sockobj2.listen(5)
        self.addString("等待心电连接......")
        while True:
            # 接收到信号并记录下地址信息
            connection2, address2 = sockobj2.accept()
            # 在左侧文本框显示出来
            show = '已连接上这个地址：' + str(address2)
            self.addString(show)
            # 开启一个死循环
            while True:
                # 收取数据，最长为1024个字节，此时的data是字节码格式的
                data = connection2.recv(1024)
                # 把data解码成字符串格式
                data = data.decode()
                # 有时候头尾会多出"\n"影响操作，先删掉
                data = data.strip('\n')  # 这样才得到能用的字符串
                # 之后处理出0和1，我这里假设直接得到0和1
                if data == 0:
                    Img.wood = 2  # 看不见手
                elif data == 1:
                    Img.wood = 0  # 看得见手
                if not data: break
            # 关闭连接
            connection2.close()


    def noECG(self):
        # 如果点了“心电关”则会用新的按键“心电开”来覆盖之前的按键
        ttk.Button(self.root, text="心电开", command=self.ECG).place(x=760, y=30)

        self.addString("心电已断开")
        return 0


    # 退出界面
    def _quit(self):
        self.root.quit()
        self.root.destroy()
        # 释放摄像头。
        # 如果打开界面而没有打开监视器，就直接关闭界面，会报错。是因为没有定义Img和cap
        Img.cap.release()
        # 关闭opencv的窗口
        cv2.destroyAllWindows()
        print("控制台已退出")
        return 0


    # 把设置了的数值传入cvImg.py
    def setting(self):
        global Img
        # 判断角度设置是不是空，如果不是空则传入Img类
        if self.angle.get() != '':
            # angle是一个Entry，用".get()"可以得到其中的数据，是字符串格式的
            # int()可以把字符串强制转换成整型数
            Img.theta = int(self.angle.get())
        if self.scal_x.get() != '':
            Img.scal_x = float(self.scal_x.get())
        if self.scal_y.get() != '':
            Img.scal_y = float(self.scal_y.get())
        if self.dim.get() != '':
            Img.dim = int(self.dim.get())
        if self.deepth.get() != '':
            Img.deepth = - int(self.deepth.get())
        if self.lateral.get() != '':
            Img.lateral = int(self.lateral.get())
        if self.delay.get() != '':
            Img.delay = int(self.delay.get())
        if self.hand.get() != '':
            Img.wood = int(self.hand.get())

        # 还记得我们在cvImg.py的flow函数里有个changed吗？就是在这里用上的
        # 当我们改变参数之后，要把这个changed置1，这样才会重新计算和角度相关的值
        Img.changed = 1  # 信号检测mark

        hwb = ""

        # 根据不同的显示给hwb字符串赋值，为后面输出参数修改结果作准备
        if Img.wood == 0:
            hwb = "手"
        elif Img.wood == 1:
            hwb = "木头"
        elif Img.wood == 2:
            hwb = "空白"

        # 这一条语句就是输出的字符串，包括了所有的参数
        string = "手动设置参数已更新，转角：" + str(Img.theta) + " 左右放缩：" + str(Img.scal_x) +" 前后放缩：" + str(Img.scal_y)\
                 + " 模糊度：" + str(Img.dim) +" 前移：" + str(Img.deepth) + " 右移：" + str(Img.lateral) + "延迟：" + str(Img.delay) + " 显示：" + hwb
        self.addString(string)
        return 0


    # 修改被试的特征，逻辑和上面差不多
    def character(self):
        global Img
        if self.speed.get() != '':
            Img.mSpeed = int(self.speed.get())
        if self.color.get() != '':
            Img.fColor = int(self.color.get())

        self.addString("被试特征已更新")
        return 0


    # 这个函数的功能是为监听matlab开启一个新的线程
    def listen(self):
        self.addString("已打开MATLAB监听")
        t_matlab = threading.Thread(target=self.connectMATLAB)
        t_matlab.setDaemon(True)
        t_matlab.start()
        return 0


    # 关闭监视器，下面这几句话前面已经有类似的了
    def closeMonitor(self):
        Img.cap.release()
        cv2.destroyAllWindows()
        self.addString("监视器已关闭")
        ttk.Button(self.root, text="打开监视器", command=self.monitor).place(x=630, y=30)
        return 0


    # 打开监视器
    def monitor(self):
        global Img
        # 用“关闭监视器”按钮覆盖当前按钮
        ttk.Button(self.root, text="关闭监视器", command=self.closeMonitor).place(x=630, y=30)
        # 为图像处理新开一个线程
        t_opencv = threading.Thread(target=Img.flow)
        t_opencv.setDaemon(True)
        t_opencv.start()
        self.addString("监视器已打开")
        return 0


    # 监听matlab信号
    def connectMATLAB(self):
        global sockobj
        ttk.Button(self.root, text="结束监听", command=self.deconnect).place(x=450, y=455)
        myHost = ''
        # 用端口6000，和matlab程序上一致
        myPort = 6000
        # 接下来的和前面心电连接相似
        sockobj = socket(AF_INET, SOCK_STREAM)
        sockobj.bind((myHost, myPort))
        sockobj.listen(5)
        self.addString("等待MATLAB连接......")
        while True:
            connection, address = sockobj.accept()
            show = '已连接上这个地址：' + str(address)
            self.addString(show)
            while True:
                data = connection.recv(1024)
                data = data.decode()
                data = data.strip('\n')  # 这样才得到能用的字符串
                if len(data) > 1:
                    # 这个parse函数就是字符串处理，并把处理好的字符串传入图像处理部分
                    self.parse(data)
                if not data: break
            connection.close()


    # 断开matlab连接
    def deconnect(self):
        global sockobj
        ttk.Button(self.root, text="开始监听", command=self.connectMATLAB).place(x=450, y=455)
        sockobj.close()
        self.addString("MATLAB监听已断开")
        return 0


    # 字符串处理函数
    def parse(self, str):
        global Img, root, interval, show, angle, deepth
        self.addString(str)
        # 分割字符串，以","为分割标志，split数组存放分割好的字符串
        split = str.split(',')
        # 判断split数组的第一个元素是不是"rotate"
        if split[0] == 'rotate':
            # 此处的angle是一个全局变量，用作临时储存设定的角度
            angle = int(split[1])
            # 把split对应位置的数赋给Img类中的成员变量，后面都是类似操作
            Img.redX = int(split[3])
            Img.redY = int(split[4])
            Img.deepth = 0
            Img.changed = 1
            # 间隔时间，全局变量，临时存储
            interval = int(split[8])
            Img.showBlue = 0
            Img.showRed = 0
            # 这个时候不管怎样都显示手
            Img.wood = 0

            # 根据显示内容的选择情况，赋予show一个值
            # show是一个全局变量，用来临时存储显示的选择
            if split[6] == 'hand':
                show = 0
            elif split[6] == 'wood':
                show = 1
            elif split[6] == 'blank':
                show = 2

        # 如果split的第一个值是"deepth"
        elif split[0] == 'deepth':
            Img.theta = 0
            Img.redX = int(split[3])
            Img.redY = int(split[4])
            Img.changed = 1
            interval = int(split[8])
            Img.showBlue = 0
            Img.showRed = 0
            Img.wood = 0

            if split[6] == 'hand':
                show = 0
            elif split[6] == 'wood':
                show = 1
            elif split[6] == 'blank':
                show = 2

            # 下面这一句我是从C++的程序里抄过来的，因为我没找到deepth在哪里赋值
            if int(split[3]) == 320:
                deepth = - int(split[1])

        # showGreen时，显示蓝点，显示手
        elif split[0] == 'showGreen':
            Img.showBlue = 1


        # showRed时，各种骚活就出来了
        elif split[0] == 'showRed':  # 改变图像
            # 首先把蓝点关掉
            Img.showBlue = 0
            # 显示的内容变成之前选择好的东西，手，木头，或者空白
            Img.wood = show
            # 旋转角度
            if angle != 0:
                Img.theta = angle
            # 前后平移
            elif deepth != 0:
                Img.deepth = deepth
            # 停一小段时间，时间长度等于时间间隔
            time.sleep(interval * 0.001)
            # 显示红点
            Img.showRed = 1


        # 说是getRed，其实就是退出当前trial
        elif split[0] == 'getRed':
            Img.showBlue = 0
            Img.showRed = 0
            show = 0
            angle = 0
            deepth = 0
            Img.theta = 0
            Img.deepth = 0
            Img.wood = 3  # 显示黑屏

        elif split[0] == 'quit':
            Img.showBlue = 0
            Img.showRed = 0
            show = 0
            angle = 0
            deepth = 0
            Img.theta = 0
            Img.deepth = 0
            Img.wood = 3  # 显示黑屏

        return 0

    # 清空文本框
    def clear(self):
        self.scltxt.delete(1.0, END)  # 清空文本框
