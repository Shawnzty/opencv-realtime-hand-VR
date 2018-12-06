from tkinter import scrolledtext
from tkinter import ttk
from tkinter import *
from cvImg import *
from socket import *
import threading
import cv2

global sockobj, interval, Img, show, angle, deepth

class gui:
    def __init__(self, root):
        self.root = root
        self.root.title("Opencv 控制台")
        self.root.geometry('870x500')

        # 左侧文本框
        self.scltxt = scrolledtext.ScrolledText(root, width=60, height=38, wrap=WORD)
        self.scltxt.place(x=0, y=0)

        # 初始化
        ttk.Button(root, text="初始化", command=self.init).place(x=450, y=30)
        ttk.Button(root, text="打开监视器", command=self.monitor).place(x=630, y=30)
        ttk.Button(root, text="心电开", command=self.ECG).place(x=760, y=30)

        ttk.Label(root, text="手动设置", font=('16')).place(x=450, y=70)
        # 输入旋转
        ttk.Label(root, text="转角：（顺时针为正）").place(x=450, y=100)
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
        self.hand = IntVar()
        self.hand.set(0)  # 默认是手
        ttk.Label(root, text="显示：").place(x=450, y=275)
        self.hand_slt = Radiobutton(root, text='手', variable=self.hand, value=0)  # , command=handset)
        self.hand_slt.place(x=560, y=274)
        self.wood_slt = Radiobutton(root, text='木头', variable=self.hand, value=1)  # , command=handset)
        self.wood_slt.place(x=609, y=274)
        self.wood_slt = Radiobutton(root, text='空白', variable=self.hand, value=2)  # , command=handset)
        self.wood_slt.place(x=670, y=274)

        # 设置延时时间
        ttk.Label(root, text="延时：（毫秒）").place(x=450, y=250)
        self.delay = ttk.Entry(root, width=12)
        self.delay.place(x=630, y=250)

        # 确认，把修改值传递到cvImg
        set = ttk.Button(master=root, text="确认", width=10, command=self.setting)
        set.place(x=760, y=99, relheight=0.39, relwidth=0.1)

        # 猴子设定
        ttk.Label(root, text="被试特征", font='16').place(x=450, y=325)

        # 默认80，数字本身无物理意义，越小越慢，越大越快
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


    def init(self):
        global Img
        self.scltxt.delete(1.0, END)  # 清空文本框
        Img = cvImg()  # 类实例化
        self.addString("cvImg已实例化")
        return 0


    def addString(self, string):
        self.scltxt.insert(INSERT, string + '\n')


    def ECG(self):
        ttk.Button(self.root, text="心电关", command=self.noECG).place(x=760, y=30)
        self.addString("心电已启用")
        t_ECG = threading.Thread(target=self.connectECG)
        t_ECG.setDaemon(True)
        t_ECG.start()


    def connectECG(self):
        myHost2 = ''
        myPort2 = 6001  # 定义一个和matlab不同的接口
        sockobj2 = socket(AF_INET, SOCK_STREAM)
        sockobj2.bind((myHost2, myPort2))
        sockobj2.listen(5)
        self.addString("等待心电连接......")
        while True:
            connection2, address2 = sockobj2.accept()
            show = '已连接上这个地址：' + str(address2)
            self.addString(show)
            while True:
                data = connection2.recv(1024)
                data = data.decode()
                data = data.strip('\n')  # 这样才得到能用的字符串
                # 之后处理出0和1，我这里假设直接得到0和1
                if data == 0:
                    Img.wood = 2  # 看不见手
                elif data == 1:
                    Img.wood = 0  # 看得见手
                if not data: break
            connection2.close()


    def noECG(self):
        ttk.Button(self.root, text="心电开", command=self.ECG).place(x=760, y=30)

        self.addString("心电已断开")
        return 0


    def _quit(self):
        self.root.quit()
        self.root.destroy()
        Img.cap.release()
        cv2.destroyAllWindows()
        print("控制台已退出")
        return 0


    def setting(self):
        global Img
        if self.angle.get() != '':
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

        Img.changed = 1  # 信号检测mark

        hwb = ""

        if Img.wood == 0:
            hwb = "手"
        elif Img.wood == 1:
            hwb = "木头"
        elif Img.wood == 2:
            hwb = "空白"

        string = "手动设置参数已更新，转角：" + str(Img.theta) + " 左右放缩：" + str(Img.scal_x) +" 前后放缩：" + str(Img.scal_y)\
                 + " 模糊度：" + str(Img.dim) +" 前移：" + str(Img.deepth) + " 右移：" + str(Img.lateral) + "延迟：" + str(Img.delay) + " 显示：" + hwb
        self.addString(string)
        return 0


    def character(self):
        global Img
        if self.speed.get() != '':
            Img.mSpeed = int(self.speed.get())
        if self.color.get() != '':
            Img.fColor = int(self.color.get())

        self.addString("被试特征已更新")
        return 0


    def listen(self):
        self.addString("已打开MATLAB监听")
        t_matlab = threading.Thread(target=self.connectMATLAB)
        t_matlab.setDaemon(True)
        t_matlab.start()
        return 0


    def closeMonitor(self):
        Img.cap.release()
        cv2.destroyAllWindows()
        self.addString("监视器已关闭")
        ttk.Button(self.root, text="打开监视器", command=self.monitor).place(x=630, y=30)
        return 0


    def monitor(self):
        global Img
        ttk.Button(self.root, text="关闭监视器", command=self.closeMonitor).place(x=630, y=30)
        t_opencv = threading.Thread(target=Img.flow)
        t_opencv.setDaemon(True)
        t_opencv.start()
        self.addString("监视器已打开")
        return 0


    def connectMATLAB(self):
        global sockobj
        ttk.Button(self.root, text="结束监听", command=self.deconnect).place(x=450, y=455)
        myHost = ''
        myPort = 6000
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
                    self.parse(data)
                if not data: break
            connection.close()


    def deconnect(self):
        global sockobj
        ttk.Button(self.root, text="开始监听", command=self.connectMATLAB).place(x=450, y=455)
        sockobj.close()
        self.addString("MATLAB监听已断开")
        return 0


    def parse(self, str):
        global Img, root, interval, show, angle, deepth
        self.addString(str)
        split = str.split(',')
        if split[0] == 'rotate':
            angle = int(split[1])
            Img.redX = int(split[3])
            Img.redY = int(split[4])
            Img.deepth = 0
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

            if int(split[3]) == 320:
                deepth = - int(split[1])


        elif split[0] == 'showGreen':
            Img.showBlue = 1


        elif split[0] == 'showRed':  # 改变图像
            Img.showBlue = 0
            Img.wood = show
            if angle != 0:
                Img.theta = angle
            elif deepth != 0:
                Img.deepth = deepth
            time.sleep(interval * 0.001)
            Img.showRed = 1

        elif split[0] == 'getRed':
            Img.showBlue = 0
            Img.showRed = 0
            show = 0
            angle = 0
            deepth = 0
            Img.theta = 0
            Img.deepth = 0
            Img.wood = 3  # 显示黑

        elif split[0] == 'quit':
            Img.showBlue = 0
            Img.showRed = 0
            show = 0
            angle = 0
            deepth = 0
            Img.theta = 0
            Img.deepth = 0
            Img.wood = 3  # 显示黑

        return 0


    def clear(self):
        self.scltxt.delete(1.0, END)  # 清空文本框
