# 建议先看main.py，再看这个
# 这里引用了好多库，包含各种引用方式，具体区别参照main.py里的注释
# 这个部分是我前四周都在做的，主要是图像处理部分

import cv2
import numpy as np
import screeninfo, threading, math, datetime, time
from scipy.linalg import solve
from scipy import stats
import queue

# 首先讲一下图像处理的整体思路：
# 我这里对图像进行的操作包括：旋转，前后左右平移，大小缩放，画面延迟，显示内容（手，木头，空白）。
# 其中图像旋转部分是最复杂的，其他的实现起来都不困难。
# 我这里是把每一帧的画面用阈值进行分割，通过二值化，得到一个背景都是黑色，有手的部分都是白色的掩模（mask）。
# 然后在旋转时让帧和掩模同时旋转。显示在屏幕上的时候，只显示掩模上白色的区域，并用帧的对应位置代替。
# 形象地理解就是，那一块板子盖在摄像机拍下来的图像上，这块板子大部分是黑色不透光的，只有图像上是手的区域，是透明能看见下面的。
# 旋转的时候就是，把这个板子和图像一起转


# 这里先对定义一个类，类名是cvImg
class cvImg:
    # 接下来开始定义这个类的成员变量和成员函数（或者称为属性和方法）
    # 当我们对类进行实例化之后，就会得到一个“对象”，这个对象就包含了以下的成员变量和成员函数。
    # 类的实例化可以进行很多次，得到很多个“对象”，每个对象都有自己单独的成员变量和成员函数。

    # 首先是存储图像用的变量，图像其实就是一个矩阵，根据每个位置的颜色不同，值也不同。根据二值化，灰度和RGB的不同，矩阵维数也不同
    # 这里要说一点，就是在索引图像某个位置的值时，是先行再列，即[row,column]，在矩阵里面感觉很自然。
    # 但是，由于x轴是屏幕横向，y轴是屏幕纵向。所以是[y,x]，请务必注意。我的代码在有些地方是使用了比较习惯的[x,y]形式。
    # 原点是屏幕左上角
    # image
    frame = np.ndarray([])  # 这个变量用来存储摄像机的每一帧画面
    canvas = np.ndarray([])  # 这个变量是最后显示在屏幕上的画面
    mask = np.ndarray([])  # 这是掩模
    monkeySkin = np.ndarray([])  # 这是存储猴子毛发颜色的，不用修补的话就用不上
    woodPic = np.ndarray([])  # 这是存储木头图像的

    # 我们能看到，画面中间是摄像机图像处理后显示的区域，我称其为白色底板，后面的东西都是“画上去的”
    leftMg = 670  # 白色底板左侧边缘到屏幕左侧边缘的距离，单位是像素
    upMg = 567  # 白色底板上边缘到屏幕上边缘的距离，单位是像素
    bkW = 640  # 白色底板宽度，单位是像素
    bkH = 300  # 白色底板长度，单位是像素

    # 旋转角度，顺时针为正，逆时针为负，旋转中心是画布底边中点
    # 另外计算几个三角函数留着备用，避免每一帧都要计算，节约运算时间
    theta = 0
    radian = 2*math.pi*theta/360  # 有些地方要用弧度
    sin = math.sin(radian)
    cos = math.cos(radian)
    tan = math.tan(radian)
    center = [bkW/2, bkH]

    # 画圆形触点，红色和蓝色。
    redX = 0  # 红点x轴坐标
    redY = 0  # 红点Y轴坐标
    showRed = 0  # 0表示不显示红点
    blueX = 310  # 蓝点x轴坐标
    blueY = 170  # 蓝点y轴坐标
    showBlue = 0  # 0表示不显示蓝点

    # 其他控制量
    dim = 1  # 模糊度，最清晰为1
    scal_x = 1  # 左右方向放缩量
    scal_y = 1  # 前后方向放缩量
    deepth = 0  # 前后移动，前负后正，和gui.py里面不一样，gui里面是为了符合人的习惯
    lateral = 0  # 左右平移，左负右正
    wood = 0  # 0显示手，1显示木头, 2显示blank，和无心率时一样， 3显示黑屏
    mSpeed = 80  # 猴子手的活动速度，数值本身无意义， 默认80
    fColor = 40  # 猴子毛发颜色深浅，数值代表灰度值, 默认40
    delay = 0  # 延迟毫秒数

    # 辅助参数
    left = []  # 在修补时用到，用来存储直线方程
    right = []  # 同上
    changed = 0  # 有数值修改则为1，无数值修改则为0，作为一个监测有无信号的参数

    # 接下来是成员函数部分，注意每个成员函数的参数区都有self，然后成员函数在使用其他成员函数或者成员变量时，要加"self."
    # 这个是构造函数，在类的实例化时进行
    def __init__(self):
        print("cvImg类已实例化")


    # 这个函数是制作掩模mask，有一个参数thresh，是灰度图二值化分割时的阈值
    def makeMask(self, thresh):
        # 首先做一个和帧一样大的白色背景
        bg_3 = 255 * np.ones((self.frame.shape[0], self.frame.shape[1], 3), dtype=np.float32)
        # 把上面的白色背景转换成灰度图，虽然看起来还是一片白，但是矩阵维数变了，白色从[255 255 255]变成了255，黑色从[0 0 0]变成了0
        bg_g = cv2.cvtColor(bg_3, cv2.COLOR_BGR2GRAY)
        # 把帧转换成灰度图
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        # 这一步很关键，把白色背景和帧的灰度图作差，那么浅色的就会因为灰度值很大而变得很小，相反猴子的手是深色的，就会变成比较浅的颜色
        hand = bg_g - gray
        # 通过设置阈值进行二值化，得到二值化图像
        _, self.mask = cv2.threshold(hand, thresh, 255, cv2.THRESH_BINARY)  # 二值化mask
        # 这个return 0其实可以不要，但是我比较习惯在没有返回值的时候写个return 0
        return 0

    # 这个函数在C++里面有内置现成的，但是python没有，所以只好自己写，顺便再做一点修改
    # 它的功能是把一个图像放到另一个图像上面去
    def copyTo(self):
        # 按照显示的是手还是木头进行分类
        # 首先显示手
        if self.wood == 0:
            # 得到mask中所有不是黑色的点的坐标，黑色的点灰度值是0
            # 本来是要把mask遍历一遍的，但那样太费时间了，用一个np.where就能搞定，非常舒服，后面还有很多这种骚操作
            locs = np.where(self.mask != 0)  # Get the non-zero mask locations
            # 下面两行是在往屏幕上画图时，起点的位置，因为不是从屏幕左上角开始画的，而是画在白色底板上的
            # 本来是只要考虑上边距和左边距就可以了，但是由于要考虑缩放时的问题，所以把缩放量也要考虑进去
            y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
            x_offset = self.leftMg + int((self.bkW - self.frame.shape[1])/2) + 1
            # 这一步是在屏幕上画出处理后的帧
            # deepth和lateral分别是前后平移和左右平移的量，平移的操作是在这里进行的。
            # 因为在这里处理，所以如果平移量过大，会出现矩阵访问越界的问题
            self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.frame[locs[0], locs[1]]/255

        # 接下来显示木头
        if self.wood == 1:
            # 显示木头其实和显示手的方法相似，但是掩模mask不同
            # 这里我把手的图像模糊化，就不能看出手指分界了，并且边缘颜色会比中间浅。
            self.frame = cv2.blur(self.frame, (80, 80))
            # 用模糊化的图像重新制作掩模。还记得这个函数的参数的意义吗？是二值化阈值
            # 那么我们在对模糊了的手图像进行阈值二值化操作时，随着阈值的选区，mask白色的部分大小就会变。有点像把假山放在池塘里面，水位上升露出的部分就会变少。
            self.makeMask(50)  # 这个参数会影响木头的大小
            # 后面三行是一样的
            locs = np.where(self.mask != 0)  # Get the non-zero mask locations
            y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
            x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1
            # 等号右边是woodpic不是frame，因为要用木头图片填充
            self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.woodPic[locs[0], locs[1]] / 255
        return 0

    # 旋转手的函数，这个函数是再调用别的函数来执行旋转操作，我写这个函数的原因主要是想让代码简洁一点，一步一步看得清楚
    def handRotate(self, angle):
        # 这个rotate函数也是自己写的
        self.frame = self.rotate(self.frame, angle, (255, 255, 255))
        self.mask = self.rotate(self.mask, angle, (0, 0, 0))
        return 0


    # 这个是真正执行旋转的函数
    # 先看一下参数表，第一个image是传入的图像矩阵，第二个angle是旋转角度（顺时针为正），第三个bV是旋转后背景的填充色，
    # 第四个center是旋转中心，默认为白色底板底边中点。
    def rotate(self, image, angle, bV, center=None, scale=1.0):
        # 获取图像尺寸，h是高，w是宽。image.shape的返回值有三个，前两个是我们需要的
        (h, w) = image.shape[:2]

        # 若未指定旋转中心，则将底边中点设为旋转中心
        if center is None:
            center = (w / 2, h)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # borderValue=bV是指用一个指定颜色填充背景
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=bV)
        # rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated


    # 这是在修补时用到的，求一条直线和另一条直线的交点
    def intersection(self, hand):
        hand_xy = [-hand[0], 1]
        side_xy = [-self.tan, 1]
        xy_mat = np.array([hand_xy, side_xy])
        b_mat = np.array([hand[1], self.center[1] - self.tan * self.center[0]])
        return solve(xy_mat, b_mat)


    # 这也是在修补时用的，把计算好的点，旋转要给角度
    # 交点绕旋转中心旋转, sin负号, 顺时针是负夹角
    def point_rot(self, A):
        ax = A[0]
        ay = A[1]
        ox = self.center[0]
        oy = self.center[1]
        bx = int((ax - ox) * self.cos - (ay - oy) * -self.sin + ox)
        by = int((ax - ox) * -self.sin + (ay - oy) * self.cos + oy)
        return (bx + 20, by)


    # 修补时用，把猴子毛发图片修补上去，和copyTo的原理相似
    def makeUpMky(self, box):
        makeup_mask = np.zeros((self.bkH, self.bkW, 3), dtype=np.float32)
        cv2.fillPoly(makeup_mask, box, (255, 255, 255))  # 填充修补区域为白色
        locs = np.where(makeup_mask != 0)  # Get the non-zero mask locations
        y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1 # +1去白边
        x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1  # +1去白边
        self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.monkeySkin[locs[0], locs[1]] / 255
        return 0


    # 修补时用，把木头图片修补上去，和copyTo原理相似
    def makeUpWd(self, box):
        makeup_mask = np.zeros((self.bkH, self.bkW, 3), dtype=np.float32)
        cv2.fillPoly(makeup_mask, box, (255, 255, 255))  # 填充修补区域为白色
        locs = np.where(makeup_mask != 0)  # Get the non-zero mask locations
        y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
        x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1  # +1去白边
        self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.woodPic[locs[0], locs[1]] / 255
        return 0


    # 拟合直线时用，包括取点，拟合
    def linefit(self):
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for i in range(0, 10):  # 这个范围越大，理论上拟合效果越好
            left_x.append(np.where(self.mask[self.mask.shape[0] - 1 - 2 * i] != 0)[0][0])
            left_y.append(self.mask.shape[0] - 1 - 4 * i)
            right_x.append(np.where(self.mask[self.mask.shape[0] - 1 - 2 * i] != 0)[0][-1])
            right_y.append(self.mask.shape[0] - 1 - 4 * i)

        # 预处理拟合点
        left_y.pop(left_x.index(min(left_x)))
        left_x.remove(min(left_x))
        left_y.pop(left_x.index(max(left_x)))
        left_x.remove(max(left_x))
        right_y.pop(right_x.index(min(right_x)))
        right_x.remove(min(right_x))
        right_y.pop(right_x.index(max(right_x)))
        right_x.remove(max(right_x))

        # 求出拟合直线解析式[k, b]
        # left = np.polyfit(left_x, left_y, 1)
        self.left = stats.linregress(left_x, left_y)[0:2]
        # right = np.polyfit(right_x, right_y, 1)
        self.right = stats.linregress(right_x, right_y)[0:2]
        return 0


    # 这是图像处理的主要流程，我取名flow就是因为，这个是对“帧流”进行处理，颇像流水线工作
    def flow(self):
        # 选择一个显示器，编号是0号，1号，2号这个顺序
        screen_id = 1
        is_color = False
        # 选中这个这个id的显示器，得到这个显示器的尺寸信息
        screen = screeninfo.get_monitors()[screen_id]
        # 得到显示器长度和宽度，尺寸单位是像素
        width, height = screen.width, screen.height
        # 命名新窗口名字
        window_name = 'MonkeyView'
        # 定义新窗口的信息，下面三行包括了全屏，把屏幕移动到指定位置
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 创建延迟队列，队列的特点是先进先出。延时的原理是：
        # 把处理好的每一帧放进一个队列，从队列被塞满开始。每取出一帧就再放入一帧。如此反复操作。
        # 这里的计算时间我写的是20ms，我记得差不多是这个时间，你们最好根据实际每一帧处理的时间修改一下
        q = queue.Queue(maxsize=int(self.delay/20))  # 队列长度等于延迟除以每一帧的计算时间

        # 用来修补，把猴子的毛发颜色存储下来，给修补的时候用
        self.monkeySkin = cv2.imread("D:\\Intern\\cvImgPy\\pics\\nbk.png")  # 猴子毛发提取
        # self.monkeySkin = cv2.medianBlur(self.monkeySkin, 21)  # 去噪 消除边缘

        # 木头图片有两个功能，第一个是显示木头的时候代替手出现，还有显示木头的时候修补也要用到
        self.woodPic = cv2.imread("D:\\Intern\\cvImgPy\\pics\\wood.png")  # 木头图片
        # 选择0号摄像头
        self.cap = cv2.VideoCapture(0)  # 从摄像头读取
        # self.cap = cv2.VideoCapture('D:\\Intern\\cvImgPy\\pics\\real.MOV')  # 选择摄像头，把视频读取删除

        # 当摄像头开启时，程序一直循环
        while self.cap.isOpened():
            # 这行是算每每帧的运算时间用到的，作为开始时间
            starttime = datetime.datetime.now()  # 开始计算
            # 从摄像头读取当前帧，保存在frame。观察到这里frame前面有个"self."。如果不加，那么frame就会成为这个flow函数的局部变量
            # 不能是cvImg类的一个成员变量
            ret, self.frame = self.cap.read()
            # 判断读取来的帧是不是有效，是不是空的
            if isinstance(self.frame, np.ndarray):  # 因为从某一帧开始，读取的视频就不是ndarray格式的了，导致报错，所以加一个判断

                # 剪切图像，[0:293, 0:640]，这个东西的意思就是frame矩阵的0到293行，0到640列。
                # 那么在图像上，就表现为从左上角顶点开始，拉一个长640，宽293的矩形区域
                self.frame = self.frame[0:293, 0:640]
                # 下面这一句非常牛逼，可以直接把矩阵左右翻转，速度超级快，应该是内置有函数。比我写两个for循环一个个调换数字，快太多了。
                self.frame = self.frame[:, ::-1]  # 左右翻转镜像

                # 修补时用，用中值去噪把椒盐噪声去除掉，改善后面修补的效果
                # self.frame = cv2.medianBlur(self.frame, 5)

                # 刷新背景，第一句是创建全黑的背景，其实就是生成一个矩阵，长宽和屏幕尺寸一样，"3"表示使用RGB颜色，在opencv中实际上是BGR
                # 第二句是创建一个矩形，我一个个参数来讲解
                # 第一个是选择背景，这里选的就是canvas；第二个是矩形左上角点的坐标，先x后y；第三个是矩形右下角的坐标；第四个是填充颜色，1的意思其实是"255/255"
                # 想要别的颜色就改成比如"164/255"；第五个是填充选项，-1的意思是填充
                self.canvas = np.zeros((height, width, 3), dtype=np.float32)
                cv2.rectangle(self.canvas, (self.leftMg, self.upMg), (self.leftMg + self.bkW, self.upMg + self.bkH), (1, 1, 1), -1)

                # 判断角度有无改变，有改变的话重新计算一遍角度相关参数
                if self.changed == 1:
                    self.radian = 2 * math.pi * self.theta / 360
                    self.sin = math.sin(self.radian)
                    self.cos = math.cos(self.radian)
                    self.tan = math.tan(self.radian)
                    self.changed = 0

                # 尺寸变换，我来逐一讲解每个参数的意义
                # 第一个是原图像；第二个是变换后的尺寸（长，宽）；第三个是拉伸的填充算法，我这边就选的线性填充
                self.frame = cv2.resize(self.frame, (int(self.bkW * self.scal_x), int(self.bkH * self.scal_y)),
                                        interpolation=cv2.INTER_LINEAR)

                # 清晰度变换,均值去噪
                # 这个去噪起到的效果是让图像变模糊。第一个参数的意义是原图像。
                # 第二个参数的意义是模糊的程度，实际上是使用一个(x,y)的模板来进行模糊操作。
                self.frame = cv2.blur(self.frame, (self.dim, self.dim))

                # 定义旋转中心，在frame上的位置是底边中点。
                # frame.shape会返回一个三个元素的数组，其中第一个是行数，第二个是列数，第三个是颜色通道数量
                self.center = [0.5 * self.frame.shape[1], self.frame.shape[0]]  # 先x后y, mask坐标系

                # 制作掩膜，参数使用的是猴子毛发的颜色作为阈值
                self.makeMask(self.fColor)

                # 从这里开始，根据wood（显示）的取值不同，进行不同的处理
                # 首先是wood取2时，显示白色背景，白色背景前面已经画好了，所以if里面就没有再画
                # 注意到if判断里面还有个or，这个or后面的东西作用是判断图像里面是不是有手，没有手的话有些处理会出问题
                # 判断的逻辑是看制作好的掩模mask里面靠近底边的一行，有没有超过20个白色的像素点，如果没有，则显示全白
                if self.wood == 2 or len(np.where(self.mask[270] != 0)[0]) < 20:
                    # 这里是画圆点，根据showRed的取值情况来判断是否画点。这个showRed是会根据matlab传入的信号变化的
                    if self.showRed == 1:
                        # 画点的函数，第一个参数是背景，第二个参数是点的位置，先x后y，第三个参数是半径，第四个参数是BGR颜色，第五个参数是填充选项
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)

                    # 画蓝点的方法和红点一样
                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.blueX, self.upMg + self.blueY), 15, (255, 0, 0), -1)

                # wood取3，这时显示全黑的背景
                if self.wood == 3:
                    cv2.rectangle(self.canvas, (self.leftMg, self.upMg), (self.leftMg + self.bkW, self.upMg + self.bkH),
                                  (0, 0, 0), -1)

                # wood取0，这时显示手
                elif self.wood == 0:
                    # 旋转变换
                    # 调用前面的函数，注意调用自己的成员函数，前面要加"self."
                    self.handRotate(self.theta)

                    # 调用成员函数把结果画在背景上
                    self.copyTo()

                    # 下面是画圆点
                    if self.showRed == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)

                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.blueX, self.upMg + self.blueY), 15, (255, 0, 0), -1)

                # wood取1，显示木头
                elif self.wood == 1:
                    # 和手的情况基本上一样
                    # 旋转变换
                    self.handRotate(self.theta)

                    # 这个地方的成员函数操作和显示手不一样，前面也已经写了
                    self.copyTo()

                    # 画圆点
                    if self.showRed == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)
                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.blueX, self.blueY), 15, (255, 0, 0), -1)

                # 接下来是延迟操作，用到了“队列”这样一种数据结构
                # 延迟,队列真美妙
                # 首先是判断延迟时间是不是零，如果是零的话就直接显示处理好的图像
                if self.delay != 0:
                    # 如果delay不是0，那么判断队列是不是满了
                    if q.full():
                        # 如果满了，那就从队头取出一个帧，显示在屏幕上
                        cv2.imshow(window_name, q.get())
                        # 然后再把当前帧从队尾放入
                        q.put(self.canvas)
                    else:
                        # 如果队列没有满，那么把当前帧放入队列
                        q.put(self.canvas)
                else:
                    # 如果延时是0，那么直接显示处理好的当前帧
                    cv2.imshow(window_name, self.canvas)

                # 下面这两行，和之前的starttime是一起用来计算运算时间的，你们算延时的时候最好先算一下现在每一帧处理多少时间
                # interval = datetime.datetime.now() - starttime
                # print(interval)
            else:
                break
            # 设置一个退出键“Q”
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                break

        # 循环结束后，释放摄像头
        self.cap.release()
        # 关闭所有opencv的窗口
        cv2.destroyAllWindows()
