import cv2
import numpy as np
import screeninfo, threading, math, datetime, time
from scipy.linalg import solve
from scipy import stats
import queue


class cvImg:
    # image
    frame = np.ndarray([])
    canvas = np.ndarray([])
    mask = np.ndarray([])
    monkeySkin = np.ndarray([])
    woodPic = np.ndarray([])

    # 画布尺寸
    leftMg = 670  # 左边距
    upMg = 567  # 上边距
    bkW = 640  # 白色底板宽度
    bkH = 300  # 白色底板长度

    # 旋转角度，顺时针为正，逆时针为负，旋转中心是画布底边中点
    theta = 0
    radian = 2*math.pi*theta/360
    sin = math.sin(radian)
    cos = math.cos(radian)
    tan = math.tan(radian)
    center = [bkW/2, bkH]

    # 画圆饼
    redX = 0
    redY = 0
    showRed = 0  # 0表示不蓝点红点
    blueX = 310
    blueY = 170
    showBlue = 0  # 0表示不显示蓝点

    # 其他控制量
    dim = 1  # 模糊度，最清晰为1
    scal_x = 1  # 左右方向放缩量
    scal_y = 1  # 前后方向放缩量
    deepth = 0  # 前后移动，前负后正
    lateral = 0  # 左右平移，左负右正
    wood = 0  # 0显示手，1显示木头, 2显示blank，和无心率时一样
    mSpeed = 80  # 猴子手的活动速度，数值本身无意义， 默认80
    fColor = 40  # 猴子毛发颜色深浅，数值代表灰度值, 默认40
    delay = 0  # 延迟毫秒数

    # 辅助参数
    left = []
    right = []
    changed = 0  # 有数值修改则为1，无数值修改则为0，作为一个监测有无信号的参数

    def __init__(self):
        print("cvImg类已实例化")


    def makeMask(self, thresh):
        # mask制作
        bg_3 = 255 * np.ones((self.frame.shape[0], self.frame.shape[1], 3), dtype=np.float32)
        bg_g = cv2.cvtColor(bg_3, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        hand = bg_g - gray
        _, self.mask = cv2.threshold(hand, thresh, 255, cv2.THRESH_BINARY)  # 二值化mask
        # print(mask[mask.shape[0]-1])
        return 0


    def copyTo(self):
        if self.wood == 0:
            locs = np.where(self.mask != 0)  # Get the non-zero mask locations
            y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
            x_offset = self.leftMg + int((self.bkW - self.frame.shape[1])/2) + 1
            self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.frame[locs[0], locs[1]]/255

        if self.wood == 1:
            self.frame = cv2.blur(self.frame, (80, 80))
            self.makeMask(50)  # 这个参数会影响木头的大小
            locs = np.where(self.mask != 0)  # Get the non-zero mask locations
            y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
            x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1
            self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.woodPic[locs[0], locs[1]] / 255
        return 0


    def handRotate(self, angle):
        self.frame = self.rotate(self.frame, angle, (255, 255, 255))
        self.mask = self.rotate(self.mask, angle, (0, 0, 0))
        return 0


    # 旋转且裁剪，不会缩放
    def rotate(self, image, angle, bV, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]

        # 若未指定旋转中心，则将底边中点设为旋转中心
        if center is None:
            center = (w / 2, h)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=bV)
        # rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated


    #求交点
    def intersection(self, hand):
        hand_xy = [-hand[0], 1]
        side_xy = [-self.tan, 1]
        xy_mat = np.array([hand_xy, side_xy])
        b_mat = np.array([hand[1], self.center[1] - self.tan * self.center[0]])
        return solve(xy_mat, b_mat)


    # 交点绕旋转中心旋转, sin负号, 顺时针是负夹角
    def point_rot(self, A):
        ax = A[0]
        ay = A[1]
        ox = self.center[0]
        oy = self.center[1]
        bx = int((ax - ox) * self.cos - (ay - oy) * -self.sin + ox)
        by = int((ax - ox) * -self.sin + (ay - oy) * self.cos + oy)
        return (bx + 20, by)


    def makeUpMky(self, box):
        makeup_mask = np.zeros((self.bkH, self.bkW, 3), dtype=np.float32)
        cv2.fillPoly(makeup_mask, box, (255, 255, 255))  # 填充修补区域为白色
        locs = np.where(makeup_mask != 0)  # Get the non-zero mask locations
        y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1 # +1去白边
        x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1  # +1去白边
        self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.monkeySkin[locs[0], locs[1]] / 255
        return 0


    def makeUpWd(self, box):
        makeup_mask = np.zeros((self.bkH, self.bkW, 3), dtype=np.float32)
        cv2.fillPoly(makeup_mask, box, (255, 255, 255))  # 填充修补区域为白色
        locs = np.where(makeup_mask != 0)  # Get the non-zero mask locations
        y_offset = self.upMg + (self.bkH - self.frame.shape[0]) + 1  # +1去白边
        x_offset = self.leftMg + int((self.bkW - self.frame.shape[1]) / 2) + 1  # +1去白边
        self.canvas[y_offset + self.deepth + locs[0], x_offset + self.lateral + locs[1]] = self.woodPic[locs[0], locs[1]] / 255
        return 0


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

    def flow(self):
        screen_id = 1
        is_color = False
        # get the size of the screen
        screen = screeninfo.get_monitors()[screen_id]
        width, height = screen.width, screen.height
        window_name = 'MonkeyView'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 创建延迟队列
        q = queue.Queue(maxsize=int(self.delay/20))  # 队列长度等于延迟除以每一帧的计算时间

        self.monkeySkin = cv2.imread("D:\\Intern\\cvImgPy\\pics\\nbk.png")  # 猴子毛发提取
        # self.monkeySkin = cv2.medianBlur(self.monkeySkin, 21)  # 去噪 消除边缘

        self.woodPic = cv2.imread("D:\\Intern\\cvImgPy\\pics\\wood.png")  # 木头图片
        self.cap = cv2.VideoCapture(0)  # 从摄像头读取
        # self.cap = cv2.VideoCapture('D:\\Intern\\cvImgPy\\pics\\real.MOV')  # 选择摄像头，把视频读取删除


        while self.cap.isOpened():
            starttime = datetime.datetime.now()  # 开始计算
            ret, self.frame = self.cap.read()
            if isinstance(self.frame, np.ndarray):  # 因为从某一帧开始，读取的视频就不是ndarray格式的了，导致报错，所以加一个判断

                # 剪切图像
                self.frame = self.frame[0:293, 0:640]
                self.frame = self.frame[:, ::-1]  # 左右翻转镜像

                # 中值去噪，真去噪
                # self.frame = cv2.medianBlur(self.frame, 5)

                # 刷新背景
                self.canvas = np.zeros((height, width, 3), dtype=np.float32)
                cv2.rectangle(self.canvas, (self.leftMg, self.upMg), (self.leftMg + self.bkW, self.upMg + self.bkH), (1, 1, 1), -1)

                # 判断角度有无改变
                if self.changed == 1:
                    self.radian = 2 * math.pi * self.theta / 360
                    self.sin = math.sin(self.radian)
                    self.cos = math.cos(self.radian)
                    self.tan = math.tan(self.radian)
                    self.changed = 0

                # 尺寸变换
                self.frame = cv2.resize(self.frame, (int(self.bkW * self.scal_x), int(self.bkH * self.scal_y)),
                                        interpolation=cv2.INTER_LINEAR)
                # 清晰度变换,均值去噪
                self.frame = cv2.blur(self.frame, (self.dim, self.dim))

                # 开始修补
                self.center = [0.5 * self.frame.shape[1], self.frame.shape[0]]  # 先x后y, mask坐标系

                # 制作掩膜
                self.makeMask(self.fColor)

                # 如果没有手，也空白
                # 开始按照手和木头进行分类
                # 显示空白
                if self.wood == 2 or len(np.where(self.mask[270] != 0)[0]) < 20:
                    if self.showRed == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)

                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.blueX, self.upMg + self.blueY), 15, (255, 0, 0), -1)

                if self.wood == 3:
                    cv2.rectangle(self.canvas, (self.leftMg, self.upMg), (self.leftMg + self.bkW, self.upMg + self.bkH),
                                  (0, 0, 0), -1)

                # 显示手
                elif self.wood == 0:
                    # start = int(np.where(self.mask[self.mask.shape[0] - 1] != 0)[0][0])
                    # end = int(np.where(self.mask[self.mask.shape[0] - 1] != 0)[0][-1])
                    #
                    # # 拟合直线
                    # self.linefit()
                    #
                    # self.monkeySkin = self.monkeySkin[0:self.frame.shape[0], 0:self.frame.shape[1]]
                    #
                    # # 分旋转方向，计算直线和交点
                    # if self.theta > 0:
                    #     # 根据手的左中右位置，修补区域不同
                    #     if start > 0.5 * self.bkW:
                    #         # 没有加斜率不存在的判断，考虑到拟合时有20个点
                    #         # 求交点
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         # 求交点旋转后的位置
                    #         lp_rot = self.point_rot(left_point)
                    #         rp_rot = self.point_rot(right_point)
                    #         # 会出问题
                    #         # if rp_rot[0] < 0:
                    #         #     print(right_x)
                    #         #     print(right)
                    #
                    #         # 绘制四边形
                    #         start_point = [self.center[0] + int((start - self.center[0]) * self.cos), self.center[1] - int((start - self.center[0]) * self.sin)]  # 先x后y
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos), self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理
                    #         if 'previous_1' in locals().keys():
                    #             if rp_rot[0] < 0 or abs(rp_rot[0] - self.previous_1[0]) > 80:  # 根据猴子手的速度不同进行调节，这地方要有调参的
                    #                 rp_rot = self.previous_1
                    #             else:
                    #                 self.previous_1 = rp_rot
                    #         else:
                    #             self.previous_1 = rp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, rp_rot, end_point]], dtype = np.int32)
                    #
                    #         self.makeUpMky(box)
                    #
                    #
                    #     elif start <= 0.5 * self.bkW and end >= 0.5 * self.bkW:
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         rp_rot = self.point_rot(right_point)
                    #
                    #         start_point = self.center
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos), self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理，暂时没什么要处理的，先留着
                    #         if 'previous_2' in locals().keys():
                    #             if rp_rot[0] < 0:
                    #                 rp_rot = self.previous_2
                    #             else:
                    #                 self.previous_2 = rp_rot
                    #         else:
                    #             self.previous_2 = rp_rot
                    #
                    #         box = np.array([[start_point, rp_rot, end_point]], dtype=np.int32)
                    #         # 画三角形
                    #         self.makeUpMky(box)
                    #
                    # if self.theta < 0:
                    #     if end < 0.5 * self.bkW:
                    #
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         lp_rot = self.point_rot(left_point)
                    #         rp_rot = self.point_rot(right_point)
                    #
                    #         start_point = [self.center[0] + int((start - self.center[0]) * self.cos), self.center[1] - int((start - self.center[0]) * self.sin)]  # 先x后y
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos), self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理
                    #         if 'previous_3' in locals().keys():
                    #             if rp_rot[0] < 0 or abs(rp_rot[0] - self.previous_3[0]) > 80:  # 根据猴子手的速度不同进行调节，这地方要有调参的
                    #                 rp_rot = self.previous_3
                    #             else:
                    #                 self.previous_3 = rp_rot
                    #         else:
                    #             self.previous_3 = rp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, rp_rot, end_point]], dtype=np.int32)
                    #         self.makeUpMky(box)
                    #
                    #     elif start < 0.5 * self.bkW and end > 0.5 * self.bkW:
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         lp_rot = self.point_rot(left_point)
                    #
                    #         start_point = self.center
                    #         end_point = [self.center[0] + int((start - self.center[0]) * self.cos), self.center[1] - int((start - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理，暂时没什么要处理的，先留着
                    #         if 'previous_4' in locals().keys():
                    #             if lp_rot[0] < 0:
                    #                 lp_rot = self.previous_4
                    #             else:
                    #                 self.previous_4 = lp_rot
                    #         else:
                    #             self.previous_4 = lp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, end_point]], dtype=np.int32)
                    #         # 画三角形
                    #         self.makeUpMky(box)
                    # # 到此修补完毕

                    # 旋转变换
                    self.handRotate(self.theta)

                    self.copyTo()

                    if self.showRed == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)

                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.blueX, self.upMg + self.blueY), 15, (255, 0, 0), -1)


                # 显示木头
                elif self.wood == 1:
                    # start = int(np.where(self.mask[self.mask.shape[0] - 1] != 0)[0][0])
                    # end = int(np.where(self.mask[self.mask.shape[0] - 1] != 0)[0][-1])
                    #
                    # # 拟合直线
                    # self.linefit()
                    #
                    # self.woodPic = self.woodPic[0:self.frame.shape[0], 0:self.frame.shape[1]]
                    #
                    # # 分旋转方向，计算直线和交点
                    # if self.theta > 0:
                    #     # 根据手的左中右位置，修补区域不同
                    #     if start > 0.5 * self.bkW:
                    #         # 没有加斜率不存在的判断，考虑到拟合时有20个点
                    #         # 求交点
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         # 求交点旋转后的位置
                    #         lp_rot = self.point_rot(left_point)
                    #         rp_rot = self.point_rot(right_point)
                    #         # 会出问题
                    #         # if rp_rot[0] < 0:
                    #         #     print(right_x)
                    #         #     print(right)
                    #
                    #         # 绘制四边形
                    #         start_point = [self.center[0] + int((start - self.center[0]) * self.cos),
                    #                        self.center[1] - int((start - self.center[0]) * self.sin)]  # 先x后y
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos),
                    #                      self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理
                    #         if 'previous_1' in locals().keys():
                    #             if rp_rot[0] < 0 or abs(rp_rot[0] - self.previous_1[0]) > 80:  # 根据猴子手的速度不同进行调节，这地方要有调参的
                    #                 rp_rot = self.previous_1
                    #             else:
                    #                 self.previous_1 = rp_rot
                    #         else:
                    #             self.previous_1 = rp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, rp_rot, end_point]], dtype=np.int32)
                    #
                    #         self.makeUpMky(box)
                    #
                    #
                    #     elif start <= 0.5 * self.bkW and end >= 0.5 * self.bkW:
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         rp_rot = self.point_rot(right_point)
                    #
                    #         start_point = self.center
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos),
                    #                      self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理，暂时没什么要处理的，先留着
                    #         if 'previous_2' in locals().keys():
                    #             if rp_rot[0] < 0:
                    #                 rp_rot = self.previous_2
                    #             else:
                    #                 self.previous_2 = rp_rot
                    #         else:
                    #             self.previous_2 = rp_rot
                    #
                    #         box = np.array([[start_point, rp_rot, end_point]], dtype=np.int32)
                    #         # 画三角形
                    #         self.makeUpMky(box)
                    #
                    # if self.theta < 0:
                    #     if end < 0.5 * self.bkW:
                    #
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         if self.right[0] == self.right[0]:
                    #             right_point = self.intersection(self.right)
                    #         else:
                    #             right_point = [end, self.center[1] + int(end * self.tan)]
                    #
                    #         lp_rot = self.point_rot(left_point)
                    #         rp_rot = self.point_rot(right_point)
                    #
                    #         start_point = [self.center[0] + int((start - self.center[0]) * self.cos),
                    #                        self.center[1] - int((start - self.center[0]) * self.sin)]  # 先x后y
                    #         end_point = [self.center[0] + int((end - self.center[0]) * self.cos),
                    #                      self.center[1] - int((end - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理
                    #         if 'previous_3' in locals().keys():
                    #             if rp_rot[0] < 0 or abs(rp_rot[0] - self.previous_3[0]) > 80:  # 根据猴子手的速度不同进行调节，这地方要有调参的
                    #                 rp_rot = self.previous_3
                    #             else:
                    #                 self.previous_3 = rp_rot
                    #         else:
                    #             self.previous_3 = rp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, rp_rot, end_point]], dtype=np.int32)
                    #         self.makeUpMky(box)
                    #
                    #     elif start < 0.5 * self.bkW and end > 0.5 * self.bkW:
                    #         if self.left[0] == self.left[0]:  # 判断前面算的斜率是不是NaN
                    #             left_point = self.intersection(self.left)
                    #         else:
                    #             left_point = [start, self.center[1] + int(start * self.tan)]
                    #
                    #         lp_rot = self.point_rot(left_point)
                    #
                    #         start_point = self.center
                    #         end_point = [self.center[0] + int((start - self.center[0]) * self.cos),
                    #                      self.center[1] - int((start - self.center[0]) * self.sin)]
                    #
                    #         # 错误情况处理，暂时没什么要处理的，先留着
                    #         if 'previous_4' in locals().keys():
                    #             if lp_rot[0] < 0:
                    #                 lp_rot = self.previous_4
                    #             else:
                    #                 self.previous_4 = lp_rot
                    #         else:
                    #             self.previous_4 = lp_rot
                    #
                    #         box = np.array([[start_point, lp_rot, end_point]], dtype=np.int32)
                    #         # 画三角形
                    #         self.makeUpMky(box)
                    # # 到此修补完毕

                    # 旋转变换
                    self.handRotate(self.theta)

                    self.copyTo()

                    if self.showRed == 1:
                        cv2.circle(self.canvas, (self.leftMg + self.redX, self.upMg + self.redY), 10, (0, 0, 255), -1)
                    if self.showBlue == 1:
                        cv2.circle(self.canvas, (self.blueX, self.blueY), 15, (255, 0, 0), -1)

                # 延迟,队列真美妙
                if self.delay != 0:
                    if q.full():
                        cv2.imshow(window_name, q.get())
                        q.put(self.canvas)
                    else:
                        q.put(self.canvas)
                else:
                    cv2.imshow(window_name, self.canvas)

                # cv2.imshow(window_name, self.canvas)
                # interval = datetime.datetime.now() - starttime
                # print(interval)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                break

        self.cap.release()
        cv2.destroyAllWindows()
