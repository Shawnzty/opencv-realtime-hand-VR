from gui import *
import queue, threading, time


global root, threads


# 主流程
if __name__ == '__main__':
    root = Tk()
    main_ui = gui(root)
    t_gui = threading.Thread(target=root.mainloop())
    t_gui.setDaemon(True)
    t_gui.start()
