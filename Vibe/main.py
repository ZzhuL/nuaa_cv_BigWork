# -*- encoding: utf-8 -*-
"""
    @Project: Vibe.py
    @File   : main.py.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/13  12:58
"""
import cv2
import numpy as np
from ViBe import ViBe


def main():
    # 打开视频文件
    vc = cv2.VideoCapture('car1.avi')
    # 获取视频的宽度和高度
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化变量
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()  # 读取第一帧
    else:
        rval = False

    # 将帧转换为灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 创建视频写入器
    output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height), isColor=False)

    # 初始化 ViBe 背景减除算法
    vibe = ViBe()
    vibe.__FirstFrame__(frame)

    # 创建用于显示帧的窗口
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("segMat", cv2.WINDOW_NORMAL)

    # 处理帧直到视频结束或用户中断
    while rval:
        rval, frame = vc.read()  # 读取下一帧
        if rval:
            # 将帧转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 更新 ViBe 背景减除
            vibe.__Update__(gray)
            segMat = vibe.__getFG_mask__()  # 获取前景掩码
            segMat = segMat.astype(np.uint8)

            # 保存segMat为视频帧
            output_video.write(segMat)
            # 显示帧和前景掩码
            cv2.imshow("frame", frame)
            cv2.imshow("SegMat", segMat)

            # 保存segMat为图像文件
            # cv2.imwrite(f"segMat_{c}.jpg", segMat)

            # 等待用户输入（按下 'Esc' 键退出）
            k = cv2.waitKey(1)
            if k == 27:
                vc.release()  # 释放视频文件
                cv2.destroyAllWindows()  # 关闭窗口
                break

            c += 1
        else:
            break
    output_video.release()
    vc.release()  # 释放视频文件
    cv2.destroyAllWindows()  # 关闭窗口


if __name__ == '__main__':
    main()
