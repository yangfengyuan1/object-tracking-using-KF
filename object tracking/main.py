import os
# import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import *

# 状态初始化
# initial_target_box = [555, 631, 791, 715] # output.avi
# initial_target_box = [64, 521, 340, 695]  # 目标初始bouding box
initial_target_box = [193 ,342 ,250 ,474] # videotest1
# initial_target_box = [328 ,145 ,350 ,220]
IOU_Threshold = 0.3 # 匹配时的阈值
trace_list_len = 100 # 跟踪列表保留的轨迹
Q_xishu = 0.02 # 调整过程噪声协方差数乘矩阵系数
R_xishu = 1 # 调整观测噪声协方差数乘矩阵系数，R>>Q
SAVE_VIDEO = True
rmse_win_len = 10


initial_box_state = xyxy_to_xywh(initial_target_box) # xyxy->xywh
ground_true = [] # maskrcnn
estimations = [] # filter
jiance = []      # yolo


# 卡尔曼滤波：假设涉及的运动都是线性运动
# 状态转移矩阵，上一时刻的状态转移到当前时刻
# 用2d bbox描述状态，用x,y,w,h,dx,dy等信息描述框状态，初试设置为1
A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])
# 观测值丢失时使用的状态转移矩阵
A_ = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
               [0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])

# 控制输入矩阵B
# 如果系统没有外部控制输入或者对系统的控制输入了解不足，可以将控制矩阵 B 设置为零矩阵或者忽略掉。
B = None
# 状态观测矩阵
# 状态观测矩阵 H 经常被初始化为单位矩阵，表示系统的状态完全可观测
H = np.eye(8)
# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
# 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
Q = np.eye(8) * Q_xishu
# 观测噪声协方差矩阵R，p(v)~N(0,R)
# 观测噪声来自于检测框丢失、重叠等
R = np.eye(8) * R_xishu
# 状态估计协方差矩阵P初始化
P = np.eye(8)

# 假设状态转移函数为非线性函数
# EKF状态转移矩阵,状态向量x=[x,y,w,h,dx,dy,dw,dh,ddx,ddy]T
def ekf_state_transition(X, dt):
    ddt = 0.5 * dt * dt
    A_ekf = np.array([[1, 0, 0, 0, dt, 0, 0, 0, ddt, 0],
                      [0, 1, 0, 0, 0, dt, 0, 0, 0, ddt],
                      [0, 0, 1, 0, 0, 0, dt, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, dt, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, dt],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    return A_ekf,np.dot(A_ekf, X)

# 计算状态向量X关于观测的雅可比矩阵H
def calculate_jacobian(X):
    # 提取状态向量中的位置和大小信息
    x, y, w, h, dx, dy, dw, dh, ddx, ddy = X

    # 计算雅可比矩阵H
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 对x的偏导数
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 对y的偏导数
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 对w的偏导数
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 对h的偏导数
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ])
    # H = np.eye(8)
    return H

# EKF观测函数
def ekf_observation(X):
    H_ekf = calculate_jacobian(X)
    return H_ekf,np.dot(H_ekf, X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='VIDEO source(e.g.: ./data/testvideo1.mp4)')
    parser.add_argument('--filter',type=str,help='type of filter(KF/EKF/UKF)',default='KF')
    args = parser.parse_args()
    source = args.source
    filter_type = args.filter
    # [中心x,中心y,宽w,高h,dx,dy,dw,dh] ,initial dx=dy=0
    if filter_type.upper() == "KF":
        initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3], 0, 0, 0, 0]]).T  
    elif filter_type.upper() == "EKF":
        initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3], 0, 0, 0, 0, 0, 0]]).T  
        # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
        # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
        Q = np.eye(10) * Q_xishu
        # 观测噪声协方差矩阵R，p(v)~N(0,R)
        # 观测噪声来自于检测框丢失、重叠等
        R = np.eye(10) * R_xishu
        # 状态估计协方差矩阵P初始化
        P = np.eye(10)
        # 观测值丢失时使用的状态转移矩阵
        A_ = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                       ])
    if not os.path.exists(source):
        print('Error: Source VIDEO path does not exist.')
        exit(1)
    _, file_extension = os.path.splitext(source)
    if file_extension.lower() not in ['.mp4', '.avi']:
        print('Error: Unsupported file format. Only .mp4 and .avi are supported.')
        exit(1)
    if filter_type.upper() not in ["KF", "EKF", "UKF"]:
        print('Error: Invalid filter type. Please choose from KF, EKF, or UKF.')
        exit(1)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Error: Failed to open the video file.')
        exit(1)
    print("Filter type:", filter_type, "\n")

    video_path = source
    label_path = "./data/labels"
    ext = os.path.splitext(os.path.basename(video_path))
    file_name = ext[0]
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    # cv2.namedWindow("track", cv2.WINDOW_NORMAL)

    if SAVE_VIDEO:
        output_name = filter_type+"{}_output.avi".format(file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # 编码器，avi格式
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_name, fourcc, video_fps,(video_width,video_height))

    # ---------状态初始化----------------------------------------
    frame_counter = 1
    X_posterior = np.array(initial_state) # [x,y,w,h,dx,dy]
    P_posterior = np.array(P) # 状态估计协方差矩阵P初始化
    Z = np.array(initial_state) # [x,y,w,h,dx,dy]
    trace_list = []  # 用于保存目标box的轨迹
    rmse_list = [] #用于保存rmse结果


    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        last_box_posterior = xywh_to_xyxy(X_posterior[0:4])
        # 白色框；将指定物体对应框设为初始上一帧最好估计
        plot_one_box(last_box_posterior, frame, color=(255, 255, 255), target=False)
        if not ret:
            break
        print("frame num:"+str(frame_counter))        
        # 从视觉算法检测结果获取2d bbox信息,yolo结果为检测值，maskrcnn结果为groundtruth
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        label_ground_truth_path = os.path.join(label_path, file_name + "yolo_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            max_iou = IOU_Threshold
            max_iou_matched = False
            # ---------使用最大IOU来寻找观测值------------
            # 视觉检测算法
            for j, data_ in enumerate(content): 
                data = data_.replace('\n', "").split(" ") # 删除回车干扰，使用空格分隔
                xyxy = np.array(data[1:5], dtype="float")
                # mask R-CNN 检测出的所有2d bbox，用绿色框
                plot_one_box(xyxy, frame)
                # 与所有框与目标框的iou
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                if iou > max_iou:# 筛出最大IOU及对应box
                    target_box = xyxy
                    max_iou = iou 
                    max_iou_matched = True

            if max_iou_matched == True:
                if filter_type.upper() == "KF":
                    # 如果找到了最大IOU BOX,则认为该框为观测值
                    # 跟踪目标画成red框
                    plot_one_box(target_box, frame, target=True)
                    xywh = xyxy_to_xywh(target_box)
                    jiance.append(0.25*(xywh[0]+xywh[1]+xywh[2]+xywh[3]))       
                    print(filter_type.upper()+ "jiance"+str(0.25*(xywh[0]+xywh[1]+xywh[2]+xywh[3])))
                    box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
                    trace_list = updata_trace_list(box_center, trace_list, trace_list_len)
                    # 框上方blue字
                    cv2.putText(frame, filter_type.upper()+"IoU tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    # 计算dx,dy
                    dx = xywh[0] - X_posterior[0]
                    dy = xywh[1] - X_posterior[1]
                    dw = xywh[2] - X_posterior[2]
                    dh = xywh[3] - X_posterior[3]
                    Z[0:4] = np.array([xywh]).T
                    Z[4::] = np.array([dx, dy, dw, dh])
                    
                elif filter_type.upper() == "EKF":
                    # 如果找到了最大IOU BOX,则认为该框为观测值
                    # 跟踪目标画成red框
                    plot_one_box(target_box, frame, target=True)
                    xywh = xyxy_to_xywh(target_box)
                    jiance.append(0.25*(xywh[0]+xywh[1]+xywh[2]+xywh[3]))
                    print(filter_type.upper()+ "jiance"+str(0.25*(xywh[0]+xywh[1]+xywh[2]+xywh[3])))       

                    box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
                    trace_list = updata_trace_list(box_center, trace_list, trace_list_len)
                    # 框上方blue字
                    cv2.putText(frame, filter_type.upper()+"IoU tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    # 计算dx,dy
                    dx = xywh[0] - X_posterior[0]
                    dy = xywh[1] - X_posterior[1]
                    dw = xywh[2] - X_posterior[2]
                    dh = xywh[3] - X_posterior[3]
                    ddx = dx - X_posterior[4]
                    ddy = dy - X_posterior[5]
                    Z[0:4] = np.array([xywh]).T
                    Z[4::] = np.array([dx, dy, dw, dh, ddx, ddy])
                    # Z[4:] = np.concatenate([dx, dy, dw, dh, ddx, ddy])
                    
 
            f.close()              

        with open(label_ground_truth_path, "r") as f2:
            content2 = f2.readlines()
            max_iou2 = IOU_Threshold
            max_iou_matched2 = False
            # ---------使用最大IOU来寻找观测值------------
            # 视觉检测算法
            for j2, data2_ in enumerate(content2): 
                data2 = data2_.replace('\n', "").split(" ") # 删除回车干扰，使用空格分隔
                xyxy2 = np.array(data2[1:5], dtype="float")
                # 与所有框与目标框的iou
                iou2 = cal_iou(xyxy2, xywh_to_xyxy(X_posterior[0:4]))
                if iou2 > max_iou2:# 筛出最大IOU及对应box
                    target_box2 = xyxy2
                    max_iou2 = iou 
                    max_iou_matched2 = True               
            if max_iou_matched2 == True:
                xywh2 = xyxy_to_xywh(target_box2)
                # plot_one_box(xyxy2, frame,color = (0,0,0))
                ground_true.append(0.25*(xywh2[0]+xywh2[1]+xywh2[2]+xywh2[3]))
                print("groundtruth:"+str(0.25*(xywh2[0]+xywh2[1]+xywh2[2]+xywh2[3])))       
            f2.close()

        if max_iou_matched == True and max_iou_matched2 == True :
            if filter_type.upper() == "KF":
                # -----进行先验估计-----------------
                X_prior = np.dot(A, X_posterior) # B=0
                # ground_true.append()
                box_prior = xywh_to_xyxy(X_prior[0:4])                
                # plot_one_box(box_prior, frame, color=(0, 0, 0), target=False)
                # -----计算状态估计协方差矩阵P--------
                P_prior = np.dot(np.dot(A, P_posterior), A.T) + Q
                # ------计算卡尔曼增益---------------------
                k1 = np.dot(P_prior, H.T)
                k2 = np.dot(np.dot(H, P_prior), H.T) + R
                K = np.dot(k1, np.linalg.inv(k2))
                # --------------后验估计------------
                Z1 = np.dot(H, X_prior)
                
                X_posterior_1 = Z - Z1
                X_posterior = X_prior + np.dot(K, X_posterior_1)
                box_posterior = xywh_to_xyxy(X_posterior[0:4])
                estimations.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print("est:"+str(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))+"\n")
                # plot_one_box(box_posterior, frame, color=(0, 0, 0), target=False)
                # ---------更新状态估计协方差矩阵P-----
                P_posterior_1 = np.eye(8) - np.dot(K, H)
                P_posterior = np.dot(P_posterior_1, P_prior)  
            elif filter_type.upper() == "EKF":
                # print("EKF tracking")
                dt = 30 / video_fps
                W = np.eye(10) #* Q_xishu
                V = np.eye(10) #* R_xishu
                A_ekf,X_prior = ekf_state_transition(X_posterior, dt)
                P_prior_1 = np.dot(A_ekf, P_posterior)
                P_prior = np.dot(P_prior_1, A_ekf.T) + np.dot(np.dot(W,Q),W.T) # W=I

                H,Z1 = ekf_observation(X_prior)
                k1 = np.dot(P_prior, H.T)
                k2 = np.dot(np.dot(H, P_prior), H.T) + np.dot(np.dot(V,R),V.T) # V=I
                K = np.dot(k1, np.linalg.inv(k2))      
                
                X_posterior_1 = Z - Z1
                X_posterior = X_prior + np.dot(K, X_posterior_1)
                estimations.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print("est:"+str(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))+"\n")
                P_posterior_1 = np.eye(10) - np.dot(K, H)
                P_posterior = np.dot(P_posterior_1, P_prior)
            elif filter_type.upper() == "UKF":
                pass
                        
        else:
            if filter_type.upper() == "KF":
                # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
                # 此时直接迭代，不使用卡尔曼滤波
                # X_posterior = np.dot(A, X_posterior)
                X_posterior = np.dot(A_, X_posterior)
                estimations.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print(filter_type.upper()+"jiance:null")
                jiance.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print("est_loss:"+str(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))+"\n")
                box_posterior = xywh_to_xyxy(X_posterior[0:4])
                # plot_one_box(box_posterior, frame, color=(255, 255, 255), target=False)
                box_center = ((int(box_posterior[0] + box_posterior[2]) // 2), int((box_posterior[1] + box_posterior[3]) // 2))
                trace_list = updata_trace_list(box_center, trace_list, 20)
                cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif filter_type.upper() == "EKF":
                # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
                # 此时直接迭代，不使用卡尔曼滤波
                # X_posterior = np.dot(A, X_posterior)
                X_posterior = np.dot(A_, X_posterior)
                estimations.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print(filter_type.upper()+"jiance:null")
                jiance.append(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3]))
                print("est_loss:"+str(np.array(0.25*(X_posterior[0]+X_posterior[1]+X_posterior[2]+X_posterior[3])))+"\n")
                box_posterior = xywh_to_xyxy(X_posterior[0:4])
                # plot_one_box(box_posterior, frame, color=(255, 255, 255), target=False)
                box_center = ((int(box_posterior[0] + box_posterior[2]) // 2), int((box_posterior[1] + box_posterior[3]) // 2))
                trace_list = updata_trace_list(box_center, trace_list, 20)
                cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif filter_type.upper() == "UKF":
                pass

        draw_trace(frame, trace_list)
        # rmse = cal_rmse(ground_true, estimations)
        moving_rmse = cal_moving_rmse(ground_true, estimations, rmse_win_len)
        rmse_list.append(moving_rmse)

        cv2.putText(frame,"mov_rmse:"+str(moving_rmse),(25,25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Last frame best estimation(White)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            out.write(frame)
        frame_counter = frame_counter + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if frame_counter == 100:
            break
    
    plt.subplot(1, 2, 1)
    y_label = np.array(estimations)
    x_label = np.arange(len(y_label))
    plt.plot(x_label,np.array(ground_true),"r", label = "groundtrue")
    plt.plot(x_label,np.array(estimations),"g", label = "estiations")
    plt.plot(x_label,np.array(jiance),"b", label = "yolojiance")
    plt.xlabel("time")
    plt.ylabel(filter_type.upper()+"position")
    plt.title("tracking result")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    y_label1 = np.array(rmse_list)
    x_label1 = np.arange(len(y_label1))
    plt.plot(x_label1, y_label1, marker = "o", color = "y", label = "mov_rmse")
    plt.title("tracking rmse")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

