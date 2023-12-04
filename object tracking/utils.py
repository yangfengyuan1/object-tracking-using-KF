import cv2
import numpy as np

# 坐标格式转换
def xyxy_to_xywh(xyxy):
    center_x = (xyxy[0] + xyxy[2]) / 2
    center_y = (xyxy[1] + xyxy[3]) / 2
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return (center_x, center_y, w, h)

# 颜色默认是绿色
def plot_one_box(xyxy, img, color=(0, 200, 0), target=False):
    xy1 = (int(xyxy[0]), int(xyxy[1]))
    xy2 = (int(xyxy[2]), int(xyxy[3]))
    if target:
        # red
        color = (0, 0, 255)
    cv2.rectangle(img, xy1, xy2, color, 1, cv2.LINE_AA)  # filled

# 固定大小栈，存中心位置
def updata_trace_list(box_center, trace_list, max_list_len=50):
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
    return trace_list


def draw_trace(img, trace_list):
    """
    更新trace_list,绘制trace
    :param trace_list:
    :param max_list_len:
    :return:
    """
    for i, item in enumerate(trace_list):

        if i < 1:
            continue
        cv2.line(img,
                 (trace_list[i][0], trace_list[i][1]), (trace_list[i - 1][0], trace_list[i - 1][1]),
                 (255, 255, 0), 3)


def cal_iou(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou


def cal_distance(box1, box2):
    """
    计算两个box中心点的距离
    :param box1: xyxy 左上右下
    :param box2: xyxy
    :return:
    """
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
    dis = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return dis


def xywh_to_xyxy(xywh):
    x1 = xywh[0] - xywh[2]//2
    y1 = xywh[1] - xywh[3]//2
    x2 = xywh[0] + xywh[2] // 2
    y2 = xywh[1] + xywh[3] // 2

    return [x1, y1, x2, y2]

def cal_rmse(ground_true_x, estimation_x):
    # Convert lists to numpy arrays
    ground_true_x = np.array(ground_true_x)
    estimation_x = np.array(estimation_x)
    # Check if arrays have the same length
    if ground_true_x.size != estimation_x.size:
        raise ValueError("Arrays must have the same length")
    
    # Calculate RMSE
    mse = np.mean((ground_true_x - estimation_x) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# 每次调用都重新算，时间复杂度高
# 直接计算整体rmse并不合理，改为计算最近几次估计的rmse
# def moving_rmse1(ground_truth, estimations, window_size=10):
#     # Convert lists to numpy arrays
#     ground_true = np.array(ground_true)
#     estimation = np.array(estimation)
#     # Check if arrays have the same length
#     if ground_true.size != estimation.size:
#         raise ValueError("Arrays must have the same length")
#     errors = []
#     for i in range(len(estimations) - window_size + 1):
#         window_estimations = estimations[i:i+window_size]
#         window_ground_truth = ground_truth[i:i+window_size]
#         # 计算每个窗口的 RMSE
#         rmse = np.sqrt(np.mean((np.array(window_estimations) - np.array(window_ground_truth)) ** 2))
#         errors.append(rmse) 
#     return errors

# 直接计算整体rmse并不合理，改为计算最近几次估计的rmse
def cal_moving_rmse(ground_truth, estimation, window_size=10):
    # Convert lists to numpy arrays
    ground_truth = np.array(ground_truth)
    estimation = np.array(estimation)
    # Check if arrays have the same length
    if ground_truth.size == estimation.size:
        if len(estimation) > window_size:
            ground_truth_buffer = ground_truth[-window_size:]
            estimations_buffer = estimation[-window_size:]
            rmse = np.sqrt(np.mean((np.array(estimations_buffer) - np.array(ground_truth_buffer)) ** 2))
        else:
            rmse = np.sqrt(np.mean((np.array(estimation) - np.array(ground_truth)) ** 2))
    else:
        rmse = 10000
    return rmse



if __name__ == "__main__":
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 200, 300]
    iou = cal_iou(box1, box2)
    print(iou)
    box1.pop(0)
    box1.append(555)
    print(box1)
