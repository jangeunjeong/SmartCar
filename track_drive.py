#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# TX2+VESC+Lidar 장착 모델 
# Ubuntu 18.04 + ROS Melodic
# Yolo 사용
# ar_track_alvar를 xycar_ws/src 아래에 복사(설치)
#=============================================

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import Int32MultiArray
from ar_track_alvar_msgs.msg import AlvarMarkers
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import LaserScan

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge() # OpenCV 함수를 사용하기 위한 브릿지 
motor = None # 모터 토픽을 담을 변수
lidar_msg = None  # lidar 토픽을 담을 변수
ultra_msg = None  # 초음파 토픽을 담을 변수
ar_msg = {"ID":[],"DZ":[]}  # AR태그 토픽을 담을 변수
img_ready = False # 카메라 토픽이 도착했는지의 여부 표시 
lidar_ready = False # lidar 토픽이 도착했는지의 여부 표시
ultra_ready = False # 초음파 토픽이 도착했는지의 여부 표시
ar_ready = False  # AR태그 토픽이 도착했는지의 여부 표시
Fix_Speed = 4  # 모터 속도값
box_data = None  # 객체인식 Bounding Box 정보를 담을 변수
box_ready = False  # 객체인식 토픽이 도착했는지의 여부 표시

#=============================================
# 차선 인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 300   # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 380   # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW   # ROI 영역의 세로 크기  
L_ROW = 40  # 차선의 위치를 찾기 위한 기준선(수평선)의 Row값

#=============================================
# 프로그램에서 사용할 이동평균필터 클래스
#=============================================
class MovingAverage:

    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]
            
    def get_sample_count(self):
        return len(self.data)
        
    def get_mm(self):
        return float(sum(self.data)) / len(self.data)

    def get_wmm(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

#=============================================
# 이동평균필터 클래스의 인스턴스 선언부
#=============================================
lidar1_mvavg = MovingAverage(10)
lidar1_mvavg.add_sample(0) 
lidar2_mvavg = MovingAverage(10)
lidar2_mvavg.add_sample(0)
lidar3_mvavg = MovingAverage(10)
lidar3_mvavg.add_sample(0)
lidar4_mvavg = MovingAverage(10)
lidar4_mvavg.add_sample(0)
lidar5_mvavg = MovingAverage(10)
lidar5_mvavg.add_sample(0)
lidar6_mvavg = MovingAverage(10)
lidar6_mvavg.add_sample(0)
lidar7_mvavg = MovingAverage(10)
lidar7_mvavg.add_sample(0)
lidar8_mvavg = MovingAverage(10)
lidar8_mvavg.add_sample(0)
lidar9_mvavg = MovingAverage(10)
lidar9_mvavg.add_sample(0)

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수.
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수.
# 토픽에서 이미지 정보를 꺼내 image 라는 변수에 옮겨 담음.
# 카메라 토픽의 도착을 표시하는 img_ready 값을 True로 바꿈.
#=============================================
def img_callback(data):
    global image, img_ready
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    img_ready = True

#=============================================
# 콜백함수 - 라이다 토픽을 처리하는 콜백함수.
# 토픽에서 정보를 꺼내 lidar_msg 라는 변수에 옮겨 담음.
# 라이다 토픽의 도착을 표시하는 lidar_ready 값을 True로 바꿈.
#=============================================
def lidar_callback(data):
    global lidar_msg, lidar_ready
    lidar_msg = data.ranges
    lidar_ready = True

#=============================================
# 콜백함수 - 초음파 토픽을 처리하는 콜백함수.
# 토픽에서 정보를 꺼내 ultra_msg 라는 변수에 옮겨 담음.
# 초음파 토픽의 도착을 표시하는 ultra_ready 값을 True로 바꿈.
#=============================================
def ultra_callback(data):
    global ultra_msg, ultra_ready
    ultra_msg = data.data
    ultra_ready = True

#=============================================
# 콜백함수 - AR태그 토픽을 처리하는 콜백함수.
# 토픽에서 정보를 꺼내 ar_msg 라는 변수에 옮겨 담음.
# AR태그 토픽의 도착을 표시하는 ar_ready 값을 True로 바꿈.
#=============================================
def ar_callback(data):
    global ar_msg, ar_ready

    ar_msg["ID"] = []
    ar_msg["DX"] = []
    ar_msg["DZ"] = []

    for i in data.markers:
        ar_msg["ID"].append(i.id)
        ar_msg["DX"].append(i.pose.pose.position.x)
        ar_msg["DZ"].append(i.pose.pose.position.z)

    ar_ready = True

#=============================================
# 콜백함수 - YOLO 객체인식 토픽을 처리하는 콜백함수.
# 토픽에서 정보를 꺼내 box_date 라는 변수에 옮겨 담음.
# YOLO 토픽의 도착을 표시하는 box_ready 값을 True로 바꿈.
#=============================================
def box_callback(data):
    global box_data, box_ready

    box_data = data
    box_ready = True
    
#=============================================
# 모터 토픽을 발행하는 함수.  
# 입력으로 받은 angle과 speed 값을 
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)
    
#=============================================
# 카메라 영상 이미지에서 차선을 찾아
# 차선을 벗어나지 않으며 주행하는 코드
#=============================================
def lane_drive():
    global image, img_ready
    global motor
    prev_x_left = 0
    prev_x_right = WIDTH

    while img_ready == False:
        continue

    img = image.copy() # 이미지처리를 위한 카메라 원본이미지 저장
    display_img = img  # 디버깅을 위한 디스플레이용 이미지 저장
    img_ready = False  # 카메라 토픽이 도착하면 콜백함수 안에서 True로 바뀜

    #=========================================
    # 원본 칼라이미지를 그레이 회색톤 이미지로 변환하고 
    # 블러링 처리를 통해 노이즈를 제거한 후에 (약간 뿌옇게, 부드럽게)
    # Canny 변환을 통해 외곽선 이미지로 만들기
    #=========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)
    
    """
    # HSV - Canny Edge
    blur_img = cv2.GaussianBlur(img,(5,5), 0)
    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    threshed_img = cv2.inRange(hsv, (0,0,120), (255,255,255))
    edge_img = cv2.Canny(np.uint8(threshed_img), 60, 75)

    cv2.imshow("threshed img", threshed_img)
    """
	
    # img(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    
    # edge_img의 특정영역(ROI Area)을 잘라내기
    roi_edge_img = edge_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    cv2.imshow("roi edge img", roi_edge_img)

    # 잘라낸 이미지에서 HoughLinesP 함수를 사용하여 선분들을 찾음
    all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180,50,50,20)
    
    if all_lines is None:
        return

    line_draw_img = roi_img.copy()
    
    #=========================================
    # 선분들의 기울기 값을 각각 모두 구한 후에 리스트에 담음. 
    # 기울기의 절대값이 너무 작은 경우 (수평선에 가까운 경우)
    # 해당 선분을 빼고 담음. 
    #=========================================
    slopes = []
    filtered_lines = []

    for line in all_lines:
        x1, y1, x2, y2 = line[0]

        if (x2 == x1):
            slope = 1000.0
        else:
            slope = float(y2-y1) / float(x2-x1)
    
        if 0.2 < abs(slope):
            slopes.append(slope)
            filtered_lines.append(line[0])

    # print("Number of lines after slope filtering : %d" % len(filtered_lines))

    if len(filtered_lines) == 0:
        return

    #=========================================
    # 왼쪽 차선에 해당하는 선분과 오른쪽 차선에 해당하는 선분을 구분하여 
    # 각각 별도의 리스트에 담음.
    #=========================================
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = filtered_lines[j]
        slope = slopes[j]

        x1,y1, x2,y2 = Line

        # 기울기 값이 음수이고 화면의 왼쪽에 있으면 왼쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 - Margin값)
        Margin = 0
        
        if (slope < 0) and (x2 < WIDTH/2-Margin):
            left_lines.append(Line.tolist())

        # 기울기 값이 양수이고 화면의 오른쪽에 있으면 오른쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 + Margin값)
        elif (slope > 0) and (x1 > WIDTH/2+Margin):
            right_lines.append(Line.tolist())

    # print("Number of left lines : %d" % len(left_lines))
    # print("Number of right lines : %d" % len(right_lines))

    # 디버깅을 위해 차선과 관련된 직선과 선분을 그리기 위한 도화지 준비
    line_draw_img = roi_img.copy()
    
    # 왼쪽 차선에 해당하는 선분은 빨간색으로 표시
    for line in left_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), (0,0,255), 2)

    # 오른쪽 차선에 해당하는 선분은 노란색으로 표시
    for line in right_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), (0,255,255), 2)

    #=========================================
    # 왼쪽/오른쪽 차선에 해당하는 선분들의 데이터를 적절히 처리해서 
    # 왼쪽차선의 대표직선과 오른쪽차선의 대표직선을 각각 구함.
    # 기울기와 Y절편값으로 표현되는 아래와 같은 직선의 방적식을 사용함.
    # (직선의 방정식) y = mx + b (m은 기울기, b는 Y절편)
    #=========================================

    # 왼쪽 차선을 표시하는 대표직선을 구함        
    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 왼쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(left_lines)
    if size != 0:
        for line in left_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0                
            
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_left = m_sum / size
        b_left = y_avg - m_left * x_avg

        if m_left != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_left) / m_left)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_left) / m_left)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), (255,0,0), 2)

    # 오른쪽 차선을 표시하는 대표직선을 구함      
    m_right, b_right = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 오른쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(right_lines)
    if size != 0:
        for line in right_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0     
       
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_right = m_sum / size
        b_right = y_avg - m_right * x_avg

        if m_right != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_right) / m_right)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_right) / m_right)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), (255,0,0), 2)

    #=========================================
    # 차선의 위치를 찾기 위한 기준선(수평선)은 아래와 같음.
    # (직선의 방정식) y = L_ROW 
    # 위에서 구한 2개의 대표직선, 
    # (직선의 방정식) y = (m_left)x + (b_left)
    # (직선의 방정식) y = (m_right)x + (b_right)
    # 기준선(수평선)과 대표직선과의 교점인 x_left와 x_right를 찾음.
    #=========================================

    #=========================================        
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_left == 0.0:
        x_left = prev_x_left  # 변수에 저장해 놓았던 이전 값을 가져옴
		
        # x_left = prev_left_mvavg.get_mm()  # 그동안 모아 놓았던 이동평균값을 가져옴 
        # x_left = int(prev_left_mvavg.get_wmm())  # 그동안 모아 놓았던 이동평균값을 가져옴 
 
    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_left)x + (b_left)
    #=========================================
    else:
        x_left = int((L_ROW - b_left) / m_left)
                        
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_right == 0.0:
	x_right = prev_x_right  # 변수에 저장해 놓았던 이전 값을 가져옴	
	
		# 이동평균값을 이용할 수도 있음
        # x_right = prev_right_mvavg.get_mm()  # 그동안 모아 놓았던 이동평균값을 가져옴 
        # x_right = int(prev_right_mvavg.get_wmm())  # 그동안 모아 놓았던 이동평균값을 가져옴 

    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_right)x + (b_right)
    #=========================================
    else:
        x_right = int((L_ROW - b_right) / m_right)
       
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에 반대쪽 차선의 위치 정보를 이용해서 내 위치값을 정함 
    #=========================================
    if m_left == 0.0 and m_right != 0.0:
        x_left = x_right - 380

    if m_left != 0.0 and m_right == 0.0:
        x_right = x_left + 380

    # 이번에 구한 값으로 예전 값을 업데이트 함			
    prev_x_left = x_left
    prev_x_right = x_right
	
	# 이동평균값을 이용할 수도 있음
    # prev_left_mvavg.add_sample(x_left) 
    # prev_right_mvavg.add_sample(x_right)  

    # 왼쪽 차선의 위치와 오른쪽 차선의 위치의 중간 위치를 구함
    x_midpoint = (x_left + x_right) // 2 

    # 화면의 중앙지점(=카메라 이미지의 중앙지점)을 구함
    view_center = WIDTH//2
  
    #=========================================
    # 디버깅용 이미지 그리기
    # (1) 수평선 그리기 (직선의 방정식) y = L_ROW 
    # (2) 수평선과 왼쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (3) 수평선과 오른쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (4) 왼쪽 교점과 오른쪽 교점의 중점 위치에 작은 파란색 사각형 그리기
    # (5) 화면의 중앙점 위치에 작은 빨간색 사각형 그리기 
    #=========================================
    cv2.line(line_draw_img, (0,L_ROW), (WIDTH,L_ROW), (0,255,255), 2)
    cv2.rectangle(line_draw_img, (x_left-5,L_ROW-5), (x_left+5,L_ROW+5), (0,255,0), 4)
    cv2.rectangle(line_draw_img, (x_right-5,L_ROW-5), (x_right+5,L_ROW+5), (0,255,0), 4)
    cv2.rectangle(line_draw_img, (x_midpoint-5,L_ROW-5), (x_midpoint+5,L_ROW+5), (255,0,0), 4)
    cv2.rectangle(line_draw_img, (view_center-5,L_ROW-5), (view_center+5,L_ROW+5), (0,0,255), 4)

    # 위 이미지를 디버깅용 display_img에 overwrite해서 화면에 디스플레이 함
    display_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH] = line_draw_img
    cv2.imshow("Lanes positions", display_img)
    cv2.waitKey(1)

    # ====================================
    # 핸들을 얼마나 꺾을지 결정 
    # ====================================
    angle = (x_midpoint-view_center) // 2

    # ====================================
    # 주행 속도를 결정  
    # ====================================
    speed = Fix_Speed

    # ====================================
    # 모터쪽으로 토픽을 전송  
    # ====================================
    drive(angle, speed)


# ====================================
# image 안에서 타겟 객체를 찾는다.
# 여러 개 있으면 Probability 가장 높은 걸 선택
# Bounding Box의 중앙점 x좌표값을 반환
# 타겟 객체를 찾지 못하면 0 값을 반환
# ====================================
def object_detection():

    boxes = box_data
    target_found = False
    
    for i in range(len(boxes.bounding_boxes)):
        bbox = boxes.bounding_boxes[i]
        class_name = bbox.Class
        probability = round(bbox.probability, 2)
                
        if ((class_name == "bottle") and (probability > 0.3)):
            target_found = True                 

    return target_found

        
#=============================================
# 카메라 영상 이미지 img를 받아서 흑백으로 변환한 다음에 
# 아래 부분에 ROI 영역을 설정하고
# 작은 사각형 안에 흰색 점이 기준치 이상이면 
# 정지선을 발견한 것으로 하고
# True 반환, 아니면 False 반환함.
#=============================================
def check_stopline():
    global image

    img = image.copy()  # 이미지처리를 위한 카메라 원본이미지 저장

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0, 0, 180])
    stop_line_img = cv2.inRange(hsv, lower_white, upper_white)

    cv2.imshow("stop line", stop_line_img)
    cv2.waitKey(1)

    area = stop_line_img[380:420, 200:440] # Total 4800 points
    stopline_count = cv2.countNonZero(area)
    print("Stop Line Count = " + str(stopline_count))

    area2 = stop_line_img[300:420,200:440]
    crosswalk_count = cv2.countNonZero(area2)
    print("Cross Walk Count = " + str(crosswalk_count))

    if stopline_count > 6000:
        print("Stop Line Count = " + str(stopline_count))
        return "Stopline"

    elif crosswalk_count > 10000:
        print("Cross Walk Count = " + str(crosswalk_count))
        return "Crosswalk"


    return False

# ====================================
# 라이다 센서 데이터 필터링 함수
# 라이다 데이터 중 전방 180도 센서만 사용
# 왼쪽부터 20도 간격으로 나누고 중앙 값을 계산
# 0이 나오면 이동평균필터 적용
# ====================================
def lidar_filtering():
    global lidar_msg, lidar_ready

    while lidar_ready == False:
        continue
    lidar_ready = False

    mid_list = []
    mid_list.append(lidar_msg[491:504])
    mid_list.append(lidar_msg[1:14])

    # 중앙값 찾기
    median_value1 = np.percentile(lidar_msg[99:126], 50)
    median_value2 = np.percentile(lidar_msg[71:98], 50)
    median_value3 = np.percentile(lidar_msg[43:70], 50)
    median_value4 = np.percentile(lidar_msg[15:42], 50)
    median_value5 = np.percentile(mid_list, 50)
    median_value6 = np.percentile(lidar_msg[463:490], 50)
    median_value7 = np.percentile(lidar_msg[435:462], 50)
    median_value8 = np.percentile(lidar_msg[407:434], 50)
    median_value9 = np.percentile(lidar_msg[379:406], 50)

    # 값이 0이면 이동평균필터 값 사용
    if median_value1 == 0.0:
        median_value1 = lidar1_mvavg.get_mm()
    if median_value2 == 0.0:
        median_value2 = lidar2_mvavg.get_mm()
    if median_value3 == 0.0:
        median_value3 = lidar3_mvavg.get_mm()
    if median_value4 == 0.0:
        median_value4 = lidar4_mvavg.get_mm()
    if median_value5 == 0.0:
        median_value5 = lidar5_mvavg.get_mm()
    if median_value6 == 0.0:
        median_value6 = lidar6_mvavg.get_mm()
    if median_value7 == 0.0:
        median_value7 = lidar7_mvavg.get_mm()
    if median_value8 == 0.0:
        median_value8 = lidar8_mvavg.get_mm()
    if median_value9 == 0.0:
        median_value9 = lidar9_mvavg.get_mm()

    # 최종 결과를 저장
    lidar1_mvavg.add_sample(median_value1)
    lidar2_mvavg.add_sample(median_value2)
    lidar3_mvavg.add_sample(median_value3)
    lidar4_mvavg.add_sample(median_value4)
    lidar5_mvavg.add_sample(median_value5)
    lidar6_mvavg.add_sample(median_value6)
    lidar7_mvavg.add_sample(median_value7)
    lidar8_mvavg.add_sample(median_value8)
    lidar9_mvavg.add_sample(median_value9)
    
    return [median_value1, median_value2, median_value3, median_value4, median_value5, median_value6, median_value7, median_value8, median_value9]

#=============================================
# 라이다와 초음파 센서를 이용해서 벽까지의 거리를 알아내서
# 벽과 충돌하지 않으며 주행하도록 핸들 조정함.
#=============================================
def sensor_drive(lidar_data):
    global ultra_msg
    left_data = 0.0
    right_data = 0.0

    for i in lidar_data[0:4]:
        left_data += i

    for i in lidar_data[5:8]:
        right_data += i

    angle = 0
    speed = Fix_Speed

    # Turn left
    if (left_data > right_data):
        print("Trun left")
        angle = -40
        speed = Fix_Speed
                
    # Turn right
    elif (left_data < right_data):
        print("Trun right")
        angle = 40
        speed = Fix_Speed

    # Turn left
    if (ultra_msg[0] - ultra_msg[4] > 15):
        angle = -50
        speed = Fix_Speed
        print("Turn left2 : ", ultra_msg)

    # Turn right
    elif (ultra_msg[4] - ultra_msg[0] > 15):
        angle = 50
        speed = Fix_Speed
        print("Turn right2 : ", ultra_msg)

    # Stop
    if (lidar_data[4] < 0.15 and lidar_data[4] != 0.0):
        print("stop")
        if (left_data > right_data):
            angle = 40
        if (left_data < right_data):
            angle = -40
        speed = -Fix_Speed

    drive(angle, speed)


#=============================================
# AR 감지 패지키지가 발행하는 토픽을 받아서
# 전방에 AR Tag가 있는지 체크하고
# 제일 가까이 있는 AR Tag에 적힌 ID 값을 반환함.
# 없으면 None 을 반환함.
#=============================================
def check_AR():
    global ar_msg
    
    if len(ar_msg["ID"]) == 0:
        return 99, 1, 0

    id_value = float(ar_msg["ID"][0])
    distance = ar_msg["DZ"][0]
    x_pos = ar_msg["DX"][0]

    for i in range(len(ar_msg["ID"])):
        if(distance > ar_msg["DZ"][i]):
            id_value = ar_msg["ID"][i]
            distance = ar_msg["DZ"][i]
            x_pos = ar_msg["DX"][i]

    return id_value, distance, x_pos

#=============================================
# 평행 주차를 진행함.
# 하드코딩으로 구현함.
# 시간, 속도, 핸들조향각을 튜닝해야 함.
#=============================================
def do_parking():
    ar_ID, distance, x_pos= check_AR()
    print("AR ID = " + str(ar_ID))
    print("AR DISTANCE = " + str(distance))

    while(distance > 0.4):
        ar_ID, distance, x_pos= check_AR()
        drive(int(x_pos * 150), Fix_Speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #1 Stop & Wait")
    for i in range(5): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

    # 오른쪽으로 꺾은 후에 후진함
    print("Part - Step #2 Right & Back")
    for i in range(20): 
        drive(50, -Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #3 Stop & Wait")
    for i in range(10): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

    # 왼쪽으로 꺾은 후에 전진함
    print("Part - Step #4 Left & Forward")
    for i in range(10): 
        drive(-10, Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 왼쪽으로 꺾은 후에 전진함
    print("Part - Step #5 Left & Forward")
    for i in range(15): 
        drive(-50, Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #6 Stop & Wait")
    for i in range(5): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

    # 오른쪽으로 꺾은 후에 후진함
    print("Part - Step #7 Right & Back")
    for i in range(23): 
        drive(45, -Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 왼쪽으로 꺾은 후에 후진함
    print("Part - Step #8 Left & Back")
    for i in range(15): 
        drive(-43, -Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 후진함
    print("Part - Step #9 Back")
    for i in range(10): 
        drive(0, -Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #10 Stop & Wait")
    for i in range(10): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

    # 일단 조금 전진함.
    print("Part - Step #11 Forward once")
    for i in range(20): 
        drive(0, Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #12 Stop & Wait")
    for i in range(5): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

#=============================================
# 차량의 방향을 변경함
# 죄로 꺾은 후에 후진 + 잠시 기다린 후에 + 조금 전진 
# 다시 차선으로 복귀할 수 있는 위치와 자세가 됩니다. 
#=============================================
def do_turn():
  
    # 잠시 대기함
    print("Part - Step #1 Stop & Wait")
    for i in range(5): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)  

    # 왼쪽으로 꺾은 후에 후진함
    print("Part - Step #2 Left & Back")
    for i in range(30): 
        drive(-50, -Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

    # 잠시 대기함
    print("Part - Step #3 Stop & Wait")
    for i in range(5): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2) 

    # 일단 조금 전진함.
    print("Part - Step #4 Forward once")
    for i in range(7): 
        drive(10, Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

#=============================================
# 갈림길을 선택함
# 하드코딩으로 구현함
# 객체인식 결과에 따라 방향이 결정됨. 
#=============================================
def do_fork(data):

    Direction = data # left or right

    # 오른쪽 갈림길 선택
    if(Direction == "right"):
        print("Find object !!!")
        for i in range(15): 
            drive(40, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

    # 왼쪽 갈림길 선택
    if(Direction == "left"):
        print("No object !!!")
        for i in range(12): 
            drive(-40, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        for i in range(16): 
            drive(50, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)
  
#=============================================
# 머신러닝 모델 SSD로 전방의 사진을 인식함.
# 타겟 사진을 3초안에 인식하면 원하는 방향을
# 사진을 3초동안 못찾으면 반대 방향을 반환함.
#=============================================
def yolo_drive():
    global box_ready

    start_time = time.time()
  
    while (time.time() - start_time) < 3:
        if box_ready == False:
            drive(0, 0)
            continue

        box_ready = False

        # 사진에서 타겟 객체 위치 찾기
        target_found = object_detection()
        if target_found == True:
            return "right"

    return "left"
            
            
#=============================================
# 횡단보도에서 대기함.
# 대기 후 출발함.
#=============================================
def wait():
    # 잠시 대기함
    print("Part - Step #1 Stop & Wait")
    for i in range(30): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2)

    print("Part - Step #2 Forward")
    for i in range(15): 
        drive(10, Fix_Speed)  # drive(angle, speed)
        time.sleep(0.2)

#=============================================
# AR 방향으로 주행함.
# AR 앞에서 정지함.
# 잠시 대기하고 출발지점으로 돌아감
#=============================================
def ar_drive(direction):
    ar_ID, distance, x_pos= check_AR()
    print("AR ID = " + str(ar_ID))
    print("AR DISTANCE = " + str(distance))

    while(distance > 0.4):
        ar_ID, distance, x_pos= check_AR()
        print("AR DISTANCE = " + str(distance))
        drive(int(x_pos * 200), Fix_Speed)
        time.sleep(0.2)

    print("----- Arrive !!! -----")
    # 잠시 대기함
    print("Part - Step #1 Stop & Wait")
    for i in range(20): 
        drive(0, 0)  # drive(angle, speed)
        time.sleep(0.2) 

    # 다시 출발지점으로 가기 위함
    if direction == "left":
        # 왼쪽으로 꺾은 후에 후진함
        print("Part - Step #2 Left & Back")
        for i in range(10): 
            drive(-30, -Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 잠시 대기함
        print("Part - Step #3 Stop & Wait")
        for i in range(5): 
            drive(0, 0)  # drive(angle, speed)
            time.sleep(0.2) 

        # 일단 조금 전진함.
        print("Part - Step #4 Forward once")
        for i in range(5): 
            drive(-10, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 오른쪽으로 꺾은 후에 전진함.
        print("Part - Step #5 Right & Forward")
        for i in range(25): 
            drive(50, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 왼쪽으로 꺾은 후에 후진함.
        print("Part - Step #6 Left & Back")
        for i in range(5): 
            drive(-35, -Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

    if direction == "right":
        # 왼쪽으로 꺾은 후에 후진함
        print("Part - Step #2 Left & Back")
        for i in range(15): 
            drive(-50, -Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 잠시 대기함
        print("Part - Step #3 Stop & Wait")
        for i in range(5): 
            drive(0, 0)  # drive(angle, speed)
            time.sleep(0.2) 

        # 일단 조금 전진함.
        print("Part - Step #4 Forward once")
        for i in range(7): 
            drive(-20, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 일단 조금 전진함.
        print("Part - Step #5 Right & Forward")
        for i in range(25): 
            drive(50, Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

        # 왼쪽으로 꺾은 후에 후진함.
        print("Part - Step #6 Left & Back")
        for i in range(5): 
            drive(-35, -Fix_Speed)  # drive(angle, speed)
            time.sleep(0.2)

#=============================================
# 실질적인 메인 함수 
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함. 
#=============================================
def start():

    global image, img_ready
    global motor
    global lidar_msg, ultra_msg
    prev_x_left = 0
    prev_x_right = WIDTH
    LANE = 1
    PARKING = 2
    TURN = 3
    YOLO = 4
    SENSOR = 5
    FINISH = 6
    END = 7
    x_pos = 0

    direction = ""
    drive_mode = 0
    drive_mode = LANE

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('h_drive')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)
    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback)
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, box_callback)

    print ("----- Xycar self driving -----")

    # 첫번째 토픽이 도착할 때까지 기다림.

    while ar_ready == False:
        continue
    print("AR Detector Ready ---------")

    while not image.size == (WIDTH * HEIGHT * 3):
        continue
    print("Camera Ready --------------")

    while lidar_msg == None:
        continue
    print("LiDAR Ready ----------")

    while ultra_msg == None:
        continue
    print("UltraSonic Ready ----------")

    #while box_data == None:
    #    continue
    #print("Object Detection Ready ----------")
    
    #=========================================
    # 메인 루프 
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서 
    # "이미지처리 +차선위치찾기 +조향각결정 +모터토픽발행" 
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # ======================================
        # 차선을 보고 주행합니다.
        # AR 또는 정지선에 따라 모드를 변경합니다.
        # ======================================
        while drive_mode == LANE:
           
            #print("LANE Drive")

            lane_drive()
            ar_ID, distance, x_pos = check_AR()

            if (ar_ID != 99):
                print("AR ID = " + str(ar_ID))

            if (ar_ID == 0):
                drive_mode = PARKING 

            if (ar_ID == 1 and distance < 1.0): 
                drive_mode = YOLO
            
            result = check_stopline()

            if (result == "Stopline"):
                print("Stop Line Detected")
                drive_mode = TURN 

            if (result == "Crosswalk"): 
                print("Cross Walk Detected") 
                wait()
                            
        # ======================================
        # 여기서부터는 주차공간에 차를 세웠다가
        # 잠시 기다린 후에 
        # 다시 차선으로 복귀합니다. 
        # ======================================
        if drive_mode == PARKING:
           
            print("Parking Parking")
            do_parking()

            drive_mode = LANE
            time.sleep(1)

        # ======================================
        # 여기서부터는 차량의 방향을 변경합니다.
        # 죄로 꺾은 후에 후진 + 잠시 기다린 후에 + 조금 전진 
        # 다시 차선으로 복귀할 수 있는 위치와 자세가 됩니다. 
        # ======================================
        if drive_mode == TURN:
           
            print("TURN TURN")
            do_turn()

            drive_mode = LANE
            time.sleep(1)

        # ======================================
        # 학습된 머신러닝 모델을 사용하여
        # 앞에 있는 2개의 사진을 인식해서
        # 정답 사진쪽 갈림길로 들어서고 
        # 센서 주행 모드로 변경합니다. 
        # ======================================
        if drive_mode == YOLO:
           
            print("YOLO Drive")
            direction = yolo_drive()
            do_fork(direction)        

            drive_mode = SENSOR
            time.sleep(1)

        # ======================================
        # 라이다 및 초음파 센서로 주행합니다.
        # AR이 보이면 FINISH 모드로 변경합니다.
        # ======================================
        while drive_mode == SENSOR:
           
            print("Sensor Drive") 
            lidar_data = []
            lidar_data = lidar_filtering()
            sensor_drive(lidar_data)

            ar_ID, distance, x_pos = check_AR()

            if (ar_ID == 2 and distance < 1.8):
                drive_mode = FINISH  

        # ======================================
        # AR을 보고 주차합니다.
        # AR 방향으로 주행 후 앞에서 정차합니다. 
        # 정차 후 다시 출발지점으로 돌아갑니다.
        # ======================================
        if drive_mode == FINISH:
           
            print("Finish Finish")
            ar_drive(direction)
            drive_mode = LANE
            time.sleep(1)
     
    cv2.destroyAllWindows()


#=============================================
# 메인 함수 호툴
# if 조건문 안으로 들어가 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()


