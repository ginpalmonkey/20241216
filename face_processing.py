import cv2
import mediapipe as mp
import time
import threading
import numpy as np
import requests
from keras_facenet import FaceNet
from eye_blink import detect_eye_blink
from head_turn import detect_head_turn
from draw_utils import draw_text_kor
from time_measurement import initialize_time_variables, update_sleep_time, update_turn_time
from calculate import cal_headturn, cal_sleeptime, cal_blinks_per_minute, cal_blink_points, totalscore

SERVER_URL = "http://192.168.0.117:9000"


# # Mediapipe Face Mesh 초기화
facenet = FaceNet()
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

class FaceProcessingThread(threading.Thread):
    def __init__(self, video_capture, user_id):
        super().__init__()
        self.video_capture = video_capture
        self.user_id = user_id
        self.running = True
        self.face_matched = False
        self.name = "Unknown"
        self.frame_to_display = None

        # 시간 측정 Param
        self.SLEEP_TIME = 2.0
        self.CALIBRATION_TIME = 5.0

        # 페이스메쉬 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 집중도 관련 변수
        self.calibration_start = time.time()
        self.calibration_ears = []
        self.EAR_THRESH = None
        self.blink_start = None
        self.blink_count = 0
        self.eyes_closed = False
        # 동적 임계값 조정에 필요함
        self.ear_values = []
        self.calibration_window = 100
        self.frames_closed = 0

        # 시간 함수 초기화
        self.start_time, self.total_sleep_time, self.sleep_start, self.total_turn_time, self.turn_start = initialize_time_variables()

    def run(self):

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
             mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            while self.running:
                frame = self.video_capture.read()
                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_results = face_detection.process(rgb_frame)
                mesh_results = face_mesh.process(rgb_frame)

                if detection_results.detections:
                    largest_face = max(
                        detection_results.detections,
                        key=lambda det: det.location_data.relative_bounding_box.width *
                                        det.location_data.relative_bounding_box.height
                    )
                    bboxC = largest_face.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    face = frame[max(0, y):max(0, y + height), max(0, x):max(0, x + width)]
                    face_resized = cv2.resize(face, (160, 160))
                    embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

                    try:
                        data = {"id": self.user_id, "embedding": embedding.tolist()}
                        response = requests.post(f"{SERVER_URL}/verify-face", json=data)
                        if response.status_code == 200:
                            self.name = response.json().get("name", "Unknown")
                            self.face_matched = True
                        else:
                            self.face_matched = False
                    except Exception:
                        self.face_matched = False

                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    display_texts = []
                    # 얼굴 매칭 성공시 집중도 분석
                    if self.face_matched and mesh_results.multi_face_landmarks:
                        for facial_landmarks in mesh_results.multi_face_landmarks:
                            h, w, _ = frame.shape

                            # Detect eye blink
                            (self.EAR_THRESH, self.blink_start, self.blink_count, self.eyes_closed, current_state,
                             calibration_message, self.frames_closed) = detect_eye_blink(
                                facial_landmarks, w, h, self.EAR_THRESH, self.SLEEP_TIME, self.CALIBRATION_TIME,
                                self.calibration_start, self.calibration_ears, self.blink_start, self.blink_count,
                                self.eyes_closed, self.ear_values, calibration_window=self.calibration_window,
                                min_blink_duration=0.15, frames_closed=self.frames_closed, frames_required=3
                            )

                            # Detect head turn
                            head_direction = detect_head_turn(facial_landmarks, w, h)

                            # 수면시간, 고개돌림시간 추가
                            self.sleep_start, self.total_sleep_time = update_sleep_time(current_state, self.sleep_start,
                                                                                        self.total_sleep_time,
                                                                                        current_time)
                            self.turn_start, self.total_turn_time = update_turn_time(head_direction, self.turn_start,
                                                                                     self.total_turn_time, current_time)

                            if calibration_message:
                                display_texts.append(calibration_message)

                            display_texts.append(
                                f"머리 방향: {head_direction}, 상태: {current_state}, blinks: {self.blink_count}")

                            current_sleep = 0 if self.sleep_start is None else (current_time - self.sleep_start)
                            current_turn = 0 if self.turn_start is None else (current_time - self.turn_start)

                            display_texts.append(f"총 녹화 시간: {int(elapsed_time)}s")
                            display_texts.append(f"총 수면 시간: {int(self.total_sleep_time + current_sleep)}s")
                            display_texts.append(f"총 고개 돌린 시간: {int(self.total_turn_time + current_turn)}s")
                    else:
                        cv2.putText(frame, "Face Not Matched", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    for i, text in enumerate(display_texts):
                        frame = draw_text_kor(frame, text, (10, 30 + i * 40))

                self.frame_to_display = frame

                # 종료시 수면, 고개돌림 시간 처리

            if self.sleep_start is not None:
                self.total_sleep_time += (time.time() - self.sleep_start)
            if self.turn_start is not None:
                self.total_turn_time += (time.time() - self.turn_start)


    def stop(self):
        # 점수 계산
        final_elapsed_time = time.time() - self.start_time
        sleep_point = cal_sleeptime(self.total_sleep_time, final_elapsed_time)
        ht_point = cal_headturn(int(final_elapsed_time - self.total_turn_time), int(final_elapsed_time))
        blink_perm = cal_blinks_per_minute(self.blink_count, final_elapsed_time)
        blink_point = cal_blink_points(int(blink_perm))

        total = totalscore(sleep_point, ht_point, blink_point)

        # 결과 출력
        print(int(final_elapsed_time))
        print(f"수면시간 : {int(self.total_sleep_time)}")
        print(f"고개돌림 : {int(self.total_turn_time)}")
        print(f"집중 점수 : {total}")