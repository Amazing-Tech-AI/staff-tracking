import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QSlider
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10

import csv
import datetime  # Thêm thư viện datetime
import matplotlib.pyplot as plt

# Min/Max số người trong camera
max_people_count = 0
min_people_count = float('inf')  # Để tìm giá trị nhỏ nhất ban đầu

# Min/Max số người trong từng zone
zone_min_people = {}
zone_max_people = {}

# Thời gian xuất hiện lâu nhất/ngắn nhất trong camera
person_appearance_times = {}  # {track_id: {'first_seen': timestamp, 'last_seen': timestamp, 'total_time': seconds}}
longest_appearance = {'track_id': None, 'time': 0}
shortest_appearance = {'track_id': None, 'time': float('inf')}
# Thêm biến toàn cục để lưu giá trị trong 30s gần nhất
max_longest_time_30s = 0
min_shortest_time_30s = float('inf')

people_count_log = []  # Lưu số lượng người vào/ra theo thời gian


# Biến toàn cục để vẽ và quản lý zone
drawing = False
start_point = (0, 0)
end_point = (0, 0)
fixed_zones = []
zone_ids = []
track_to_fixed_id = {}
track_start_time = {}

class VideoApp(QMainWindow):
    def __init__(self):
        self.hide_tracking = False
        self.zone_counter = 1  # Bắt đầu đánh số zone từ 1
        
        super().__init__()

        self.start_time = cv2.getTickCount() / cv2.getTickFrequency()  # Lưu thời điểm bắt đầu (giây)
        self.total_people_count = 0  # Tổng số người đã vào camera (cộng dồn)
        self.previous_people_count = 0  # Số người trong camera ở lần cập nhật trước
        self.person_out_last_30s = 0  # Số người rời khỏi camera trong 30 giây gần nhất
        
        self.setWindowTitle("Object Tracking with Zones")
        self.setGeometry(100, 100, 1024, 768)
        # Create URL input field
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("Enter RTMP URL")
        
        # Connect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_stream)
        
        # QLabel để hiển thị video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.video_label.setCursor(Qt.CursorShape.CrossCursor)
        self.video_label.setMouseTracking(True)

        # Các nút điều khiển
        # self.start_button = QPushButton("Start")
        # self.start_button.clicked.connect(self.start_video)

        # self.pause_button = QPushButton("Pause")
        # self.pause_button.clicked.connect(self.pause_video)
        self.toggle_button = QPushButton("Start")
        self.toggle_button.clicked.connect(self.toggle_video)
        self.toggle_button.setEnabled(False)  # Disable until connected

        self.stop_button = QPushButton("Stop")  # Nút STOP
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        self.slider = QSlider(Qt.Orientation.Horizontal)  # Thanh trượt video
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.url_input)
        layout.addWidget(self.connect_button)        
        layout.addWidget(self.video_label)
        # layout.addWidget(self.start_button)
        # layout.addWidget(self.pause_button)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.slider)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Khởi tạo video và YOLO
        # self.cap = cv2.VideoCapture("./data/coffeshop.mp4")
        self.model = YOLOv10("./weights/yolov10x.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = DeepSort(max_age=20, n_init=3)

        # Timer để cập nhật frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Gắn sự kiện chuột vào QLabel
        self.video_label.mousePressEvent = self.mousePressEvent
        self.video_label.mouseReleaseEvent = self.mouseReleaseEvent

        # Frame cuối cùng khi pause (để vẫn vẽ được zone)
        self.cap = None
        self.last_frame = None
        self.is_running = False  # Trạng thái video (đang chạy hay tạm dừng)
        self.video_duration = 0  # Tổng thời gian video

        # Timer để xuất file mỗi 30s
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.export_statistics)
        self.stats_timer.start(30000)  # 30 giây


    def connect_to_stream(self):
        """Connect to RTMP stream."""
        rtmp_url = self.url_input.text().strip()
        if not rtmp_url:
            return

        # Close existing capture if any
        if self.cap is not None:
            self.cap.release()

        # Try to connect to the stream
        self.cap = cv2.VideoCapture(rtmp_url)
        
        if not self.cap.isOpened():
            self.video_label.setText("Failed to connect to stream")
            return

        # Enable controls if connection successful
        self.toggle_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.slider.setEnabled(True)
        self.connect_button.setText("Reconnect")

        self.video_duration = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.video_duration)
        
    def toggle_video(self):
        """Chuyển đổi giữa Start và Pause."""
        if self.is_running:
            # Đang chạy → Tạm dừng
            self.is_running = False
            self.timer.stop()
            self.toggle_button.setText("Start")  # Đổi tên nút
        else:
            # Đang dừng → Chạy tiếp
            self.is_running = True
            self.timer.start(30)
            self.toggle_button.setText("Pause")  # Đổi tên nút

            # Khi chạy tiếp, giữ lại những zone đã vẽ trước đó
            if self.last_frame is not None:
                updated_frame = self.last_frame.copy()
                updated_frame = self.draw_fixed_zones(updated_frame)  # Hiển thị zone đã vẽ
                self.display_frame(updated_frame)  # Cập nhật giao diện

    def stop_video(self):
        """Dừng video và đặt về đầu."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Đặt về frame đầu tiên
            self.is_running = False
            self.timer.stop()
            self.toggle_button.setText("Start")
            self.slider.setValue(0)  # Cập nhật thanh trượt

            # Reset tất cả dữ liệu tracking nhưng giữ vùng zone và file CSV
            self.reset_tracking_data()

            # Cập nhật lại frame đầu tiên
            ret, frame = self.cap.read()
            if ret:
                frame = self.draw_fixed_zones(frame)  # Giữ lại vùng zone
                self.display_frame(frame)  # Cập nhật giao diện
    
    def reset_tracking_data(self):
        """Reset tất cả dữ liệu tracking nhưng giữ vùng zone và dữ liệu CSV."""
        global person_appearance_times, longest_appearance, shortest_appearance, track_to_fixed_id, track_start_time

        # Xóa tất cả dữ liệu tracking
        # time_camera.clear()  # Xóa danh sách theo dõi thời gian của từng người
        track_to_fixed_id.clear()  # Xóa ID của tracking
        track_start_time.clear()  # Xóa thời gian bắt đầu của tracking

        self.total_people_count = 0  # Reset tổng số người vào
        self.previous_people_count = 0  # Reset số người trong camera
        self.person_out_last_30s = 0  # Reset số người rời đi

        longest_appearance = {'track_id': None, 'time': 0}
        shortest_appearance = {'track_id': None, 'time': float('inf')}


        # Reset bộ đếm ID của tracker (bắt đầu lại từ 0)
        self.tracker = DeepSort(max_age=20, n_init=3)

    def update_frame(self):
        """Cập nhật frame video và xử lý tracking."""
        global max_people_count, min_people_count  
        ret, frame = self.cap.read()

        if not ret:  # Video đã kết thúc
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            # if ret:
            #     self.last_frame = frame.copy()
            #     self.display_frame(frame)
            self.is_running = False
            self.toggle_button.setText("Start")  # Đổi lại nút "Start"
            self.slider.setValue(0)  # Reset thanh trượt về đầu

            # Reset toàn bộ tracking như khi bấm "Stop"
            self.reset_tracking_data()

            # Cập nhật lại frame đầu tiên với vùng zone giữ nguyên
            ret, frame = self.cap.read()
            if ret:
                frame = self.draw_fixed_zones(frame)  # Giữ vùng zone
                self.display_frame(frame)  # Cập nhật giao diện

            return  # Không xử lý tiếp khi video đã kết thúc
        
        zone_counts = {zone_id: 0 for zone_id in zone_ids}  # Tạo dictionary đếm số lượng người trong từng zone

        global longest_time_camera, longest_time_zone

        # Chỉ chạy YOLO khi video đang chạy (không chạy khi tua)
        if self.is_running:
            results = self.model(frame, verbose=False)[0]
            detections = []
            for det in results.boxes:
                label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)

                if class_id == 0:  # Chỉ tracking người
                    detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Đếm số người có mặt trong camera
            person_count = sum(1 for track in tracks if track.is_confirmed())

            # **Tính số người vào/ra**
            if person_count > self.previous_people_count:
                self.total_people_count += (person_count - self.previous_people_count)  # Cộng dồn số người vào

            if person_count < self.previous_people_count:
                self.person_out_last_30s = self.previous_people_count - person_count  # Số người rời đi trong 30s gần nhất
            else:
                self.person_out_last_30s = 0  # Nếu số lượng không giảm, không có ai rời đi

            self.previous_people_count = person_count  # Cập nhật số người trong camera hiện tại

            # Theo dõi thời gian của từng người trong camera
            # Thời gian hiện tại
            current_timestamp = cv2.getTickCount() / cv2.getTickFrequency()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # Nếu lần đầu xuất hiện, lưu thời gian bắt đầu
                if track_id not in person_appearance_times:
                    person_appearance_times[track_id] = {
                        'first_seen': current_timestamp,
                        'last_seen': current_timestamp,
                        'total_time': 0
                    }
                else:
                    person_appearance_times[track_id]['last_seen'] = current_timestamp
                    appearance_duration = current_timestamp - person_appearance_times[track_id]['first_seen']
                    person_appearance_times[track_id]['total_time'] = appearance_duration

                    # Cập nhật thời gian lâu nhất
                    if appearance_duration > longest_appearance['time']:
                        longest_appearance['track_id'] = track_id
                        longest_appearance['time'] = appearance_duration

                    # Cập nhật thời gian ngắn nhất (chỉ tính người đã xuất hiện ít nhất 2s)
                    if appearance_duration >= 2.0 and appearance_duration < shortest_appearance['time']:
                        shortest_appearance['track_id'] = track_id
                        shortest_appearance['time'] = appearance_duration

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                fixed_id = self.get_fixed_id_for_track(track_id, (x1, y1, x2, y2))
                if fixed_id is not None:
                    zone_counts[fixed_id] += 1  # Cộng số lượng người trong zone đó

            # Cập nhật min/max số người trong từng zone
            for zone_id, count in zone_counts.items():
                if zone_id not in zone_max_people:
                    zone_max_people[zone_id] = 0
                if zone_id not in zone_min_people:
                    zone_min_people[zone_id] = 0

                if count > 0:
                    if zone_min_people[zone_id] == 0:  
                        zone_min_people[zone_id] = count  # Nếu trước đó min là 0, cập nhật thành số người đầu tiên
                    else:
                        zone_min_people[zone_id] = min(zone_min_people[zone_id], count)  # Tính min như bình thường

                if count > zone_max_people[zone_id]:  # Cập nhật max như bình thường
                    zone_max_people[zone_id] = count

            # Cập nhật giá trị min/max số người trong camera
            if person_count > max_people_count:
                max_people_count = person_count

            if person_count > 0 and person_count < min_people_count:
                min_people_count = person_count

            # Vẽ vùng zone
            frame = self.draw_fixed_zones(frame)

            # Nếu không phải Pause thì hiển thị tracking
            if not self.hide_tracking:
                frame = self.draw_tracks(frame, tracks)

        # Cập nhật thanh trượt
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)  # Ngăn chặn sự kiện khi cập nhật thanh trượt
        self.slider.setValue(current_frame)
        self.slider.blockSignals(False)

        # Tạo overlay nền để hiển thị thông tin
        overlay = frame.copy()
        stats_x = frame.shape[1] - 250  # Góc trên bên phải
        line_spacing = 25  # Khoảng cách giữa các dòng

        # Tính toán chiều cao của overlay dựa trên số lượng zone
        base_height = 180  # Chiều cao tối thiểu chứa các thông tin chung
        zone_height = len(zone_counts) * 40  # Mỗi zone chiếm khoảng 40 pixels
        overlay_height = base_height + zone_height  # Tổng chiều cao

        cv2.rectangle(overlay, (stats_x, 10), (frame.shape[1] - 10, overlay_height), (0, 0, 0), -1)

        alpha = 0.6  # Độ trong suốt
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Hiển thị số liệu min/max trong camera
        cv2.putText(frame, f"People: {person_count}", (stats_x + 10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Max: {max_people_count}", (stats_x + 10, 40 + line_spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Min: {min_people_count if min_people_count != float('inf') else 'N/A'}", 
                    (stats_x + 10, 40 + 2 * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Hiển thị thời gian longest/shortest trong camera
        longest_time = f"{longest_appearance['time']:.1f}s" if longest_appearance['track_id'] is not None else "N/A"
        shortest_time = f"{shortest_appearance['time']:.1f}s" if shortest_appearance['track_id'] is not None else "N/A"

        cv2.putText(frame, f"Longest: {longest_time}", (stats_x + 10, 40 + 3 * line_spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Shortest: {shortest_time}", (stats_x + 10, 40 + 4 * line_spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # Hiển thị số người trong từng zone ngay dưới Shortest
        y_offset = 40 + 5 * line_spacing  # Đặt vị trí ngay sau "Shortest"

        for zone_id, count in zone_counts.items():
            min_people = zone_min_people.get(zone_id, "N/A")
            max_people = zone_max_people.get(zone_id, 0)

            cv2.putText(frame, f"Zone {zone_id}", (stats_x + 10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Min: {min_people}, Max: {max_people}", 
                        (stats_x + 10, y_offset + line_spacing), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += 2 * line_spacing  # Di chuyển xuống cho Zone tiếp theo

        self.last_frame = frame.copy()
        self.display_frame(frame)


    def export_statistics(self):
        """Ghi dữ liệu vào file CSV mỗi 30 giây."""
        global people_count_log, longest_appearance, shortest_appearance # Tính số giây từ lúc bắt đầu
        global max_longest_time_30s, min_shortest_time_30s  # Biến lưu giá trị theo 30 giây

        if not self.is_running:
            return  # Nếu video không chạy thì không ghi file

        # Lấy ngày & giờ hiện tại
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")  # Định dạng ngày: YYYY-MM-DD
        time_str = now.strftime("%H:%M:%S")  # Định dạng giờ: HH:MM:SS

        file_exists = False
        try:
            with open("people_statistics.csv", "r") as file:
                file_exists = bool(file.readline())  # Kiểm tra xem file có dữ liệu chưa
        except FileNotFoundError:
            pass  # File chưa tồn tại, sẽ tạo mới

        # Cập nhật Longest Camera (thời gian lớn nhất trong 30s)
        if longest_appearance['track_id'] is not None:
            max_longest_time_30s = max(max_longest_time_30s, longest_appearance['time'])

        # Cập nhật Shortest Camera (thời gian nhỏ nhất trong 30s)
        if shortest_appearance['track_id'] is not None:
            min_shortest_time_30s = min(min_shortest_time_30s, shortest_appearance['time'])

        # Lấy ID của longest/shortest trong zone
        longest_id = longest_appearance['track_id'] if longest_appearance['track_id'] is not None else "N/A"
        shortest_id = shortest_appearance['track_id'] if shortest_appearance['track_id'] is not None else "N/A"

        # Ghi vào file CSV
        with open("people_statistics.csv", "a", newline="") as file:
            writer = csv.writer(file)

            # Nếu file mới, ghi header
            if not file_exists:
                writer.writerow(["Date", "Time", "People Count", "Person Out",
                                "Longest Camera", "Shortest Camera",
                                "Longest Zone", "Shortest Zone"])
                
            writer.writerow([date_str, time_str, self.total_people_count, self.person_out_last_30s, 
                            round(max_longest_time_30s, 2), round(min_shortest_time_30s, 2), longest_id, shortest_id])

        # Reset lại giá trị sau khi ghi file để bắt đầu tính cho 30 giây tiếp theo
        max_longest_time_30s = 0
        min_shortest_time_30s = float('inf')

        # Cập nhật biểu đồ
        self.update_chart()

    def update_chart(self):
        """Vẽ biểu đồ từ dữ liệu CSV."""
        counts,out_counts, cam_times, zone_times = [], [], [], []

        with open("people_statistics.csv", "r") as file:
            reader = csv.reader(file)
            next(reader, None)  # Bỏ qua dòng tiêu đề (header)

            for row in reader:
                if len(row) < 6:
                    continue # Bỏ qua dòng không đủ dữ liệu
                try:
                    counts.append(int(row[2]))
                    out_counts.append(int(row[3]))
                    cam_times.append(float(row[4]))
                    zone_times.append(float(row[5]))
                except ValueError:
                    continue  # Bỏ qua dòng lỗi

        if not counts:  # Nếu không có dữ liệu hợp lệ thì không vẽ
            return

        plt.figure(figsize=(10, 5))

        # Biểu đồ 1: Số người theo thời gian
        plt.subplot(2, 1, 1)
        plt.plot(range(len(counts)),counts, label="People Count", marker="o")
        plt.plot(range(len(out_counts)), out_counts, label="Person Out", marker="x", color="red")
        plt.xlabel("Measurement Inde")
        plt.ylabel("People Count")
        plt.legend()
        plt.grid(True)

        # Biểu đồ 2: Thời gian quan sát lâu nhất
        plt.subplot(2, 1, 2)
        plt.plot(range(len(cam_times)), cam_times, label="Longest in Camera", color="r", marker="s")
        plt.plot(range(len(zone_times)), zone_times, label="Longest in Zone", color="g", marker="d")
        plt.xlabel("Measurement Index")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("people_chart.png")
        plt.close()

    def set_position(self, position):
        """Di chuyển video đến vị trí mới khi kéo thanh trượt."""
        if self.cap:
            was_running = self.is_running  # Lưu trạng thái trước khi tua
            self.is_running = False  # Tạm dừng video khi tua
            self.timer.stop()

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)  # Hiển thị frame mới mà không chạy YOLO
            # Khôi phục trạng thái trước khi tua
            self.is_running = was_running
            if was_running:
                self.timer.start(15)  

    def draw_fixed_zones(self, frame):
        """Vẽ các vùng zone cố định."""
        for i, zone in enumerate(fixed_zones):
            cv2.rectangle(frame, zone[0], zone[1], (0, 255, 255), 2)
            cv2.putText(frame, f"Zone {zone_ids[i]}", (zone[0][0] + 5, zone[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame

    def draw_tracks(self, frame, tracks):
        """Vẽ bounding box, ID và kiểm tra thời gian trong zone."""
        global track_start_time, track_to_fixed_id, person_appearance_times, longest_appearance, shortest_appearance

        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Thời gian xuất hiện trong camera
            if track_id not in person_appearance_times:
                person_appearance_times[track_id] = {'first_seen': current_time}

            time_in_camera = current_time - person_appearance_times[track_id]['first_seen']

            # Xác định longest/shortest
            bbox_color = (0, 255, 0)  # Mặc định màu xanh lá
            status_text = None  # Không hiển thị gì nếu không phải longest/shortest
            text_color = (255, 255, 255)  # Màu trắng mặc định

            if track_id == longest_appearance['track_id']:
                bbox_color = (0, 0, 255)  # Bounding box màu đỏ
                status_text = "Longest"
                text_color = (0, 0, 255)  # Chữ màu đỏ
            elif track_id == shortest_appearance['track_id']:
                bbox_color = (255, 0, 0)  # Bounding box màu xanh dương
                status_text = "Shortest"
                text_color = (255, 0, 0)  # Chữ màu xanh dương

            fixed_id = self.get_fixed_id_for_track(track_id, (x1, y1, x2, y2))
            track_in_zone = fixed_id is not None  # Kiểm tra xem track có trong zone không

            # # Nếu object đã có ID trước đó
            # has_id = track_id in track_to_fixed_id

            if track_in_zone:
                # Tính thời gian ở trong zone
                elapsed_time = current_time - track_start_time.get(track_id, 0)
                # Hiển thị ID và Time trong zone
                display_text = f"ID: {fixed_id} Time: {elapsed_time:.1f}s"

                # Nếu là longest/shortest, vẽ chữ trạng thái **trên dòng ID Time**
                if status_text:
                    cv2.putText(frame, status_text, (x1, y1 - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                # Vẽ ID Time
                cv2.putText(frame, display_text, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                continue
            else:
                # Nếu track rời khỏi zone, không hiển thị ID và Time
                if track_id in track_start_time:
                    del track_start_time[track_id]  # Xóa thời gian theo dõi của track

            # Vẽ bounding box với màu tương ứng
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

            # Nếu là longest/shortest, vẽ chữ trạng thái **trên dòng P.ID Time**
            if status_text:
                cv2.putText(frame, status_text, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            cv2.putText(frame, f"P.{track_id} Time: {time_in_camera:.1f}s", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                #P. {track_id}
        return frame

    def get_fixed_id_for_track(self, track_id, track_bbox):
        """Xác định xem track_id có nằm trong vùng zone với ít nhất 80% diện tích không."""
        global track_to_fixed_id, track_start_time

        # if track_id in track_to_fixed_id:
        #     return track_to_fixed_id[track_id]

        x1, y1, x2, y2 = track_bbox
        bbox_area = (x2 - x1) * (y2 - y1)  # Diện tích bounding box
        
        best_zone_id = None
        max_overlap_ratio = 0

        for i, zone in enumerate(fixed_zones):
            fx1, fy1 = zone[0]
            fx2, fy2 = zone[1]

            # Tính giao nhau giữa bounding box và zone
            ix1 = max(x1, fx1)
            iy1 = max(y1, fy1)
            ix2 = min(x2, fx2)
            iy2 = min(y2, fy2)

            if ix1 < ix2 and iy1 < iy2:  # Kiểm tra nếu có giao nhau
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection_area / bbox_area  # Tính phần trăm diện tích giao nhau

                if overlap_ratio > max_overlap_ratio:  # Nếu ít nhất 80% diện tích trong vùng zone
                    max_overlap_ratio = overlap_ratio
                    best_zone_id = zone_ids[i]
                    # return zone_ids[i]
                
    # Nếu object rời khỏi zone, xóa khỏi danh sách theo dõi
        if max_overlap_ratio >= 0.8:
            track_to_fixed_id[track_id] = best_zone_id
            if track_id not in track_start_time:  # Chỉ gán thời gian lần đầu vào zone
                track_start_time[track_id] = cv2.getTickCount() / cv2.getTickFrequency()
            return best_zone_id
        
        # if track_id in track_start_time:
        #     # del track_to_fixed_id[track_id]
        #     del track_start_time[track_id]

        return None

    def display_frame(self, frame):
        """Chuyển đổi frame OpenCV thành QPixmap để hiển thị trên QLabel."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        """Bắt sự kiện chuột để vẽ và xóa zone nhưng chỉ khi video đang Pause."""
        global drawing, start_point

        if self.is_running:  # Nếu video đang chạy, không cho phép vẽ hoặc xóa
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Bắt đầu vẽ zone
            drawing = True

            # Lấy kích thước thực tế của frame video
            if self.last_frame is not None:
                frame_height, frame_width, _ = self.last_frame.shape  # Lấy kích thước frame thực sự
                label_width = self.video_label.width()
                label_height = self.video_label.height()

                # Chuyển đổi tọa độ từ QLabel sang frame video
                scale_x = frame_width / label_width
                scale_y = frame_height / label_height

                start_point = (int(event.pos().x() * scale_x), int(event.pos().y() * scale_y))

        elif event.button() == Qt.MouseButton.RightButton:
            # Xóa zone khi bấm chuột phải
            zone_deleted = False  # Đánh dấu nếu có zone nào bị xóa
            for i in range(len(fixed_zones) - 1, -1, -1):  # Lặp ngược để xóa đúng vị trí
                fx1, fy1 = fixed_zones[i][0]
                fx2, fy2 = fixed_zones[i][1]

                # Chuyển đổi tọa độ chuột hiện tại sang frame video trước khi so sánh
                if self.last_frame is not None:
                    frame_height, frame_width, _ = self.last_frame.shape
                    label_width = self.video_label.width()
                    label_height = self.video_label.height()

                    scale_x = frame_width / label_width
                    scale_y = frame_height / label_height

                    mouse_x = int(event.pos().x() * scale_x)
                    mouse_y = int(event.pos().y() * scale_y)

                    if fx1 <= mouse_x <= fx2 and fy1 <= mouse_y <= fy2:
                        deleted_zone_id = zone_ids[i]  # Lấy ID của vùng bị xóa
        
                        # Xóa vùng khỏi danh sách
                        del fixed_zones[i]
                        del zone_ids[i]

                        # Xóa ID của những người thuộc vùng bị xóa để họ được tracking lại
                        global track_to_fixed_id, track_start_time
                        track_to_fixed_id = {tid: zid for tid, zid in track_to_fixed_id.items() if zid != deleted_zone_id}
                        track_start_time = {tid: time for tid, time in track_start_time.items() if tid not in track_to_fixed_id}

                        zone_deleted = True  # Đánh dấu rằng có zone bị xóa
                        break  # Thoát vòng lặp sau khi xóa một vùng

            # **Hiển thị ngay lập tức sau khi xóa**
            if zone_deleted and self.last_frame is not None:
                updated_frame = self.last_frame.copy()
                updated_frame = self.draw_fixed_zones(updated_frame)  # Vẽ lại frame không có zone bị xóa
                self.last_frame = updated_frame.copy()  # Lưu lại frame mới nhất
                self.display_frame(updated_frame)  # Hiển thị ngay lập tức  
              

    def mouseReleaseEvent(self, event):
        """Kết thúc vẽ zone nhưng chỉ khi video đang Pause."""
        global drawing, end_point, fixed_zones, zone_ids

        if self.is_running:  # Nếu video đang chạy, không cho phép vẽ
            return

        if event.button() == Qt.MouseButton.LeftButton:
            drawing = False

            # Lấy kích thước frame thực tế
            if self.last_frame is not None:
                frame_height, frame_width, _ = self.last_frame.shape
                label_width = self.video_label.width()
                label_height = self.video_label.height()

                local_pos = self.video_label.mapFromGlobal(event.globalPosition().toPoint())
                mouse_x = local_pos.x()
                mouse_y = local_pos.y()

                # Chuyển đổi tọa độ từ QLabel sang frame video
                scale_x = frame_width / label_width
                scale_y = frame_height / label_height

                end_point = (int(mouse_x * scale_x), int(mouse_y * scale_y))

                x1, y1 = start_point
                x2, y2 = end_point

                # Đảm bảo luôn lưu theo thứ tự (trái, trên) -> (phải, dưới)
                fixed_zones.append(((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))))
                zone_ids.append(self.zone_counter)  
                self.zone_counter += 1  # Tăng số thứ tự cho zone tiếp theo

                # Cập nhật frame ngay sau khi vẽ zone
                updated_frame = self.last_frame.copy()
                updated_frame = self.draw_fixed_zones(updated_frame)  # Vẽ lại với zone mới

                self.last_frame = updated_frame.copy()  # Cập nhật frame mới nhất
                self.display_frame(updated_frame)  # Hiển thị ngay lập tức

    def mouseMoveEvent(self, event):
        """Hiển thị khung tạm thời khi đang vẽ vùng."""
        global drawing, end_point

        if self.is_running:  # Nếu video đang chạy, không cho phép vẽ
            return

        # self.video_label.setCursor(Qt.CursorShape.CrossCursor)

        if drawing:
            # Lấy kích thước gốc của frame video
            if self.last_frame is not None:
                frame_height, frame_width, _ = self.last_frame.shape  # Lấy kích thước frame thực sự
                label_width = self.video_label.width()
                label_height = self.video_label.height()

                # Chuyển đổi tọa độ từ QLabel sang frame video
                local_pos = self.video_label.mapFromGlobal(event.globalPosition().toPoint())
                mouse_x = local_pos.x()
                mouse_y = local_pos.y()

                scale_x = frame_width / label_width
                scale_y = frame_height / label_height

                end_point = (int(mouse_x * scale_x), int(mouse_y * scale_y))

                # Hiển thị khung vẽ khi đang kéo chuột
                updated_frame = self.last_frame.copy()
                cv2.rectangle(updated_frame, start_point, end_point, (255, 0, 0), 2)  # Vẽ khung xanh

                updated_frame = self.draw_fixed_zones(updated_frame)  # Vẽ lại các zone đã có
                self.display_frame(updated_frame)  # Cập nhật hiển thị

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
