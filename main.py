import sys
import cv2
import datetime
import psycopg2
import csv
from collections import deque  # Для ковзного середнього
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QScrollArea, QLineEdit, QMessageBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# === Параметри підключення до PostgreSQL ===
DB_CONFIG = {
    "dbname": "tracker",
    "user": "postgres",
    "password": "Admin",
    "host": "localhost",
    "port": "5432"
}

# === Підключення до бази даних ===
conn = None
cur = None

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("Підключення до бази даних успішне!")

    # Видаляємо таблиці, якщо вони існують, і створюємо нові
    cur.execute("DROP TABLE IF EXISTS TrackingData CASCADE;")
    cur.execute("DROP TABLE IF EXISTS Objects CASCADE;")
    cur.execute("DROP TABLE IF EXISTS Videos CASCADE;")

    cur.execute("""
    CREATE TABLE Videos (
        video_id SERIAL PRIMARY KEY,
        path VARCHAR(255) NOT NULL,
        duration INTEGER,
        fps INTEGER
    );

    CREATE TABLE Objects (
        track_id INTEGER PRIMARY KEY,
        object_class VARCHAR(50) NOT NULL,
        first_detected TIMESTAMP WITHOUT TIME ZONE
    );

    CREATE TABLE TrackingData (
        id SERIAL PRIMARY KEY,
        video_id INTEGER NOT NULL,
        track_id INTEGER NOT NULL,
        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        x1 INTEGER NOT NULL,
        y1 INTEGER NOT NULL,
        x2 INTEGER NOT NULL,
        y2 INTEGER NOT NULL,
        speed DOUBLE PRECISION NOT NULL,
        FOREIGN KEY (video_id) REFERENCES Videos(video_id) ON DELETE CASCADE,
        FOREIGN KEY (track_id) REFERENCES Objects(track_id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    print("Таблиці створено успішно!")
except Exception as e:
    print(f"Помилка підключення до БД: {e}")
    if conn:
        conn.rollback()
    sys.exit(1)

# === Завантаження моделі YOLOv10 ===
model = YOLO("yolov10s.pt")

# Ініціалізація DeepSORT
tracker = DeepSort(max_age=60)

# Завантаження відео
video_path = "2103099-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Помилка: не вдалося відкрити відео.")
    app = QApplication(sys.argv)
    QMessageBox.critical(None, "Помилка", "Не вдалося відкрити відео. Перевірте шлях до файлу.")
    sys.exit(1)

# Отримання FPS і тривалості відео
fps = cap.get(cv2.CAP_PROP_FPS) or 30
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

# Додаємо відео до бази даних
try:
    cur.execute("""
        INSERT INTO Videos (path, duration, fps)
        VALUES (%s, %s, %s)
        RETURNING video_id
    """, (video_path, duration, fps))
    video_id = cur.fetchone()[0]
    conn.commit()
    print(f"Відео додано до бази даних з video_id: {video_id}")
except Exception as e:
    print(f"Помилка при додаванні відео у БД: {e}")
    conn.rollback()
    cap.release()
    cur.close()
    conn.close()
    sys.exit(1)

# Збереження попередніх координат і швидкостей
previous_positions = {}  # Для координат і часу
speed_history = {}  # Для згладжування швидкості
scale_factor = 0.05
track_ids = set()
SMOOTHING_WINDOW = 5  # Кількість кадрів для ковзного середнього
MIN_TIME_DIFF = 0.01  # Мінімальний поріг для time_diff (у секундах)

def calculate_speed(track_id, x1, y1, x2, y2, current_time_ms, scale_factor_value):
    """
    Покращений розрахунок швидкості з згладжуванням.
    Використовуємо центр рамки для обчислення відстані та ковзне середнє для швидкості.
    """
    # Обчислюємо центр рамки
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Ініціалізуємо історію швидкостей для нового track_id
    if track_id not in speed_history:
        speed_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)

    if track_id in previous_positions:
        prev_center_x, prev_center_y, prev_time_ms = previous_positions[track_id]
        time_diff = (current_time_ms - prev_time_ms) / 1000.0  # Переводимо в секунди

        # Перевіряємо, чи time_diff достатньо великий
        if time_diff >= MIN_TIME_DIFF:
            # Обчислюємо відстань між центрами рамок у пікселях
            distance_px = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
            speed_px_per_sec = distance_px / time_diff
            raw_speed_kmh = speed_px_per_sec * scale_factor_value * 3.6  # Конвертація в км/год
            print(f"Track ID {track_id}: distance={distance_px:.2f}px, time_diff={time_diff:.3f}s, raw_speed={raw_speed_kmh:.1f}km/h")
        else:
            raw_speed_kmh = 0.0
    else:
        raw_speed_kmh = 0.0

    # Зберігаємо центр рамки та час для наступного кадру
    previous_positions[track_id] = (center_x, center_y, current_time_ms)

    # Додаємо нову швидкість до історії
    speed_history[track_id].append(raw_speed_kmh)

    # Обчислюємо згладжену швидкість (ковзне середнє)
    if len(speed_history[track_id]) > 0:
        smoothed_speed = sum(speed_history[track_id]) / len(speed_history[track_id])
    else:
        smoothed_speed = raw_speed_kmh

    print(f"Track ID {track_id}: smoothed_speed={smoothed_speed:.1f}km/h")
    return smoothed_speed

class VideoMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracking System")

        # Головний віджет та лайаут
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Відео відображення
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 400)
        layout.addWidget(self.video_label)

        # Статистика та звіт секція
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)

        # Статистика об'єктів
        object_stats_widget = QWidget()
        object_stats_layout = QVBoxLayout(object_stats_widget)
        self.object_stats_label = QLabel("Статистика об'єктів:")
        self.object_stats_text = QLabel()
        self.object_stats_text.setStyleSheet("font-size: 12px; margin: 5px;")
        object_stats_scroll = QScrollArea()
        object_stats_scroll.setWidget(self.object_stats_text)
        object_stats_scroll.setWidgetResizable(True)
        object_stats_scroll.setMinimumWidth(400)
        object_stats_layout.addWidget(self.object_stats_label)
        object_stats_layout.addWidget(object_stats_scroll)
        stats_layout.addWidget(object_stats_widget)

        # Секція звіту
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        self.report_label = QLabel("Звіт:")
        self.report_text = QLabel()
        self.report_text.setStyleSheet("font-size: 12px; margin: 5px;")
        report_scroll = QScrollArea()
        report_scroll.setWidget(self.report_text)
        report_scroll.setWidgetResizable(True)
        report_scroll.setMinimumWidth(400)
        self.generate_report_button = QPushButton("Згенерувати звіт")
        self.generate_report_button.clicked.connect(self.generate_report)
        report_layout.addWidget(self.report_label)
        report_layout.addWidget(report_scroll)
        report_layout.addWidget(self.generate_report_button)
        stats_layout.addWidget(report_widget)

        layout.addWidget(stats_widget)

        # Пошук за ID та кнопка очищення
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Введіть track_id")
        self.search_input.setMaximumWidth(150)
        self.search_button = QPushButton("Пошук")
        self.search_button.clicked.connect(self.search_by_track_id)
        self.clear_button = QPushButton("Очистити")
        self.clear_button.clicked.connect(self.clear_search)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.clear_button)
        layout.addLayout(search_layout)

        # Додавання поля для введення scale_factor
        scale_layout = QHBoxLayout()
        scale_layout.addStretch(1)
        scale_label = QLabel("Scale Factor (пікселі в км/год):")
        scale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_input = QDoubleSpinBox(self)
        self.scale_input.setRange(0.01, 1.0)
        self.scale_input.setSingleStep(0.01)
        self.scale_input.setValue(scale_factor)
        self.scale_input.setDecimals(3)
        self.scale_input.setFixedWidth(100)
        self.scale_input.valueChanged.connect(self.update_scale_factor)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_input)
        scale_layout.setSpacing(5)
        scale_layout.addStretch(1)
        layout.addLayout(scale_layout)

        # Кнопки управління
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)

        # Таймлайн
        self.timeline_label = QLabel("00:00")
        layout.addWidget(self.timeline_label)

        # Таймер для оновлення відео
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Підключення кнопок
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)

        self.is_processing = False
        self.searched_track_id = None
        self.tracked_object = None
        self.frame_count = 0

        # Встановлюємо повноекранний режим
        self.showFullScreen()

    def update_scale_factor(self):
        """Оновлення scale_factor при зміні значення користувачем."""
        global scale_factor
        scale_factor = self.scale_input.value()
        print(f"Scale Factor оновлено: {scale_factor}")

    def keyPressEvent(self, event):
        """Закриває програму при натисканні Esc."""
        if event.key() == Qt.Key.Key_Escape:
            print("Закриваємо програму")
            self.close()

    def start_processing(self):
        if not self.is_processing and cap.isOpened():
            self.timer.start(int(1000 / fps))
            self.is_processing = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.frame_count = 0
        else:
            QMessageBox.warning(self, "Попередження", "Не вдалося запустити обробку. Перевірте, чи відео відкрито.")

    def stop_processing(self):
        if self.is_processing:
            self.timer.stop()
            self.is_processing = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def search_by_track_id(self):
        try:
            self.searched_track_id = int(self.search_input.text())
            self.tracked_object = None
            self.update_stats()
            print(f"Пошук активовано для track_id: {self.searched_track_id}")
        except ValueError:
            self.object_stats_text.setText("Невірний формат ID. Введіть ціле число.")
            self.searched_track_id = None
            self.tracked_object = None

    def clear_search(self):
        """Очищення пошуку."""
        self.searched_track_id = None
        self.tracked_object = None
        self.search_input.clear()
        self.update_stats()
        global tracker
        tracker = DeepSort(max_age=60)
        print("Пошук очищено")

    def generate_report(self):
        try:
            query = """
                SELECT td.timestamp, td.track_id, o.object_class, td.x1, td.y1, td.x2, td.y2, td.speed
                FROM TrackingData td
                JOIN Objects o ON td.track_id = o.track_id
                WHERE td.video_id = %s
            """
            params = [video_id]
            if self.searched_track_id is not None:
                query += " AND td.track_id = %s"
                params.append(self.searched_track_id)
            query += " ORDER BY td.timestamp DESC LIMIT 50"

            cur.execute(query, params)
            rows = cur.fetchall()
            if rows:
                # Генеруємо унікальний шлях до файлу звіту
                report_filename = f"tracking_report_video_{video_id}.csv"
                report_text = "Звіт по відстежуванню об'єктів:\n\n"
                report_text += "Час | Track ID | Клас | Координати | Швидкість\n"
                report_text += "-" * 50 + "\n"
                for row in rows:
                    timestamp, track_id, obj_class, x1, y1, x2, y2, speed = row
                    report_text += f"{timestamp} | {track_id} | {obj_class} | ({x1}, {y1}, {x2}, {y2}) | {speed:.1f} км/год\n"
                self.report_text.setText(report_text)

                # Зберігаємо звіт у CSV
                with open(report_filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Track ID", "Object Class", "x1", "y1", "x2", "y2", "Speed (km/h)"])
                    for row in rows:
                        writer.writerow(row)

                self.report_text.setText(report_text + f"\n\nЗвіт збережено у '{report_filename}'")
            else:
                self.report_text.setText("Жодних даних для звіту не знайдено.")
        except Exception as e:
            self.report_text.setText(f"Помилка при генерації звіту: {e}")

    def update_stats(self):
        if self.searched_track_id is not None:
            try:
                cur.execute("""
                    SELECT td.timestamp, o.object_class, td.x1, td.y1, td.x2, td.y2, td.speed
                    FROM TrackingData td
                    JOIN Objects o ON td.track_id = o.track_id
                    WHERE td.track_id = %s AND td.video_id = %s
                    ORDER BY td.timestamp DESC
                    LIMIT 10
                """, (self.searched_track_id, video_id))
                rows = cur.fetchall()
                if rows:
                    stats_text = f"Дані для track_id {self.searched_track_id}:\n"
                    for row in rows:
                        timestamp, obj_class, x1, y1, x2, y2, speed = row
                        stats_text += f"Час: {timestamp}, Клас: {obj_class}, Координати: ({x1}, {y1}, {x2}, {y2}), Швидкість: {speed:.1f} км/год\n"
                    self.object_stats_text.setText(stats_text)
                else:
                    self.object_stats_text.setText(f"Жодних даних для track_id {self.searched_track_id} не знайдено.")
            except Exception as e:
                self.object_stats_text.setText(f"Помилка при пошуку: {e}")
        else:
            try:
                cur.execute("""
                    SELECT DISTINCT ON (td.track_id) td.track_id, o.object_class, td.speed
                    FROM TrackingData td
                    JOIN Objects o ON td.track_id = o.track_id
                    WHERE td.video_id = %s
                    ORDER BY td.track_id, td.timestamp DESC
                """, (video_id,))
                rows = cur.fetchall()
                if rows:
                    stats_text = "Статистика об'єктів:\n"
                    for row in rows:
                        track_id, obj_class, speed = row
                        stats_text += f"Track ID: {track_id}, Клас: {obj_class}, Останній вимір швидкості: {speed:.1f} км/год\n"
                    self.object_stats_text.setText(stats_text)
                else:
                    self.object_stats_text.setText("Жодних даних для відображення.")
            except Exception as e:
                self.object_stats_text.setText(f"Помилка при оновленні статистики: {e}")

    def update_frame(self):
        global track_ids
        ret, frame = cap.read()
        if not ret:
            self.stop_processing()
            QMessageBox.warning(self, "Попередження", "Не вдалося прочитати кадр з відео. Відео завершилося або сталася помилка.")
            return

        # Обчислення поточного часу з відео (у мілісекундах)
        self.frame_count += 1
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Виявлення об'єктів
        try:
            results = model(frame)
        except Exception as e:
            QMessageBox.warning(self, "Помилка", f"Помилка при виявленні об'єктів: {e}")
            return

        # Виявлення всіх об'єктів
        all_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if confidence > 0.5:
                    all_detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, label))

        # Оновлення трекера з усіма виявленнями
        tracks = tracker.update_tracks([(det[0], det[1], det[2]) for det in all_detections], frame=frame)

        # Статистика та відображення
        object_counts = {}
        track_speeds = {}

        # Якщо пошук активний, шукаємо лише потрібний track_id
        if self.searched_track_id is not None:
            found = False
            for track, detection in zip(tracks, all_detections):
                if not track.is_confirmed():
                    continue
                if track.track_id == self.searched_track_id:
                    found = True
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    label = detection[2]
                    speed_kmh = calculate_speed(track.track_id, x1, y1, x2, y2, current_time_ms, scale_factor)

                    print(f"Знайдено track_id {self.searched_track_id}: Клас: {label}, Координати: ({x1}, {y1}, {x2}, {y2}), Швидкість: {speed_kmh:.1f} км/год")

                    # Малюємо рамку та дані
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    font_scale = 1.5
                    thickness = 4
                    text = f"ID {track.track_id} ({label}) - {speed_kmh:.1f} km/h"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_y = max(y1 - text_height - 5, text_height)
                    cv2.putText(frame, text, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                    # Запис у базу даних
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        if track.track_id not in track_ids:
                            track_ids.add(track.track_id)
                            cur.execute("""
                                INSERT INTO Objects (track_id, object_class, first_detected)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (track_id) DO NOTHING
                            """, (track.track_id, label, timestamp))
                            conn.commit()

                        cur.execute("""
                            INSERT INTO TrackingData (video_id, track_id, timestamp, x1, y1, x2, y2, speed)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (video_id, track.track_id, timestamp, x1, y1, x2, y2, speed_kmh))
                        conn.commit()
                    except Exception as e:
                        print(f"Помилка при записі у БД: {e}")
                        conn.rollback()
                    break

            if not found:
                cv2.putText(frame, f"Track ID {self.searched_track_id} not found", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                print(f"Об'єкт з track_id {self.searched_track_id} не знайдено в цьому кадрі")
        else:
            # Якщо пошук не активний, відображаємо всі об'єкти
            for track, detection in zip(tracks, all_detections):
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                label = detection[2]
                speed_kmh = calculate_speed(track.track_id, x1, y1, x2, y2, current_time_ms, scale_factor)

                print(f"Track ID {track.track_id}: Клас: {label}, Координати: ({x1}, {y1}, {x2}, {y2}), Швидкість: {speed_kmh:.1f} км/год")

                object_counts[label] = object_counts.get(label, 0) + 1
                track_speeds[track.track_id] = speed_kmh

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                font_scale = 1.5
                thickness = 4
                text = f"ID {track.track_id} ({label}) - {speed_kmh:.1f} km/h"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_y = max(y1 - text_height - 5, text_height)
                cv2.putText(frame, text, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    if track.track_id not in track_ids:
                        track_ids.add(track.track_id)
                        cur.execute("""
                            INSERT INTO Objects (track_id, object_class, first_detected)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (track_id) DO NOTHING
                        """, (track.track_id, label, timestamp))
                        conn.commit()

                    cur.execute("""
                        INSERT INTO TrackingData (video_id, track_id, timestamp, x1, y1, x2, y2, speed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (video_id, track.track_id, timestamp, x1, y1, x2, y2, speed_kmh))
                    conn.commit()
                except Exception as e:
                    print(f"Помилка при записі у БД: {e}")
                    conn.rollback()

        # Оновлення статистики
        if self.searched_track_id is None:
            stats_text = "Статистика об'єктів:\n"
            for label, count in object_counts.items():
                relevant_speeds = [speed for tid, speed in track_speeds.items() if any(d[2] == label for d in all_detections for t in tracks if t.track_id == tid)]
                avg_speed = sum(relevant_speeds) / len(relevant_speeds) if relevant_speeds else 0
                stats_text += f"{label}: {count} (Середня швидкість: {avg_speed:.1f} км/год)\n"
            self.object_stats_text.setText(stats_text)
        else:
            self.update_stats()

        # Конвертація кадру для відображення
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            QMessageBox.warning(self, "Помилка", f"Помилка при відображенні кадру: {e}")

        # Оновлення таймлайну
        total_seconds = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        self.timeline_label.setText(f"{minutes:02d}:{seconds:02d}")

    def closeEvent(self, event):
        """Обробка закриття вікна: зупиняємо обробку та закриваємо ресурси."""
        print("Закриваємо програму: зупиняємо обробку та звільняємо ресурси")
        self.stop_processing()
        cap.release()
        if cur:
            cur.close()
        if conn:
            conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoMonitorWindow()
    window.show()
    sys.exit(app.exec())