import cv2
import numpy as np
import os
import pickle
import time
import pyttsx3
import threading
from collections import deque
from datetime import datetime

TTS_RATE = 150
TTS_VOLUME = 0.9
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "face_encodings_opencv.pkl"
CHECK_INTERVAL = 1.5
CONFIDENCE_THRESHOLD = 0.55

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

class SmartTTS:
    def __init__(self, rate=150, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self._lock = threading.Lock()
        self._queue = deque()
        self._is_speaking = False
        self._running = True
        self._last_spoken = {}
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def _process_queue(self):
        while self._running:
            if self._queue and not self._is_speaking:
                text = self._queue.popleft()
                self._is_speaking = True
                with self._lock:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass
                self._is_speaking = False
            else:
                time.sleep(0.05)

    def speak(self, text):
        if text and text.strip():
            self._queue.append(text.strip())
            print(f"[TTS] {text}")

    def speak_once(self, text, cooldown=5):
        now = time.time()
        if text in self._last_spoken:
            if now - self._last_spoken[text] < cooldown:
                return
        self._last_spoken[text] = now
        self.speak(text)

    def stop(self):
        self._running = False
        self.engine.stop()

class FaceDatabase:
    def __init__(self):
        self.known_names = []
        self.known_images = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load()

    def load(self):
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.known_names = data['names']
                    self.known_images = data['images']
                print(f"[DB] {len(self.known_names)} դեմք բեռնված")
            except:
                print("[DB] Ֆայլը չկարդացվեց")

    def save(self):
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({'names': self.known_names, 'images': self.known_images}, f)
        print(f"[DB] {len(self.known_names)} դեմք պահված")

    def add_face(self, image, name):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return False, "Դեմք չգտնվեց"
        
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (100, 100))
        
        self.known_names.append(name)
        self.known_images.append(face_resized)
        self.save()
        
        cv2.imwrite(f"{KNOWN_FACES_DIR}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", face_resized)
        return True, f"{name} ավելացված է"

    def recognize(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None, False
        
        if len(self.known_names) == 0:
            return None, False
        
        results = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))
            
            best_match = None
            best_score = 0
            
            for i, known_face in enumerate(self.known_images):
                diff = cv2.absdiff(face_resized, known_face)
                score = 1.0 - (np.mean(diff) / 255.0)
                
                if score > best_score:
                    best_score = score
                    best_match = i
            
            if best_score > CONFIDENCE_THRESHOLD:
                results.append(self.known_names[best_match])
        
        if results:
            return results[0], True
        return None, False

class SmartCamera:
    def __init__(self):
        self.cap = None
        self._init_camera()

    def _init_camera(self):
        try:
            from picamera2 import Picamera2
            self.use_picamera2 = True
            self.camera = Picamera2()
            self.config = self.camera.create_still_configuration()
            self.camera.configure(self.config)
            print("[Camera] Raspberry Pi Camera")
        except:
            self.use_picamera2 = False
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("[Camera] USB Camera")
            else:
                print("[Camera] Camera չի գտնվել")

    def capture(self):
        if self.use_picamera2:
            try:
                self.camera.start()
                time.sleep(0.5)
                frame = self.camera.capture_array()
                self.camera.stop()
                return frame
            except:
                return None
        else:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
            return None

    def cleanup(self):
        if self.use_picamera2:
            try:
                self.camera.stop()
            except:
                pass
        else:
            if self.cap:
                self.cap.release()

class AutoFaceRecognizer:
    def __init__(self):
        self.tts = SmartTTS(rate=TTS_RATE, volume=TTS_VOLUME)
        self.db = FaceDatabase()
        self.camera = SmartCamera()
        self.running = True
        self.mode = "auto"
        self.last_check = 0
        self.last_face = None

    def learn_new_face(self):
        print("\n Սովորելու ռեժիմ – 3 վայրկյան")
        self.tts.speak("Սովորելու ռեժիմ, նայիր տեսախցիկին")
        
        time.sleep(2)
        image = self.camera.capture()
        
        if image is None:
            self.tts.speak("Camera-ն չի աշխատում")
            return
        
        name = input("Մուտքագրիր անունը: ").strip()
        if name:
            success, msg = self.db.add_face(image, name)
            self.tts.speak(msg)
        else:
            self.tts.speak("Անունը դատարկ է")

    def show_faces(self):
        if self.db.known_names:
            print("\n Պահված դեմքերը:")
            for i, name in enumerate(self.db.known_names, 1):
                print(f"   {i}. {name}")
            print(f"   Ընդամենը: {len(self.db.known_names)} դեմք")
            self.tts.speak(f"Պահված է {len(self.db.known_names)} դեմք")
        else:
            print("\n Դեմքեր չկան")
            self.tts.speak("Դեմքեր չկան")

    def auto_recognize_loop(self):
        print("\nԱվտոմատ ճանաչում – սպասում եմ...")
        print("   Սեղմիր 'l' նոր դեմք սովորելու համար")
        print("   Սեղմիր 'f' դեմքերի ցանկը տեսնելու համար")
        print("   Սեղմիր 'q' դուրս գալու համար\n")
        
        while self.running:
            try:
                import sys
                if sys.stdin.isatty():
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        cmd = sys.stdin.read(1).lower()
                        if cmd == 'q':
                            break
                        elif cmd == 'l':
                            self.learn_new_face()
                        elif cmd == 'f':
                            self.show_faces()
                        continue
            except:
                pass
            
            now = time.time()
            if now - self.last_check < CHECK_INTERVAL:
                time.sleep(0.05)
                continue
            
            self.last_check = now
            frame = self.camera.capture()
            
            if frame is None:
                continue
            
            name, known = self.db.recognize(frame)
            
            if known and name:
                if name != self.last_face:
                    self.last_face = name
                    self.tts.speak_once(f"Բարև {name}", cooldown=5)
                    print(f"Ճանաչվեց: {name}")
            else:
                self.last_face = None

    def run(self):
        print("""
╔════════════════════════════════════════════════════════════╗
║              SMART GLASSES - FACE RECOGNITION              ║
║                                                            ║
║ ️  ԱՎՏՈՄԱՏ ՌԵԺԻՄ                                      ║
║     Ակնոցը ինքնուրույն ճանաչում է դեմքերը                ║
║     Ծանոթ դեմք տեսնելիս ասում է "Բարև [անուն]"           ║
║                                                            ║
      Հրամաններ (ցանկացած պահի)                           ║
║     l - նոր դեմք ավելացնել (սովորել)                     ║
║     f - ցույց տալ բոլոր պահված դեմքերը                   ║
║     q - դուրս գալ                                         ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        print("Ծրագիրը պատրաստ է")
        print("Ակնոցը աշխատում է...\n")
        
        self.auto_recognize_loop()
        self.cleanup()

    def cleanup(self):
        self.tts.stop()
        self.camera.cleanup()
        print("\nԾրագիրն ավարտված է")

if __name__ == "__main__":
    print("Տեղադրեք անհրաժեշտ գրադարանները:")
    print("pip install opencv-python pyttsx3 numpy\n")
    
    app = AutoFaceRecognizer()
    app.run()