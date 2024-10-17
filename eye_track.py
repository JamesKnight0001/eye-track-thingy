import cv2
import mediapipe as mp
import numpy as np
import ctypes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import math
import time
import os
import joblib
import keyboard
import win32api
import win32con

mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh


"""
Todo:
idk
"""


class Eyetracker:
    def __init__(self):
        self.face = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("I cant find the webcam.")
            exit()
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        self.scr_w = user32.GetSystemMetrics(0)
        self.scr_h = user32.GetSystemMetrics(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.eye_mod = None
        self.comb_mod = None
        self.eye_scl = None
        self.comb_scl = None
        self.eye_poly = None
        self.comb_poly = None

        self.pause = False
        self.gaze_hist = []
        self.gaze_hist_lim = 1000
        self.heatmap_on = False

        self.req_frames = 60

        self.tweak_num = 10
        self.angle_tweak = 0.0
        self.acc_pct = 100.0

        self.smooth_win = 10  
        self.move_thresh = 5  
        self.last_gaze_pt = None

    def h_ang(self, lndmrks, sz):
        mp_lndmrks = lndmrks.landmark
        img_pts = np.array([
            (mp_lndmrks[1].x * sz[1], mp_lndmrks[1].y * sz[0]),
            (mp_lndmrks[152].x * sz[1], mp_lndmrks[152].y * sz[0]),
            (mp_lndmrks[33].x * sz[1], mp_lndmrks[33].y * sz[0]),
            (mp_lndmrks[263].x * sz[1], mp_lndmrks[263].y * sz[0]),
            (mp_lndmrks[61].x * sz[1], mp_lndmrks[61].y * sz[0]),
            (mp_lndmrks[291].x * sz[1], mp_lndmrks[291].y * sz[0])
        ], dtype='double')

        mod_pts = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        f_len = sz[1]
        ctr = (sz[1]/2, sz[0]/2)
        cam_mtx = np.array(
            [[f_len, 0, ctr[0]],
             [0, f_len, ctr[1]],
             [0, 0, 1]], dtype='double'
        )
        dist_coeffs = np.zeros((4,1))
        success, rot_vec, trans_vec = cv2.solvePnP(
            mod_pts, img_pts, cam_mtx, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return [0, 0, 0]
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        sy = math.sqrt(rot_mat[0,0] ** 2 + rot_mat[1,0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rot_mat[2,1], rot_mat[2,2])
            y = math.atan2(-rot_mat[2,0], sy)
            z = math.atan2(rot_mat[1,0], rot_mat[0,0])
        else:
            x = math.atan2(-rot_mat[1,2], rot_mat[1,1])
            y = math.atan2(-rot_mat[2,0], sy)
            z = 0
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)
        return [x, y, z]

    def get_eye_feat(self, lndmrks, sz):
        l_eye_idx = [33, 133, 160, 159, 158, 157, 173]
        r_eye_idx = [362, 263, 387, 386, 385, 384, 398]
        l_eye = []
        r_eye = []
        for idx in l_eye_idx:
            x = lndmrks.landmark[idx].x * sz[1]
            y = lndmrks.landmark[idx].y * sz[0]
            l_eye.append((x, y))
        for idx in r_eye_idx:
            x = lndmrks.landmark[idx].x * sz[1]
            y = lndmrks.landmark[idx].y * sz[0]
            r_eye.append((x, y))

        def comp_ear(eye):
            A = math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
            B = math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
            C = math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
            ear = (A + B) / (2.0 * C)
            return ear
        l_ear = comp_ear(l_eye)
        r_ear = comp_ear(r_eye)
        ear_avg = (l_ear + r_ear) / 2.0
        l_eye_flat = [coord for pt in l_eye for coord in pt]
        r_eye_flat = [coord for pt in r_eye for coord in pt]
        eye_feat = l_eye_flat + r_eye_flat + [l_ear, r_ear, ear_avg]
        return eye_feat

    def calib(self):
        eye_data1 = []
        scr_pos1 = []

        eye_data2 = []
        head_data2 = []
        scr_pos2 = []

        calib_pts = [
            ('Left', (0.1 * self.scr_w, 0.5 * self.scr_h)),
            ('Right', (0.9 * self.scr_w, 0.5 * self.scr_h)),
            ('Up', (0.5 * self.scr_w, 0.1 * self.scr_h)),
            ('Down', (0.5 * self.scr_w, 0.9 * self.scr_h)),
            ('Center', (0.5 * self.scr_w, 0.5 * self.scr_h)),
            ('Top Left', (0.1 * self.scr_w, 0.1 * self.scr_h)),
            ('Top Right', (0.9 * self.scr_w, 0.1 * self.scr_h)),
            ('Bottom Left', (0.1 * self.scr_w, 0.9 * self.scr_h)),
            ('Bottom Right', (0.9 * self.scr_w, 0.9 * self.scr_h))
        ]

        cv2.namedWindow('Calib', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calib', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Calibration Step 1: KEEP YOUR HEAD STILL, USE YOUR EYES TO FOLLOW THE TARGET.")
        time.sleep(1)

        for label, pt in calib_pts:
            img = np.zeros((self.scr_h, self.scr_w, 3), dtype=np.uint8)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)
            cv2.putText(img, f"Look at the {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Calibration Step 1: KEEP YOUR HEAD STILL, USE YOUR EYES TO FOLLOW THE TARGET.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Calib', img)
            cv2.waitKey(1)
            time.sleep(1)

            success_frames = 0
            frames_captured = 0
            while success_frames < self.req_frames and frames_captured < 100:
                ret, frame = self.cap.read()
                frames_captured +=1
                if not ret:
                    continue
                sz = frame.shape
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face.process(img_rgb)
                if not results.multi_face_landmarks:
                    continue
                lndmrks = results.multi_face_landmarks[0]

                eye_feat = self.get_eye_feat(lndmrks, sz)

                eye_data1.append(eye_feat)
                scr_pos1.append([pt[0], pt[1]])

                success_frames +=1

        print("Calibration Step 2: MOVE YOUR HEAD AND USE YOUR EYES TO FOLLOW THE TARGET.")
        time.sleep(1)

        for label, pt in calib_pts:
            img = np.zeros((self.scr_h, self.scr_w, 3), dtype=np.uint8)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 20, (0, 255, 0), -1)
            cv2.putText(img, f"Look at the {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Calibration Step 2: MOVE YOUR HEAD AND USE YOUR EYES TO FOLLOW THE TARGET.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Calib', img)
            cv2.waitKey(1)
            time.sleep(1)

            success_frames = 0
            frames_captured = 0
            while success_frames < self.req_frames and frames_captured < 100:
                ret, frame = self.cap.read()
                frames_captured +=1
                if not ret:
                    continue
                sz = frame.shape
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face.process(img_rgb)
                if not results.multi_face_landmarks:
                    continue
                lndmrks = results.multi_face_landmarks[0]

                eye_feat = self.get_eye_feat(lndmrks, sz)
                head_ang = self.h_ang(lndmrks, sz)

                eye_data2.append(eye_feat)
                head_data2.append(head_ang)
                scr_pos2.append([pt[0], pt[1]])

                success_frames +=1

        cv2.destroyWindow('Calib')

        if len(eye_data1) == 0 or len(eye_data2) == 0:
            print("Not enough data. (add more now)")
            self.cap.release()
            exit()

        eye_data1 = np.array(eye_data1)
        scr_pos1 = np.array(scr_pos1)

        self.eye_poly = PolynomialFeatures(degree=2)
        eye_feat_poly = self.eye_poly.fit_transform(eye_data1)
        self.eye_scl = StandardScaler()
        eye_feat_scl = self.eye_scl.fit_transform(eye_feat_poly)
        X_train_eye, X_val_eye, y_train_eye, y_val_eye = train_test_split(eye_feat_scl, scr_pos1, test_size=0.2, random_state=42)
        self.eye_mod = Ridge(alpha=1.0)
        self.eye_mod.fit(X_train_eye, y_train_eye)

        eye_data2 = np.array(eye_data2)
        head_data2 = np.array(head_data2)
        scr_pos2 = np.array(scr_pos2)

        comb_feat = np.hstack((eye_data2, head_data2))
        self.comb_poly = PolynomialFeatures(degree=2)
        comb_feat_poly = self.comb_poly.fit_transform(comb_feat)
        self.comb_scl = StandardScaler()
        comb_feat_scl = self.comb_scl.fit_transform(comb_feat_poly)
        X_train_comb, X_val_comb, y_train_comb, y_val_comb = train_test_split(comb_feat_scl, scr_pos2, test_size=0.2, random_state=42)
        self.comb_mod = Ridge(alpha=1.0)
        self.comb_mod.fit(X_train_comb, y_train_comb)

        joblib.dump({
            'eye_mod': self.eye_mod,
            'comb_mod': self.comb_mod,
            'eye_scl': self.eye_scl,
            'comb_scl': self.comb_scl,
            'eye_poly': self.eye_poly,
            'comb_poly': self.comb_poly
        }, 'model.pkl')
        print("Models saved")

    def adj_angle_tweak(self):
        print("Adjustment thingy")
        eye_data_new = []
        head_data_new = []
        scr_pos_new = []

        test_pts = [
            ('Random Point', (np.random.uniform(0.1, 0.9) * self.scr_w, np.random.uniform(0.1, 0.9) * self.scr_h))
            for _ in range(self.tweak_num)
        ]

        cv2.namedWindow('Angle Adjustment', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Angle Adjustment', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        errs = []

        for idx, (label, pt) in enumerate(test_pts):
            img = np.zeros((self.scr_h, self.scr_w, 3), dtype=np.uint8)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 20, (255, 0, 0), -1)
            cv2.putText(img, f"Look at the point {idx+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Angle Adjustment', img)
            cv2.waitKey(1)
            time.sleep(1)

            success = False
            attempts = 0
            while not success and attempts < 10:
                ret, frame = self.cap.read()
                attempts +=1
                if not ret:
                    print("trash webcam")
                    continue

                if self.angle_tweak != 0.0:
                    (h, w) = frame.shape[:2]
                    ctr = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(ctr, self.angle_tweak, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h))

                sz = frame.shape
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face.process(img_rgb)
                if not results.multi_face_landmarks:
                    continue
                lndmrks = results.multi_face_landmarks[0]

                eye_feat = self.get_eye_feat(lndmrks, sz)
                head_ang = self.h_ang(lndmrks, sz)

                eye_data_new.append(eye_feat)
                head_data_new.append(head_ang)
                scr_pos_new.append([pt[0], pt[1]])

                eye_feat = np.array(eye_feat).reshape(1, -1)
                eye_feat_poly = self.eye_poly.transform(eye_feat)
                eye_feat_scl = self.eye_scl.transform(eye_feat_poly)
                gaze_pt_eye = self.eye_mod.predict(eye_feat_scl)

                comb_feat = np.hstack((eye_feat, head_ang)).reshape(1, -1)
                comb_feat_poly = self.comb_poly.transform(comb_feat)
                comb_feat_scl = self.comb_scl.transform(comb_feat_poly)
                gaze_pt_comb = self.comb_mod.predict(comb_feat_scl)

                acc_wt = self.acc_pct / 100.0

                gaze_pt = gaze_pt_eye * (1 - acc_wt) + gaze_pt_comb * acc_wt

                pred_x = gaze_pt[0][0]
                pred_y = gaze_pt[0][1]

                err = math.hypot(pred_x - pt[0], pred_y - pt[1])
                errs.append(err)
                success = True

        cv2.destroyWindow('Angle Adjustment')

        if eye_data_new and head_data_new:
            eye_data_new = np.array(eye_data_new)
            head_data_new = np.array(head_data_new)
            scr_pos_new = np.array(scr_pos_new)

            eye_feat_poly_new = self.eye_poly.transform(eye_data_new)
            eye_feat_scl_new = self.eye_scl.transform(eye_feat_poly_new)
            X_eye = np.vstack((eye_feat_scl_new))
            y_eye = np.vstack((scr_pos_new))
            self.eye_mod.fit(X_eye, y_eye)

            comb_feat_new = np.hstack((eye_data_new, head_data_new))
            comb_feat_poly_new = self.comb_poly.transform(comb_feat_new)
            comb_feat_scl_new = self.comb_scl.transform(comb_feat_poly_new)
            X_comb = np.vstack((comb_feat_scl_new))
            y_comb = np.vstack((scr_pos_new))
            self.comb_mod.fit(X_comb, y_comb)

            joblib.dump({
                'eye_mod': self.eye_mod,
                'comb_mod': self.comb_mod,
                'eye_scl': self.eye_scl,
                'comb_scl': self.comb_scl,
                'eye_poly': self.eye_poly,
                'comb_poly': self.comb_poly
            }, 'model.pkl')
            print("Models updated")

        if errs:
            avg_err = sum(errs) / len(errs)
            print(f"Average error: {avg_err:.2f} pixels")

            if avg_err > 100:
                self.angle_tweak += 2.0
            elif avg_err < 50:
                self.angle_tweak -= 2.0
            print(f"Angle tweak to {self.angle_tweak} degrees")
        else:
            print("No errors")

    def start(self):
        print("eye tracker started | Press 'Ctrl+C' to quit.")
        try:
            while True:
                if keyboard.is_pressed('ctrl') and keyboard.is_pressed('num 1'):
                    self.heatmap_on = True
                    self.show_heatmap()
                    self.heatmap_on = False
                    time.sleep(0.5)
                elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('num 2'):
                    self.calib()
                    time.sleep(0.5)
                elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('num 3'):
                    self.adj_angle_tweak()
                    time.sleep(0.5)
                elif keyboard.is_pressed('num 1'):
                    self.pause = not self.pause
                    if self.pause:
                        print("eye tracker paused.")
                    else:
                        print("eye tracker resumed.")
                    time.sleep(0.5)

                if self.pause or self.heatmap_on:
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam.")
                    break

                if self.angle_tweak != 0.0:
                    (h, w) = frame.shape[:2]
                    ctr = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(ctr, self.angle_tweak, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h))

                sz = frame.shape
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face.process(img_rgb)
                if not results.multi_face_landmarks:
                    continue
                lndmrks = results.multi_face_landmarks[0]

                eye_feat = self.get_eye_feat(lndmrks, sz)
                head_ang = self.h_ang(lndmrks, sz)

                eye_feat = np.array(eye_feat).reshape(1, -1)
                eye_feat_poly = self.eye_poly.transform(eye_feat)
                eye_feat_scl = self.eye_scl.transform(eye_feat_poly)
                gaze_pt_eye = self.eye_mod.predict(eye_feat_scl)

                comb_feat = np.hstack((eye_feat, head_ang)).reshape(1, -1)
                comb_feat_poly = self.comb_poly.transform(comb_feat)
                comb_feat_scl = self.comb_scl.transform(comb_feat_poly)
                gaze_pt_comb = self.comb_mod.predict(comb_feat_scl)

                acc_wt = self.acc_pct / 100.0

                gaze_pt = gaze_pt_eye * (1 - acc_wt) + gaze_pt_comb * acc_wt

                gaze_x = gaze_pt[0][0]
                gaze_y = gaze_pt[0][1]

                margin = 20
                gaze_x = max(margin, min(self.scr_w - margin, gaze_x))
                gaze_y = max(margin, min(self.scr_h - margin, gaze_y))

                self.gaze_hist.append((gaze_x, gaze_y))
                if len(self.gaze_hist) > self.gaze_hist_lim:
                    self.gaze_hist.pop(0)

                if len(self.gaze_hist) >= self.smooth_win:
                    recent_pts = np.array(self.gaze_hist[-self.smooth_win:])
                    wts = np.linspace(1, 0.5, self.smooth_win)
                    gaze_x = np.average(recent_pts[:,0], weights=wts)
                    gaze_y = np.average(recent_pts[:,1], weights=wts)
                    
                if self.last_gaze_pt is None or math.hypot(gaze_x - self.last_gaze_pt[0], gaze_y - self.last_gaze_pt[1]) > self.move_thresh:
                    win32api.SetCursorPos((int(gaze_x), int(gaze_y)))
                    self.last_gaze_pt = (gaze_x, gaze_y)
                    
                win32api.SetCursorPos((int(gaze_x), int(gaze_y)))

        except KeyboardInterrupt:
            pass

        self.cap.release()
        cv2.destroyAllWindows()

    def show_heatmap(self):
        heatmap_img = np.zeros((self.scr_h, self.scr_w), dtype=np.float32)
        for gaze in self.gaze_hist:
            x = int(gaze[0])
            y = int(gaze[1])
            cv2.circle(heatmap_img, (x, y), 30, 1, -1)
        heatmap_img = cv2.GaussianBlur(heatmap_img, (0, 0), sigmaX=99, sigmaY=99)
        heatmap_img = np.minimum(heatmap_img * 255, 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imshow('Heatmap', heatmap_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Heatmap')
