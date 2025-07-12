import cv2
import numpy as np
import json
import os
import sys


class MultiFaceColorExtractor:
    def __init__(self, draw=True):
        # DNN face detector
        proto = "deploy.prototxt"
        model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(proto, model)
        # Facemark LBF (68‑point)
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("lbfmodel.yaml")
        self.draw = draw

        # Regions → clusters of landmark indices
        self.region_landmarks = {
            "forehead": [19, 24],    # above eyebrows
            "left_cheek": [2, 3, 4],
            "right_cheek": [12, 13, 14]
        }

        # skin-mask thresholds in YCrCb
        self.ycrcb_mins = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_maxs = np.array([255, 173, 127], dtype=np.uint8)

    def _load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        return img

    def _detect_faces(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)),
                                     1.0, (300,300),
                                     (104,177,123))
        self.detector.setInput(blob)
        dets = self.detector.forward()[0,0]
        boxes = []
        for d in dets:
            conf = float(d[2])
            if conf < 0.5: continue
            x1, y1, x2, y2 = (d[3:7] * [w,h,w,h]).astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1, x2-x1, y2-y1))
        return boxes

    def _get_skin_mask(self, img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        return cv2.inRange(ycrcb, self.ycrcb_mins, self.ycrcb_maxs)

    def _rotate(self, img, angle, center):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0])), M

    def _transform_points(self, pts, M):
        # apply affine matrix M to list of (x,y)
        pts_arr = np.array(pts, dtype=np.float32)
        ones = np.ones((pts_arr.shape[0],1))
        pts_hom = np.hstack([pts_arr, ones])
        transformed = M.dot(pts_hom.T).T
        return [tuple(map(int,p)) for p in transformed]

    def _avg_rgb_masked(self, img, mask, pts, patch_radius):
        samples = []
        h, w = img.shape[:2]
        for x,y in pts:
            x1,x2 = max(0, x-patch_radius), min(w, x+patch_radius+1)
            y1,y2 = max(0, y-patch_radius), min(h, y+patch_radius+1)
            patch = img[y1:y2, x1:x2]
            mpatch = mask[y1:y2, x1:x2]
            # only skin pixels
            skin_pixels = patch[mpatch>0]
            if len(skin_pixels)==0:
                continue
            avg = skin_pixels.mean(axis=0)  # BGR
            samples.append(avg)
        if not samples:
            return {"r":0,"g":0,"b":0}
        mean_b,mean_g,mean_r = np.mean(samples, axis=0)
        return {"r":int(mean_r),"g":int(mean_g),"b":int(mean_b)}

    def process_image(self, image_path):
        base = os.path.basename(image_path)
        img = self._load_image(image_path)
        orig = img.copy()  # keep original for annotation

        # 1) Detect faces
        faces = self._detect_faces(img)
        if not faces:
            return {"error": f"No face detected in {base}"}

        # 2) Fit landmarks on original image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes_np = np.array(faces, dtype=np.int32)
        ok, lm_all = self.facemark.fit(gray, boxes_np)
        if not ok:
            return {"error": "Landmark fitting failed"}

        out = {"image": base, "faces": []}

        # 3) Process each face
        for fid, (x, y, wf, hf) in enumerate(faces):
            lm68 = lm_all[fid][0].astype(int)  # shape (68,2)

            # 3a) Record original landmarks per region
            original_pts = {
                region: [tuple(lm68[i]) for i in idxs]
                for region, idxs in self.region_landmarks.items()
            }

            # 3b) Compute eye‐angle and align
            left_eye_pts  = lm68[36:42]
            right_eye_pts = lm68[42:48]
            le_c = left_eye_pts.mean(axis=0)
            re_c = right_eye_pts.mean(axis=0)
            dy, dx = re_c[1] - le_c[1], re_c[0] - le_c[0]
            angle = np.degrees(np.arctan2(dy, dx))
            center = (int((le_c[0] + re_c[0]) / 2), int((le_c[1] + re_c[1]) / 2))
            aligned, M = self._rotate(orig, angle, center)

            # 3c) Skin mask on aligned image
            mask = self._get_skin_mask(aligned)

            # 3d) Transform original landmarks into aligned space
            aligned_pts_all = self._transform_points(lm68.tolist(), M)  # list of (x,y)

            # 3e) Dynamic patch radius
            patch_r = max(1, int(0.02 * wf))

            # 3f) Sample colors from aligned + masked
            face_info = {
                "id": fid,
                "landmarks_original": {},
                "colors": {}
            }
            for region, idxs in self.region_landmarks.items():
                pts_orig    = original_pts[region]
                pts_aligned = [aligned_pts_all[i] for i in idxs]

                # store original
                face_info["landmarks_original"][region] = [
                    {"x": int(px), "y": int(py)} for px, py in pts_orig
                ]

                # sample color
                face_info["colors"][region] = self._avg_rgb_masked(
                    aligned, mask, pts_aligned, patch_r
                )

            out["faces"].append(face_info)

            # 4) Annotate the **original** image with boxes & points
            if self.draw:
                # face rectangle
                cv2.rectangle(orig, (x, y), (x + wf, y + hf), (0, 255, 0), 2)
                # sampled points
                for region, pts in face_info["landmarks_original"].items():
                    for coord in pts:
                        px, py = coord["x"], coord["y"]
                        cv2.circle(orig, (px, py), patch_r, (0, 0, 255), -1)
                        cv2.putText(orig, region, (px + 5, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 5) Save annotated original image
        if self.draw:
            annotated_path = os.path.splitext(base)[0] + "_annotated.jpg"
            cv2.imwrite(annotated_path, orig)

        return out

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python main.py <image1> [...]"); sys.exit(1)
    ext = MultiFaceColorExtractor(draw=True)
    for img_path in sys.argv[1:]:
        print(json.dumps(ext.process_image(img_path), indent=2))
