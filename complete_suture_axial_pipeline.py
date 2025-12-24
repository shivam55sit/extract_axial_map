import cv2
import numpy as np
import pandas as pd
import math


# =========================================================
# 1. LOAD SECTOR INTENSITY CSV (ROBUST)
# =========================================================
def load_sector_intensity(csv_path):
    df = pd.read_csv(csv_path)

    # ---- Clean column names ----
    df.columns = df.columns.str.strip()

    required_cols = [
        "Sector",
        "Angle_Start_deg",
        "Angle_End_deg",
        "Mean_Intensity"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    sectors = []
    for _, row in df.iterrows():
        sectors.append({
            "sector": int(row["Sector"]),
            "start": float(row["Angle_Start_deg"]),
            "end": float(row["Angle_End_deg"]),
            "intensity": float(row["Mean_Intensity"])
        })

    return sectors


# =========================================================
# 2. COMPUTE ANGLE OF A POINT (0° at 3 o'clock, CCW)
# =========================================================
def compute_angle(cx, cy, x, y):
    dx = x - cx
    dy = cy - y  # invert y-axis for image coordinates

    angle = math.degrees(math.atan2(dy, dx))
    angle = angle % 360

    return angle


# =========================================================
# 3. ASSIGN SECTOR BASED ON ANGLE
# =========================================================
def get_sector_for_angle(angle, sectors):
    for s in sectors:
        if s["start"] <= angle < s["end"]:
            return s
    return None


# =========================================================
# 4. FIND SUTURE CENTROIDS
# =========================================================
def extract_suture_centroids(mask_img):
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        centroids.append((cx, cy))

    return centroids


# =========================================================
# 5. ANNOTATE SUTURE RANKS
# =========================================================
def annotate_suture_ranks(
    suture_mask_path,
    sector_csv_path,
    output_path
):
    img = cv2.imread(suture_mask_path)
    H, W = img.shape[:2]

    cx, cy = W // 2, H // 2  # center of axial map

    sectors = load_sector_intensity(sector_csv_path)
    centroids = extract_suture_centroids(img)

    # ---- Assign sector + intensity to each suture ----
    suture_info = []
    for (x, y) in centroids:
        angle = compute_angle(cx, cy, x, y)
        sector = get_sector_for_angle(angle, sectors)

        if sector is None:
            continue

        suture_info.append({
            "x": x,
            "y": y,
            "angle": angle,
            "sector": sector["sector"],
            "intensity": sector["intensity"]
        })

    if not suture_info:
        raise ValueError("No sutures matched any sector")

    # ---- Rank sutures (higher intensity = higher rank) ----
    suture_info = sorted(
        suture_info,
        key=lambda x: x["intensity"],
        reverse=True
    )

    # ---- Assign ranks (same intensity → same rank) ----
    rank = 1
    prev_intensity = None

    for s in suture_info:
        if prev_intensity is None:
            s["rank"] = rank
        elif s["intensity"] < prev_intensity:
            rank += 1
            s["rank"] = rank
        else:
            s["rank"] = rank

        prev_intensity = s["intensity"]

    # ---- Draw ranks on image ----
    output = img.copy()

    for s in suture_info:
        cv2.putText(
            output,
            str(s["rank"]),
            (s["x"] + 5, s["y"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, output)
    print(f"✅ Annotated image saved: {output_path}")


# =========================================================
# 6. MAIN
# =========================================================
def main():
    annotate_suture_ranks(
        suture_mask_path=r"C:\Users\shivam.prajapati\Documents\lvp-projects\suture_radilaity\suture_angles_visual.png",
        sector_csv_path=r"C:\Users\shivam.prajapati\Documents\lvp-projects\suture_radilaity\axial_map_16_sector_intensity.csv",
        output_path="suture_ranked.png"
    )


if __name__ == "__main__":
    main()
