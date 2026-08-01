import os
import random
import cv2
import numpy as np

INPUT_DIR = "data/spoof"
OUTPUT_DIR = "data/spoof_generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = [f for f in os.listdir(INPUT_DIR)
          if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp"))]


def motion_blur(img):
    k = random.choice([5, 7, 9, 11])
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def screen_lines(img):
    out = img.copy()
    for i in range(0, out.shape[0], 3):
        out[i:i+1] = np.clip(out[i:i+1] * 0.75, 0, 255)
    return out


def jpeg_artifacts(img):
    quality = random.randint(20, 60)
    _, enc = cv2.imencode(".jpg", img,
                          [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 1)


def gaussian_noise(img):
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    out = img.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness(img):
    alpha = random.uniform(0.75, 1.25)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def perspective(img):
    h, w = img.shape[:2]

    src = np.float32([[0, 0],
                      [w, 0],
                      [0, h],
                      [w, h]])

    shift = 12

    dst = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [w-random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h-random.randint(0, shift)],
        [w-random.randint(0, shift), h-random.randint(0, shift)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def reflection(img):
    overlay = img.copy()

    x1 = random.randint(0, img.shape[1] // 2)
    x2 = random.randint(img.shape[1] // 2, img.shape[1])

    cv2.rectangle(
        overlay,
        (x1, 0),
        (x2, img.shape[0]),
        (255, 255, 255),
        -1
    )

    alpha = random.uniform(0.05, 0.18)

    return cv2.addWeighted(
        overlay,
        alpha,
        img,
        1-alpha,
        0
    )


effects = [
    motion_blur,
    screen_lines,
    jpeg_artifacts,
    gaussian_noise,
    brightness,
    gaussian_blur,
    perspective,
    reflection
]

count = 1

for image_name in images:

    img = cv2.imread(os.path.join(INPUT_DIR, image_name))

    if img is None:
        continue

    for _ in range(20):

        out = img.copy()

        random.shuffle(effects)

        for fx in effects[:random.randint(3, 6)]:
            out = fx(out)

        filename = f"spoof_{count:04d}.jpeg"

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, filename),
            out
        )

        count += 1

print("=" * 40)
print("Finished")
print("Generated :", count - 1)
print("Saved in  :", OUTPUT_DIR)
print("=" * 40)