import os
import json
import argparse
from PIL import Image
from tqdm import tqdm


CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]


def visdrone2coco(image_dir, txt_dir, save_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    ann_id = 1
    image_id = 1

    img_files = sorted(os.listdir(image_dir))

    for img_name in tqdm(img_files):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(image_dir, img_name)
        width, height = Image.open(img_path).size

        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        txt_path = os.path.join(
            txt_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    x, y, w, h, score, cls, trunc, occ = map(int, line.split(","))

                    # ignore区域
                    if score == 0:
                        continue

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    ann_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(save_json), exist_ok=True)

    with open(save_json, "w") as f:
        json.dump(coco, f)

    print(f"Saved to {save_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VisDrone annotations to COCO format"
    )

    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory of images"
    )

    parser.add_argument(
        "--txt_dir",
        required=True,
        help="Directory of VisDrone txt annotations"
    )

    parser.add_argument(
        "--save_json",
        required=True,
        help="Output COCO json file"
    )

    args = parser.parse_args()

    visdrone2coco(
        args.image_dir,
        args.txt_dir,
        args.save_json
    )

'''
python projects/VLM4RSDet/vlm4rsdet/visdrone2coco.py \
    --image_dir data/visdrone/train/images \
    --txt_dir data/visdrone/train/annotations \
    --save_json data/visdrone/train/annotations/result.json
'''

'''
python projects/VLM4RSDet/vlm4rsdet/visdrone2coco.py \
    --image_dir data/visdrone/val/images \
    --txt_dir data/visdrone/val/annotations \
    --save_json data/visdrone/val/annotations/result.json
'''