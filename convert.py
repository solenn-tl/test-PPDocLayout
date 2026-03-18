import json 
import requests

def download_manifest(manifest_url):
    response = requests.get(manifest_url)
    print(response)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to download manifest: {response.status_code}")
    

def download_iiif_images(img_url):
    response = requests.get(img_url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def retrieve_images_url(manifest,offset):
    with open(manifest, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    images_urls = []
    sequences = manifest_data["sequences"]
    for sequence in sequences:
        canvases = sequence["canvases"]
        for canvas in canvases:
            images = canvas["images"]
            for image in images:
                resource = image["resource"]
                img_url = resource["@id"]
                if img_url:
                    images_urls.append(img_url)
    return images_urls[offset:]

def retrieveLayoutAnnotations(annotations_file, imgs_ls, output_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)

    if isinstance(annotations_data, dict):
        annotations = annotations_data.get("items", [])
        images_bboxes = annotations_data.get("images_bboxes", [])
    else:
        annotations = annotations_data
        images_bboxes = []

    pages_by_id = {}
    pages_order = []
    counter = 0
    for annotation in annotations:
        lines_resolved = annotation.get("lines_resolved", [])
        bboxes = annotation.get("bboxes", [])
        alignment = annotation.get("alignment", annotation.get("alignement"))

        for idx, line in enumerate(lines_resolved):
            page_id = line.get("page_id", line.get("page"))
            if page_id is None:
                continue

            if page_id not in pages_by_id:
                page_img = imgs_ls[counter] if counter < len(imgs_ls) else ""
                page_dims = images_bboxes[counter] if counter < len(images_bboxes) else None
                page_entry = {
                    "page": page_id,
                    "rows": [],
                    "img": page_img
                }
                if page_dims and len(page_dims) >= 4:
                    page_entry["original_width"] = page_dims[2] - page_dims[0]
                    page_entry["original_height"] = page_dims[3] - page_dims[1]
                pages_by_id[page_id] = page_entry
                pages_order.append(page_id)
                counter += 1

            bbox = bboxes[idx] if idx < len(bboxes) else None
            pages_by_id[page_id]["rows"].append({
                "row_id": line.get("line_id", line.get("line", idx)),
                "row_type": line.get("layout_label"),
                "text": line.get("text"),
                "bboxes": bbox,
                "alignment": alignment,
            })

    pages_annotations = [pages_by_id[page_id] for page_id in pages_order]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages_annotations, f, ensure_ascii=False)

def layoutAnnotationsToLabelStudio(annotations_file, output_file):
    """
     Convert layout annotations to Label Studio format.
     {
        "data": {
            "ocr": "/data/upload/receipt_00523.png"
        },
        "predictions": [
            {
                "model_version": "best_ocr_model_1_final",
                "result": [
                    {
                    "original_width": 864,
                    "original_height": 1296,
                    "image_rotation": 0,
                    "value": {
                        "x": 48.9333,
                        "y": 61.3336,
                        "width": 9.73333,
                        "height": 2.8446,
                        "rotation": 0
                    },
                    "id": "bb1",
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectangle"
                    },
                    {
                    "original_width": 864,
                    "original_height": 1296,
                    "image_rotation": 0,
                    "value": {
                        "x": 48.9333,
                        "y": 61.3336,
                        "width": 9.7333,
                        "height": 2.8446,
                        "rotation": 0,
                        "labels": [
                            "Text"
                        ]
                    },
                    "id": "bb1",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "labels"
                    },
                    {
                    "original_width": 864,
                    "original_height": 1296,
                    "image_rotation": 0,
                    "value": {
                        "x": 48.9333,
                        "y": 61.3336,
                        "width": 9.7333,
                        "height": 2.8446,
                        "rotation": 0,
                        "text": [
                            "TOTAL"
                        ]
                    },
                    "id": "bb1",
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea"
                    }
                ],
                "score": 0.89
            }
        ]
        }
    """
    with open(annotations_file, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)

    if isinstance(raw_annotations, dict):
        pages = raw_annotations.get("pages", raw_annotations.get("items", []))
    else:
        pages = raw_annotations

    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_xyxy(bbox):
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        x1 = _to_float(bbox[0])
        y1 = _to_float(bbox[1])
        x2 = _to_float(bbox[2])
        y2 = _to_float(bbox[3])
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width <= 0 or height <= 0:
            return None
        return left, top, width, height

    def _infer_page_size(rows):
        max_x = 0.0
        max_y = 0.0
        for row in rows:
            normalized = _normalize_xyxy(row.get("bboxes"))
            if normalized is None:
                continue
            left, top, width, height = normalized
            max_x = max(max_x, left + width)
            max_y = max(max_y, top + height)
        return max(max_x, 1.0), max(max_y, 1.0)

    label_studio_tasks = []

    for page_idx, page in enumerate(pages):
        rows = page.get("rows", []) if isinstance(page, dict) else []

        page_width = _to_float(
            page.get("original_width") if isinstance(page, dict) else 0
        )
        page_height = _to_float(
            page.get("original_height") if isinstance(page, dict) else 0
        )
        if page_width <= 0 or page_height <= 0:
            page_width, page_height = _infer_page_size(rows)

        results = []
        confidence_sum = 0.0
        confidence_count = 0

        for row_idx, row in enumerate(rows):
            normalized = _normalize_xyxy(row.get("bboxes"))
            if normalized is None:
                continue

            left, top, width, height = normalized
            x_pct = (left / page_width) * 100.0
            y_pct = (top / page_height) * 100.0
            w_pct = (width / page_width) * 100.0
            h_pct = (height / page_height) * 100.0

            region_id = f"p{page_idx}_r{row_idx}"
            label = row.get("row_type") or "Text"
            text = row.get("text") or ""

            results.append({
                "original_width": int(round(page_width)),
                "original_height": int(round(page_height)),
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "labels": [label],
                },
                "id": region_id,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
            })

            results.append({
                "original_width": int(round(page_width)),
                "original_height": int(round(page_height)),
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "text": [text],
                },
                "id": region_id,
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea",
            })

            alignment = row.get("alignment")
            if isinstance(alignment, (int, float)):
                confidence_sum += float(alignment)
                confidence_count += 1

        image_ref = ""
        if isinstance(page, dict):
            image_ref = page.get("img") or page.get("image") or page.get("ocr") or ""

        prediction = {
            "model_version": "soduco-v4-surya-ocr",
            "result": results,
        }
        if confidence_count > 0:
            prediction["score"] = confidence_sum / confidence_count

        task = {
            "data": {
                "ocr": image_ref,
                "page": page.get("page", page_idx) if isinstance(page, dict) else page_idx,
            },
            "predictions": [prediction],
        }
        label_studio_tasks.append(task)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(label_studio_tasks, f, ensure_ascii=False, indent=2)

    return label_studio_tasks

def labelStudiosToCOCO(annotations_file, output_file):
    return "Not implemented yet"

if __name__ == "__main__":
    FILES_PATH = "./dataset/annuaires-test/"
    with open(FILES_PATH + 'manifests.json', 'r') as f:
        manifest_ls = json.load(f)

    for elem in manifest_ls:
        manifest_url = elem['manifest_url']
        annotations_file = FILES_PATH + elem['annotations']
        layout_output = FILES_PATH + elem['annotations'].replace('.json', '_layout.json')
        imgs = retrieve_images_url(FILES_PATH + 'manifest_' + elem['annotations'], 21)
        retrieveLayoutAnnotations(annotations_file, imgs, layout_output)
        labelstudio_output = FILES_PATH + elem['annotations'].replace('.json', '_label_studio.json')
        layoutAnnotationsToLabelStudio(layout_output, labelstudio_output)
        #manifest_file = download_manifest(manifest_url)