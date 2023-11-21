import sys, os

ABSOLUTE_PATH = '/home/junwoo/nuclei_test'
FONT_PATH = '/usr/share/fonts/truetype/ubuntu'
SAVE_PATH = '/mnt/c/Users/jwoo/Desktop/HttpServer/save'
DATA_PATH = '/mnt/c/Users/jwoo/Desktop/HttpServer/data'
MODEL_PATH = f'{ABSOLUTE_PATH}/model'
sys.path.insert(0, os.path.abspath(f'{ABSOLUTE_PATH}/detectron2'))

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import nd2
from PIL import Image, ImageFont, ImageDraw
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from czifile import CziFile
import imageio
import scipy


def read_nd2(filepath):
    # nd2reader is for metadata
    with nd2.ND2File(filepath) as ndfile:
        greens = np.array([np.array(ndfile)[x][0] for x in range(ndfile.shape[0])]).astype(np.double)
        reds = np.array([np.array(ndfile)[x][1] for x in range(ndfile.shape[0])]).astype(np.double)
        transs = np.array([np.array(ndfile)[x][2] for x in range(ndfile.shape[0])]).astype(np.double)

        y_size = reds.shape[1]
        x_size = reds.shape[2]
        zero_base = np.zeros((y_size, x_size), dtype=np.uint8)
        one_base = np.ones((y_size, x_size), dtype=np.uint8)
        r_max = np.mean(np.max(reds, axis=(1, 2)))
        g_max = np.mean(np.max(greens, axis=(1, 2)))
        t_max = np.mean(np.max(transs, axis=(1, 2)))

        for i, (r, g, t) in enumerate(zip(reds, greens, transs)):
            r_min = np.min(r)
            g_min = np.min(g)
            t_min = np.min(t)
            r_mode = scipy.stats.mode(r.reshape(r.shape[0] * r.shape[1]), keepdims=False)[0]
            g_mode = scipy.stats.mode(g.reshape(g.shape[0] * g.shape[1]), keepdims=False)[0]
            t_mode = scipy.stats.mode(t.reshape(t.shape[0] * t.shape[1]), keepdims=False)[0]
            r = r - r_mode
            g = g - g_mode
            t = t - t_mode
            r = np.maximum(r, zero_base)
            g = np.maximum(g, zero_base)
            t = np.maximum(t, zero_base)
            r = r / (r_max - r_min)
            g = g / (g_max - g_min)
            t = t / (t_max - t_min)
            r = np.minimum(r, one_base)
            g = np.minimum(g, one_base)
            t = np.minimum(t, one_base)
            reds[i] = r
            greens[i] = g
            transs[i] = t

        red = np.array(np.stack([reds, np.zeros(reds.shape), np.zeros(reds.shape)], axis=3) * 255).astype(np.uint8)
        green = np.array(np.stack([np.zeros(greens.shape), greens, np.zeros(greens.shape)], axis=3) * 255).astype(np.uint8)
        trans = np.array(np.stack([np.zeros(transs.shape), np.zeros(transs.shape), transs], axis=3) * 255).astype(np.uint8)
        ndfile.close()
    return red, green, trans


def read_czi(filepath):
    with CziFile(filepath) as czi:
        metadata = czi.metadata()
        pixelType = metadata.split('<PixelType>')[1].split('</PixelType>')[0]
        dyeName = metadata.split('<DyeName>')[1].split('</DyeName>')[0]
        dyeId = metadata.split('<DyeId>')[1].split('</DyeId>')[0]
        img = czi.asarray()
        nb_channel = img.shape[1]
        z_depth = img.shape[2]
        y_size = img.shape[3]
        x_size = img.shape[4]
        zero_base = np.zeros((y_size, x_size), dtype=np.uint8)
        one_base = np.ones((y_size, x_size), dtype=np.uint8)
        if img.shape[0] == 1 and img.shape[5] == 1:
            img = img.reshape((nb_channel, z_depth, y_size, x_size))
        else:
            print('czi file array format recheck')
            exit(1)
        reds = np.array(img[0]).astype(np.double)
        greens = np.array(img[1]).astype(np.double)

        r_max = np.mean(np.max(reds, axis=(1, 2)))
        g_max = np.mean(np.max(greens, axis=(1, 2)))

        for i, (r, g) in enumerate(zip(reds, greens)):
            r_min = np.min(r)
            g_min = np.min(g)
            r_mode = scipy.stats.mode(r.reshape(r.shape[0] * r.shape[1]), keepdims=False)[0]
            g_mode = scipy.stats.mode(g.reshape(g.shape[0] * g.shape[1]), keepdims=False)[0]
            r = r - r_mode
            g = g - g_mode
            r = np.maximum(r, zero_base)
            g = np.maximum(g, zero_base)
            r = r / (r_max - r_min)
            g = g / (g_max - g_min)
            r = np.minimum(r, one_base)
            g = np.minimum(g, one_base)
            reds[i] = r
            greens[i] = g
        reds = np.array(np.stack([reds, np.zeros(reds.shape), np.zeros(reds.shape)], axis=3) * 255).astype(np.uint8)
        greens = np.array(np.stack([np.zeros(greens.shape), greens, np.zeros(greens.shape)], axis=3) * 255).astype(
            np.uint8)

    return reds, greens, {'zDepth': z_depth, 'xSize': x_size, 'ySize': y_size,
                          'pixelType': pixelType, 'dyeName': dyeName, 'dyeId': dyeId, 'pixelMicrons': 'Unknown'}


def old_overlay_instances(img, masks, color=(255, 0, 0), opacity=0.05, score=None):
    font_size = 8
    font1 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-B.ttf', size=font_size)
    font2 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-B.ttf', size=font_size)

    masks = np.array(masks)
    img_shape = np.array(img).shape[:2]
    img = Image.fromarray(np.uint8(img)).convert('RGBA')
    for index, (mask, sc) in enumerate(zip(masks, score)):
        back_mask = []
        font_x, font_y = 0, 0
        for y in range(len(mask)):
            row = []
            for x in range(len(mask[y])):
                if mask[y][x]:
                    font_x = x
                    font_y = y
                    row.append([255, 255, 255, int(opacity * 255)])
                else:
                    row.append([0, 0, 0, 0])
            back_mask.append(row)
        back_mask = Image.fromarray(np.uint8(back_mask))
        overlay_img = Image.new("RGB", img_shape, color=color)
        img.paste(overlay_img, (0, 0), back_mask)

        im = ImageDraw.Draw(img)
        im.text((font_x, font_y), f'{index}',
                (color[0], color[1], color[2], int(opacity * 255)), font=font1)
        im.text((font_x, font_y + font_size + 0.5), f'.{int(sc*100)}' if sc < 1 else f'1.00',
                (color[0], color[1], color[2], int(opacity * 255)), font=font2)
    return img


def overlay_instances(img, masks, bboxs, color=(255, 0, 0), opacity=0.05, score=None):
    font_size = 7
    font1 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-MI.ttf', size=font_size)
    font2 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-M.ttf', size=font_size)
    img_shape = np.array(img).shape
    img = Image.fromarray(np.uint8(img), mode='RGB')
    masks = np.array(masks, dtype=np.int8).reshape((len(masks), img_shape[0], img_shape[1], 1))
    back_mask = np.zeros((img_shape[0], img_shape[1], 4), dtype=np.int8)
    back_masks = (np.ones((len(masks), img_shape[0], img_shape[1], 1), dtype=np.int8) *
                  masks * np.array([255, 255, 255, int(opacity * 255)], dtype=np.int8))
    for mask in back_masks:
        back_mask += mask
    del masks
    del back_masks
    back_mask = Image.fromarray(np.uint8(back_mask), mode='RGBA')
    overlay_img = Image.new("RGB", (img_shape[1], img_shape[0]), color=color)
    img.paste(overlay_img, (0, 0), back_mask)
    for index, (bbox, sc) in enumerate(zip(bboxs, score)):
        im = ImageDraw.Draw(img)
        im.text((bbox[0] + 3, bbox[3]), f'{index}',
                (255, 255, 255), font=font1)
        im.text((bbox[0] + 3, bbox[3] + font_size + 0.5), f'.{int(sc*100)}' if sc < 1 else f'1.00',
                (255, 255, 255), font=font2)
    return img


def center_coord(box_coord):
    x_center = box_coord[0] + box_coord[2] / 2.
    y_center = box_coord[1] + box_coord[3] / 2.
    return x_center, y_center


def check_overlay(small_box_coord, big_boxs_coord):
    for i, big_box_coord in enumerate(big_boxs_coord):
        if IoU(small_box_coord, big_box_coord) > 0.5:
            return True, i
    return False, -1


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    interArea = interArea / ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    return interArea


def read_file(directory):
    files = os.listdir(directory)
    for file in files:
        if '.nd2' in file or '.czi' in file:
            return f'{directory}/{file}'
    return None


def read_params(path: str) -> dict:
    """
    @params : path(String)
    @return : parameters(dict)
    Read a configuration file in a given path and return the parameters.
    """
    file = f'{path}/config.txt'
    params = {}

    with open(file, 'r') as f:
        input = f.readlines()
        for line in input:
            if '=' not in line:
                continue
            line = line.strip().split('=')
            param_name = line[0].strip()
            param_val = line[1].split('#')[0].strip()
            if param_name in ['score']:
                params[param_name] = int(param_val)
            else:
                params[param_name] = param_val
    return params


def write_result(nuclei_boxs, protein_boxs, overlays, save_path):
    save_file = f'{save_path}/report.txt'
    nb_nuclei = len(nuclei_boxs)
    nb_protein = len(protein_boxs)
    nb_overlay = len(overlays)

    with open(save_file, 'w') as f:
        output_string = ''
        output_string += f'Nb of nuclei: {nb_nuclei}\n'
        output_string += f'Nb of rad51clumps: {nb_protein}\n'
        output_string += f'Nb of overlaps: {nb_overlay}\n'
        output_string += f'overlap indices (rad51, nucleus):\n'
        for i in range(nb_overlay):
            output_string += f'({overlays[i][0]}, {overlays[i][1]})\n'
        f.write(output_string)
        f.close()


def score_filter(boxs, cls, masks, scores, threshold=0.5):
    filtered_boxs = []
    filtered_cls = []
    filtered_masks = []
    filtered_scores = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            filtered_boxs.append(boxs[i])
            filtered_cls.append(cls[i])
            filtered_masks.append(masks[i])
            filtered_scores.append(scores[i])
    return filtered_boxs, filtered_cls, filtered_masks, filtered_scores


def nms(coord_list, scores, threshold=0.2):
    coord_list = np.array(coord_list)
    scores = np.array(scores)
    score_indices = np.argsort(scores)[::-1]
    ret_indices = []
    while 1:
        if len(score_indices) == 1:
            ret_indices.append(score_indices[0])
            break
        elif len(score_indices) == 0:
            break
        else:
            ret_indices.append(score_indices[0])
            preserv_list = []
            length = len(score_indices)
            for a in range(1, length):
                if IoU(coord_list[score_indices[0]], coord_list[score_indices[a]]) < threshold:
                    preserv_list.append(score_indices[a])
            score_indices = np.array(preserv_list)
    return np.array(ret_indices)


def prediction(predictor, image, score_threshold):
    outputs = predictor(image)
    boxs = np.array(
        [[box[0], box[1], box[2], box[3]] for box in outputs["instances"].to("cpu").pred_boxes])
    cls = np.array(outputs["instances"].to("cpu").pred_classes)
    masks = np.array(outputs["instances"].to("cpu").pred_masks)
    scores = np.array(outputs["instances"].to("cpu").scores)
    boxs, cls, masks, scores = (
        score_filter(boxs, cls, masks, scores, score_threshold))
    return boxs, cls, masks, scores


if __name__ == '__main__':
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        sys.exit(1)

    save_folder = f'{SAVE_PATH}/{job_id}'
    data_foler = f'{DATA_PATH}/{job_id}'

    file = read_file(data_foler)
    params = read_params(data_foler)
    overlay_indice = []
    protein_boxs = []
    nuclei_boxs = []

    if file is None:
        exit(1)
    if 'score' in params:
        score_threshold = params['score'] / 100.
    else:
        score_threshold = .90

    selected_file = file
    if '.nd2' in selected_file:
        reds, greens, transs = read_nd2(selected_file)
    elif '.czi' in selected_file:
        reds, greens, info = read_czi(selected_file)
    else:
        exit(1)
    nb_proj, height, width, n_channel = reds.shape

    nuclei_cfg = get_cfg()
    nuclei_cfg.merge_from_file(f'{ABSOLUTE_PATH}/config/rad51_config.yaml')
    nuclei_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    nuclei_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    nuclei_cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/nuclei_model.pth"
    nuclei_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    nuclei_cfg.TEST.DETECTIONS_PER_IMAGE = 400
    nuclei_predictor = DefaultPredictor(nuclei_cfg)

    protein_cfg = get_cfg()
    protein_cfg.merge_from_file(f'{ABSOLUTE_PATH}/config/rad51_config.yaml')
    protein_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    protein_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    protein_cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/rad51protein_model2.pth"
    protein_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    protein_cfg.TEST.DETECTIONS_PER_IMAGE = 200
    protein_predictor = DefaultPredictor(protein_cfg)

    for predic, images, target in zip([nuclei_predictor, protein_predictor], [reds, greens], ['nuclei', 'rad51']):
        all_boxs = []
        all_scores = []
        all_masks = []
        all_cls = []
        for i in range(nb_proj):
            boxs, cls, masks, scores = prediction(predic, images[i], score_threshold)
            all_boxs.extend(boxs)
            all_scores.extend(scores)
            all_masks.extend(masks)
            all_cls.extend(cls)

        indices = nms(all_boxs, all_scores, 0.2)
        filtered_boxs = []
        filtered_scores = []
        filtered_masks = []
        filtered_cls = []
        for index in indices:
            filtered_boxs.append(all_boxs[index])
            filtered_scores.append(all_scores[index])
            filtered_masks.append(all_masks[index])
            filtered_cls.append(all_cls[index])
        filtered_boxs = np.array(filtered_boxs)
        filtered_scores = np.array(filtered_scores)
        filtered_masks = np.array(filtered_masks)
        filtered_cls = np.array(filtered_cls)
        all_boxs.clear()
        all_scores.clear()
        all_masks.clear()
        all_cls.clear()
        if target == 'nuclei':
            color = (0, 255, 255)
            nuclei_boxs = filtered_boxs.copy()
        else:
            color = (255, 0, 255)
            protein_boxs = filtered_boxs.copy()

        img = overlay_instances(img=np.int8(np.sum(images, axis=0, keepdims=True).reshape(images.shape[1:]) / images.shape[0]),
                                masks=filtered_masks, bboxs=filtered_boxs,
                                color=color, opacity=0.3, score=filtered_scores)
        img.save(f'{save_folder}/{target}.png')

    overlay_img = Image.fromarray(
        np.uint8(reds[int(reds.shape[0]/2)] + greens[int(greens.shape[0]/2)])
    ).convert('RGBA')
    overlay_img.save(f'{save_folder}/overlay.png')

    for p_index, p_box in enumerate(protein_boxs):
        check, nucleus_index = check_overlay(p_box, nuclei_boxs)
        if check:
            overlay_indice.append([p_index, nucleus_index])
    overlay_indice = np.array(overlay_indice)  # rad51 , nucleus
    write_result(nuclei_boxs, protein_boxs, overlay_indice, save_path=save_folder)
