import sys
import os

ABSOLUTE_PATH = '/home/junwoo/Rad51_RCNN'
FONT_PATH = '/usr/share/fonts/truetype/ubuntu'
SAVE_PATH = '/mnt/c/Users/jwoo/Desktop/HttpServer/save'
DATA_PATH = '/mnt/c/Users/jwoo/Desktop/HttpServer/data'
MODEL_PATH = f'{ABSOLUTE_PATH}/model'
sys.path.insert(0, os.path.abspath(f'{ABSOLUTE_PATH}/detectron2'))

import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import nd2
from PIL import Image, ImageFont, ImageDraw
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from czifile import CziFile
import tifffile
from sklearn.cluster import KMeans


def read_nd2(filepath, target_dim=(2048, 2048), erase=False):
    # nd2reader is for metadata
    with nd2.ND2File(filepath) as ndfile:
        greens = np.array([np.array(ndfile)[x][0] for x in range(ndfile.shape[0])]).astype(np.double)
        reds = np.array([np.array(ndfile)[x][1] for x in range(ndfile.shape[0])]).astype(np.double)
        transs = np.array([np.array(ndfile)[x][2] for x in range(ndfile.shape[0])]).astype(np.double)
        if erase:
            reds, dead_cell_masks = erase_dead_cells(reds, d_masks=None)
            greens, _ = erase_dead_cells(greens, d_masks=dead_cell_masks)
            transs, _ = erase_dead_cells(transs, d_masks=dead_cell_masks)

        if reds.shape[1] * reds.shape[2] > target_dim[0] * target_dim[1]:
            img_arr = image_down_sampling(reds, greens, transs, dim=target_dim)
            reds = img_arr[0]
            greens = img_arr[1]
            transs = img_arr[2]

        reds = normalization(reds)
        greens = normalization(greens)
        transs = normalization(transs)
        reds = add_max_layer(reds)

        z_depth = reds.shape[0]
        y_size = reds.shape[1]
        x_size = reds.shape[2]

        red = np.array(np.stack([reds, np.zeros(reds.shape), np.zeros(reds.shape)], axis=3) * 255).astype(np.uint8)
        green = np.array(np.stack([np.zeros(greens.shape), greens, np.zeros(greens.shape)], axis=3) * 255).astype(np.uint8)
        trans = np.array(np.stack([np.zeros(transs.shape), np.zeros(transs.shape), transs], axis=3) * 255).astype(np.uint8)
        ndfile.close()
    return red, green, trans


def read_czi(filepath, target_dim=(2048, 2048), erase=False):
    with CziFile(filepath) as czi:
        metadata = czi.metadata()
        pixelType = metadata.split('<PixelType>')[1].split('</PixelType>')[0]
        dyeName = metadata.split('<DyeName>')[1].split('</DyeName>')[0]
        dyeId = metadata.split('<DyeId>')[1].split('</DyeId>')[0]
        img = czi.asarray()
        if img.shape[0] == 1 and img.shape[5] == 1:
            img = img.reshape((img.shape[1], img.shape[2], img.shape[3], img.shape[4]))
        else:
            print('czi file array format recheck')
            exit(1)

        reds = np.array(img[0]).astype(np.double)
        greens = np.array(img[1]).astype(np.double)
        if erase:
            reds, dead_cell_masks = erase_dead_cells(reds, d_masks=None)
            greens, _ = erase_dead_cells(greens, d_masks=dead_cell_masks)

        if reds.shape[1] * reds.shape[2] > target_dim[0] * target_dim[1]:
            img_arr = image_down_sampling(reds, greens, dim=target_dim)
            reds = img_arr[0]
            greens = img_arr[1]

        reds = normalization(reds)
        greens = normalization(greens)
        reds = add_max_layer(reds)

        z_depth = reds.shape[0]
        y_size = reds.shape[1]
        x_size = reds.shape[2]

        reds = np.array(np.stack([reds, np.zeros(reds.shape), np.zeros(reds.shape)], axis=3) * 255).astype(np.uint8)
        greens = np.array(np.stack([np.zeros(greens.shape), greens, np.zeros(greens.shape)], axis=3) * 255).astype(
            np.uint8)
    return reds, greens, {'zDepth': z_depth, 'xSize': x_size, 'ySize': y_size,
                          'pixelType': pixelType, 'dyeName': dyeName, 'dyeId': dyeId, 'pixelMicrons': 'Unknown'}


def read_tif(filepath, target_dim=(2048, 2048), erase=False):
    reds = []
    greens = []
    imgs = tifffile.imread(filepath).astype(np.double)
    z_depth = imgs.shape[0]
    for z_level in range(z_depth):
        reds.append(imgs[z_level][0])
        greens.append(imgs[z_level][1])
    reds = np.array(reds)
    greens = np.array(greens)
    if erase:
        reds, dead_cell_masks = erase_dead_cells(reds, d_masks=None)
        greens, _ = erase_dead_cells(greens, d_masks=dead_cell_masks)

    if reds.shape[1] * reds.shape[2] > target_dim[0] * target_dim[1]:
        img_arr = image_down_sampling(reds, greens, dim=target_dim)
        reds = img_arr[0]
        greens = img_arr[1]

    reds = normalization(reds)
    greens = normalization(greens)
    reds = add_max_layer(reds)

    z_depth = reds.shape[0]
    y_size = reds.shape[1]
    x_size = reds.shape[2]
    reds = np.array(np.stack([reds, np.zeros(reds.shape), np.zeros(reds.shape)], axis=3) * 255).astype(np.uint8)
    greens = np.array(np.stack([np.zeros(greens.shape), greens, np.zeros(greens.shape)], axis=3) * 255).astype(
        np.uint8)
    return reds, greens, {'zDepth': z_depth, 'xSize': x_size, 'ySize': y_size,
                          'pixelType': 'Unknown', 'dyeName': 'Unknown', 'dyeId': 'Unknown', 'pixelMicrons': 'Unknown'}


def image_down_sampling(*args, dim=(2048, 2048)):
    ret_arr = []
    for arg in args:
        tmp = []
        for i, img in enumerate(arg):
            tmp.append(cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA))
        ret_arr.append(tmp)
    return np.array(ret_arr)


def normalization(imgs):
    val_max = np.max(np.max(imgs, axis=(1, 2)))
    val_min = np.min(np.min(imgs, axis=(1, 2)))
    for i, img in enumerate(imgs):
        img -= val_min
        img = img / (val_max - val_min)
        imgs[i] = img
    return imgs


def add_max_layer(imgs):
    max_layer = np.amax(imgs, axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    topHat = cv2.morphologyEx(max_layer, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(max_layer, cv2.MORPH_BLACKHAT, kernel)
    res = max_layer + np.maximum(10. * (topHat - blackHat), np.zeros(max_layer.shape))

    res = np.minimum(res, np.ones(res.shape))
    res = np.maximum(res, np.zeros(res.shape))
    res = res - np.min(res) / (np.max(res) - np.min(res))
    # Stacking the original image with the enhanced image
    """
    result = np.hstack((max_layer, res, res - max_layer))
    plt.figure()
    plt.imshow(result)
    plt.show()
    """
    new_imgs = np.concatenate((imgs, [res]))
    return new_imgs


def erase_dead_cells(imgs, d_masks):
    """
    img: stacked grey scale image without channel
    """
    if d_masks is None:
        max_layer = np.amax(imgs, axis=0)
        d_masks = np.zeros(max_layer.shape)
        clustering = KMeans(n_clusters=2, init='k-means++', n_init='auto').fit(max_layer.reshape(-1, 1))
        if np.sum(clustering.labels_) < (max_layer.shape[0] * max_layer.shape[1]) / 2.:
            d_masks = ((clustering.labels_ + 1) % 2).reshape(max_layer.shape)
        else:
            d_masks = clustering.labels_.reshape(max_layer.shape)
        d_masks = np.array([d_masks for _ in range(imgs.shape[0])])
        """
        d_masks = np.zeros(imgs.shape, dtype=np.uint8)
        for i, img in enumerate(imgs):
            #clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', eigen_solver="arpack").fit(img.reshape(-1, 1))
            clustering = KMeans(n_clusters=2, init='k-means++', n_init='auto').fit(img.reshape(-1, 1))
            #clustering = AgglomerativeClustering(n_clusters=2).fit(img.reshape(-1, 1))
            #clustering = OPTICS(min_samples=2).fit(img.reshape(-1, 1))
            if np.sum(clustering.labels_) < (img.shape[0] * img.shape[1])/2.:
                d_masks[i] = ((clustering.labels_ + 1) % 2).reshape(img.shape)
            else:
                d_masks[i] = clustering.labels_.reshape(img.shape)
    """
    masked_imgs = imgs * d_masks
    min_val = np.min(np.min(imgs, axis=(1, 2)))
    for i, masked_img in enumerate(masked_imgs):
        for row in range(len(masked_img)):
            for col in range(len(masked_img[row])):
                if masked_img[row][col] == 0:
                    masked_img[row][col] = min_val
    return masked_imgs, d_masks.astype(np.double)


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


def overlay_instances(img, args, masks, bboxs, opacity=0.05, score=None, color=None):
    font_size = 7
    font1 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-MI.ttf', size=font_size)
    font2 = ImageFont.truetype(font=f'{FONT_PATH}/Ubuntu-M.ttf', size=font_size)
    img_shape = np.array(img).shape
    img = Image.fromarray(np.uint8(img), mode='RGB')
    masks = np.array(masks, dtype=np.int8).reshape((len(masks), img_shape[0], img_shape[1], 1))
    back_mask = np.zeros((img_shape[0], img_shape[1], 4), dtype=np.int8)
    back_masks = (np.ones((len(masks), img_shape[0], img_shape[1], 1), dtype=np.uint8) *
                  masks * np.array([255, 255, 255, int(opacity * 255)], dtype=np.uint8))
    for mask in back_masks:
        back_mask += mask
    del masks
    del back_masks
    back_mask = Image.fromarray(np.uint8(back_mask), mode='RGBA')
    overlay_img = Image.new("RGB", (img_shape[1], img_shape[0]), color=color)
    img.paste(overlay_img, (0, 0), back_mask)
    for arg, bbox, sc in zip(args, bboxs, score):
        im = ImageDraw.Draw(img)
        im.text((bbox[0] + 3, bbox[3]), f'{arg}',
                (255, 255, 255), font=font1)
        im.text((bbox[0] + 3, bbox[3] + font_size + 0.5), f'.{int(sc*100)}' if sc < 1 else f'1.00',
                (255, 255, 255), font=font2)
    return img


def center_coord(box_coord):
    x_center = box_coord[0] + box_coord[2] / 2.
    y_center = box_coord[1] + box_coord[3] / 2.
    return x_center, y_center


def check_overlay(small_box_coord, big_boxs_coord,):
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
        if '.nd2' in file or '.czi' in file or '.tif' in file or '.tiff' in file:
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
            if 'score' in param_name:
                params[param_name] = int(param_val)
            else:
                params[param_name] = param_val
    return params


def write_result(nuclei_boxs, protein_boxs, overlays, circle_args, ellipse_args, save_path):
    save_file = f'{save_path}/report.txt'
    nb_nuclei = len(nuclei_boxs)
    nb_protein = len(protein_boxs)
    nb_overlay = len(overlays)

    with open(save_file, 'w') as f:
        output_string = ''
        output_string += f'Nb of nuclei: {nb_nuclei}\n'
        output_string += f'Nb of rad51clumps: {nb_protein}\n'
        output_string += f'Nb of overlaps: {nb_overlay}\n'
        output_string += f'overlap indices (rad51, nucleus) and number of shape(circle:0, ellipse:1)\n'
        for i in range(nb_overlay):
            if overlays[i][0] in circle_args:
                output_string += f'({overlays[i][0]}, {overlays[i][1]}), {0}\n'
            else:
                output_string += f'({overlays[i][0]}, {overlays[i][1]}), {1}\n'
        f.write(output_string)
        f.close()


def score_filter(boxs, cls, masks, scores, threshold=0.9):
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


def nms(coord_list, scores, masks, threshold=0.2, sortby='score'):
    coord_list = np.array(coord_list)
    scores = np.array(scores)
    masks = np.array(masks)
    if sortby == 'size':
        order_indices = np.argsort(np.sum(masks, axis=(1, 2)))[::-1]
    else:
        order_indices = np.argsort(scores)[::-1]
    ret_indices = []
    while 1:
        if len(order_indices) == 1:
            ret_indices.append(order_indices[0])
            break
        elif len(order_indices) == 0:
            break
        else:
            ret_indices.append(order_indices[0])
            preserv_list = []
            length = len(order_indices)
            for a in range(1, length):
                if IoU(coord_list[order_indices[0]], coord_list[order_indices[a]]) < threshold:
                    preserv_list.append(order_indices[a])
            order_indices = np.array(preserv_list)
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


def post_processing(masks, bins=25):
    mask_sums = np.array([np.sum(masque) for masque in masks])
    args = np.arange(len(masks))
    post_mask_args = args.copy()
    for _ in range(2):
        mask_sums_mode = (np.argmax(np.histogram(mask_sums[post_mask_args],
                                                 bins=np.arange(0, np.max(mask_sums[post_mask_args]) + bins, bins))[0])
                          * bins + (bins / 2))
        mask_std = np.std(mask_sums[post_mask_args])
        post_mask_args = np.array([arg for arg, val in zip(args, mask_sums) if (mask_sums_mode - 3. * mask_std) < val < (mask_sums_mode + 3. * mask_std)])
    return post_mask_args


def get_eccentricity_from_masks(masks):
    eccs = []
    for mask in masks:
        pts = get_boundary(mask)
        X = pts[:, 1].reshape(-1, 1)
        Y = pts[:, 0].reshape(-1, 1)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
        eta_matrix = np.array([[x[0], x[1]/2, x[3]/2],
                               [x[1]/2, x[2], x[4]/2],
                               [x[3]/2, x[4]/2, -1]])
        if np.linalg.det(eta_matrix) < 0:
            eta = 1
        else:
            eta = -1
        eccentricity = np.sqrt((2 * np.sqrt((x[0] - x[2])**2 + x[1]**2))/
                               (eta * (x[0] + x[2]) + (np.sqrt((x[0] - x[2])**2 + x[1]**2))))
        """
        plt.figure()
        plt.scatter(X, Y, label='segmented Rad51 boundary')
        x_coord = np.linspace(np.min(X.flatten()), np.max(X.flatten()) + 1, 300)
        y_coord = np.linspace(np.min(Y.flatten()), np.max(Y.flatten()) + 1, 300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.imshow(mask)
        plt.show()
        """

        eccs.append(eccentricity)
    return np.array(eccs)


def get_boundary(mask):
    boundaries = []
    pts = np.argwhere(mask == 1)
    for pt in pts:
        for delta in [[-1, -1], [-1, 0], [-1, 1],
                      [0, -1], [0, 1],
                      [1, -1], [1, 0], [1, 1]]:
            if mask[max(0, min(pt[0] - delta[0], mask.shape[0])), max(0, min(pt[1] - delta[1], mask.shape[1]))] == 0:
                boundaries.append(pt)
                break
    boundaries = np.array(boundaries)
    return boundaries


if __name__ == '__main__':
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        sys.exit(1)
    #job_id = 'mytest'

    save_folder = f'{SAVE_PATH}/{job_id}'
    data_foler = f'{DATA_PATH}/{job_id}'
    target_dim = (2048, 2048)

    file = read_file(data_foler)
    params = read_params(data_foler)
    overlay_indice = []
    protein_boxs = []
    nuclei_boxs = []
    if (file is None or 'erase' not in params
            or 'nuclei_score' not in params or 'rad51_score' not in params):
        exit(1)

    eccentricity_threshold = .75
    nuclei_iou_threshold = .2
    rad51_iou_threshold = .35
    nuclei_score = params['nuclei_score'] / 100.
    rad51_score = params['rad51_score'] / 100.
    if params['erase'] == 'True':
        erase = True
    else:
        erase = False

    selected_file = file
    if '.nd2' in selected_file:
        reds, greens, transs = read_nd2(selected_file, target_dim=target_dim, erase=erase)
    elif '.czi' in selected_file:
        reds, greens, info = read_czi(selected_file, target_dim=target_dim, erase=erase)
    elif '.tif' in selected_file or '.tiff' in selected_file:
        reds, greens, info = read_tif(selected_file, target_dim=target_dim, erase=erase)
    else:
        exit(1)

    print(f'### Prediction starts ###')
    nuclei_cfg = get_cfg()
    nuclei_cfg.merge_from_file(f'{ABSOLUTE_PATH}/config/rad51_config.yaml')
    nuclei_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    nuclei_cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 1
    nuclei_cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/nuclei_model.pth"
    nuclei_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.005
    nuclei_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8
    nuclei_cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]
    nuclei_cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    nuclei_predictor = DefaultPredictor(nuclei_cfg)

    protein_cfg = get_cfg()
    protein_cfg.merge_from_file(f'{ABSOLUTE_PATH}/config/rad51_config.yaml')
    protein_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    protein_cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/rad51protein_model2.pth"
    #protein_cfg.MODEL.WEIGHTS = f"{MODEL_PATH}/rad51protein_model.pth"
    protein_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    protein_cfg.TEST.DETECTIONS_PER_IMAGE = 200
    protein_predictor = DefaultPredictor(protein_cfg)

    for predic, images, target, score in zip([nuclei_predictor, protein_predictor],
                                             [reds, greens], ['nuclei', 'rad51'],
                                             [nuclei_score, rad51_score]):
        all_boxs = []
        all_scores = []
        all_masks = []
        all_cls = []
        for i in range(images.shape[0]):
            boxs, cls, masks, scores = prediction(predic, images[i], score)

            """
            img = Image.fromarray(np.uint8(images[i]), mode='RGB')
            img.save(f'{save_folder}/{target}_{i}.png')
            if i >= (images.shape[0] - 1):
                img = overlay_instances(
                    img=images[i],
                    masks=masks, bboxs=boxs,
                    color=(0, 255, 255), opacity=0.3, score=scores)
                img.save(f'{save_folder}/{target}_{i}.png')
            """

            all_boxs.extend(boxs)
            all_scores.extend(scores)
            all_masks.extend(masks)
            all_cls.extend(cls)

        if target=='nuclei':
            indices = nms(all_boxs, all_scores, all_masks, nuclei_iou_threshold, sortby='score')
        else:
            indices = nms(all_boxs, all_scores, all_masks, rad51_iou_threshold, sortby='score')
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
        del all_boxs
        del all_scores
        del all_masks
        del all_cls

        if target == 'nuclei':
            post_mask_args = post_processing(filtered_masks)
            filtered_masks = filtered_masks[post_mask_args]
            filtered_boxs = filtered_boxs[post_mask_args]
            filtered_scores = filtered_scores[post_mask_args]
            filtered_cls = filtered_cls[post_mask_args]
        if target == 'rad51':
            eccs = get_eccentricity_from_masks(filtered_masks)
            circle_args = np.argwhere(eccs < eccentricity_threshold).flatten()
            ellipse_args = np.argwhere(eccs >= eccentricity_threshold).flatten()

        if target == 'nuclei':
            nuclei_boxs = filtered_boxs.copy()
            img = overlay_instances(
                img=np.int8(np.sum(images[:-1], axis=0, keepdims=True).reshape(images.shape[1:]) / images.shape[0]),
                args=np.arange(len(filtered_scores)),
                masks=filtered_masks, bboxs=filtered_boxs,
                opacity=0.4, score=filtered_scores, color=(0, 255, 255))
        else:
            protein_boxs = filtered_boxs.copy()
            circle_masks = filtered_masks[circle_args]
            circle_boxs = filtered_boxs[circle_args]
            circle_scores = filtered_scores[circle_args]
            circle_cls = filtered_cls[circle_args]
            ellipse_masks = filtered_masks[ellipse_args]
            ellipse_boxs = filtered_boxs[ellipse_args]
            ellipse_scores = filtered_scores[ellipse_args]
            ellipse_cls = filtered_cls[ellipse_args]

            img = overlay_instances(
                img=np.int8(np.max(images, axis=0, keepdims=True).reshape(images.shape[1:])),
                args=circle_args,
                masks=circle_masks, bboxs=circle_boxs,
                opacity=0.4, score=circle_scores, color=(255, 0, 255))
            img = overlay_instances(
                img=img,
                args=ellipse_args,
                masks=ellipse_masks, bboxs=ellipse_boxs,
                opacity=0.4, score=ellipse_scores, color=(0, 0, 255))
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
    write_result(nuclei_boxs, protein_boxs, overlay_indice, circle_args, ellipse_args, save_path=save_folder)
