import cv2
import os.path as osp
from basicsr.utils import scandir
import glob


folder_path = '/mnt/ai2022/dxw/DATASET/HZXF_dataset/MGTV_6_IMG/Val_HR'
data_path = '/mnt/ai2022/dxw/DATASET/HZXF_dataset/MGTV_6_IMG'


def prepare_keys_ours(folder_path):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    txt_file = open(osp.join(data_path, 'Val_HR.txt'), 'w')
    filelist = glob.glob(folder_path + "/*")
    filelist.sort()
    print(filelist)
    for i in filelist:
        filenum = glob.glob(i + "/*")
        filenum.sort()
        filenums = len(filenum)  #读取一个文件夹共有多少帧
        file_name = i[-4:]                      #获取这个文件夹的名字
        h, w, c = 1080, 1920, 3
        txt_file.write(f'{file_name} {filenums} ({h},{w},{c})\n')
        # for j in filenum:
        #     img = cv2.imread(j, cv2.IMREAD_UNCHANGED)
        #     if img.ndim == 2:
        #         h, w = img.shape
        #         c = 1
        #     else:
        #         h, w, c = img.shape
        # print((h, w, c))

    # img_path_list = sorted(
    #     list(scandir(folder_path, suffix='png', recursive=True)))
    # KEYS = [40]
    # for idx, k in enumerate(filelist):
    #     T = k.split('/')[-1]
    #     filenum = glob.glob(k + "/*")
    #     nums = str(len(filenum))
    #     KEYS[idx] = T + " " + nums
    #     print(KEYS)


    # keys = [v.split('/')[-1] for v in filelist]  # example: 000/00000000
    # print("img_path_list", img_path_list)
    # print("keys:", keys)

    #return img_path_list, keys


def read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def write_data(img_path_list, keys):
    compress_level = 1
    txt_file = open(osp.join(data_path, 'meta_info_train.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        _, img_byte, img_shape = read_img_worker(
            osp.join(folder_path, path), key, compress_level)
        h, w, c = img_shape
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
    txt_file.close()


# img_path_list, keys = prepare_keys_ours(folder_path)
# write_data(img_path_list, keys)
prepare_keys_ours(folder_path)