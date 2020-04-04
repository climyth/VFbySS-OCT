from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
import xlrd
import numpy as np
import os

# Setup ======================================
# column number starts form 0
filename_col = 0   # image file column
pid_col = 1        # patients' id column.
eye_col = 2        # eye column ('OD' or 'OS')
thv_col = 129      # THV beginning column
# =============================================


# Data loading function ===============================
def LoadData(base_folder, vf_file, image_read_from_previous_numpy=False, post_fix="train", sheet_name="Train"):
    worksheet = xlrd.open_workbook(vf_file).sheet_by_name(sheet_name)
    nrows = worksheet.nrows - 1
    pids = np.empty([0, 2])
    y_data = np.empty([0, 52])
    x_data = np.empty([0, 200, 480, 3])  # OCT image shape = 480 x 200 (w x h)
    print("reading %d rows" % (nrows))

    if image_read_from_previous_numpy == False:
        for r in range(1, nrows + 1):
            img_filename = base_folder + "/" + worksheet.cell_value(r, filename_col)
            if os.path.isfile(img_filename) == False:
                print("Not found file: " + img_filename)
                continue

            pid_row = np.empty([1, 2])
            pid_row[0, 0] = worksheet.cell_value(r, pid_col)   # pid
            if worksheet.cell_value(r, eye_col) == "OD":   # Eye. OD=0, OS=1
                pid_row[0, 1] = 0
            else:
                pid_row[0, 1] = 1
            pids = np.concatenate((pids, pid_row), axis=0)

            img = load_img(img_filename)
            x_img = img_to_array(img)
            x_img = x_img.reshape((1,) + x_img.shape)
            x_data = np.concatenate((x_data, x_img), axis=0)
            print("[%d] concatenated: %s" % (r, img_filename))

            c1 = 0
            y_row = np.empty([1, 52])
            for c in range(0, 54):
                if c != 25 and c != 34:  # 암점은 제외한다.
                    y_row[0, c1] = worksheet.cell_value(r, c + thv_col)
                    c1 = c1+1
            y_data = np.concatenate((y_data, y_row), axis=0)

    if image_read_from_previous_numpy:
        x_data = np.load(base_folder + "/img_data24_" + post_fix + ".npy")
        y_data = np.load(base_folder + "/vf_data24_" + post_fix + ".npy")
        pids = np.load(base_folder + "/pid_data24_" + post_fix + ".npy")
    else:
        np.save(base_folder + "/img_data24_" + post_fix, x_data)
        np.save(base_folder + "/vf_data24_" + post_fix, y_data)
        np.save(base_folder + "/pid_data24_" + post_fix, y_data)
        print("Image array saved")

    print("Loading completed. X data shape is %s and Y data shape is %s"%((x_data.shape), (y_data.shape)))
    print("Printing sample y_data:")
    print(y_data[0])
    return x_data, y_data, pids

