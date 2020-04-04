import xlrd
import numpy as np
from DataLoad import *
from Model import *
from matplotlib import pyplot as plt
from pathlib import Path


# Setup ====================================================================================
image_file = "TestSet/OCT019.jpg"    # combined OCT image
vf_file = "TestSet/test_data.xlsm"   # ground truth visual field file
vf_sheet = "Test"  # data sheet in excel file
oct_filename_col = 0   # column number (starts from 0) of OCT filename
thv_start_col = 113    # column number (starts from 0) where THV values begin
weight_file1 = "Weights/InceptionResnet_SS_OCT.hdf5"   # InceptionResnet V2
weight_file2 = "Weights/InceptionV3_SS_OCT.hdf5"       # inception V3
weight_file3 = "Weights/InceptionV4_SS_OCT.hdf5"       # inception V4
# ===========================================================================================


def read_visual_field(excel_file, sheet_name, oct_filename, filename_col=0, thv_col=113):
    worksheet = xlrd.open_workbook(excel_file).sheet_by_name(sheet_name)
    nrows = worksheet.nrows
    y_data = np.empty([52])

    for r in range(1, nrows):
        cur_filename = worksheet.cell_value(r, filename_col)
        if Path(cur_filename).name == Path(oct_filename).name:
            c1 = 0
            for c in range(0, 54):
                if c != 25 and c != 34:  # physiological scotomas excluded
                    y_data[c1] = worksheet.cell_value(r, c + thv_col)
                    c1 = c1+1
            break
    return y_data


def draw_visual_field(field_values, mplt, start_pos=(0, 0), rect_size=(35, 25)):
    #  ===== config =====================================================
    rect_pos = [                  [3,0],[4,0],[5,0],[6,0],
                            [2,1],[3,1],[4,1],[5,1],[6,1],[7,1],
                      [1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],
                [0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],      [8,3],
                [0,4],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],      [8,4],
                      [1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],
                            [2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
                                  [3,7],[4,7],[5,7],[6,7]]
    min_vf = 5   # for color range calculation
    max_vf = 35  # for color range calculation
    # ====================================================================

    for i in range(0, 52):
        x = start_pos[0] + rect_pos[i][0] * rect_size[0]
        y = start_pos[1] + rect_pos[i][1] * rect_size[1]
        vf = field_values[i]
        bg_col = (vf - min_vf) / (max_vf - min_vf)   # max = 35, min = 5
        if bg_col < 0:
            bg_col = 0
        if bg_col > 1.0:
            bg_col = 1.0
        txt_color = 'black'
        if bg_col < 0.5:
            txt_color = 'white'
        rect = mplt.Rectangle((x, y), rect_size[0], rect_size[1], fill=True, fc=(bg_col, bg_col, bg_col))
        mplt.gca().add_patch(rect)
        mplt.gca().text(x+rect_size[0]/2, y+rect_size[1]/2, "{:.1f}".format(vf), fontsize=6,
                        horizontalalignment='center', verticalalignment='center', color=txt_color)


# make deep learning model
model1 = GetModel('InceptionResnet')
model1.load_weights(weight_file1)

model2 = GetModel('InceptionV3')
model2.load_weights(weight_file2)

model3 = GetModel('InceptionV4')
model3.load_weights(weight_file3)

# load OCT image
img = load_img(image_file)
x_img = img_to_array(img)
x_img = x_img.reshape((1,) + x_img.shape)

# load ground-truth values
truth = read_visual_field(vf_file, vf_sheet, image_file, oct_filename_col, thv_start_col)

# get visual field prediction
pred1 = model1.predict(x_img)
pred2 = model2.predict(x_img)
pred3 = model3.predict(x_img)

# draw prediction
plt.clf()
plt.imshow(img)   # draw OCT image
# draw ground-truth visual field
plt.gca().text(10, 240, "Ground truth visual field", fontsize=6, color='black')
draw_visual_field(truth, plt, start_pos=(5, 250))
# draw predicted visual field
plt.gca().text(10, 500, "Predicted by InceptionResnet V2", fontsize=6, color='black')
draw_visual_field(pred1[0], plt, start_pos=(5, 510))
plt.gca().text(10, 750, "Predicted by Inception V3", fontsize=6, color='black')
draw_visual_field(pred2[0], plt, start_pos=(5, 760))
plt.gca().text(10, 1000, "Predicted by Inception V4", fontsize=6, color='black')
draw_visual_field(pred3[0], plt, start_pos=(5, 1010))
# show figure
plt.gcf().set_size_inches(4.5, 10)
plt.axis("scaled")
plt.show()
