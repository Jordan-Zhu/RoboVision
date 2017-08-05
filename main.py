import cv2
import numpy as np
import scipy.io as sio
import settings
import matplotlib.pyplot as plt
import Line_feat_contours as lfc
import classify_curves as cc
import label_curves as lc
import merge_lines as merge_lines
import util as util
from edge_detect import edge_detect
from line_match import line_match
from line_seg import line_seg
from crop_image import crop_image
import draw_img as draw
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
    settings.init()
    settings.dev_mode = True
    # print(settings.dev_mode)

    for num_img in [4]:
        #Crops the image according to the user's mouse clicks
        #First click is top left, second click is bottom right
        mouse_X, mouse_Y = crop_image(num_img)


        # Read in depth image, -1 means w/ alpha channel.
        # This keeps in the holes with no depth data as just black.

        depth_im = 'img/learn%d.png'%num_img
        old_img = cv2.imread(depth_im, -1)

        #crops the depth image
        img = old_img[mouse_Y[0]:mouse_Y[1], mouse_X[0]:mouse_X[1]]
        final_img = util.normalize_depth(img, colormap=cv2.COLORMAP_BONE)

        #USE COPY.DEEPCOPY if you don't want to edit variables passed in
        #AKA THE IMAGES, DON'T DO THIS
        P = {"path": 'outputImg\\',
        "num_img": num_img,
        "old_img": old_img,
        "old_height": old_img.shape[0],
        "old_width": old_img.shape[1],
        "img": img,
        "height": img.shape[0],
        "width":img.shape[1],
        "img_size":img.shape,
        "mouse_X": mouse_X,
        "mouse_Y": mouse_Y,
        "cx": 320,
        "cy": 240,
        "focal_length": 300,
        "thresh": 20,
        "window_size": 10,
        "min_len": 20,
        "min_dist": 10,
        "max_dist": 200,
        "delta_angle": 20
        }

        #Values that depend on values in P
        P2 = {"old_blank_image": np.zeros((P["old_height"], P["old_width"], 3), np.uint8),
        "blank_image": np.zeros((P["height"], P["width"], 3), np.uint8)}

        #adds all these new values to P
        P.update(P2)

        #Creates point cloud map
        point_cloud = util.depth_to_PC(P)

    
        # Open a copy of the depth image
        # to change the contrast on the full-sized image
        img2 = cv2.imread(depth_im, -1)
        img2 = util.normalize_depth(img2)
        img2 = util.clahe(img2, iter=2)
        
        # crops the image
        img2 = img2[mouse_Y[0]:mouse_Y[1], mouse_X[0]:mouse_X[1]]
        P["img2"] = img2

        # *********************************** SECTION 1 *****************************************

        # FIND DEPTH / CURVATURE DISCONTINUITIES.
        curve_disc, depth_disc, edgelist = edge_detect(P)

        #CREATES LINE SEGMENTS
        seglist = line_seg(edgelist, tol=5)
        if settings.dev_mode is True:
            draw.draw_edge_list(seglist, P)

        line_pairs = []

        # ******* SECTION 2 *******
        # SEGMENT AND LABEL THE CURVATURE LINES (CONVEX/CONCAVE).
        for j in range(seglist.shape[0]):
            #First round of creating line features- creating these features first to use in line merge
            #Line features created:
            line_feature, list_point = lfc.create_line_features(seglist[j], j, edgelist, P)

            #Merge lines that are next to each other and have a similar slope
            line_new, list_point_new, line_merged = merge_lines.merge_lines(line_feature, list_point, P)

            #Draw the contour with the merged lines
            if settings.dev_mode is True:
                draw.draw_merged(line_new, P)

            #Classify curves as either discontinuity or curvature
            line_new = cc.classify_curves(curve_disc, depth_disc, line_new, list_point_new, P)
            
            #Draws curves with different colors according to what kind of discontinuity that line is
            #KEY: Green is depth discontinuity
            #KEY: Blue is curvature discontinuity
            if settings.dev_mode is True:
                draw.draw_curve(line_new, j, P)

            #Label curves further
            #Curvature - convex or concave
            #Depth - Right or Left or Doesn't belong to this contour at all 
            line_new = lc.label_curves(img, line_new, list_point_new, edgelist[j])
    
            #Draws curves with different colors according to what kind of discontinuity that line is
            #KEY: Curvature-
            #Convex: Pink
            #Concave: Purple
            #KEY: Depth-
            #Left: Blue
            #Right: Orange
            #Does not belong to this contour: Yellow
            if settings.dev_mode is True:
                draw.draw_label(line_new, j, P)

                # START PAIRING THE LINES
            # Delete lines that are concave OR less than the minimum length OR shouldn't be part of that contour (it belongs to the object in front of it or next to it)
            delete_these = np.where(np.logical_or(line_new[:, 12] == 4, line_new[:, 12] == -1, line_new[:, 4] < P["min_len"]))
            line_new = np.delete(line_new, delete_these, axis=0)
            
            #Starts pairing lines that passed minimum requirements
            list_pair, matched_lines = line_match(line_new, P)
            for k in range(len(matched_lines)):
                line_pairs.append(matched_lines[k])

                #Draws the pairs that were found
            #Same colors are paired together
            draw.draw_listpair(list_pair, line_new, final_img)
        
        #Final drawing of all the pairs that were found
        cv2.imshow("ALL THE PAIRS", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()