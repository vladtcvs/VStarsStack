import vstarstack.data
import vstarstack.cfg
import cv2
import numpy as np

import json
import matplotlib.pyplot as plt
import os

def find_kps(files):
    kps = {}
    orb = cv2.ORB_create()
    for fname in files:
        name = os.path.splitext(os.path.basename(fname))[0]
        print(name)
        dataframe = vstarstack.data.DataFrame.load(fname)
        for channel in dataframe.get_channels():
            image,opts = dataframe.get_channel(channel)
            if opts["weight"]:
                continue
            if opts["encoded"]:
                continue
            if not opts["brightness"]:
                continue
            if channel not in kps:
                kps[channel] = {}
            image = (image / np.amax(image) * 255).astype(np.uint8)
            kp, descs = orb.detectAndCompute(image, None)
            kps[channel][name] = (kp, descs, fname)
    return kps

def match_images(kps):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    crd_clusters = {}
    for channel in kps:
        print("Channel = %s" % channel)
        names = sorted(list(kps[channel].keys()))
        clusters = []
        for ind in range(len(names)-1):
            next = ind+1
            name1 = names[ind]
            name2 = names[next]
            print("\t%s <-> %s" % (name1, name2))
            kp1,des1,fname1 = kps[channel][name1]
            kp2,des2,fname2 = kps[channel][name2]
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            nm = int(len(matches)/3)
            matches = matches[:nm]
            matches = matches[:20]    

            for match in matches:
                ind2 = match.trainIdx
                ind1 = match.queryIdx
                for cluster in clusters:
                    if name1 in cluster and cluster[name1] == ind1:
                        cluster[name2] = ind2
                        break
                    if name2 in cluster and cluster[name2] == ind2:
                        cluster[name1] = ind1
                        break
                else:
                    cluster = {}
                    cluster[name1] = ind1
                    cluster[name2] = ind2
                    clusters.append(cluster)

            if vstarstack.cfg.debug:
                d1 = vstarstack.data.DataFrame.load(fname1)
                img1,_ = d1.get_channel(channel)
                d2 = vstarstack.data.DataFrame.load(fname2)
                img2,_ = d2.get_channel(channel)

                img1 = (img1 / np.amax(img1) * 255).astype(np.uint8)
                img2 = (img2 / np.amax(img2) * 255).astype(np.uint8)

                img3 = cv2.drawMatches(img1,kp1,
                                        img2,kp2,
                                        matches,
                                        None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img3)
                plt.show()

        crd_clusters[channel] = []
        for cluster in clusters:
            crd_cluster = {}
            for name in cluster:
                kp,_,_ = kps[channel][name]
                point = kp[cluster[name]]
                crd_cluster[name] = {"x":point.pt[0], "y":point.pt[1]}
            crd_clusters[channel].append(crd_cluster)

    return crd_clusters

def run(argv):
    inputs = argv[0]
    clusters_fname = argv[1]
    
    files = vstarstack.common.listfiles(inputs, ".zip")
    files = [filename for name,filename in files]
    kps = find_kps(files)
    clusters = match_images(kps)

    total_clusters = []
    for channel in clusters:
        ch_clusters = clusters[channel]
        total_clusters += ch_clusters
    with open(clusters_fname, "w") as f:
        json.dump(total_clusters, f, indent=4, ensure_ascii=False)
