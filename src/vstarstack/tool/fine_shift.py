#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#


import os
import json
import multiprocessing as mp

from vstarstack.library.fine_shift.fine_shift import Aligner
import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.library.data
import vstarstack.library.common

import vstarstack.tool.common

ncpu = vstarstack.tool.cfg.nthreads

def create_aligner(project: vstarstack.tool.cfg.Project):
    """Create aligner for the project"""
    W = project.config.telescope.camera.w
    H = project.config.telescope.camera.h

    num_steps = project.config.fine_shift.Nsteps
    dh = project.config.fine_shift.dh
    gridW = project.config.fine_shift.gridW
    gridH = project.config.fine_shift.gridH
    spk = project.config.fine_shift.stretchPenlatyCoefficient
    min_points = project.config.fine_shift.points_min_len

    aligner = Aligner(W, H, gridW, gridH, spk, num_steps, min_points, dh)
    return aligner

def process_alignment(name : str,
                      inpath : str,
                      outpath : str,
                      aligner : Aligner,
                      clusters : list):
    """Find alignment for image"""
    print(f"Processing: {name}")
    if os.path.isfile(inpath):
        with open(inpath, encoding='utf8') as f:
            descriptor = json.load(f)
    else:
        descriptor = {}

    data = aligner.process_alignment(name, clusters)
    descriptor["fine_shift"] = {}
    descriptor["fine_shift"]["descriptor"] = data

    vstarstack.tool.common.check_dir_exists(outpath)
    with open(outpath, "w", encoding='utf8') as f:
        json.dump(descriptor, f, indent=4, ensure_ascii=False)

def _process_alignment_wrapper(arg):
    return process_alignment(*arg)

def find_alignment(project: vstarstack.tool.cfg.Project, argv: list):
    """Find alignment for each image file"""
    clusters = argv[0]
    outpath = argv[1]

    with open(clusters, encoding='utf8') as f:
        clusters = json.load(f)

    aligner = create_aligner(project)
    cluster_len_k = project.config.fine_shift.cluster_len_k
    maxcllen = max([len(cluster) for cluster in clusters])
    names = []

    good_clusters = []
    for cluster in clusters:
        if len(cluster.keys()) < maxcllen * cluster_len_k:
            continue

        names += list(cluster.keys())
        good_clusters.append(cluster)
    names = sorted(list(set(names)))

    with mp.Pool(ncpu) as pool:
        args = [(name, outpath, outpath, aligner, good_clusters) for name in names]
        for _ in pool.imap_unordered(_process_alignment_wrapper, args):
            pass


def apply_alignment_file(name : str,
                         image_f : str,
                         descriptor_f : str,
                         aligner : Aligner,
                         output_f : str):
    """Apply alignment to each file"""
    print(name)
    if not os.path.exists(descriptor_f):
        return
    with open(descriptor_f, encoding='utf8') as f:
        descriptor = json.load(f)
        descriptor = descriptor["fine_shift"]["descriptor"]

    dataframe = vstarstack.library.data.DataFrame.load(image_f)
    dataframe = aligner.apply_alignment(dataframe, descriptor)
    vstarstack.tool.common.check_dir_exists(output_f)
    dataframe.store(output_f)

def _apply_alignment_file_wrapper(arg):
    apply_alignment_file(*arg)


def apply_alignment(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 3:
        npys = argv[0]
        descs = argv[1]
        outputs = argv[2]
    else:
        npys = project.config.paths.npy_fixed
        descs = project.config.paths.descs
        outputs = project.config.paths.aligned

    aligner = create_aligner(project)
    if os.path.isdir(npys):
        files = vstarstack.tool.common.listfiles(npys, ".zip")
        with mp.Pool(ncpu) as pool:
            args = [(name,
                     fname,
                     os.path.join(descs, name + ".json"),
                     aligner,
                     os.path.join(outputs, name + ".zip"))
                for name, fname in files]
            for _ in pool.imap_unordered(_apply_alignment_file_wrapper, args):
                pass
    else:
        apply_alignment_file(os.path.basename(npys), npys, descs, aligner, outputs)

commands = {
    "align-features": (find_alignment, "find alignment of images", "clusters.json alignments/"),
    "apply-aligns": (apply_alignment, "apply alignments to images", "npys/ alignments/ output/"),
}
