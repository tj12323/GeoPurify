import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

from dataset.point_loader import Point3DLoader


class FusedFeatureLoader(Point3DLoader):
    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_feat,
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.aug = aug
        self.input_color = input_color

        self.datapath_feat = datapath_prefix_feat

        if "nuscenes" in self.dataset_name:
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if "scannet" in self.dataset_name:
                    scene_name = data_path[:-15].split("/")[-1]
                else:
                    scene_name = data_path[:-4].split("/")[-1]
                    scene_name = data_path[:-4].split("/")[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + "_*.pt"))
                self.list_occur.append(len(file_dirs))

            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur) == 0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach(
                "shm://%s_%s_%06d_locs_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            feats_in = SA.attach(
                "shm://%s_%s_%06d_feats_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            labels_in = SA.attach(
                "shm://%s_%s_%06d_labels_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:

                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.0) * 127.5

        if self.dataset_name == "scannet_3d":
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        if "nuscenes" not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(
                join(self.datapath_feat, scene_name + "_%d.pt" % (nn_occur))
            )
        else:

            processed_data = torch.load(join(self.datapath_feat, scene_name + ".pt"))

        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data["feat"], processed_data["mask_full"]
            if isinstance(mask_chunk, np.ndarray):
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != "train":
                feat_3d_new = torch.zeros(
                    (locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype
                )
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(mask_chunk)
        elif len(processed_data.keys()) > 2:
            feat_3d, mask_visible, mask_chunk = (
                processed_data["feat"],
                processed_data["mask"],
                processed_data["mask_full"],
            )
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        if self.split == "train" and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)

            indices = index3[chunk_ind] - 1

            feat_3d = feat_3d[indices]
        elif self.split == "train" and not flag_mask_merge:
            feat_3d = feat_3d[mask]
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)

            indices = index3[chunk_ind] - 1

            feat_3d = feat_3d[indices]
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk],
                feats_in[mask_chunk],
                labels_in[mask_chunk],
                return_ind=True,
            )
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        else:

            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return (
                coords,
                feats,
                labels,
                feat_3d,
                mask,
                torch.from_numpy(inds_reconstruct).long(),
            )
        return coords, feats, labels, feat_3d, mask


def collation_fn(batch):

    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask_chunk),
    )


def collation_fn_eval_all(batch):

    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask),
        torch.cat(inds_recons),
    )
