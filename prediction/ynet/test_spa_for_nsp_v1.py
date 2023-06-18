import torch
import torch.nn as nn
from utils.image_utils import get_patch, sampling, image2world
from utils.kmeans import kmeans
from utils.image_utils import get_patch, image2world, get_image_crop, get_patch_spa, get_gt_patch_spa
from utils.raster_util import raster_one_hot
from tqdm import tqdm

import pandas as pd
import os 
import numpy as np
import pdb
import pickle as pkl

def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
	"""
	Create Gaussian Kernel for CWS
	"""
	ax = torch.linspace(0, H, H, device=device) - coordinates[1]
	ay = torch.linspace(0, W, W, device=device) - coordinates[0]
	xx, yy = torch.meshgrid([ax, ay])
	meshgrid = torch.stack([yy, xx], dim=-1)
	radians = torch.atan2(dist[0], dist[1])

	c, s = torch.cos(radians), torch.sin(radians)
	R = torch.Tensor([[c, s], [-s, c]]).to(device)
	if rot:
		R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
	dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

	conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
	conv = torch.square(conv)
	T = torch.matmul(R, conv)
	T = torch.matmul(T, R.T)

	kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
	kernel = torch.exp(-0.5 * kernel)
	return kernel / kernel.sum()


def evaluate(model, val_loader, val_images, num_goals, num_traj, obs_len, batch_size, device, input_template, waypoints, resize, temperature, params, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None, dataset_name=None, homo_mat=None, mode='val', gt_template=torch.ones(1500,1500)):
	"""

	:param model: torch model
	:param val_loader: torch dataloader
	:param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param num_goals: int, number of goals
	:param num_traj: int, number of trajectories per goal
	:param obs_len: int, observed timesteps
	:param batch_size: int, batch_size
	:param device: torch device
	:param input_template: torch.Tensor, heatmap template
	:param waypoints: number of waypoints
	:param resize: resize factor
	:param temperature: float, temperature to control peakiness of heatmap
	:param use_TTST: bool
	:param use_CWS: bool
	:param rel_thresh: float
	:param CWS_params: dict
	:param dataset_name: ['sdd','ind','eth']
	:param params: dict with hyperparameters
	:param homo_mat: dict with homography matrix
	:param mode: ['val', 'test']
	:return: val_ADE, val_FDE for one epoch
	"""

	model.eval()
	val_ADE1 = [[],[],[],]
	val_FDE1 = [[],[],[],]
	val_ADE2 = [[],[],[],]
	val_FDE2 = [[],[],[],]
	val_ADE3 = [[],[],[],]
	val_FDE3 = [[],[],[],]
	val_ADE4 = [[],[],[],]
	val_FDE4 = [[],[],[],]
	val_ADE5 = [[],[],[],]
	val_FDE5 = [[],[],[],]
	val_trajFDE5 = [[],[],[],]
 
	out_val_ADE1 = [[],[],[],]
	out_val_FDE1 = [[],[],[],]
	out_val_ADE2 = [[],[],[],]
	out_val_FDE2 = [[],[],[],]
	out_val_ADE3 = [[],[],[],]
	out_val_FDE3 = [[],[],[],]
	out_val_ADE4 = [[],[],[],]
	out_val_FDE4 = [[],[],[],]
	out_val_ADE5 = [[],[],[],]
	out_val_FDE5 = [[],[],[],]
	out_val_trajFDE5 = [[],[],[],]
 
	in_val_ADE1 = [[],[],[],]
	in_val_FDE1 = [[],[],[],]
	in_val_ADE2 = [[],[],[],]
	in_val_FDE2 = [[],[],[],]
	in_val_ADE3 = [[],[],[],]
	in_val_FDE3 = [[],[],[],]
	in_val_ADE4 = [[],[],[],]
	in_val_FDE4 = [[],[],[],]
	in_val_ADE5 = [[],[],[],]
	in_val_FDE5 = [[],[],[],]
	in_val_trajFDE5 = [[],[],[],]
	
	indoor_list = ['hanyang', 'inter']
 
	W = params['crop_W']
	H = params['crop_H']
	counter = 0
	with torch.no_grad():
		# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
		for trajectory, meta, scene, trackID, frame_list, class_id_list, x_list, y_list in tqdm(val_loader):
			processed_pd_list = [[],[],[]]
   			# processed_pd_list = []
			# processed_pd_list = []	
			# Get scene image and apply semantic segmentation
			scene_image = val_images[scene].to(device).unsqueeze(0)
			scene_image = model.segmentation(scene_image)
			scene_image = torch.Tensor(raster_one_hot(scene_image)).permute(0,3,1,2).cuda()
			for indoor in indoor_list:
				if indoor in scene:
					ind = True
					print("indoor!!:", scene)
					break
				ind = False
				print("outdoor!!:", scene)
			
			for i in range(0, len(trajectory), batch_size):
				# Create Heatmaps for past and ground-truth future trajectories
				
				hist = trajectory[i:i+batch_size, :obs_len, :]
				track = trackID[i:i+batch_size]
				frame = frame_list[i:i+batch_size]
				label = class_id_list[i:i+batch_size]
				x = x_list[i:i+batch_size]
				y = y_list[i:i+batch_size]
				len_batch = trajectory[i:i+batch_size].shape[0]
				
				_, _, H, W = scene_image.shape  # image shape
				print('\revaluate, batch:', i, end=" ")
				observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
				observed_map = get_patch(input_template, observed, H, W)
				observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

				gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
				semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)
				feature_input = torch.cat([semantic_image, observed_map], dim=1)
				features = model.pred_features(feature_input)

				# Predict goal and waypoint probability distributions
				pred_waypoint_map = model.pred_goal(features)
				pred_waypoint_map = pred_waypoint_map[:, waypoints]

				pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
				pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

				################################################ TTST ##################################################
				for idx, goal_point_ in enumerate([1,5,20]):
					if use_TTST:
						# TTST Begin
						# sample a large amount of goals to be clustered
						goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
						goal_samples = goal_samples.permute(2, 0, 1, 3)

						num_clusters = goal_point_ - 1
						goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample

						# Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
						goal_samples_list = []
						for person in range(goal_samples.shape[1]):
							goal_sample = goal_samples[:, person, 0]

							# Actual k-means clustering, Outputs:
							# cluster_ids_x -  Information to which cluster_idx each point belongs to
							# cluster_centers - list of centroids, which are our new goal samples
							cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
							goal_samples_list.append(cluster_centers)

						goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
						goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
						# TTST End
					# Not using TTST
					else:
						goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=goal_point_)
						goal_samples = goal_samples.permute(2, 0, 1, 3)

					# Predict waypoints:
					# in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
					if len(waypoints) == 1:
						waypoint_samples = goal_samples

					################################################ CWS ###################################################
					# CWS Begin
					if use_CWS and len(waypoints) > 1:
						sigma_factor = CWS_params['sigma_factor']
						ratio = CWS_params['ratio']
						rot = CWS_params['rot']

						goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
						last_observed = trajectory[i:i+batch_size, obs_len-1].to(device)  # [N, 2]
						waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
						for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
							waypoint_list = []  # for each K sample have a separate list
							waypoint_list.append(waypoint_samples)

							for waypoint_num in reversed(range(len(waypoints)-1)):
								distance = last_observed - waypoint_samples
								gaussian_heatmaps = []
								traj_idx = g_num // goal_point_  # idx of trajectory for the same goal
								for dist, coordinate in zip(distance, waypoint_samples):  # for each person
									length_ratio = 1 / (waypoint_num + 2)
									gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
									sigma_factor_ = sigma_factor - traj_idx
									gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
								gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

								waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
								waypoint_map = waypoint_map_before * gaussian_heatmaps
								# normalize waypoint map
								waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

								# For first traj samples use softargmax
								if g_num // goal_point_ == 0:
									# Softargmax
									waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
									waypoint_samples = waypoint_samples.squeeze(0)
								else:
									waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
									waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
									waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
								waypoint_list.append(waypoint_samples)

							waypoint_list = waypoint_list[::-1]
							waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
							waypoint_samples_list.append(waypoint_list)
						waypoint_samples = torch.stack(waypoint_samples_list)

						# CWS End

					# If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
					elif not use_CWS and len(waypoints) > 1:
						waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=goal_point_ * num_traj)
						waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
						goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
						waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)
					# import pdb;pdb.set_trace()
					# Interpolate trajectories given goal and waypoints
					future_samples = []
					for waypoint in waypoint_samples:
						waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
						waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

						waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
						waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

						traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]

						pred_traj_map = model.pred_traj(traj_input)
						pred_traj = model.softargmax(pred_traj_map)
						future_samples.append(pred_traj)
					future_samples = torch.stack(future_samples)

					gt_goal = gt_future[:, -1:]

					# converts ETH/UCY pixel coordinates back into world-coordinates
					if dataset_name == 'eth':
						waypoint_samples = image2world(waypoint_samples, scene, homo_mat, resize)
						pred_traj = image2world(pred_traj, scene, homo_mat, resize)
						gt_future = image2world(gt_future, scene, homo_mat, resize)
					# print(gt_goal.shape)
					# print(waypoint_samples.shape)
					# print(future_samples.shape)
					# if ind:
					# 	print(hist[0][0], gt_goal[0,-1], waypoint_samples[0,0,-1], future_samples[0,0,-1])
					# if torch.sqrt(((gt_goal[0,-1] - future_samples[0,0,-1])**2).sum()) > 100:
					# 	print(hist[0][0], gt_goal[0,-1], waypoint_samples[0,0,-1], future_samples[0,0,-1])
					# if trajectory.any() < 0:
					# 	print(hist[0][0], gt_goal[0,-1], waypoint_samples[0,0,-1], future_samples[0,0,-1])
					# import pdb;pdb.set_trace()
					val_FDE1[idx].append(((((gt_future[:, 4:5] - future_samples[:, :, 4:5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					val_ADE1[idx].append(((((gt_future[:, :5] - future_samples[:, :, :5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
					val_FDE2[idx].append(((((gt_future[:, 9:10] - future_samples[:, :, 9:10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					val_ADE2[idx].append(((((gt_future[:, :10] - future_samples[:, :, :10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
					val_FDE3[idx].append(((((gt_future[:, 14:15] - future_samples[:, :, 14:15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					val_ADE3[idx].append(((((gt_future[:, :15] - future_samples[:, :, :15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
					val_FDE4[idx].append(((((gt_future[:, 19:20] - future_samples[:, :, 19:20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					val_ADE4[idx].append(((((gt_future[:, :20] - future_samples[:, :, :20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
					val_FDE5[idx].append(((((gt_goal - waypoint_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					val_ADE5[idx].append(((((gt_future[:, :25] - future_samples[:, :, :25]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
					val_trajFDE5[idx].append(((((gt_goal - future_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					if ind:
						in_val_FDE1[idx].append(((((gt_future[:, 4:5] - future_samples[:, :, 4:5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						in_val_ADE1[idx].append(((((gt_future[:, :5] - future_samples[:, :, :5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						in_val_FDE2[idx].append(((((gt_future[:, 9:10] - future_samples[:, :, 9:10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						in_val_ADE2[idx].append(((((gt_future[:, :10] - future_samples[:, :, :10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						in_val_FDE3[idx].append(((((gt_future[:, 14:15] - future_samples[:, :, 14:15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						in_val_ADE3[idx].append(((((gt_future[:, :15] - future_samples[:, :, :15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						in_val_FDE4[idx].append(((((gt_future[:, 19:20] - future_samples[:, :, 19:20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						in_val_ADE4[idx].append(((((gt_future[:, :20] - future_samples[:, :, :20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						in_val_FDE5[idx].append(((((gt_goal - waypoint_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						in_val_ADE5[idx].append(((((gt_future[:, :25] - future_samples[:, :, :25]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						in_val_trajFDE5[idx].append(((((gt_goal - future_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					else:
						out_val_FDE1[idx].append(((((gt_future[:, 4:5] - future_samples[:, :, 4:5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						out_val_ADE1[idx].append(((((gt_future[:, :5] - future_samples[:, :, :5]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						out_val_FDE2[idx].append(((((gt_future[:, 9:10] - future_samples[:, :, 9:10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						out_val_ADE2[idx].append(((((gt_future[:, :10] - future_samples[:, :, :10]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						out_val_FDE3[idx].append(((((gt_future[:, 14:15] - future_samples[:, :, 14:15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						out_val_ADE3[idx].append(((((gt_future[:, :15] - future_samples[:, :, :15]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						out_val_FDE4[idx].append(((((gt_future[:, 19:20] - future_samples[:, :, 19:20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						out_val_ADE4[idx].append(((((gt_future[:, :20] - future_samples[:, :, :20]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						out_val_FDE5[idx].append(((((gt_goal - waypoint_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
						out_val_ADE5[idx].append(((((gt_future[:, :25] - future_samples[:, :, :25]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
						out_val_trajFDE5[idx].append(((((gt_goal - future_samples[:, :, -1:]) *4 / 50 / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
					# print(future_samples.shape)
					# pdb.set_trace()
					# if goal_point_ != 1:
					# 	pdb.set_trace()
					breakpoint()
					for new_i in range(len_batch):
						for new_j in range(obs_len):
							new_dict = {
								'frame':[frame[new_i][new_j]],
								'trackID':[track[new_i][new_j]],
								'xmin':x[new_i][new_j],
								'xmax':x[new_i][new_j],
								'ymin':y[new_i][new_j],
								'ymax':y[new_i][new_j],
								'sceneId':scene,
								'label':label[new_i][new_j],
								'lost':False,
								'occluded': [0],
								'generated': [0],
								'target': [1],
								'pred':np.array((future_samples[:,new_i] / resize).cpu().detach().numpy()),
							}
							
							# if np.array(waypoint_samples[:, :, -1:][:,new_i].cpu().detach().numpy()).shape !=  (20, 1, 2) :
							# 	print(np.array(waypoint_samples[:, :, -1:][:,new_i].cpu().detach().numpy()).shape)
							processed_pd_list[idx].append(new_dict)     
					
			for idx, goal_point_ in enumerate([1,5,20]):
				processed_pd = pd.DataFrame(processed_pd_list[idx])
				save_dir = f'result/result_goal{goal_point_}'
			

				if not os.path.exists(save_dir):
					os.mkdir(save_dir)
				# with open(f'./{save_dir}/{scene}.pkl', 'wb') as f:
				# 	pkl.dump(processed_pd, f)
				processed_pd.to_pickle(f'./{save_dir}/{scene}.pkl')
				# cur_ynet_pkl = pd.read_pickle(f'./{save_dir}/{scene}.pkl')

				# pred = {}
				# try:
				# 	for meta_i, meta_df in cur_ynet_pkl.groupby('trackID'):
				# 		pred[int(meta_df['trackID'].values[0])] = meta_df['pred'].values
				# except:
				# 	breakpoint()

		for i in range(3):
			# pdb.set_trace()
			val_ADE1[i] = torch.cat(val_ADE1[i]).mean()
			val_FDE1[i] = torch.cat(val_FDE1[i]).mean()
			val_ADE2[i] = torch.cat(val_ADE2[i]).mean()
			val_FDE2[i] = torch.cat(val_FDE2[i]).mean()
			val_ADE3[i] = torch.cat(val_ADE3[i]).mean()
			val_FDE3[i] = torch.cat(val_FDE3[i]).mean()
			val_ADE4[i] = torch.cat(val_ADE4[i]).mean()
			val_FDE4[i] = torch.cat(val_FDE4[i]).mean()
			val_ADE5[i] = torch.cat(val_ADE5[i]).mean()
			val_FDE5[i] = torch.cat(val_FDE5[i]).mean()
			val_trajFDE5[i] = torch.cat(val_trajFDE5[i]).mean()
   
			in_val_ADE1[i] = torch.cat(in_val_ADE1[i]).mean()
			in_val_FDE1[i] = torch.cat(in_val_FDE1[i]).mean()
			in_val_ADE2[i] = torch.cat(in_val_ADE2[i]).mean()
			in_val_FDE2[i] = torch.cat(in_val_FDE2[i]).mean()
			in_val_ADE3[i] = torch.cat(in_val_ADE3[i]).mean()
			in_val_FDE3[i] = torch.cat(in_val_FDE3[i]).mean()
			in_val_ADE4[i] = torch.cat(in_val_ADE4[i]).mean()
			in_val_FDE4[i] = torch.cat(in_val_FDE4[i]).mean()
			in_val_ADE5[i] = torch.cat(in_val_ADE5[i]).mean()
			in_val_FDE5[i] = torch.cat(in_val_FDE5[i]).mean()
			in_val_trajFDE5[i] = torch.cat(in_val_trajFDE5[i]).mean()

			out_val_ADE1[i] = torch.cat(out_val_ADE1[i]).mean()
			out_val_FDE1[i] = torch.cat(out_val_FDE1[i]).mean()
			out_val_ADE2[i] = torch.cat(out_val_ADE2[i]).mean()
			out_val_FDE2[i] = torch.cat(out_val_FDE2[i]).mean()
			out_val_ADE3[i] = torch.cat(out_val_ADE3[i]).mean()
			out_val_FDE3[i] = torch.cat(out_val_FDE3[i]).mean()
			out_val_ADE4[i] = torch.cat(out_val_ADE4[i]).mean()
			out_val_FDE4[i] = torch.cat(out_val_FDE4[i]).mean()
			out_val_ADE5[i] = torch.cat(out_val_ADE5[i]).mean()
			out_val_FDE5[i] = torch.cat(out_val_FDE5[i]).mean()
			out_val_trajFDE5[i] = torch.cat(out_val_trajFDE5[i]).mean()
   
	return val_ADE1, val_FDE1, val_ADE2, val_FDE2, val_ADE3, val_FDE3, val_ADE4, val_FDE4, val_ADE5, val_FDE5, val_trajFDE5,\
	in_val_ADE1, in_val_FDE1, in_val_ADE2, in_val_FDE2, in_val_ADE3, in_val_FDE3, in_val_ADE4, in_val_FDE4, in_val_ADE5, in_val_FDE5, in_val_trajFDE5, \
    out_val_ADE1, out_val_FDE1, out_val_ADE2, out_val_FDE2, out_val_ADE3, out_val_FDE3, out_val_ADE4, out_val_FDE4, out_val_ADE5, out_val_FDE5, out_val_trajFDE5, 