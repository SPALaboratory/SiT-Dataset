from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

class SceneDataset(Dataset):
	def __init__(self, data, resize, total_len):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""

		self.trajectories, self.meta, self.scene_list, self.trackID, self.frame_list, self.class_id_list, self.x_list, self.y_list = self.split_trajectories_by_scene(data, total_len)
		self.trajectories = self.trajectories * resize

	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, idx):
		trajectory 	  = self.trajectories[idx]
		meta 		  = self.meta[idx]
		scene 		  = self.scene_list[idx]
  
		trackID 	  = self.trackID[idx]
		frame_list 	  = self.frame_list[idx]
		class_id_list = self.class_id_list[idx]
		x_list 		  = self.x_list[idx]
		y_list 		  = self.y_list[idx]

		return trajectory, meta, scene, trackID, frame_list, class_id_list, x_list, y_list

	def split_trajectories_by_scene(self, data, total_len):
		trajectories = []
		meta = []
		scene_list = []
		trackID_list = []
		frame_list = []
		class_id_list = []
		xmin_list = []
		xmax_list = []
		ymin_list = []
		ymax_list = []

		for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
			
			try:
				trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
			except:
				import pdb;pdb.set_trace()
			meta.append(meta_df)
			scene_list.append(meta_df.iloc()[0:1].sceneId.item())
			# import pdb;pdb.set_trace()
			# trackID_LIST = []
			# for i in range(len(meta_df['trackID'])):
			# 	s_ = meta_df['trackID'].tolist()[i].split('_')
			# 	if s_[1] == "hanyang":
			# 		if s_[2] == 'plz':
			# 			loc = '0'
			# 		else:
			# 			loc = '1'
			# 	elif s_[1] == 'inter':
			# 		loc = '2'
			# 	elif s_[1] == 'sungsu_alley':
			# 		if s_[1] == 'cross':
			# 			loc = '3'
			# 		else:
			# 			loc = '4'
			# 	elif s_[1] == '2exit':
			# 		loc = '5'
			# 	else:
			# 		loc = '6'
			# 	trackID_LIST.append(int(s_[0]+loc+s_[-2]+s_[-1].zfill(3)))
			# trackID_list.append(np.array(trackID_LIST).astype('int32').reshape(-1, total_len))
			trackID_list.append(meta_df['trackID'].to_numpy().reshape(-1, total_len))
			frame_list.append(meta_df['frame'].to_numpy().astype('float32').reshape(-1, total_len))
			class_id_list.append(meta_df['class'].to_numpy().astype('float32').reshape(-1, total_len))
			xmin_list.append(meta_df['x'].to_numpy().astype('float32').reshape(-1, total_len))
			ymin_list.append(meta_df['y'].to_numpy().astype('float32').reshape(-1, total_len))
   			

		# for meta_id, meta_df in tqdm(data.groupby('metaId', as_index=False), desc='Prepare Dataset'):
		# 	trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape( total_len, 2))
		# 	meta.append(meta_df)
		# 	scene_list.append(meta_df.iloc()[0:1].sceneId.item())
		return np.array(trajectories), meta, scene_list, np.array(trackID_list), np.array(frame_list), np.array(class_id_list), np.array(xmin_list), np.array(ymin_list)


def scene_collate(batch):
	trajectories = []
	meta = []
	scene = []
	trackID_list = []
	frame_list = []
	class_id_list = []
	xmin_list = []
	ymin_list = []
 
 
	for _batch in batch:
		trajectories.append(_batch[0])
		meta.append(_batch[1])
		scene.append(_batch[2])
  
		trackID_list.append(_batch[3])
		frame_list.append(_batch[4])
		class_id_list.append(_batch[5])
		xmin_list.append(_batch[6])
		ymin_list.append(_batch[7])
  
	return torch.Tensor(trajectories).squeeze(0), meta, scene[0], np.array(trackID_list).squeeze(0), np.array(frame_list).squeeze(0), np.array(class_id_list).squeeze(0), np.array(xmin_list).squeeze(0),  np.array(ymin_list).squeeze(0)

