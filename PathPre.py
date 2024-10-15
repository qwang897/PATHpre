from model import *
import mdtraj as md
import argparse
import numpy as np
import torch
import torch.nn as nn


m = nn.Sigmoid()
device = "cuda"

def get_args():
	parser = argparse.ArgumentParser(description='Predict')
	parser.add_argument('--pre_path', action="store_true", help='whether to predict the path of allosteric')
	parser.add_argument('--pdb1', help='one structure of static state')
	parser.add_argument('--pdb2', help='one structure of static state')
	parser.add_argument('--cutoff', default=1.0, type=float, help='contactmap cutoff (nm)')
	args = parser.parse_args()
	return args

def calcmap_all(frame,resnum):
	map1 = np.zeros((resnum,resnum))
	num_contact = 0
	for index0 in range(resnum):
		for index1 in range(index0+4,resnum):
			res5 = frame[index0]
			res6 = frame[index1]
			dist_atom=((float(res5[0])-float(res6[0]))**2+(float(res5[1])-float(res6[1]))**2+(float(res5[2])-float(res6[2]))**2)**0.5
			map1[index0,index1]=dist_atom
			map1[index1,index0]=dist_atom
	map1 = 1-m(torch.tensor(map1)-1)
	for index0 in range(resnum):
		for index1 in range(index0,index0+4):
			if index1 < resnum:
				map1[index0,index1]=0.
				map1[index1,index0]=0.
	return np.expand_dims(map1,0)

def code_matrix(matrix,dis_cut):
	matrix[matrix < dis_cut] = 0.
	matrix[matrix >= dis_cut] = 1.
	return matrix

def constraint(A,B,C):
	great = A > torch.max(B,C)
	less = A < torch.min(B,C)
	result = torch.where(great,torch.max(B,C),torch.where(less,torch.min(B,C),A))
	return result

if __name__ == "__main__":
	args = get_args()
	dis_cut = 1-m(torch.tensor(args.cutoff)-1)
	traj1 = md.load(args.pdb1)
	traj2 = md.load(args.pdb2)
	ca_indices1 = traj1.topology.select("name CA")
	ca_indices2 = traj2.topology.select("name CA")
	length = len(ca_indices1)
	ca_coordinates1 = traj1.xyz[0][ca_indices1]
	ca_coordinates2 = traj2.xyz[0][ca_indices2]
	cry1 = torch.from_numpy(calcmap_all(ca_coordinates1,length)[0]).unsqueeze(2).to(device)
	cry2 = torch.from_numpy(calcmap_all(ca_coordinates2,length)[0]).unsqueeze(2).to(device)
	print(args.pre_path)
	if args.pre_path:
		model = ResNet(Block,[8]).to(device).half()
		model.load_state_dict(torch.load("./parameter/path.pth"))
		model.eval()
		with torch.no_grad():
			inputseq_matrix1_0 = torch.concat((cry1,cry2),dim=2).to(dtype=torch.float).half()
			inputseq_matrix1_1 = torch.concat((cry2,cry1),dim=2).to(dtype=torch.float).half()
			outputs1_0 = model(inputseq_matrix1_0.transpose(0,2).unsqueeze(0))
			outputs1_1 = model(inputseq_matrix1_1.transpose(0,2).unsqueeze(0))
			outputs1 = 0.5*(outputs1_0+outputs1_1)
			inputseq_matrix0_0 = torch.concat((cry1,outputs1[0,0].unsqueeze(2)),dim=2).to(dtype=torch.float).half()
			inputseq_matrix0_1 = torch.concat((outputs1[0,0].unsqueeze(2),cry1),dim=2).to(dtype=torch.float).half()
			inputseq_matrix2_0 = torch.concat((cry2,outputs1[0,0].unsqueeze(2)),dim=2).to(dtype=torch.float).half()
			inputseq_matrix2_1 = torch.concat((outputs1[0,0].unsqueeze(2),cry2),dim=2).to(dtype=torch.float).half()
			outputs0_0 = model(inputseq_matrix0_0.transpose(0,2).unsqueeze(0))
			outputs0_1 = model(inputseq_matrix0_1.transpose(0,2).unsqueeze(0))
			outputs2_0 = model(inputseq_matrix2_0.transpose(0,2).unsqueeze(0))
			outputs2_1 = model(inputseq_matrix2_1.transpose(0,2).unsqueeze(0))
			outputs1 = constraint(0.5*(outputs1_0+outputs1_1)[0,0], cry1[:,:,0].half(), cry2[:,:,0].half()) 
			outputs0 = constraint(0.5*(outputs0_0+outputs0_1)[0,0], cry1[:,:,0].half(), cry2[:,:,0].half())
			outputs2 = constraint(0.5*(outputs2_0+outputs2_1)[0,0], cry1[:,:,0].half(), cry2[:,:,0].half())
			#outputs0 = 0.5*(outputs0_0+outputs0_1)[0,0]
			#outputs2 = 0.5*(outputs2_0+outputs2_1)[0,0]
			#outputs1 = outputs1[0,0]
			np.save("./state2_con.npy",code_matrix(outputs0.cpu(),dis_cut))
			np.save("./state3_con.npy",code_matrix(outputs1.cpu(),dis_cut))
			np.save("./state4_con.npy",code_matrix(outputs2.cpu(),dis_cut))
	elif not args.pre_path:
		model = ResNet(Block,[8]).to(device)
		model.load_state_dict(torch.load("./parameter/hes.pth"))
		model.eval()
		with torch.no_grad():
			inputseq_matrix1_0 = torch.concat((cry1,cry2),dim=2).to(dtype=torch.float)
			inputseq_matrix1_1 = torch.concat((cry2,cry1),dim=2).to(dtype=torch.float)
			outputs1_0 = model(inputseq_matrix1_0.transpose(0,2).unsqueeze(0))
			outputs1_1 = model(inputseq_matrix1_1.transpose(0,2).unsqueeze(0))
			outputs1 = outputs1 = constraint(0.5*(outputs1_0+outputs1_1)[0,0], cry1[:,:,0].to(dtype=torch.float), cry2[:,:,0].to(dtype=torch.float))
			np.save("./HES.npy",code_matrix(outputs1.cpu(),dis_cut))
