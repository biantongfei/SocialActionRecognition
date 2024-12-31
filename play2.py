import pickle

with open('../30hz/avo_act1_30hz.pkl', 'rb') as f_1:
    data = pickle.load(f_1)
    print(data[0].keys())
    print(data[0]['human_joints_3d'])