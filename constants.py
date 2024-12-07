import torch
import os

bless_str = ("                         _oo0oo_\n"
             "                        o8888888o\n"
             "                        88\" . \"88\n"
             "                        (| -_- |)\n"
             "                        0\  =  /0\n"
             "                      ___/`---'\___\n"
             "                    .' \\|     |// '.\n"
             "                   / \\|||  :  |||// \ \n"
             "                  / _||||| -:- |||||- \ \n"
             "                 |   | \\\  - /// |   |\n"
             "                 | \_|  ''\---/''  |_/ |\n"
             "                 \  .-\__  '-'  ___/-. /\n"
             "               ___'. .'  /--.--\  `. .'___\n"
             "            .\"\" '<  `.___\_<|>_/___.' >' \"\".\n"
             "           | | :  `- \`.;`\ _ /`;.`/ - ` : | |\n"
             "           \  \ `_.   \_ __\ /__ _/   .-` /  /\n"
             "       =====`-.____`.___ \_____/___.-`___.-'=====\n"
             "                         `=---='\n"
             "       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
             "                 BLESS ME WITH NO BUGS\n"
             )
print(bless_str)
avg_batch_size = 128
perframe_batch_size = 2048
rnn_batch_size = 128
conv1d_batch_size = 128
tran_batch_size = 128
gcn_batch_size = 128
stgcn_batch_size = 32
msgcn_batch_size = 8
dgstgcn_batch_size = 16
r3d_batch_size = 16
learning_rate = 1e-3

if torch.cuda.is_available():
    print('Using CUDA for training')
    device = torch.device("cuda:0")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # device = torch.device('cpu')
elif torch.backends.mps.is_available():
    print('Using MPS for training')
    device = torch.device('mps')
else:
    print('Using CPU for training')
    device = torch.device('cpu')
dtype = torch.float32
intention_classes = ['Interacting', 'Interested', 'Not_Interested']
attitude_classes = ['Positive', 'Negative', 'Not_Interacting']
jpl_action_classes = ['Handshake', 'Hug', 'Pet', 'Wave', 'Punch', 'Throw', 'Point', 'Gaze', 'Leave',
                      'No_Response']
harper_action_class = ['Walk_Crash', 'Walk_Stop', 'Walk_Avoid', 'Walk_Touch', 'Walk_Kick', 'Walk_Punch',
                       'Circular_Walk', 'Circular_Follow_Touch', 'Circular_Follow_Avoid', 'Circular_Follow_Crash']
attack_class = ['Attack', 'Normal', 'Danger', 'Not_Interacting']

body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
valset_rate = 0.1
testset_rate = 0.4
video_fps = 30

body_l_pair = [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
               [5, 7], [7, 9], [6, 8], [8, 10],  # Body
               [5, 6], [11, 12], [5, 11], [6, 12],
               # [0, 5], [0, 6], [1, 2],
               [11, 13], [12, 14], [13, 15], [14, 16],
               [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]]
head_l_pair = [[23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
               [32, 33], [33, 34], [34, 35],
               [35, 36], [36, 37], [37, 38], [38, 39], [40, 41], [41, 42], [42, 43], [43, 44], [45, 46],
               [46, 47], [47, 48], [48, 49],
               [50, 51], [51, 52], [52, 53], [54, 55], [55, 56], [56, 57], [57, 58], [59, 60], [60, 61],
               [61, 62], [62, 63], [63, 64],
               [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [71, 72], [72, 73], [73, 74], [74, 75],
               [75, 76], [76, 77], [77, 78],
               [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87],
               [87, 88], [88, 89], [89, 90]]
hand_l_pair = [[91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
               [100, 101], [101, 102],
               [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
               [110, 111], [112, 113],
               [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
               [121, 122], [122, 123],
               [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
               [131, 132]]
visible_threshold_score = 0.5
# visible_threshold_score = 0.7
