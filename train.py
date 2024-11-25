from train_val import train_jpl, draw_save, send_email, train_harper
from Dataset import get_jpl_dataset, get_harper_dataset
import wandb

body_part = [True, True, True]
model = 'gcn_lstm'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
# framework = 'parallel'
# framework = 'tree'
framework = 'chain'
ori_video = False
frame_sample_hop = 1
sequence_length = 30


# JPL Dataset
# trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed',
#                                             ori_videos=ori_video)


# HARPER Dataset
# trainset, valset, testset = get_harper_dataset(body_part, sequence_length)


def train():
    p_m = train_jpl(wandb=wandb, model=model, body_part=body_part, framework=framework, sequence_length=sequence_length,
                    frame_sample_hop=frame_sample_hop, trainset=trainset, valset=valset, testset=testset)
    # p_m = train_harper(wandb=wandb, model=model, sequence_length=sequence_length, trainset=trainset, valset=valset,
    #                    testset=testset)
    # draw_save(framework, performance_model, framework)
    # send_email(result_str)


# sweep_config = {
#     'method': 'grid',
#     'metric': {
#         'name': 'avg_f1',
#         'goal': 'maximize',
#     },
#     'parameters': {
#         'epochs': {"values": [30, 35, 40, 45, 50, 55, 60]},
#         'loss_type': {'values': ['sum']},
#         'pretrained': {'values': [False]},
#         'new_classifier': {'values': [False]},
#         'times': {'values': [ii for ii in range(10)]}
#     }
# }
# sweep_id = wandb.sweep(sweep_config,
#                        project='SocialEgoNet_HARPER_fps%d' % int(sequence_length / frame_sample_hop))
# wandb.agent(sweep_id, function=train)
#
model_list = ['msgcn', 'r3d']
for model in model_list:
    trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length,
                                                augment_method='mixed', ori_videos=ori_video)
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'avg_f1',
            'goal': 'maximize',
        },
        'parameters': {
            'epochs': {"values": [1]},
            'keypoints_hidden_dim': {"values": [16]},
            'time_hidden_dim': {"values": [4]},
            'loss_type': {"values": ['sum']},
            'model': {'values': [model]},
            'times': {'values': [ii for ii in range(6)]}
        }
    }
    # wandb.init(project='SocialEgoNet', name='%s_%s' % (name, datetime.now().strftime("%Y-%m-%d_%H:%M")), config=config)
    sweep_id = wandb.sweep(sweep_config,
                           project='SocialEgoNet_JPL_fps%d_sota_test' % int(sequence_length / frame_sample_hop))
    wandb.agent(sweep_id, function=train)
