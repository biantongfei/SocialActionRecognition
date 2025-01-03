from train_val import train_jpl, send_email, train_harper
from Dataset import get_jpl_dataset
import wandb

body_part = [True, False, False]
model = 'gcn_lstm'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
# framework = 'parallel'
# framework = 'tree'
framework = 'chain'
frame_sample_hop = 3
sequence_length = 30

# JPL Dataset
trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed')


# HARPER Dataset
# trainset, valset, testset = get_harper_dataset(body_part, sequence_length)


def train():
    p_m = train_jpl(model=model, body_part=body_part, framework=framework, sequence_length=sequence_length,
                    frame_sample_hop=frame_sample_hop, trainset=trainset, valset=valset, testset=testset)
    # p_m = train_harper(model=model, sequence_length=sequence_length, trainset=trainset, valset=valset,
    #                    testset=testset)
    # draw_save(framework, performance_model, framework)
    # send_email(result_str)


sweep_config = {
        # 'method': 'bayes',
        # 'method': 'random',
        'method': 'grid',
        'metric': {
            'name': 'avg_f1',
            'goal': 'maximize',
        },
        'parameters': {
            'epochs': {"values": [50]},
            'loss_type': {"values": ['sum']},
            'keypoints_hidden_dim': {'values': [16]},
            'time_hidden_dim': {'values': [8]},
            'times': {'values': [ii for ii in range(10)]},
        }
    }
sweep_id = wandb.sweep(sweep_config, project='SocialEgoNet_JPL_fps%d' % int(sequence_length / frame_sample_hop))
wandb.agent(sweep_id, function=train)

# sweep_config = {
#     'method': 'grid',
#     'metric': {
#         'name': 'avg_f1',
#         'goal': 'maximize',
#     },
#     'parameters': {
#         'epochs': {"values": [30, 35, 40, 45, 50, 55, 60]},
#         'loss_type': {"values": ['sum']},
#         'times': {'values': [ii for ii in range(10)]},
#         'new_classifier': {"values": [False]},
#         'pretrained': {"values": [False]}
#     }
# }
# sweep_id = wandb.sweep(sweep_config, project='SocialEgoNet_HARPER_fps%d' % int(sequence_length / frame_sample_hop))
# wandb.agent(sweep_id, function=train)

# model_list = ['gcn_conv1d', 'gcn_tran', 'gcn_gcn']
# for model in model_list:
#     # trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length,
#     #                                             augment_method='mixed')
#     sweep_config = {
#         'method': 'grid',
#         'metric': {
#             'name': 'avg_f1',
#             'goal': 'maximize',
#         },
#         'parameters': {
#             'epochs': {"values": [40]},
#             'keypoints_hidden_dim': {"values": [16]},
#             'time_hidden_dim': {"values": [4]},
#             'loss_type': {"values": ['sum']},
#             'model': {'values': [model]},
#             'times': {'values': [ii for ii in range(10)]}
#         }
#     }
#     # wandb.init(project='SocialEgoNet', name='%s_%s' % (name, datetime.now().strftime("%Y-%m-%d_%H:%M")), config=config)
#     sweep_id = wandb.sweep(sweep_config,
#                            project='SocialEgoNet_JPL_fps30_sota')
#     wandb.agent(sweep_id, function=train)
