from train_val import train_jpl, draw_save, send_email, train_harper
from Dataset import get_jpl_dataset
import wandb
from datetime import datetime

body_part = [True, True, True]
model = 'gcn_lstm'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
# framework = 'parallel'
# framework = 'tree'
framework = 'chain'
ori_video = False
frame_sample_hop = 3
sequence_length = 30
trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed',
                                            ori_videos=ori_video)


def train():
    p_m = train_jpl(wandb=wandb, model=model, body_part=body_part, framework=framework, sequence_length=sequence_length,
                    frame_sample_hop=frame_sample_hop, trainset=trainset, valset=valset, testset=testset)
    # pretrained = True
    # new_classifier = False
    # if_train = False
    # p_m = train_harper(wandb=wandb, model=model, sequence_length=sequence_length, body_part=body_part,
    #                    pretrained=pretrained, new_classifier=new_classifier, train=if_train)
    # draw_save(framework, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sequence_length: %d, frame_hop: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sequence_length, frame_sample_hop)
    print(result_str)
    # send_email(result_str)


if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'avg_f1',
            'goal': 'maximize',
        },
        'parameters': {
            'epochs': {'values': [30, 40, 50]},
            'keypoint_hidden_dim': {'values': [16, 32, 64]},
            'time_hidden_dim': {'values': [1, 2, 4]},
            'fc_hidden1': {'values': [32, 64]},
            'fc_hidden2': {'values': [8, 16]}
        }
    }
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {
    #         'name': 'avg_f1',
    #         'goal': 'maximize',
    #     },
    #     'parameters': {
    #         'epochs': {'values': [5, 10, 15, 20]},
    #     },
    #     'early_terminate': {
    #         'type': 'hyperband',
    #         'min_iter': 5,
    #         'eta': 2,
    #         's': 5
    #     }
    # }
    # wandb.init(project='SocialEgoNet', name='%s_%s' % (name, datetime.now().strftime("%Y-%m-%d_%H:%M")), config=config)
    sweep_id = wandb.sweep(sweep_config, project='SocialEgoNet_JPL_fps30')
    wandb.agent(sweep_id, function=train)
