from train_val import train_jpl, draw_save, send_email, train_harper
from Dataset import get_jpl_dataset, HARPER_Dataset
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
frame_sample_hop = 1
sequence_length = 10
# JPL Dataset
trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed',
                                            ori_videos=ori_video)

# HARPER Dataset
data_path = '../HARPER/pose_sequences/'

trainset = HARPER_Dataset(data_path=data_path, files=train_files, body_part=body_part, sequence_length=10,
                          train=True)
valset = HARPER_Dataset(data_path=data_path, files=val_files, body_part=body_part, sequence_length=10)
testset = HARPER_Dataset(data_path=data_path, files=test_files, body_part=body_part, sequence_length=10)

print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (len(trainset), len(valset), len(testset)))


def train():
    # p_m = train_jpl(wandb=wandb, model=model, body_part=body_part, framework=framework, sequence_length=sequence_length,
    #                 frame_sample_hop=frame_sample_hop, trainset=trainset, valset=valset, testset=testset)
    p_m = train_harper(wandb=wandb, model=model, sequence_length=sequence_length)
    # draw_save(framework, performance_model, framework)
    # send_email(result_str)


if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'avg_f1',
            'goal': 'maximize',
        },
        'parameters': {
            'epochs': {'values': [5, 10, 15, 20]},
            'new_classifier': {'values': [True, False]},
            'times': {'values': [0, 1, 2, 3]}
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
    sweep_id = wandb.sweep(sweep_config, project='SocialEgoNet_HARPER_fps10')
    wandb.agent(sweep_id, function=train)
