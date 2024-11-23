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
sequence_length = 10
# JPL Dataset
# trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed',
#                                             ori_videos=ori_video)


# HARPER Dataset
trainset, valset, testset = get_harper_dataset(body_part, sequence_length)


def train():
    # p_m = train_jpl(wandb=wandb, model=model, body_part=body_part, framework=framework, sequence_length=sequence_length,
    #                 frame_sample_hop=frame_sample_hop, trainset=trainset, valset=valset, testset=testset)
    p_m = train_harper(wandb=wandb, model=model, sequence_length=sequence_length, trainset=trainset, valset=valset,
                       testset=testset)
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
            'epochs': {"values": [40, 45, 50]},
            'pretrained': {'values': [True]},
            'new_classifier': {'values': [True]},
            'times': {'values': [i for i in range(10)]}
        }
    }
    # wandb.init(project='SocialEgoNet', name='%s_%s' % (name, datetime.now().strftime("%Y-%m-%d_%H:%M")), config=config)
    sweep_id = wandb.sweep(sweep_config, project='SocialEgoNet_HARPER_fps%d' % int(sequence_length / frame_sample_hop))
    wandb.agent(sweep_id, function=train)
