from train_val import train, draw_save, send_email

body_part = [True, True, True]

model = 'gcn_lstm'
# model = 'gcn_tran'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
framework = 'chain'
# framework = 'tree'
# framework = 'chain'
ori_video = False
frame_sample_hop = 1
sequence_length = 30
dataset = 'crop+coco'
oneshot = False
for model in ['gcn_gcn']:
    # for body_part in [[True, False, False], [True, True, False], [True, False, True]]:
    # for framework in ['parallel', 'tree', 'chain']:
    # for model in ['msgcn', 'dgstgcn']:
    performance_model = []
    i = 0
    while i < 1:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if sequence_length:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, sequence_length=sequence_length, dataset=dataset, oneshot=oneshot)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, dataset=dataset, oneshot=oneshot)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(model, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sequence_length: %d, frame_hop: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sequence_length, frame_sample_hop)
    print(result_str)
    # send_email(result_str)
