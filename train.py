from train_val import train_jpl, draw_save, send_email

body_part = [True, True, True]
model = 'gcn_lstm'
# model = 'gcn_tran'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
# framework = 'parallel'
framework = 'tree'
# framework = 'chain'
ori_video = False
frame_sample_hop = 1
sequence_length = 30
dataset = 'mixed+coco'
oneshot = False
for model in ['gcn_lstm']:
    # for body_part in [[False, True, False], [False, False, True], [False, True, True]]:
    # for framework in ['parallel', 'tree', 'chain']:
    # for model in ['msgcn', 'dgstgcn']:
    # for sequence_length in [5,10,15,20,25,35,40]:
    performance_model = []
    i = 0
    while i < 5:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if sequence_length:
            p_m = train_jpl(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, sequence_length=sequence_length, dataset=dataset, oneshot=oneshot)
        else:
            p_m = train_jpl(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, dataset=dataset, oneshot=oneshot)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(framework, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sequence_length: %d, frame_hop: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sequence_length, frame_sample_hop)
    print(result_str)
    send_email(result_str)
