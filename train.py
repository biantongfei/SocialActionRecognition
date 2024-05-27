from train_val import train, draw_save, send_email

body_part = [True, True, True]

model = 'gcn_lstm'
# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
framework = 'parallel'
# framework = 'tree'
# framework = 'chain'
ori_video = True
frame_sample_hop = 1
sequence_length = 30
# for framework in ['parallel']:
for framework in ['intention', 'attitude', 'action', 'chain', 'tree']:
    performance_model = []
    i = 0
    while i < 10:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if sequence_length:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, sequence_length=sequence_length)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(framework, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sequence_length: %d, frame_hop: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sequence_length, frame_sample_hop)
    print(result_str)
    send_email(result_str)
