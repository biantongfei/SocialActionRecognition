from train_val import train, draw_save, send_email

# model = 'gnn_keypoint_conv1d'
body_part = [True, True, True]

# framework = 'intention'
# framework = 'attitude'
# framework = 'action'
framework = 'parallel'
# framework = 'tree'
# framework = 'chain'
ori_video = False
sample_fps = 30
video_len = 2
# for model in ['avg', 'perframe', 'conv1d', 'lstm', 'gru']:
for model in ['gcn_lstm', 'gcn_gru']:
# for model in ['gcn_lstm', 'gcn_gru']:
    performance_model = []
    i = 0
    while i < 10:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if video_len:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video, video_len=video_len)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(model, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sample_fps: %d, video_len: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sample_fps, str(video_len))
    print(result_str)
    send_email(result_str)
