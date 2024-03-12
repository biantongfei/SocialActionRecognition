from train_val import train, draw_save

model = 'lstm'
body_part = [True, True, True]
# data_format = 'coordinates'
data_format = 'manhattan'
# data_format = 'dis_angel'

# framework = 'intent'
# framework = 'attitude'
# framework = 'action'
# framework = 'parallel'
framework = 'tree'
# framework = 'chain'
ori_video = False
sample_fps = 30
video_len = 2
empty_frame = 'same'
performance_model = []
i = 0
while i < 10:
    print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
    # try:
    if video_len:
        p_m = train(model=model, body_part=body_part, data_format=data_format, framework=framework,
                    sample_fps=sample_fps, ori_videos=ori_video, video_len=video_len, empty_frame=empty_frame)
    else:
        p_m = train(model=model, body_part=body_part, data_format=data_format, framework=framework,
                    sample_fps=sample_fps, ori_videos=ori_video, empty_frame=empty_frame)
    # except ValueError:
    #     continue
    performance_model.append(p_m)
    i += 1
draw_save(performance_model, framework)
print('model: %s, body_part:' % model, body_part,
      ', framework: %s, sample_fps: %d, video_len: %s, empty_frame: %s' % (
          framework, sample_fps, str(video_len), str(empty_frame)))
