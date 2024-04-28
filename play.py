# edge_index = torch.tensor(np.array(
#                             [[i[0] + int(len(x_list) * input_size / 2), i[1] + int(len(x_list) * input_size / 2)] for i
#                              in get_l_pair(self.is_coco, self.body_part)]), dtype=torch.long).t().contiguous()
a = [1, 2, 3]
a += [1]
print(a)
