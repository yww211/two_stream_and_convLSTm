require 'torch'
tv_rgb_split1_test30_part1= torch.load('/data_yww/sum_tv_rgb_split1_test30_half1.t7')
tv_rgb_split1_test30_part2 = torch.load('/data_yww/sum_tv_rgb_split1_test30_half2.t7')
tv_rgb_split1_test30 = {}
table.insert(tv_rgb_split1_test30, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
tv_rgb_split1_test30.featMats = torch.cat(tv_rgb_split1_test30_part1.featMats,tv_rgb_split1_test30_part2.featMats,1)
  label = torch.IntTensor(tv_rgb_split1_test30.featMats:size()[1])
    for i = 1, 51 do 
      for j =1,30 do
          label[30*(i-1)+j]=i
          end
      end
     tv_rgb_split1_test30.labels = label
 print(tv_rgb_split1_test30.labels)
torch.save('tv_rgb_split1_test30.t7',tv_rgb_split1_test30)