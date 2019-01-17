require 'torch'
--[[
tv_split1_test30_half1 = torch.load('/data_yww/feat_collection/tv_split1_test30_half1.t7')
rgb_split1_test30_half1 = torch.load('/data_yww/rgb_split1_test30_half1.t7')
--sum_tv_rgb_split1_half1 = {}
sum_tv_rgb_split1_test30_half1 = {}
		table.insert(sum_tv_rgb_split1_test30_half1, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
   -- table.insert(sum_tv_rgb_split1_half2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
sum_tv_rgb_split1_test30_half1.featMats = tv_split1_test30_half1.featMats+rgb_split1_test30_half1.featMats
--if tv_split1_test30.labels:equal(rgb_split1_test30.labels) then

--sum_tv_rgb_split1_test30.labels = tv_split1_test30.labels or rgb_split1_test30.labels

print(sum_tv_rgb_split1_test30_half1.featMats:size())
torch.save('sum_tv_rgb_split1_test30_half1.t7',sum_tv_rgb_split1_test30_half1)
--end

]]



tv_split1_test30_half2 = torch.load('/data_yww/feat_collection/tv_split1_test30_half2.t7')
rgb_split1_test30_half2 = torch.load('/data_yww/rgb_split1_test30_half2.t7')
--sum_tv_rgb_split1_half1 = {}
sum_tv_rgb_split1_test30_half2 = {}
		table.insert(sum_tv_rgb_split1_test30_half2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
   -- table.insert(sum_tv_rgb_split1_half2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
sum_tv_rgb_split1_test30_half2.featMats = tv_split1_test30_half2.featMats+rgb_split1_test30_half2.featMats
--if tv_split1_test30.labels:equal(rgb_split1_test30.labels) then

--sum_tv_rgb_split1_test30.labels = tv_split1_test30.labels or rgb_split1_test30.labels

print(sum_tv_rgb_split1_test30_half2.featMats:size())
torch.save('sum_tv_rgb_split1_test30_half2.t7',sum_tv_rgb_split1_test30_half2)