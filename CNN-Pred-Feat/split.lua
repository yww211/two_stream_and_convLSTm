   require 'torch'
   --rgb_split1_half2 = torch.load('/data_yww/feat_collection/convlstm_split1_half2_rgb-feattwo_video.t7')
   --tv_split1_half2 = torch.load('/data_yww/convlstm_split1_half2_tv-feat.t7')
   rgb_split1_test30 = torch.load('/data_yww/convlstm_split1_rgb-feattest30.t7')
   --tv_split1_test30 = torch.load('/data_yww/convlstm_split1_tv-feattest30.t7')

   print(rgb_split1_test30.featMats:size())
  
   --tv_split1_half2_half1  = torch.load()
   featTe_tv_half1={}
   featTe_tv_half2={}

   featTe_rgb_half1={}
   featTe_rgb_half2={}

   rgb_split1_test30_half1 = torch.DoubleTensor()
   rgb_split1_test30_half2 = torch.DoubleTensor()
   tv_split1_test30_half1 = torch.DoubleTensor()
   tv_split1_test30_half2 = torch.DoubleTensor()
		--== Temporal
		table.insert(featTe_tv_half1, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    table.insert(featTe_tv_half2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    table.insert(featTe_rgb_half1, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    table.insert(featTe_rgb_half2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
   --[[
   tv_split1_test30_half1 = tv_split1_test30.featMats:narrow(1,1,tv_split1_test30.featMats:size()[1]/2)
    print(tv_split1_test30_half1:size())
    ddd = tv_split1_test30.featMats:size()[1]-tv_split1_test30_half1:size()[1]
    --print(ddd)
    featTe_tv_half1.featMats = tv_split1_test30_half1:clone()
    
    tv_split1_test30_half2 = tv_split1_test30.featMats:narrow(1,tv_split1_test30.featMats:size()[1]/2+1,ddd)
    
    print(tv_split1_test30_half2:size())
    featTe_tv_half2.featMats = tv_split1_test30_half2:clone()
    torch.save('tv_split1_test30_half1.t7',featTe_tv_half1)
    torch.save('tv_split1_test30_half2.t7',featTe_tv_half2)
]]
    rgb_split1_test30_half1 = rgb_split1_test30.featMats:narrow(1,1,rgb_split1_test30.featMats:size()[1]/2)
    print(rgb_split1_test30_half1:size())
    ddd = rgb_split1_test30.featMats:size()[1]-rgb_split1_test30_half1:size()[1]
    --print(ddd)
    featTe_rgb_half1.featMats = rgb_split1_test30_half1:clone()
    
    rgb_split1_test30_half2 = rgb_split1_test30.featMats:narrow(1,rgb_split1_test30.featMats:size()[1]/2+1,ddd)
    
    print(rgb_split1_test30_half2:size())
    featTe_rgb_half2.featMats = rgb_split1_test30_half2:clone()
    torch.save('rgb_split1_test30_half1.t7',featTe_rgb_half1)
    torch.save('rgb_split1_test30_half2.t7',featTe_rgb_half2)
    --featTe.featMats = temp:clone() 