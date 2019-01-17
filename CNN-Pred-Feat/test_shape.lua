    
    require 'torch'
   --[[ featTe1 = torch.load('/data_yww/feat_collection/tv+rgb_split1_half1_half1.t7')
    featTe2 = torch.load('/data_yww/feat_collection/tv+rgb_split1_half1_half2.t7')
    temp = torch.DoubleTensor()
    --temp = torch.cat(featTe1.featMats,featTe2.featMats,1)
    
    --featTe1 = torch.load('/data_yww/convlstm_75_tv-feattwo_video.t7')
    featTe={}
   -- featTe2={}
    --temp = torch.DoubleTensor()
		--== Temporal
		table.insert(featTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    	--	table.insert(featTe2, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    featTe.featMats = torch.cat(featTe1.featMats,featTe2.featMats,1)
    featTe1 =nil
    featTe2 =nil
    -- temp = featTe1.featMats:narrow(1,1,featTe1.featMats:size()[1]/2)
    
    --featTe.featMats = temp:clone()  --use clone to reduce the memeroy size of the tensor after narrow because the narrow dont shrink the storage size 
    
    torch.save('tv+rgb_split1_half1.t7',featTe)
    print(featTe.featMats:size())
   featTe =nil
   featTe1=nil
   featTe2 =nil
   temp =nil]]
   --featTe1 = torch.load('convlstm_split1_half2_tv-feat.t7')
-- featTe1 = torch.load('/data_yww/convlstm_split1_tv-feattest30.t7')
 featTe2 = torch.load('tv_rgb_split1_test30.t7')
 print(featTe2.featMats:size())
  --[[ testpermute = torch.IntTensor(7,7,100,25)
   testpermute = testpermute:permute(1,2,3,0)
   print(testpermute:size())]]
  --featTe2 = torch.load('/data_yww/tv+rgb_split1_half1.t7')
  --[[label = torch.IntTensor(featTe2.featMats:size()[1])
    for i = 1, 51 do 
      for j =1,30 do
          label[30*(i-1)+j]=i
          end
      end
     featTe2.labels = label]]
 print(featTe2.labels)
 --print(featTe2.featMats[1][1][1])
--torch.save('tv_rgb_split1_test30.t7',featTe2)