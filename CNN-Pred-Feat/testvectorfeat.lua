require 'xlua'
require 'torch'

dir1='/home/yww/Activity-Reco/RNN/'
dir2='/home/yww/Activity-Reco/CNN-Pred-Feat/'
feat0 = torch.load(dir2..'orderadjust-data_feat_test_two_video.t7')
feat1 = torch.load(dir2..'orderadjust-data_feat_test_RGB_two_video.t7')
feat2 = torch.load(dir2..'test1-rgb_data_feat_test_RGB_yww.t7')
feat3 = torch.load(dir2..'test1-tv_data_feat_test_yww.t7')

testName0 = feat0.name[4535]
testName1 = feat1.name[4535]
testMat1 = feat1.featMats[4536]

testName2 = feat2.name[1]
testName3 = feat3.name[1]
testMat2 = feat2.featMats[1]
print('testName0 is ')
print(testName0)
print('testName1 is ')
print(testName1)
print('testName2 is ')
print(testName2)
print('testName3 is ')
print(testName3)
print(testMat1)
print(testMat2)