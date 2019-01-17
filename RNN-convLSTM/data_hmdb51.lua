----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN

--  This is a testing code for trained models

----------------------------------------------------------------
require 'torch'
require 'sys'

if opt.dataset == 'ucf101' then
    -- classes in UCF-101
    classes = {
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam", "BandMarching",
    "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress", "Biking", "Billiards",
    "BlowDryHair", "BlowingCandles", "BodyWeightSquats", "Bowling", "BoxingPunchingBag",
    "BoxingSpeedBag", "BreastStroke", "BrushingTeeth", "CleanAndJerk", "CliffDiving",
    "CricketBowling", "CricketShot", "CuttingInKitchen", "Diving", "Drumming", "Fencing",
    "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch", "FrontCrawl", "GolfSwing",
    "Haircut", "HammerThrow", "Hammering", "HandstandPushups", "HandstandWalking",
    "HeadMassage", "HighJump", "HorseRace", "HorseRiding", "HulaHoop", "IceDancing",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks", "ParallelBars",
    "PizzaTossing", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar",
    "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse",
    "PullUps", "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing", "Rowing",
    "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding", "Skiing", "Skijet", "SkyDiving",
    "SoccerJuggling", "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "Swing",
    "TableTennisShot", "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard", "YoYo"
    }
elseif opt.dataset == 'hmdb51' then
    -- classes in HMDB51
    --classes = {
   -- "brush_hair", "kick_ball", "ride_horse", "pour", "jump", "smile", "stand", "shake_hands", "flic_flac", "golf", "wave", "cartwheel", "clap", "dive", "ride_bike", "turn", "chew", "draw_sword", "push", "hug", "shoot_gun", "pullup", "sit", "smoke", "somersault", "shoot_bow", "kick", "kiss", "shoot_ball", "run", "walk", "situp", "sword", "drink", "pushup", "fall_floor", "climb", "hit", "laugh", "eat", "pick", "swing_baseball", "dribble", "talk", "climb_stairs", "catch", "fencing", "punch", "throw", "sword_exercise", "handstand"
  --}
      classes = {
    "brush_hair","cartwheel","catch","chew","clap","climb", "climb_stairs","dive","draw_sword","dribble","drink", "eat","fall_floor","fencing","flic_flac","golf", "handstand","hit", "hug","jump","kick","kick_ball","kiss","laugh", "pick", "pour", "pullup", "punch","push","pushup", "ride_bike",  "ride_horse", "run", "shake_hands",   "shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke", "somersault", "stand","swing_baseball", "sword","sword_exercise", "talk","throw",  "turn", "walk",   "wave"
    }
else
    error('Unknown dataset: ' .. opt.dataset)
end
nClass = #classes -- UCF101 has 101 categories

------------------------------------------------------------
-- Only use a certain number of frames from each video
------------------------------------------------------------
function extractFrames(inputData, rho)
   print(sys.COLORS.green ..  '==> Extracting only ' .. rho .. ' frames per video')

   if inputData:size(3) == rho then 
      return inputData
   end

   local timeStep = inputData:size(3) / rho
   local dataOutput = torch.Tensor(inputData:size(1), inputData:size(2), rho)

   local idx = 1
   for j = 1,inputData:size(3),timeStep do
      dataOutput[{{},{},idx}] = inputData[{{},{},j}]
      idx = idx + 1
   end
   return dataOutput
end

------------------------------------------------------------
-- Only use a certain number of consecutive frames from each video
------------------------------------------------------------
function extractConsecutiveFrames(inputData, rho)
   print(sys.COLORS.green ..  '==> Extracting random ' .. rho .. ' consecutive frames per video')
--  Consecutive periods of time or events happen one after the other without interruption.
   local dataOutput = torch.Tensor(inputData:size(1), inputData:size(2), rho)
   local nProb = inputData:size(3) - rho
   local ind_start = torch.Tensor(1):random(1,nProb)
   print('rho is :'..rho)
   print('nProb is :'..nProb)
   print('ind_start is :'..ind_start)

   local index = torch.range(ind_start[1], ind_start[1]+rho-1)
   local indLong = torch.LongTensor():resize(index:size()):copy(index)

   -- extracting data according to the Index
   local dataOutput = inputData:index(3,indLong)

   return dataOutput
end

------------------------------------------------------------
-- n-fold cross-validation function
-- this is only use a certain amount of data for training, and the rest of data for testing
------------------------------------------------------------
function crossValidation(dataset, target, nFolds)
   print(sys.COLORS.green ..  '==> Train on ' .. (1-1/nFolds)*100 .. '% of data ..')
   print(sys.COLORS.green ..  '==> Test on ' .. 100/nFolds .. '% of data ..')
   -- shuffle the dataset
   local shuffle = torch.randperm(dataset:size(1))
   local index = torch.ceil(dataset:size(1)/nFolds)
   -- extract test data
   local testIndices = shuffle:sub(1,index)
   local test_ind = torch.LongTensor():resize(testIndices:size()):copy(testIndices)
   local testData = dataset:index(1,test_ind)
   local testTarget = target:index(1,test_ind)
   -- extract train data
   local trainIndices = shuffle:sub(index+1,dataset:size(1))
   local train_ind = torch.LongTensor():resize(trainIndices:size()):copy(trainIndices)
   local trainData = dataset:index(1,train_ind)
   local trainTarget = target:index(1,train_ind)

   return trainData, trainTarget, testData, testTarget
end

print(sys.COLORS.green .. '==> Reading UCF101 external feature vector and target file ...')

----------------------------------------------
--  feature matrix from CNN model
----------------------------------------------
local TrainData, TestData, TrainTarget, TestTarget
local testName
if  true then

   local feat_train_name = '/data_yww/tv+rgb_split1_half1.t7'
   local feat_test_name = '/data_yww/tv_rgb_split1_half2.t7'

   assert((paths.filep(feat_train_name)), 'no  training feature file found.')
   assert((paths.filep(feat_test_name)), 'no  testing feature file found.')

   
    local feat_train = torch.load(feat_train_name)
    local feat_test = torch.load(feat_test_name)

   TrainData = feat_train.featMats:permute(1,5,2,3,4)
   TestData = feat_test.featMats:permute(1,5,2,3,4)


   print('TrainData size is :')
   print(TrainData:size())
   print('TestData size is :')
   print(TestData:size())

   TrainTarget = feat_train.labels
   TestTarget = feat_test.labels


end



if true then
   -- training and testing data from UCF101 website
   --test-tv name
   local tv_name = 'convlstm__tv-feattwo_video.t7'
  -- local tv_name = 'orderadjust-data_feat_test_two_video.t7'
   assert(paths.filep(paths.concat(opt.tempFeatDir, tv_name)), 'no temporal training feature file found.')
   local tempTrainFeatureLabels = torch.load(paths.concat(opt.tempFeatDir, tv_name))
   tempTrainData = tempTrainFeatureLabels.featMats:permute(1,5,2,3,4)
    print('tempTrainData de size is :')
   print(tempTrainData:size())
  -- testName = tempTrainFeatureLabels.name

   -- check if there are enough frames to extract and extract
   --if tempTrainData:size(3) >= opt.rho then
      -- extract #rho of frames
   --   tempTrainData = extractFrames(tempTrainData, opt.rho)  
  -- else
   --   error('total number of frames lower than the extracting frames')
  -- end

   tempTrainTarget = tempTrainFeatureLabels.labels

   assert(paths.filep(paths.concat(opt.tempFeatDir, tv_name)), 'no temporal testing feature file found.')
   local tempTestFeatureLabels = torch.load(paths.concat(opt.tempFeatDir, tv_name))
   tempTestData = tempTestFeatureLabels.featMats:permute(1,5,2,3,4)
   print('tempTestData de size is :')
   print(tempTestData:size())
  -- if opt.averagePred == false then 
  ---    if tempTestData:size(3) >= opt.rho then
         -- extract #rho of frames
    --     tempTestData = extractFrames(tempTestData, opt.rho)  
   --   else
     --    error('total number of frames lower than the extracting frames')
  --    end
  -- end
   tempTestTarget = tempTestFeatureLabels.labels

end]]

local trainData, testData


collectgarbage()

return 
{
   trainData = TrainData,
   trainTarget = TrainTarget,
   testData = TestData, 
  testTarget = TestTarget,
  -- testName = testName
}
