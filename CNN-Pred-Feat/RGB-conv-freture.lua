

-- Load all the videos & predict the video labels

-- load the trained Spatial ConvNet and Temporal-stream ConvNet




require 'xlua'
require 'torch'
require 'image'
require 'nn'
require 'cudnn' 
require 'cunn'
require 'cutorch'
t = require './transforms'

local videoDecoder = assert(require("libvideo_decoder")) -- package 3

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-sP', '--sourcePath', action='store', dest='sourcePath',
          help='source path (local | workstation)', default='workstation'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='HMDB-51'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of optical stream (FlowMap-TVL1-crop20 | FlowMap-Brox)', default='FlowMap-TVL1-crop20'}
op:option{'-iSp', '--idSplit', action='store', dest='idSplit',
          help='index of the split set', default=1}

op:option{'-iP', '--idPart', action='store', dest='idPart',
          help='index of the divided part', default=1}
op:option{'-nP', '--numPart', action='store', dest='numPart',
          help='number of parts to divide', default=2}
op:option{'-mD', '--manualDivide', action='store', dest='manualDivide',
          help='manually set the range', default=true}
op:option{'-iS', '--idStart', action='store', dest='idStart',
          help='manually set the starting class', default=1}
op:option{'-iE', '--idEnd', action='store', dest='idEnd',
          help='manually set the ending class', default=51}

op:option{'-mC', '--methodCrop', action='store', dest='methodCrop',
          help='cropping method (tenCrop | centerCrop)', default='centerCrop'}
op:option{'-mP', '--methodPred', action='store', dest='methodPred',
          help='prediction method (scoreMean | classVoting)', default='scoreMean'}

op:option{'-f', '--frame', action='store', dest='frame',
          help='frame length for each video', default=25}
-- op:option{'-fpsTr', '--fpsTr', action='store', dest='fpsTr',
--           help='fps of the trained model', default=25}
-- op:option{'-fpsTe', '--fpsTe', action='store', dest='fpsTe',
--           help='fps for testing', default=25}
op:option{'-sA', '--sampleAll', action='store', dest='sampleAll',
          help='use all the frames or not', default=false}

op:option{'-p', '--type', action='store', dest='type',
          help='option for CPU/GPU', default='cuda'}
op:option{'-tH', '--threads', action='store', dest='threads',
          help='number of threads', default=2}
op:option{'-i1', '--devid1', action='store', dest='devid1',
          help='1st GPU', default=1}      
op:option{'-i2', '--devid2', action='store', dest='devid2',
          help='2nd GPU', default=2}      
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=true}

op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=2}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}

numStream = 2 -- stream number of the input data
opt,args = op:parse()
-- convert strings to numbers --
idPart = tonumber(opt.idPart)
numPart = tonumber(opt.numPart)
idSplit = tonumber(opt.idSplit)
idStart = tonumber(opt.idStart)
idEnd = tonumber(opt.idEnd)
frame = tonumber(opt.frame)
-- fpsTr = tonumber(opt.fpsTr)
-- fpsTe = tonumber(opt.fpsTe)
devid1 = tonumber(opt.devid1)
devid2 = tonumber(opt.devid2)
threads = tonumber(opt.threads)

nameDatabase = opt.nameDatabase
methodOF = opt.stream 

print('split #: '..idSplit)
print('source path: '..opt.sourcePath)
print('Database: '..opt.nameDatabase)
print('Stream: '..opt.stream)

-- print('fps for training: '..fpsTr)
-- print('fps for testing: '..fpsTe)

print('frame length per video: '..frame)
print('Data part '..idPart)

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
source = opt.sourcePath -- local | workstation
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/yww/Activity-Reco/'
end

if nameDatabase == 'UCF-101' then
	dirDatabase = 'Models-UCF101/'
elseif nameDatabase == 'HMDB-51' then	
	 dirDatabase = 'Models-HMDB51/'
end

DIR = {}
dataFolder = {}
--pathDatabase = dirSource..'dataset/'..nameDatabase..'/'
pathDatabase = dirSource..nameDatabase..'/'--pathDatabase = /home/yww/Activity-Reco/HMDB-51/
---- Temporal ----
table.insert(DIR, {dirModel = dirSource..'HMDB-51/'..dirDatabase..'Models-TwoStreamConvNets/ResNet-'..methodOF..'-sgd-sp'..idSplit..'/', 
		pathVideoIn = '/home/yww/Activity-Reco/HMDB-51/hmdb51_org-tv_75percent/'})--methodOF = FlowMap-TVL1-crop20

---- Spatial ----
table.insert(DIR, {dirModel = dirSource..'HMDB-51/'..dirDatabase..'Models-TwoStreamConvNets/ResNet-RGB-sgd-sp'..idSplit..'/', 
	pathVideoIn = '/data_yww/hmdb51_org_split1_test/'})  --pathVideoIn = /home/yww/Activity-Reco/HMDB-51/RGB
--table.insert(DIR, {dirModel = dirSource..'HMDB-51/'..dirDatabase..'Models-TwoStreamConvNets/ResNet-RGB-sgd-sp'..idSplit..'/', 
 -- pathVideoIn = '/data_yww/hmdb51_org_split1_train_half1/'})  --pathVideoIn = /home/yww/Activity-Reco/HMDB-51/RGB
pathTxtSplit = pathDatabase..'testTrainMulti_7030_splits/' -- for HMDB-51

for nS=2,numStream do
	table.insert(dataFolder, paths.basename(DIR[nS].pathVideoIn))
end

----------------------------------------------
--  		       CPU option	 	        --
----------------------------------------------
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
-- numClass = 101
dimFeat = 2048
numTopN = 3

numStack = torch.Tensor(numStream)
nChannel = torch.Tensor(numStream)

-- Temporal
numStack[1] = 10
nChannel[1] = 2
-- Spatial
numStack[2] = 1
nChannel[2] = 3

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
-- will combine to 'parse args' later
numFrameSample = frame
numSplit = 3
softMax = false
nCrops = (opt.methodCrop == 'tenCrop') and 10 or 1
print('')
print('method for video prediction: ' .. opt.methodPred)
if softMax then
	print('Using SoftMax layer')
end
print('Using '..opt.methodCrop)

nameOutFile = 'acc_'..nameDatabase..'_'..numFrameSample..'Frames'..'-'..opt.methodCrop..'-sp'..idSplit..'_part'..idPart..'.txt' -- output the video accuracy

------------------------------------------
--  	Train/Test split (UCF-101)		--
------------------------------------------
groupSplit = {}
for sp=1,numSplit do
	if sp==1 then
		table.insert(groupSplit, {setTr = torch.Tensor({{8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}}), 
			setTe = torch.Tensor({{1,2,3,4,5,6,7}})})
	elseif sp==2 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,15,16,17,18,19,20,21,22,23,24,25}}), 
			setTe = torch.Tensor({{8,9,10,11,12,13,14}})})
	elseif sp==3 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,23,24,25}}), 
			setTe = torch.Tensor({{15,16,17,18,19,20,21}})})
	end
end

-- Output information --
namePredTr = 'data_pred_train_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'yww'..'.t7'
namePredTe = 'data_pred_test_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'yww'..'.t7'
--namePredTr = 'data_pred_train_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7'
--namePredTe = 'data_pred_test_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7'


nameFeatTr = {}
nameFeatTe = {}
---- Temporal ----
table.insert(nameFeatTr, 'convlstm_split1_tv-feat'..'test30.t7')
table.insert(nameFeatTe, 'convlstm_split1_tv-feat'..'test30.t7')
---- Spatial ----
table.insert(nameFeatTr, 'convlstm_split1_rgb-feat'..'test30.t7')
table.insert(nameFeatTe, 'convlstm_split1_rgb-feat'..'test30.t7')




------ model selection ------
-- ResNet model (from Torch) ==> need cudnn
modelNameTemporl = 'model_best-tv.t7'
modelNameSpatial = 'model_best_rgb.t7'
modelPath = {}
table.insert(modelPath, '/data_yww/'..modelNameTemporl)
table.insert(modelPath, '/data_yww/'..modelNameSpatial)

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
meanstd = {}
-- Temporal

            function table.merge( tSrc1, tSrc2 )
    tDest ={}
    for k, v in pairs( tSrc1 ) do
       
        table.insert(tDest,v)
    end
    for k, v in pairs( tSrc2 ) do
       
        table.insert(tDest,v)
    end
    return tDest
end
-- Spatial
if true then
	-- if fpsTr == 10 then
	-- 	table.insert(meanstd, {mean = { 0.392, 0.376, 0.348 },
	--    				std = { 0.241, 0.234, 0.231 }})
	-- elseif fpsTr == 25 then
	-- 	table.insert(meanstd, {mean = { 0.39234371606738, 0.37576219443075, 0.34801909196893 },
	--                std = { 0.24149100687454, 0.23453123289779, 0.23117322727131 }})
	-- else
	if nameDatabase == 'UCF-101' then
		table.insert(meanstd, {mean = {0.39743499656438, 0.38846055375943, 0.35173909269078},
								std = {0.24145608138375, 0.23480329347676, 0.2306657093885}})
	elseif nameDatabase == 'HMDB-51' then
		table.insert(meanstd, {mean = {0.36410178082273, 0.36032826208483, 0.31140866484224},
  								std = {0.20658244577568, 0.20174469333003, 0.19790770088352}})
	end
	-- end
else
    error('no mean and std defined for spatial network... ')
end

Crop = (opt.methodCrop == 'tenCrop') and t.TenCrop or t.CenterCrop

----------------------------------------------
-- 					Class		        	--
----------------------------------------------
nameClass = paths.dir(DIR[2].pathVideoIn) --pathVideoIn = /home/yww/Activity-Reco/HMDB-51/RGB/
table.sort(nameClass)
table.remove(nameClass,1) -- remove "."
table.remove(nameClass,1) -- remove ".."
numClass = #nameClass -- 101 classes 

---- divide the whole dataset into several parts ----
if not opt.manualDivide then
	numClassSub = torch.floor(numClass/numPart)
	rangeClassPart = {}
	numClassAcm = torch.zeros(numPart)
	Acm = 0
	for i=1,numPart do
		if i==numPart then
			table.insert(rangeClassPart, torch.range((i-1)*numClassSub+1,numClass))
		else
			table.insert(rangeClassPart, torch.range((i-1)*numClassSub+1,i*numClassSub))
		end
		
		Acm = Acm + rangeClassPart[i]:nElement()
		numClassAcm[i] = Acm
	end
end

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
devID = torch.Tensor(numStream)
devID[1] = devid1 -- for temporal
devID[2] = devid2 -- for spatial
net = {}


cutorch.setDevice(devID[1])
print '==> Loading the spatial model...'
netTemp1 = torch.load(modelPath[2])
	if softMax then
		softMaxLayer = cudnn.SoftMax():cuda()
		netTemp1:add(softMaxLayer)
	end
netTemp1:evaluate() -- Evaluate mode

table.insert(net, netTemp1)
--for nS=1,numStream do
	--- choose GPU ---
	--cutorch.setDevice(devID[nS])
 -- cutorch.setDevice(devID[nS])
--	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
	--print(sys.COLORS.white ..  ' ')

--	print ' '
	--if nS == 1 then
--		print '==> Loading the temporal model...'
--	elseif nS == 2 then
--		print '==> Loading the spatial model...'
--	end
	--local netTemp = torch.load(modelPath[nS]):cuda() -- Torch model
	
	------ model modification ------	
	--if softMax then
	--	softMaxLayer = cudnn.SoftMax():cuda()
	--	netTemp:add(softMaxLayer)
--	end

--	netTemp:evaluate() -- Evaluate mode

--	table.insert(net, netTemp)

	-- print(netTemp)
  --print(netTemp)
	print ' '


----------------------------------------------
-- 			Loading UCF-101 labels	  		--
----------------------------------------------
-- imagenetLabel = require './imagenet'
ucf101Label = require './ucf-101'
table.sort(ucf101Label)

print ' '

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
--fd = io.open(nameOutFile,'w')
--fd:write('S(frame) S(video) T(frame) T(video) S+T(frame) S+T(video) \n')

print '==> Processing all the videos...'



  sp = idSplit

  existTe = false
	-- Testing data --
	Te = {} -- output prediction & info
        featTe = {}
	if not existTe then
	--	Te = {} -- output prediction & info
		Te.countFrame = 0
		Te.countVideo = 0
		if opt.manualDivide then
			Te.countClass = idStart - 1
		else
			Te.countClass = rangeClassPart[idPart][1]-1
		end

		Te.accFrameClass = {}
		Te.accFrameAll = 0
		Te.accVideoClass = {}
		Te.accVideoAll = 0
		Te.c_finished = 0 

		Te.hitTestFrameAll = 0
		Te.hitTestVideoAll = 0

		--==== Prediction (Spatial & Temporal) ====--
		--== Temporal
		Te.accFrameClassT = {}
		Te.accFrameAllT = 0
		Te.accVideoClassT = {}
		Te.accVideoAllT = 0
		Te.hitTestFrameAllT = 0
		Te.hitTestVideoAllT = 0
		--== Spatial
		Te.accFrameClassS = {}
		Te.accFrameAllS = 0
		Te.accVideoClassS = {}
		Te.accVideoAllS = 0
		Te.hitTestFrameAllS = 0
		Te.hitTestVideoAllS = 0

	else
		Te = torch.load(namePredTe) -- output prediction
		featTe[1] = torch.load(nameFeatTe[1]) -- output temporal features
		featTe[2] = torch.load(nameFeatTe[2]) -- output spatial features
	end
	collectgarbage()

	timerAll = torch.Timer() -- count the whole processing time

	--if Tr.countClass == numClass and Te.countClass == numClass then
	if Te.countClass == numClass then
		print('The feature data of split '..sp..' is already in your folder!!!!!!')
	else
		local classStart, classEnd
		if opt.manualDivide then
      classStart = idStart   --if you  want to continue to catanate the featMats ,change the number of idStart
    -- classStart = 36
			classEnd = idEnd
		else
			classStart = Te.countClass + 1
			classEnd = numClassAcm[idPart]
		end
    
		for c=classStart, classEnd do
print(classEnd)
				local numTestFrameClass = 0
				local numTestVideoClass = 0

				local hitTestFrameClassB = 0
				local hitTestVideoClassB = 0
      		featTe = {}
    prefeatTe={}
    currentfeatTe={}
		--== Temporal
	table.insert(featTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    
    table.insert(prefeatTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
    table.insert(currentfeatTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
   
    if classStart > 1 or c>1 then
        
      prefeatTe[1] = torch.load(nameFeatTe[2])
      print('prefeatTe size is:')
      print(prefeatTe[1].featMats:size())
      Te.countVideo = #prefeatTe[1].name
    end
				--==== Separate Spatial & Temporal ====--
				--local hitTestFrameClassT = 0
				--local hitTestVideoClassT = 0
				local hitTestFrameClassS = 0
				local hitTestVideoClassS = 0
						
				print('Current Class: '..c..'. '..nameClass[c])

				--Tr.countClass = Tr.countClass + 1
				Te.countClass = Te.countClass + 1
			  	
			  	------ Data paths ------
			  	local pathClassIn = {}--eg:.../RGB/brush_hair/
			  	local nameSubVideo = {}-- avi files' nameTxt table of brush_hair

			  	for nS=2,numStream do
			  		pathClassInTemp = DIR[nS].pathVideoIn..nameClass[c]..'/'
            print(pathClassInTemp..'55555555555')
			  		table.insert(pathClassIn, pathClassInTemp)

			  		local nameSubVideoTemp = {}
				  	if nameDatabase == 'UCF-101' then
					  	nameSubVideoTemp = paths.dir(pathClassInTemp)
						table.sort(nameSubVideo) -- ascending order
						table.remove(nameSubVideoTemp,1) -- remove "."
						table.remove(nameSubVideoTemp,1) -- remove ".."
            elseif nameDatabase == 'HMDB-51' then
	
            nameSubVideoTemp = paths.dir(pathClassInTemp)
            table.sort(nameSubVideoTemp)
            for key, val in pairs(nameSubVideoTemp) do  -- Table iteration.
              print(key, val)
              if val=='.' or val =='..' then
              table.remove(nameSubVideoTemp,key)
               end
            end
             for key1, val1 in pairs(nameSubVideoTemp) do  -- Table iteration.
          
              if val1=='.' then
              table.remove(nameSubVideoTemp,key1)
              end
            end
             for key2, val2 in pairs(nameSubVideoTemp) do  -- Table iteration.
          
              if val2=='..' then
              table.remove(nameSubVideoTemp,key2)
              end
            end
            print('nameSubVideoTemp after remove is ')
            for key, val in pairs(nameSubVideoTemp) do  -- Table iteration.
              print(key, val)
            end

					end
			  		table.insert(nameSubVideo, nameSubVideoTemp)
			  	end

			  	local numSubVideoTotal = #nameSubVideo[1] -- videos 


			  	local timerClass = torch.Timer() -- count the processing time for one class
			  	
			  	for sv=1, numSubVideoTotal do
			      	--------------------
			      	-- Load the video --
			      	--------------------  
			   		local videoName = {}
			   		local videoPath = {}
			   		local videoPathLocal = {}
			   		--local videoIndex -- for HMDB-51

			      	for nS=2,numStream do
			      		local videoNameTemp
			      		local videoPathTemp
			       		if nameDatabase == 'UCF-101' then
							videoNameTemp = paths.basename(nameSubVideo[nS][sv],'avi')
							videoPathTemp = pathClassIn[nS]..videoNameTemp..'.avi'
						elseif nameDatabase == 'HMDB-51' then  --nameSubVideo is a table contains all videos in one class floder 
               ----yww modifiy---
               print ('nameSubVideo'..nameSubVideo[1][sv])
               videoNameTemp=paths.basename(nameSubVideo[1][sv])

							if nS == 2 then
						       	videoPathTemp = pathClassIn[nS-1]..videoNameTemp
                    print('videoPathTemp is : '..videoPathTemp)
						    else
						    	videoPathTemp = pathClassIn[nS]..videoNameTemp..'.avi'
						    end
						end
                print('videoNameTemp is : '..videoNameTemp)
		        		table.insert(videoName, videoNameTemp)
                
		        		table.insert(videoPath, videoPathTemp)
		        		table.insert(videoPathLocal, nameClass[c] .. '/' .. videoNameTemp)
					end

					----------------------------------------------
			       	--          Train/Test feature split        --
			       	----------------------------------------------
			        -- find out whether the video is in training set or testing set
					
					local flagTrain, flagTest = false
					if nameDatabase == 'UCF-101' then
						local i,j = string.find(videoName[1],'_g') -- find the location of the group info in the string
					    local videoGroup = tonumber(string.sub(videoName[1],j+1,j+2)) -- get the group#
					    
					    if groupSplit[idSplit].setTe:eq(videoGroup):sum() == 0 then -- training data
					    	flagTrain = true
					    	flagTest = false
					    else -- testing data
					    	flagTrain = false
					    	flagTest = true
					    end
					elseif nameDatabase == 'HMDB-51' then

						flagTrain = false  -- all video are  test data,no more  training data
						flagTest = true

					end
	
					--=========================================--
			        --            Process the video            --
					--=========================================--
			        --if not flagTrain and flagTest then -- testing data 
				    --== Read the video ==--
					local vidTensor = {}

					for nS=2,numStream do
						--print(videoName[nS])
						---- read the video to a tensor ----
						print(videoPath[1])
						local status, height, width, length, fps = videoDecoder.init(videoPath[1])
						local vidTensorTemp
		    			local inFrame = torch.ByteTensor(3, height, width)
		    			local countFrame = 0
						while true do
							status = videoDecoder.frame_rgb(inFrame)
							if not status then
					   			break
							end
							countFrame = countFrame + 1
					   		local frameTensor = torch.reshape(inFrame, 1, 3, height, width):double():div(255)
										
					   		if countFrame == 1 then -- the first frame
						        vidTensorTemp = frameTensor
						    else -- from the second or the following videos
						        vidTensorTemp = torch.cat(vidTensorTemp,frameTensor,1)	  --number of frame of video x3xheightxwidth     	
						    end			      
									
						end
						videoDecoder.exit()
								--	print(vidTensorTemp)	--to this is same
						table.insert(vidTensor, vidTensorTemp) -- read the whole video & turn it into a 4D tensor (e.g. 150x3x240x320)
					end
					        	
					local numFrame = vidTensor[1]:size(1) -- same frame # for two streams
					local height =  vidTensor[1]:size(3)
					local width = vidTensor[1]:size(4)
					        	
					------ Video prarmeters (same for two streams) ------				        	
				    local numFrameAvailable = numFrame - numStack[2] + 1 -- for 10-stacking   100f -10 +1 =91 91frames is available !!!!!!!!!!!!!!!! ihave changed this line differ from on no flodeer
				    local numFrameInterval = opt.sampleAll and 1 or torch.floor(numFrameAvailable/numFrameSample)    --numFrameInterval = 91/25 4
				    numFrameInterval = torch.Tensor({{1,numFrameInterval}}):max() -- make sure larger than 0
				    local numFrameUsed = opt.sampleAll and numFrameAvailable or numFrameSample -- choose frame # for one video  choose 25 frames 
					          	
				    ------ Initialization of the prediction ------
				    local predFramesB = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    local scoreFramesB = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101

				    --==== Separate Spatial & Temporal ====--
				    --local predFramesT = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    local predFramesS = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    --local scoreFramesT = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101
				    local scoreFramesS = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101

				    --==== Outputs ====--
				    local scoreFrameTS = torch.Tensor(1,numFrameUsed,numClass):zero() -- 2x25x101
				    --local featFrameTS = torch.Tensor(1,dimFeat,numFrameUsed):zero() -- 2x2048x25
					local featFrameTS = torch.Tensor(1,dimFeat,7,7,numFrameUsed):zero() -- 2x2048x25
				    ----------------------------------------------------------
				    -- Forward pass the model (get features and prediction) --
				    ----------------------------------------------------------
				    -- print '==> Begin predicting......'
            
				    for nS=2,numStream do
				      	cutorch.setDevice(devID[1])
					   	--- transform ---
					   	transform = t.Compose{t.Scale(256), t.ColorNormalize(meanstd[1], nChannel[nS]), Crop(224)}

					   	local I_all -- concatenate all sampled frames for a video
						    
					    for i=1, numFrameUsed do
				       		local f = (i-1)*numFrameInterval+5 -- current frame sample (middle in 10-stacking)
				       		f = torch.Tensor({{f,numFrame-5}}):min() -- make sure we can extract the corresponding 10-stack optcial flow
				        	
					   		-- extract the input
					   		-- Temporal:	2-channel, 10-stacking
					   		-- Spatial:		3-channel, none-stacking
					   		local inFrames = vidTensor[1][{{torch.floor(f-numStack[nS]/2)+1,torch.floor(f+numStack[nS]/2)},
					              		{3-(nChannel[nS]-1),3},{},{}}]

					   		-- change the dimension for the input to "transform" 
					   		-- Temporal:	20x240x320
					   		-- Spatial:		3x240x320					              		
					   		local netInput = torch.Tensor(inFrames:size(1)*nChannel[nS],height,width):zero()
					   		for x=0,numStack[nS]-1 do
					   			netInput[{{x*nChannel[nS]+1,(x+1)*nChannel[nS]}}] = inFrames[{{x+1},{},{},{}}]
					   		end

					  		local I = transform(netInput) -- e.g. 20x224x224 (temp)or 10x20x224x224 (tenCrop

							I = I:view(1, table.unpack(I:size():totable())) -- 20x224x224 --> 1x20x224x224
    

							-- concatenation
							if i==1 then
								I_all = I
							else
								I_all = torch.cat(I_all,I,1)--1x20x224x224-->25x20x224x224
							end

					    end
  
					    --====== prediction ======--
					    local scoreFrame_now = torch.Tensor(numFrameUsed,numClass):zero() -- 25x101
				        if (opt.methodCrop == 'tenCrop') then  --methodCrop default is centerCrop
					    	local outputTen = net[1]:forward(I_all:cuda()) -- 25x10x101
				           	scoreFrame_now = torch.mean(outputTen,1) -- 25x101
				        else
				          	--I = I:view(1, table.unpack(I:size():totable())) -- 25x20x224x224
				           	local output = net[1]:forward(I_all:cuda()) -- 25x101--
                    
          
				           	scoreFrame_now = output
				        end


				        --====== feature ======--
				        --local feat_now = net[1].modules[10].output:float() -- 25x2048
						local feat_now = net[1].modules[8].output:float() -- 25x7x7x2048
					      --feat_now = feat_now:transpose(1,2)
						feat_now = feat_now:permute(1,2,3,0)
				        local featFrame_now = feat_now:reshape(1, table.unpack(feat_now:size():totable())):double() -- 1x7x7x256x25
						-- print(featFrame_now:size())
				        featFrameTS[1] = featFrame_now --retrieve a feat of an video !!!!!!
					
				    end
					
				    ------------------------------------------------------------------
				    -- Training:	features										--
				    -- Testing:		features + frame prediction + video prediction 	--
				    ------------------------------------------------------------------
				    if flagTrain and not flagTest then -- training data
				    	--Tr.countVideo = Tr.countVideo + 1

				    	--==== Feature (Spatial & Temporal) ====--
						for nS=2,numStream do
						--	featTr[nS].name[Tr.countVideo] = videoName[nS]
						--	featTr[nS].path[Tr.countVideo] = videoPathLocal[nS]
						--	if Tr.countVideo == 1 then -- the first video
				       -- 		featTr[nS].featMats = featFrameTS[{{nS},{},{}}]
						--   		featTr[nS].labels = torch.DoubleTensor(nCrops):fill(Tr.countClass)
					   --     else 					-- from the second or the following videos
					    --    	featTr[nS].featMats = torch.cat(featTr[nS].featMats,featFrameTS[{{nS},{},{}}],1)
					   --     	featTr[nS].labels = torch.cat(featTr[nS].labels,torch.DoubleTensor(nCrops):fill(Tr.countClass),1)
					   --     end
						end

					elseif not flagTrain and flagTest then -- testing data	
						Te.countVideo = Te.countVideo + 1

						--==== Feature (Spatial & Temporal) ====--
						for nS=2,numStream do
							featTe[1].name[Te.countVideo-#prefeatTe[1].name] = videoName[nS]
							featTe[1].path[Te.countVideo-#prefeatTe[1].name] = videoPathLocal[nS]
							if (Te.countVideo-#prefeatTe[1].name) == 1 then -- the first video
				        		featTe[1].featMats = featFrameTS[{{1},{},{},{},{}}]
 
				        		featTe[1].labels = torch.DoubleTensor(nCrops):fill(Te.countClass)
					        else 					-- from the second or the following videos
					        	featTe[1].featMats = torch.cat(featTe[1].featMats,featFrameTS[{{1},{},{},{},{}}],1)
                   
     
					        	featTe[1].labels = torch.cat(featTe[1].labels,torch.DoubleTensor(nCrops):fill(Te.countClass),1)--nCrops =1
					        
          end

						-- scores --> prediction --
						Te.countFrame = Te.countFrame + numFrameUsed
						numTestFrameClass = numTestFrameClass + numFrameUsed
						
						--==== Baseline ====--
						scoreFramesB = torch.mean(scoreFrameTS,1):squeeze(1) -- 101 probabilities of the frame (baseline)  scoreFrameTS 25x101 
             --torch.mean (x,1)  returns a Tensor y of the mean of the elements in each column of x.   1x101 -->squeeze(1)  101 
						local probLogB, predLabelsB = scoreFramesB:topk(numTopN, true, true) -- 5 (probabilities + labels)        probLogB--zuidade gailv,predLabelsB--dijige
						local predFramesB = predLabelsB[{{},1}] -- predicted label of the frame
				        	local hitTestFrameB = predFramesB:eq(c):sum() 
				        	hitTestFrameClassB = hitTestFrameClassB + hitTestFrameB -- accumulate the score for frame prediction
				        
						--==== Separate Spatial & Temporal ====--
						--== Temporal
				--		scoreFramesT = scoreFrameTS[1] 
				--		local probLogT, predLabelsT = scoreFramesT:topk(numTopN, true, true) -- 5 (probabilities + labels)        
				--		local predFramesT = predLabelsT[{{},1}] -- predicted label of the frame
				--	        local hitTestFrameT = predFramesT:eq(c):sum()
				--		hitTestFrameClassT = hitTestFrameClassT + hitTestFrameT -- accumulate the score for frame prediction
					    
						--== Spatial
						scoreFramesS = scoreFrameTS[1] 
						local probLogS, predLabelsS = scoreFramesS:topk(numTopN, true, true) -- 5 (probabilities + labels)        
						local predFramesS = predLabelsS[{{},1}] -- predicted label of the frame
						local hitTestFrameS = predFramesS:eq(c):sum()
						hitTestFrameClassS = hitTestFrameClassS + hitTestFrameS -- accumulate the score for frame prediction
					

						----------------------
				        -- Video Prediction --
				        ----------------------
					    local predVideoB

					    --==== Separate Spatial & Temporal ====--
					--    local predVideoT
						local predVideoS

            if opt.methodPred == 'classVoting' then 
					       	local predVideoTensor = torch.mode(predFramesB)
                  predVideoB = predVideoTensor[1]
						elseif opt.methodPred == 'scoreMean' then--default
                  local scoreMean = torch.mean(scoreFramesB,1) -- 1x101
                  local probLogB, predLabelsB = scoreMean:topk(numTopN, true, true) -- 5 (probabilities + labels)
                  predVideoB = predLabelsB[{{},1}]


							
							--== Spatial
                  local scoreMeanS = torch.mean(scoreFramesS,1)
                  local probLogS, predLabelsS = scoreMeanS:topk(numTopN, true, true) -- 5 (probabilities + labels)  numTopN=3
                  predVideoS = predLabelsS[{{},1}]
						end
					    
				        -- accumulate the score for video prediction
				        numTestVideoClass = numTestVideoClass + 1
				        local hitTestVideoB = predVideoB:eq(c):sum()
                  hitTestVideoClassB = hitTestVideoClassB + hitTestVideoB -- baseline
						--print(predVideoT)
						--print(predVideoT:eq(c))
						--print(predVideoT:eq(c):sum())
						--error(test)
				--	local hitTestVideoT = predVideoT:eq(c):sum() 
				--	hitTestVideoClassT = hitTestVideoClassT + hitTestVideoT -- temporal
				        local hitTestVideoS = predVideoS:eq(c):sum()
                hitTestVideoClassS = hitTestVideoClassS + hitTestVideoS -- spatial
        				print(videoName[1],hitTestFrameB,hitTestFrameT,hitTestFrameS,hitTestVideoB,hitTestVideoT,hitTestVideoS)
				    end				  
            end
			      	collectgarbage()
			    end
				Te.c_finished = c -- save the index
 				
	          	----------------------------------------------
	          	--       Print the prediction results       --
	          	----------------------------------------------
					Te.hitTestFrameAll = Te.hitTestFrameAll + hitTestFrameClassB
					local acc_frame_class_ST = hitTestFrameClassB/numTestFrameClass
					acc_frame_all_ST = Te.hitTestFrameAll/Te.countFrame
					print('Class frame accuracy: '..acc_frame_class_ST)
					print('Accumulated frame accuracy: '..acc_frame_all_ST)
					Te.accFrameClass[Te.countClass] = acc_frame_class_ST
					Te.accFrameAll = acc_frame_all_ST

					-- video prediction
					Te.hitTestVideoAll = Te.hitTestVideoAll + hitTestVideoClassB
					local acc_video_class_ST = hitTestVideoClassB/numTestVideoClass
					acc_video_all_ST = Te.hitTestVideoAll/Te.countVideo
					print('Class video accuracy: '..acc_video_class_ST)
					print('Accumulated video accuracy: '..acc_video_all_ST)
					Te.accVideoClass[Te.countClass] = acc_video_class_ST
					Te.accVideoAll = acc_video_all_ST

					--==== Separate Spatial & Temporal ====--


					--== Spatial
					Te.hitTestFrameAllS = Te.hitTestFrameAllS + hitTestFrameClassS
					local acc_frame_class_S = hitTestFrameClassS/numTestFrameClass
					acc_frame_all_S = Te.hitTestFrameAllS/Te.countFrame
					print('Class frame accuracy (Spatial): '..acc_frame_class_S)
					print('Accumulated frame accuracy (Spatial): '..acc_frame_all_S)
					Te.accFrameClassS[Te.countClass] = acc_frame_class_S
					Te.accFrameAllS = acc_frame_all_S

					-- video prediction
					Te.hitTestVideoAllS = Te.hitTestVideoAllS + hitTestVideoClassS
					local acc_video_class_S = hitTestVideoClassS/numTestVideoClass
					acc_video_all_S = Te.hitTestVideoAllS/Te.countVideo
					print('Class video accuracy (Spatial): '..acc_video_class_S)
					print('Accumulated video accuracy (Spatial): '..acc_video_all_S)
					Te.accVideoClassS[Te.countClass] = acc_video_class_S
					Te.accVideoAllS = acc_video_all_S


					--fd:write(acc_frame_class_S, ' ', acc_video_class_S, ' ', acc_frame_class_T, ' ', acc_video_class_T, ' ', acc_frame_class_ST, ' ', acc_video_class_ST, '\n')



			  	print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			  	
			  	if opt.save then
				--	torch.save(namePredTr, Tr)
			  --		torch.save(namePredTe, Te)

			  		for nS=2,numStream do
					print('---------------------------->')
					print('cat the prefeatTe and featTe to currentfeatTe ')
				if classStart > 1 or c>1 then
            
                    currentfeatTe[1].name = table.merge(prefeatTe[1].name,featTe[1].name)
                    currentfeatTe[1].featMats = torch.cat(prefeatTe[1].featMats,featTe[1].featMats,1)
					        	currentfeatTe[1].labels = torch.cat(prefeatTe[1].labels,featTe[1].labels,1)--nCrops =1
                    print('currentfeatTe size is ')
                    print(currentfeatTe[1].featMats:size())
                    print(currentfeatTe[1].labels:size())
              
				elseif classStart==1 and c==1 then 
					currentfeatTe[1].name= featTe[1].name
					currentfeatTe[1].featMats = featTe[1].featMats
					currentfeatTe[1].labels = featTe[1].labels--nCrops =1
            end
				  		--torch.save(nameFeatTr[nS], featTr[nS])
              torch.save(nameFeatTe[nS], currentfeatTe[1])
              featTe[1] =nil
              currentfeatTe[1] =nil
              prefeatTe[1] =nil
              collectgarbage()
		end
	end

	end
  end
  print('The total elapsed time in the split '..sp..': ' .. timerAll:time().real .. ' seconds')

	-- Final Outputs --
	print('Total frame numbers: '..Te.countFrame)
	print('Total frame accuracy for the whole dataset: '..Te.accFrameAll)
	--print('Total frame accuracy for the whole dataset (Temporal): '..Te.accFrameAllT)
	print('Total frame accuracy for the whole dataset (Spatial): '..Te.accFrameAllS)
	print('Total video numbers: '..Te.countVideo)
	print('Total video accuracy for the whole dataset: '..Te.accVideoAll)
	--print('Total video accuracy for the whole dataset (Temporal): '..Te.accVideoAllT)
	print('Total video accuracy for the whole dataset (Spatial): '..Te.accVideoAllS)
	
	--fd:write(acc_frame_all_S, ' ', acc_video_all_S, ' ', acc_frame_all_T, ' ', acc_video_all_T, ' ', acc_frame_all_ST, ' ', acc_video_all_ST, '\n')
		


	
	print ' '

	Tr = nil
	Te = nil
	--featTr = nil
	featTe = nil
	collectgarbage()
-- end


	

