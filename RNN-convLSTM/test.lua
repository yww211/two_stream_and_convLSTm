----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
--
--  This is a testing code for implementing the RNN model with LSTM
--  written by Chih-Yao Ma.
--
--  The code will take feature vectors (from CNN model) from contiguous
--  frames and train against the ground truth, i.e. the labeling of video classes.
--
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------
local nn = require 'nn'
local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
local optim = require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model-for-convLSTM'
local model = m.model
local criterion = m.criterion

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(51)

 classes = {
    "brush_hair","cartwheel","catch","chew","clap","climb", "climb_stairs","dive","draw_sword","dribble","drink", "eat","fall_floor","fencing","flic_flac","golf", "handstand","hit", "hug","jump","kick","kick_ball","kiss","laugh", "pick", "pour", "pullup", "punch","push","pushup", "ride_bike",  "ride_horse", "run", "shake_hands",   "shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke", "somersault", "stand","swing_baseball", "sword","sword_exercise", "talk","throw",  "turn", "walk",   "wave"
    }
     
-- Logger:
--local testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

-- Batch test:
local inputs = torch.Tensor(opt.rho,2048,7,7 )
local targets = torch.Tensor(opt.batchSize)
local name_test 
local labels = {}
local prob = {}

-- if opt.averagePred == true then
-- predsFrames = torch.Tensor(opt.batchSize, nClass, opt.rho-1)
predsFrames = torch.Tensor(opt.batchSize, #classes, opt.rho - opt.numSegment + 1)
-- end

local logsoftmax = nn.LogSoftMax()

if opt.cuda == true then
	inputs = inputs:cuda()
	targets = targets:cuda()
	logsoftmax = logsoftmax:cuda()
end



-- test function
function test(testData,testTarget)
  local file = io.open("../yww211.txt","a")
 classes = {
    "brush_hair","cartwheel","catch","chew","clap","climb", "climb_stairs","dive","draw_sword","dribble","drink", "eat","fall_floor","fencing","flic_flac","golf", "handstand","hit", "hug","jump","kick","kick_ball","kiss","laugh", "pick", "pour", "pullup", "punch","push","pushup", "ride_bike",  "ride_horse", "run", "shake_hands",   "shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke", "somersault", "stand","swing_baseball", "sword","sword_exercise", "talk","throw",  "turn", "walk",   "wave"
    }
	-- local vars
	local time = sys.clock()
	local timer = torch.Timer()
	local dataTimer = torch.Timer()

	if opt.cuda == true then
		model:cuda()
	end

	-- Sets Dropout layer to have a different behaviour during evaluation.
	model:evaluate()
  model:forget()
	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')

	if opt.testOnly then
		epoch = 1
	end

	--local top1Sum, top3Sum, lossSum = 0.0, 0.0, 0.0
	local N = 0
	for t = 1,testData:size(1),1 do
    if t ~=33 then
     
    
		local dataTime = dataTimer:time().real
		-- disp progress
		xlua.progress(t, testData:size(1))

		-- create mini batch
		local idx = 1
		inputs:fill(0)
		local targets
   
		for i = t,t+1-1 do
			if i <= testData:size(1) then
				inputs = testData[i]--:float()
			  targets = testTarget[i]
       -- name_test[idx] =testName[i]
				idx = idx + 1
			end
		end
		local idxBound = idx - 1

		--local top1, top3
		if opt.averagePred == true then --false
			-- make prediction for each of the images frames, start from frame #2
			idx = 1
			for i = 2, opt.rho do
				-- extract various length of frames
				local Index = torch.range(1, i)
				local indLong = torch.LongTensor():resize(Index:size()):copy(Index)
				local inputsPreFrames = inputs:index(3, indLong)
        
				predsFrames[{{},{},idx}] = model:forward(inputsPreFrames)

				idx = idx + 1
			end

			-- Convert log probabilities back to [0, 1]
			predsFrames:exp()
			-- average all the prediction across all frames
			preds = torch.mean(predsFrames, 3):squeeze()
		else

			local inputsSegments = {}
		--	local segmentBasis = math.floor(inputs:size(3)/opt.numSegment)
		--	for s = 1, opt.numSegment do
		--		 table.insert(inputsSegments, inputs[{{}, {}, {segmentBasis*(s-1) + 1,segmentBasis*s}}])
	--		end
      if inputs:size(1)==25 then
        
          for i =1,25 do  --push every Tx into a table 
          --print(inputS[i]:size()) 
            table.insert(inputsSegments,inputs[i]:cuda())
          end
      
       else  
         error('输入不是25个时间步')
       end
			preds = model:updateOutput(inputsSegments)
      --f = criterion:updateOutput(preds,targets)
			preds = logsoftmax:forward(preds):exp()
   
		end

		-- discard the redundant predictions and targets
		if (t + 1 - 1) > testData:size(1) then
			preds = preds:sub(1,idxBound)
		end
   -- print('targets:sub(1,idxBound) is :')
    --print(targets:sub(1,idxBound))
    local res = 0
		res = computeScore(preds)
   -- print('file name  is :'..testName[t])
    print ('duiyingde class is :')
    local batchSize = preds:size(1)
    print(targets)
   -- print(targets:long():view(batchSize, 1):expandAs(preds):narrow(2, 1, 1):narrow(1,1,1):int())
    print('my prediction is :'..classes[res[1]])

    inputsSegments =nil
    collectgarbage()
		end

	


	if opt.cuda == true then
		model:cuda()
	end

	-- timing
	time = sys.clock() - time
	time = time / testData:size(1)
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

	timer:reset()
	dataTimer:reset()

  	
	if opt.plot then
		testLogger:style{['% mean class accuracy (test set)'] = '-'}
		testLogger:plot()
	end
	--confusion:zero()
end
end

function computeScore(output)

   -- Coputes the top1 and top3 error rate
   local batchSize = output:size(1)
   local result 
  -- print ('batchSize is :')
 -- print (batchSize)
    -- print ('output is :')
 -- print (output)
   local _ , predictions = output:float():sort(1, true) -- descending  down 
   indices_of_predictions = torch.LongTensor({0,1})
   --print(predictions)
   result= predictions:narrow(1,1,1):int()
   print('result is ')
     print(result)


  return result
end

-- Export:
return test
