----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN

--  This is a testing code for trained models

----------------------------------------------------------------
local nn = require 'nn'
require 'cudnn'
local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
local optim = require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
--local m = require 'model-for-convLSTM'
--epoch_dir_name = '/home/yww/Activity-Reco/RNN-convLSTM/epoch/'
epoch_dir_name = '/data_yww/epoch/'

model_names = {}
epochs = paths.dir(epoch_dir_name)
table.sort(epochs)
for i,v in ipairs(epochs) do
  
  
  print(i,v)
  end

table.remove(epochs,1) -- remove "."
table.remove(epochs,1) -- remove ".."
print('after sort:')
for i,v in ipairs(epochs) do
  
  print(i,v)
  table.insert(model_names,epoch_dir_name..v)
  end

numEpochs = #epochs -- 101 classes 
print('totoal models is :'..numEpochs)


local inputs = torch.Tensor(opt.rho,2048,7,7 )
local targets = torch.Tensor(opt.batchSize)
local name_test 
local labels = {}
local prob = {}

 classes = {
    "brush_hair","cartwheel","catch","chew","clap","climb", "climb_stairs","dive","draw_sword","dribble","drink", "eat","fall_floor","fencing","flic_flac","golf", "handstand","hit", "hug","jump","kick","kick_ball","kiss","laugh", "pick", "pour", "pullup", "punch","push","pushup", "ride_bike",  "ride_horse", "run", "shake_hands",   "shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke", "somersault", "stand","swing_baseball", "sword","sword_exercise", "talk","throw",  "turn", "walk",   "wave"
    }
-- This matrix records the current confusion across classes
--local confusion = optim.ConfusionMatrix(classes)




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
  
  for i,model_name in ipairs(model_names) do 
    correct_num_of_one_certrain_model =0
  assert(paths.filep(model_name), ' model not found: ' .. model_name)
   print('=> load model from ' .. model_name)
  local model = torch.load(model_name)
local criterion = {}
if opt.seqCriterion == true then
    criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
else
     criterion = nn.ClassNLLCriterion()
    -- criterion = nn.CrossEntropyCriterion()
end

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
   -- if t ~=33 then
     
    
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
    print ('ground truth class is :')
    local batchSize = preds:size(1)
    print(targets)
    
   -- print(targets:long():view(batchSize, 1):expandAs(preds):narrow(2, 1, 1):narrow(1,1,1):int())
    print('my prediction is :'..classes[res[1]])
    if res[1]==targets then
      correct_num_of_one_certrain_model=correct_num_of_one_certrain_model+1
    end
    inputsSegments =nil
    collectgarbage()
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

end
    acc = correct_num_of_one_certrain_model/testData:size(1)
    print('the acc on model'..paths.basename(model_name)..'is: '..acc)
    file = io.open("test_offical_acc_log.txt","a")
    file:write(paths.basename(model_name)..' '..acc..'\r')
    file:close()
   -- model =nil
    model = nn.utils.clear(model)
    criterion =nil
    collectgarbage()
        
end
end

function computeScore(output)
--compute  highest Score among all prediction class
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
