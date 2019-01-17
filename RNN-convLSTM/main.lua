----------------------------------------------------------------
--  This part is for implementing the ConvLSTM 

--  The code will take  feature maps (from CNN model) from contiguous 
--  frames and train against the ground truth, i.e. the labeling of video classes. 
-- 

----------------------------------------------------------------

local sys = require 'sys'


require 'xlua'
op = xlua.OptionParser('%prog [options]')
op:option{'-dataset', '--dataset', action='store', dest='dataset',
          help='used database (ucf101 | hmdb51)', default='hmdb51'}
op:option{'-split', '--split', action='store', dest='split',
          help='index of the split set', default=1}
op:option{'-pastalogName', '--pastalogName', action='store', dest='pastalogName',
          help='the name of your experiment, e.g. pretrain-fullsize', default='model_RNN'}
op:option{'-learningRate', '--learningRate', action='store', dest='learningRate',
          help='learningRate', default=1e-4}
op:option{'-momentum', '--momentum', action='store', dest='momentum',
          help='momentum', default=0.9}
op:option{'-weightDecay', '--weightDecay', action='store', dest='weightDecay',
          help='weightDecay', default=0}
op:option{'-gradClip', '--gradClip', action='store', dest='gradClip',
          help='gradClip', default=5}
op:option{'-optimizer', '--optimizer', action='store', dest='optimizer',
          help='Use different optimizer, e.g. sgd, adam, adamax, rmsprop for now', default='adam'}
op:option{'-lrMethod', '--lrMethod', action='store', dest='lrMethod',
          help='lrMethod', default='fixed'}
        
        
op:option{'-epochUpdateLR', '--epochUpdateLR', action='store', dest='epochUpdateLR', help='learning rate decay per epochs', default=100}
op:option{'-lrDecayFactor', '--lrDecayFactor', action='store', dest='lrDecayFactor', help='learning rate decay per epochs', default=0.1}
op:option{'-batchSize', '--batchSize', action='store', dest='batchSize',help='batchSize', default=1}
op:option{'-cuda', '--cuda', action='store', dest='cuda',help='cuda', default=true}
op:option{'-nGPU', '--nGPU', action='store', dest='nGPU',help='nGPU', default=1}
op:option{'-useDevice', '--useDevice', action='store', dest='useDevice', help='useDevice', default=1}
op:option{'-maxEpoch', '--maxEpoch', action='store', dest='maxEpoch', help='maxEpoch', default=100}
        
op:option{'-maxTries', '--maxTries', action='store', dest='maxTries', help='maxTries', default=50}
op:option{'-progress', '--progress', action='store', dest='progress', help='print progress bar', default=true}
op:option{'-silent', '--silent', action='store', dest='silent', help='silent', default=false}
op:option{'-uniform', '--uniform', action='store', dest='uniform', help='uniform', default=0.1}
op:option{'-useDevice', '--useDevice', action='store', dest='useDevice', help='useDevice', default=1}
op:option{'-seqCriterion', '--seqCriterion', action='store', dest='seqCriterion', help='seqCriterion', default=false}



op:option{'-lstm', '--lstm', action='store', dest='lstm', help='use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)', default=false}
op:option{'-bn', '--bn', action='store', dest='bn', help='use batch normalization. Only supported with --lstm', default=false}
op:option{'-gru', '--gru', action='store', dest='gru', help='gru', default=false}
op:option{'-numSegment', '--numSegment', action='store', dest='numSegment', help='numSegment', default=3}
op:option{'-rho', '--rho', action='store', dest='rho', help='useDevice', default=25}
op:option{'-fcSize', '--fcSize', action='store', dest='fcSize', help='fcSize', default='{0}'}


op:option{'-hiddenSize', '--hiddenSize', action='store', dest='hiddenSize', help='use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)', default='{512}'}
op:option{'-zeroFirst', '--zeroFirst', action='store', dest='zeroFirst', help='zeroFirst', default=false}
op:option{'-dropout', '--dropout', action='store', dest='dropout', help='dropout', default=0.5}
op:option{'-averagePred', '--averagePred', action='store', dest='averagePred', help='averagePred', default=false}
op:option{'-testOnly', '--testOnly', action='store', dest='testOnly', help='testOnly', default=true}
op:option{'-inputSize', '--inpu tSize', action='store', dest='inputSize', help='Path to directory containing checkpoint', default=2048}
--op:option{'-resume', '--resume', action='store', dest='resume', help='Path to directory containing checkpoint', default='/home/yww/Activity-Reco/RNN-convLSTM/'}
op:option{'-resume', '--resume', action='store', dest='resume', help='Path to directory containing checkpoint', default='/data_yww/epoch_dropout/'}

op:option{'-resumeFile', '--resumeFile', action='store', dest='resumeFile', help='file for resuming training: latest_best | latest_current', default='convlstm_split1_dropput'}
--op:option{'-resumeFile', '--resumeFile', action='store', dest='resumeFile', help='file for resuming training: latest_best | latest_current', default='convLSTM_model_split1_half131_iteration'}
op:option{'-save', '--save', action='store', dest='save', help='file for resuming training: latest_best | latest_current', default=' '}
op:option{'-optimState', '--optimState', action='store', dest='optimState', help='file for resuming training: latest_best | latest_current', default='none'}

--cmd:text()
--opt = cmd:parse(arg or {})



-- create log file
--cmd:log(opt.save .. '/log.txt', opt)
opt,args = op:parse()
opt.pastalogName = opt.pastalogName .. 'split' .. opt.split .. '-' .. opt.fcSize .. opt.hiddenSize
opt.save = 'log' .. '_' ..  opt.pastalogName .. '_' .. opt.fcSize .. '_' .. opt.hiddenSize ..   '_' .. opt.learningRate .. '_' .. opt.weightDecay .. '_' .. opt.dropout  --pastalogName = model_RNN
opt.fcSize = loadstring(" return "..opt.fcSize)()
opt.hiddenSize = loadstring(" return "..opt.hiddenSize)()
opt.inputSize = 2048
paths.mkdir(opt.save)
-- type:
if opt.cuda == true then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	if opt.nGPU == 1 then
		cutorch.setDevice(opt.useDevice)
		print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
	end
end

-- check if rgb or flow features wanted to be used
--opt.spatial = paths.dirp(opt.spatFeatDir) and true or false
--opt.temporal = paths.dirp(opt.tempFeatDir) and true or false

if opt.dataset == 'ucf101' then
	opt.spatFeatDir = opt.spatFeatDir .. 'UCF-101/'
	opt.tempFeatDir = opt.tempFeatDir .. 'UCF-101/'
elseif opt.dataset == 'hmdb51' then
	opt.spatFeatDir = '/home/yww/Activity-Reco/CNN-Pred-Feat/'
	opt.tempFeatDir = '/home/yww/Activity-Reco/CNN-Pred-Feat/'
end

------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

-- checkpoints
checkpoints = require 'checkpoints'

-- Load previous checkpoint, if it exists
checkpoint, optimState = checkpoints.latest(opt)



--local train = require 'train'
local test  = require 'epoch_test'
--local test  = require 'test'

--local data  = require 'data_hmdb51'
local data  = require 'data_hmdb51_test30'

-- initialize bestAcc
bestAcc = 0

if opt.testOnly then
	-- Begin testing with trained model
	test(data.testData,data.testTarget)
	return
end

------------------------------------------------------------
-- Run
------------------------------------------------------------

print(sys.COLORS.red .. '==> training!')
for iteration = 1, opt.maxEpoch do
	-- Begin training process
	loss_of_half1 =train(data.trainData, data.trainTarget,iteration)

	if iteration%3==0  then

	average_loss_of_each_epoch = (loss_of_half2+loss_of_half1)/2
	f=io.open("train_loss_log_dorpout0_5.txt",'a')
	f:write(iteration..' '..average_loss_of_each_epoch..'\r')
	f:close()
	model:clearState()      

    torch.save('convLSTM_model_split1_half1_24hao_'..iteration..'_iteration.t7', model)
    print('model has been saved')
    
	--  print('iteration:'..iteration..'test begins')
	-- test(data.testData,data.testTarget)
    end
end
--test(data.testData,data.testTarget,data.testName)
--test(data.testData, data.testTarget)