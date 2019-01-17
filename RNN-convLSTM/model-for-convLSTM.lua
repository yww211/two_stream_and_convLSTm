----------------------------------------------------------------
-- Georgia Tech 2016 Spring
-- Deep Learning for Perception
-- Final Project: LRCN model for Video Classification
--
-- 
-- This is a testing code for implementing the RNN model with LSTM 
-- written by Chih-Yao Ma. 
-- 
-- The code will take feature vectors (from CNN model) from contiguous 
-- frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

require 'nn'
require 'cudnn'
require 'rnn'
local rnn = require 'rnn'
local sys = require 'sys'
require 'UntiedConvLSTM'
require 'cunn'
require 'cutorch'
print(sys.COLORS.red ..  '==> construct RNN')
input_channel=2048
output_channel=300
Tx=25
kernel_size_in=3
kernel_size_cell=3
stride=1

function model_temporal_BN_max_LSTM_BN_FC()
    -- Video Classification model
    local model = nn.Sequential()

--(1x7x7x4096)

     local convlstm = nn.Sequential()
     convlstm:add(nn.UntiedConvLSTM(input_channel,output_channel, Tx, kernel_size_in,kernel_size_cell, stride))
     model:add(nn.Sequencer(convlstm))
      --(1xchannel_oputx7x7)
     
    
     model:add(nn.SelectTable(-1)) 
     model:add(nn.SpatialAveragePooling(7,7))
     
    -- local squeeze = nn.Sequential()
     model:add(nn.Squeeze())

   --  model:add(squeeze)
  
    
     --(channel,)
     
     --(1X1xchannel_output)


    
    -- Dropout layer
    if opt.dropout > 0 then 
      model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(output_channel, 51))
    model:add(nn.LogSoftMax())
    return model
end
-------------------------------------------------------
-------------------------------------------------------
-- 
--                        Main
-- 
-------------------------------------------------------
-------------------------------------------------------

if checkpoint then
  
   local modelPath = paths.concat(opt.resume,  opt.resumeFile .. '.t7')
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   model = torch.load(modelPath)
   print('zhixingle if------------------------------------')
else
    print('zhixingle else -----------------------------')
    -- construct model
    model = (model_temporal_BN_max_LSTM_BN_FC())
   
    
    if opt.uniform > 0 then
        for k,param in ipairs(model:parameters()) do
            param:uniform(-opt.uniform, opt.uniform)
        end
    end

    -- will recurse a single continuous sequence
    model:remember((opt.lstm or opt.gru) and 'both' or 'eval')
end

-- build criterion
local criterion = {}
if opt.seqCriterion == true then
    criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
else
    criterion = nn.ClassNLLCriterion()
    -- criterion = nn.CrossEntropyCriterion()
end


print(sys.COLORS.red ..  '==> here is the network:')
print(model)


if opt.cuda == true then
    model:cuda()
    criterion:cuda()

    -- Wrap the model with DataParallelTable, if using more than one GPU
    if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            local rnn = require 'rnn'
            -- require 'TemporalDropout'
            
            -- Set the CUDNN flags
            cudnn.fastest = true
            cudnn.benchmark = true
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
    end
end

-- Export:
return
{
    model = model,
    criterion = criterion
}