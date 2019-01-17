average_losss=0
f=io.open("train_loss_log.txt",'a')
for i=1,10 do
  average_losss= average_losss+2.33
  f:write(i..' '..average_losss..'\r')
end
print(average_losss/15000)