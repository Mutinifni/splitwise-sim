rsync -avz -e /bin/ssh sim2:/home/azureuser/splitwise-sim/results/ results/
ssh sim2 "rm -rf splitwise-sim/results/*"
rsync -avz -e /bin/ssh sim3:/home/azureuser/splitwise-sim/results/ results/
ssh sim3 "rm -rf splitwise-sim/results/*"

