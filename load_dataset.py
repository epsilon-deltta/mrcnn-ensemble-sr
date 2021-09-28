import os
# download the Penn-Fudan dataset
cmd1 = "!wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip" #axel doesn't work 

cmd2 = "!unzip -q PennFudanPed.zip"
cmdsays = os.popen(cmd1).read()
print(cmdsays)
cmdsays = os.popen(cmd2).read()
print(cmdsays)