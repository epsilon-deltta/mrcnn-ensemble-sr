import os
# download the Penn-Fudan dataset
cmd1 = "wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip" #axel doesn't work 

cmd2 = "unzip -q PennFudanPed.zip"

if not os.path.exists('./PennFudanPed.zip'):
    cmdsays = os.popen(cmd1).read()
    print(cmdsays)

dirname = './PennFudanPed'
if os.path.exists(dirname)
    try:
        nfiles = len(os.listdir(dirname))
        if nfiles > 3:
            os.rmdir(dirname)
    except:
        os.rmdir(dirname)

cmdsays = os.popen(cmd2).read()
print(cmdsays)