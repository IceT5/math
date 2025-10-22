#!/bin/bash
#
#cd ~ && git clone -b notebook https://gitcode.com/liulinxiang/ops-math.git math && bash ./math/init.sh
bash
cd /opt/huawei/edu-apaas/src/init
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/resource/gitcode/init_env.sh

bash init_env.sh

source ~/.bashrc

cd ${HOME}/workspace
source ${HOME}/Ascend/set_env.sh

cd /opt/huawei/edu-apaas/src/init
git clone https://gitcode.com/cann/ops-math.git
cd ops-math

pip3 install -r requirements.txt

chmod +x build.sh

./build.sh --pkg --soc=ascend910b --ops=abs

./build_out/cann-ops-math-custom_linux-aarch64.run

./build.sh --run_example abs eager cust --vendor_name=custom
echo "Prepare environment successfully"


echo "Pls execute command below:
bash
source ~/.bashrc"