#!/bin/bash
# setup basics for VM
# type sudo chmod +x basics.sh at the command line to make the script executable
# type " sh basics.sh" at the command line to launch this script
# RTS, mfeb 2023
#-------------------------------------------
clear
echo "WELCOME - basics for the VM install "

sudo apt-get update
sudo apt-get upgrade -y


sudo apt-get install python3-dev -y
sudo apt-get install build-essential python3-dev -y
sudo apt-get install python3-venv -y


pip3 install --upgrade pip
pip3 install --upgrade pillow
pip3 install numpy
pip3 install matplotlib
pip3 install pandas
pip3 install scipy
pip3 install scikit-learn

echo "installed python3, python3dev, python3-venv, numpy, scipy, matplotlib, scikit"

echo "hit ctrl d to close this session"
exit 0
#end this part by closing the terminal and lanuching a new SSH session
