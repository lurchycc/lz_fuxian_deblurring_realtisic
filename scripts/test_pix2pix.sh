set -ex
python test.py --dataroot ./datasets/deblur_syth --name deblur_syth_A2B_pix2pix_1024 --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm none
