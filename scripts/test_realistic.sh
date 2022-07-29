set -ex
python test.py --dataroot ./datasets/deblur_syth --name deblur_syth_B2A_pix2pix_mutli_realistic_BGAN_DBGAN --model realistic --direction BtoA --netG unet_256 --norm none --dataset_mode aligned