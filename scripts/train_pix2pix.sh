set -ex
python train.py --dataroot ./datasets/deblur_syth --name deblur_syth_B2A_pix2pix_multi_realistic_BGAN_noise --model pix2pix --direction BtoA --lambda_L1 100 --netG unet_256 --norm none --dataset_mode aligned  --pool_size 50
