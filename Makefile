.PHONY: image/gan image/lsgan/0 image/lsgan/1 resource/image image/stylegan
image/gan:
	mkdir -p $@
	python3 gan.py --image_dir /home/greatofdream/jittor/resource/color_symbol_7k	 --opt_dir $@
image/lsgan/0:
	mkdir -p $@
	python3 train.py --image_dir /home/greatofdream/jittor/resource/color_symbol_7k	 --opt_dir $@ --model lsgan
image/lsgan/1:
	mkdir -p $@
	python3 train.py --image_dir /home/greatofdream/jittor/resource/color_symbol_7k	 --opt_dir $@ --model lsgan --model_dir image/lsgan/0
resource/image:
	mkdir -p /home/greatofdream/jittor/resource/color_symbol_sample
	python3 prepare_data.py --opt_dir /home/greatofdream/jittor/resource/color_symbol_sample /home/greatofdream/jittor/resource/color_symbol_7k/128
image/stylegan:
	mkdir -p $@
	mkdir -p $@/checkpoint
	mkdir -p $@/sample
	CUDA_VISIBLE_DEVICES="1" python3 trainstylegan.py --opt_dir $@ --model_dir $@ --image_dir /home/greatofdream/jittor/resource/color_symbol_sample --mixing