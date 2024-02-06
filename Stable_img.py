from diffusers import LCMScheduler, AutoPipelineForText2Image
import torch
import time

def main():

	#* Initialize the model 
	model_id = "stabilityai/stable-diffusion-xl-base-1.0"
	adapter_id = "latent-consistency/lcm-lora-sdxl"

	pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
	pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

	#pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
	pipe.to("cuda")
	pipe.enable_model_cpu_offload()

	generator = torch.Generator("cuda").manual_seed(1024)

	# if using torch < 2.0, to run faster
	pipe.enable_xformers_memory_efficient_attention()

	#* Load and fuse lcm lora
	pipe.load_lora_weights(adapter_id)
	pipe.fuse_lora()

	#* %%%% LOAD PROMPTS %%%%
	#! Cambiar archivo de prompts
	#prompt_filename = "Bot_prompts.txt"
	prompt_filename = "Bot_prompts2.txt"

	sentences = []
	counter = 0

	#Open the prompt file in read mode
	with open(prompt_filename,'r') as file:
		for line in file:
	#		sentence = line.strip()	#Remove leading/trailing whitespace
	#		sentences.append(sentence)
			if line.strip():
				sentence = line
				sentences.append(sentence)

	#* Open the negative_prompt file in read mode
	negative_prompt_filename = "Negative_prompts.txt"
	with open(negative_prompt_filename, 'r') as file:
		#negative_prompt = file.read().strip()
		lines = file.readlines()

	negative_prompt = ' '.join(map(str.strip, lines))
	print("Negative prompt: " , negative_prompt)

	#get the start time
	st = time.time()

	for sentence in sentences:
		prompt = sentence
		counter += 1
		num_inference_step= 6
		
		for i in range(20):
			image = pipe(prompt,negative_prompt= negative_prompt, guidance_scale= 0, num_inference_steps=num_inference_step, generator=generator, height=960, width=1280).images[0]
			image.save("Stable_images/" + str(counter) + "_N" + str(num_inference_step) + "_" + str(i) + ".png")
			num_inference_step += 2 #Step to 
		
	#get the end time 
	et = time.time()

	#measure the execution time 
	ex_time = et - st
	print('Execution time:', ex_time, 'seconds')

	### Succesfull negative prompt ###
	#"worst quality,low quality, food, disordered, disproportioned, incomplete, assymetrical, error, fail, overlap, over, on top of, extra, another"
	######################################

if __name__=="__main__":
    main()