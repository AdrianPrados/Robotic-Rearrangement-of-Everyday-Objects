from diffusers import DiffusionPipeline
import torch
import time

def main():

	#torch.cuda.empty_cache()

	pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
	pipe.to("cuda")
	pipe.enable_model_cpu_offload()

	#Define generator
	generator = torch.Generator("cuda").manual_seed(16)

	# if using torch < 2.0
	pipe.enable_xformers_memory_efficient_attention()

	#! Cambiar archivo de prompts
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

	#Open the prompt file in read mode
	negative_prompt_filename = "Negative_prompts.txt"
	with open(negative_prompt_filename, 'r') as file:
		#negative_prompt = file.read().strip()
		lines = file.readlines()

	negative_prompt = ' '.join(map(str.strip, lines))
	print("Negative prompt: " , negative_prompt)

	#images = pipe(prompt=prompt).images[0]

	#get the start time
	st = time.time()

	for sentence in sentences:
		prompt = sentence
		counter += 1
		guidance_scale= 8
		
		#negative_prompt= negative_prompt

		for i in range(20):
			image = pipe(prompt, guidance_scale= guidance_scale, generator=generator, height=960, width=1280).images[0]
			image.save("Stable_images/" + str(counter) + "_G" + str(guidance_scale) + "_" + str(i) + ".png")
			guidance_scale += 1 #Step to 
		#image.save("Prompt" + str(counter) + "b.png")
		
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