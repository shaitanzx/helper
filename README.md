I would like to introduce a new fork of the popular generative neural network **Fooocus - Fooocus extend**. 
I would like to point out that this fork can be run both locally on your computer and via Google Colab. 
Let's look at everything in order. 

1.	**Launch**. If you will run it on a local machine, you can safely skip this item.
   
![image](https://github.com/user-attachments/assets/468487b8-8d4e-454c-ba92-1c9e5b60feb7)

Before launching you are offered to choose the following settings
Fooocus Profile - select which profile will be loaded at startup (default, anime, realistic).
Fooocus Theme - select a theme - light or dark.
Tunnel - select launch tunnel. When it happens that gradio stops working for some reason, you can choose cloudflared tunnel. However, the generation is a bit slower
Memory patch - adds a few keys to the launch bar that allow you to optimise your graphics card if you are using the free version of Google Colab. If you have paid access, this item can be disabled
GoogleDrive output - connects your GoogleDisk and save all of your generation directly to it.

2.	**Select the resolution and aspect ratio of the generated image**

![image](https://github.com/user-attachments/assets/ba5ce3d4-8f36-4f64-af82-760713c44c6a)

This setting is located in the generation resolution selection tab. Here you can select the number of horizontal and vertical points, aspect ratio. To apply the settings, click the Set button and select this resolution from the list of proposed resolutions. Your resolution will be the last one.

3.  **Wildcard**

![image](https://github.com/user-attachments/assets/45a4fc1f-72f6-479a-96ea-d61b6c62333e)

This module allows you not to memorise existing files with wildcard words, but to select them directly from a list of dictionaries. You can also select directly the item you need from the list.

4.	**Image Batch** (batch image processing)

![004](https://github.com/user-attachments/assets/6aa747e4-4b84-490f-b716-e1ba5b64be5b)



 In a nutshell, this module allows you to perform group upscaling of images, as well as create images based on a group of existing images using ImagePromt (ControlNet). To better understand this module, I advise you to conduct a few experiments yourself. But I want to note that its use allows you to use your images as references and change their style depending on the hint and the selected model. First, you need to create a zip archive with your images. The archive should not contain subfolders, the file names should not contain characters other than Latin. Upload the prepared archive to the "Upload zip file" window. Next, select the mode for changing the image resolution
- Not scale - the generation will not take into account the resolution of the original image
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the original image will be selected.
- to OUTPUT - in this case first the resolution of the original image will be changed to the generated one with preserving the proportions, and then the generation will start
  
Depending on what you want to do with your source images, select "Action" - Upscale or ImagePrompt. In the "Method" drop-down list, select the appropriate image processing method. In case of using ImagePrompt, you also need to select the "Stop at " and "Weight" parameters.

Clicking the Add to queue button will unpack the previously downloaded archive and put all jobs in the queue. The number of queued jobs will be indicated in brackets.

Start queue starts the queue for execution. 

Clear queue - clears the queue of existing jobs, but does not delete the last loaded archive.

When the queue is finished, click on Output->Zip to generate an archive with all previously generated images from the output folder. The archive itself will appear in the Download a Zip file window. From there you can actually download it.

Clear Output - clears the output folder. It should be noted that it clears not only the folder for the current date, but the whole folder.

If the execution proceeds without errors or interruptions, the queue will be cleared automatically, and the downloaded archive will remain in memory. Otherwise, the queue will not be cleared.

5.	**Prompt Batch** (batch processing of prompts)

![005](https://github.com/user-attachments/assets/983b2b96-9819-4c74-9c4f-44047435758b)


This module allows you to start generating several prompts in the queue for execution. To do this, you need to fill in the table In the prompt column enter a positive prompt, and in the negative prompt column enter a negative prompt respectively. Clicking on New row will add an empty row to the end of the table. Delete last row deletes the last row of the table. Start batch starts execution of the prompt queue for generation.
You can also choose to add base positive and negative prompts.
None - basic prompts will not be added.
Prefix - base prompts will be added before prompts from the table.
Suffix - base samples will be added after samples from the table.

Now let's look at the Extention panel. Here are some extensions for Stable Forge adapted for Fooocus

1.	**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions
   
![009](https://github.com/user-attachments/assets/84b0707d-2846-429e-bb79-8e4934506a24)

In the ‘Main’ tab, you can select the preset of the prompt generation theme, as well as specify unchanging prompt prefixes and suffixes.

![010](https://github.com/user-attachments/assets/2f5e788d-4be2-4bf9-8a7e-79e0a271078a)

In the ‘Workflow assist’ tab you can generate 5 prompts, transfer each of them to the Fooocus prompt field and also to the Workflow field. If you select Workflow mode, it will generate not a new prompt, but variations of the prompt specified in the Workflow text field. The level of variation is selected by the engine located just below it

![011](https://github.com/user-attachments/assets/a8045069-55b5-408c-ae59-83dc2ffe72bb)

In this tab you can select the prompt syntax for different generation models, the length of the prompt, and enable the prompt generation enhancer.

![012](https://github.com/user-attachments/assets/f3851d73-9a89-4913-8f4a-c1e425c40ce4)

This is where you can control the generation of negative prompts

![_013](https://github.com/user-attachments/assets/a56eed42-2e7e-4a57-af07-98572cea5b39)

In this tab you can start the image generation queue by generated samples. Before starting you need to specify the aspect ratio of the generated image (Size to generate), the number of generated prompts (Generation of prompts), and the models to use (Model to use).

2.	**Civitai Helper**

![014](https://github.com/user-attachments/assets/07529ca2-4347-44ce-9617-de67ded8eb68)


This extension allows you to download models for generation from the civitai website.  To download a model you first need to specify your Civitai_API_key. In the Download Model section in the Civitai URL field you need to specify a link to the required model from the browser address bar and click Get Model Info by Civitai URL. After analysing the link you will be given information about the model. You will also be able to select the version of the model before downloading. This extension also allows you to find duplicates of downloaded models and check for updates. In addition, there is a group download option.

3.	**Prompt Translate**

![006](https://github.com/user-attachments/assets/cb186d90-37fb-42ac-ac7c-3540a27785ee)

Allows to translate both positive and negative prompts from any language into English, both before generation and directly during generation.

4.	**Photopea** - a free online analogue of Photoshop

![007](https://github.com/user-attachments/assets/593af9cb-5904-49a6-aa47-0f29776b0608)

5.	**Remove Background**

![008](https://github.com/user-attachments/assets/ace804df-b4ac-48dc-93ae-771bb7a08e15)

This extension allows you to remove the background on an uploaded image with a single button

6.	**OpenPoseEditor**

![015](https://github.com/user-attachments/assets/d00bb18d-8b7e-46c3-bfcf-b77d7cf0bacb)

This module allows you to create skeletons for subsequent image creation using OpenPose ControlNet. You can also create a skeleton from an existing image.

7.	**OpenPose ControlNet**

![016](https://github.com/user-attachments/assets/68af63fb-ab4f-48cb-aa2b-989dc57166a4)

Allows you to create an image based on the pose skeleton.

8.	**Recolor ControlNet**

![017](https://github.com/user-attachments/assets/b763ee5a-fc01-4f68-948a-beaf0dc39c3a)

Allows you to colorize an image based on a black and white image.

9.	**Scribble ControlNet**

![18](https://github.com/user-attachments/assets/39b4d80f-591d-4301-bb5a-c3cc47fb5325)

Allows you to color an image based on a sketch.

10. **X/Y/Z Plot**

![19](https://github.com/user-attachments/assets/02f5f5b4-99d8-47f6-9983-0a578b7bf086)

This extension allows you to make image grids to make it easier to see the difference between different generation settings and choose the best option. You can change the following parameters - Styles, Steps, Aspect Ratio, Seed, Sharpness, CFG (Guidance) Scale, Checkpoint, Refiner, Clip skip, Sampler, Scheduler, VAE, Refiner swap method, Softness of ControlNet, and also replace words in the prompt and change their order

10. **Save Image Grid for Each Batch**

![20](https://github.com/user-attachments/assets/033a6a71-7e14-478a-a3d4-307f021fecec)

<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend without autoupdate. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

![image](https://github.com/user-attachments/assets/5cf86b6d-e378-4d85-aed1-c48920b6c107)


***Change log***
V8 (current local version)
1. Save Image Grid for Each Batch
2. Add X/Y/Z Plot Extention
3. Prompt Batch is now in the extensions panel
4. Images Batch has become easier to manage while retaining its functionality
5. Images Batch is now in the extensions panel

V7
1. Add OpenPoseEditor
2. Fix bug in Image Batch Mode
3. Added cell selection in Image Batch Mode
4. Added selection of adding base prompts in Prompt Batch Mode
5. Add OpenPose ControlNet
6. Add Recolor ControlNet
7. Add Scribble ControlNet

V6

1. Add Prompt Batch Mode
2. Rename Batch Mode to Images Batch Mode
3. Fixed an incorrect start random number in Batch Mode
4. Add visual management of Wildcard and Words/phrases of wildcard
5. Added the ability to set any resolution for the generated image
6. Add OneButtonPrompt

V5
1. Model Downloader replaced with Civitai Helper

V4
1. Add VAE download
2. Add Batch mode

V3
1. Add Photopea
2. Add Remove Background
3. Add Extention Panel
4. All extensions are available in Extention Panel

V2
1. Added a Model Downloader to Fooocus webui instead of colab

V1
1. added the ability to download models from the civitai.com
2. saving the generated image to Google Drive
3. added prompt translator
4. added a patch for the ability to work in free colab mode 
