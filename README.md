# Network Blending Interface
_Or: my interpid, but ultimately futile attempt to make GAN portraits of my friends_

### Code
**StyleGAN3 Network Blending UI**
- [Repo](https://github.com/adamdavidcole/stylegan3-fun-blend)
- [Notebook](https://github.com/adamdavidcole/stylegan3-fun-blend/blob/main/blend.ipynb)

  
###
**StyleGAN2 Network Blending UI**
- [Repo](https://github.com/adamdavidcole/stylegan2-ada-pytorch-adam) 
- [Notebook](https://github.com/adamdavidcole/stylegan2-ada-pytorch-adam/blob/main/network_blending_gui.ipynb)


### Video
[Notebook walkthrough](https://www.youtube.com/watch?v=fSDtTEVMMTo)


## Motivation
StyleGAN is well known for making people who don't exist. But while fake faces are cool, I'm more interested in creating images of real people. I began this project wondering how I could use StyleGAN to create expressive portraits of my friends.

This idea came about while I was first experimenting with StyleGAN3 on a set of butterfly images. I noticed that some of the early steps during the transfer learning from human faces to butterflys were quite interesting: they felt expressive and painterly, and I was excited by the possibility of applying similar transformations on the people in my life: freinds, family, etc..

![transfer learning grid](images/transfer_learning/transfer_learning_grid2.png)
*Sample of transfer learned images*

I did some research into how I could achieve this goal and discovered [Justin Pinkney's](https://www.justinpinkney.com/stylegan-network-blending/) network blending blogpost. At the same time, I was reading through Memo Akten's thesis on ["Realtime Continuous, Meaningful Human Control over Deep Neural Networks for Creative Expression"](https://research.gold.ac.uk/id/eprint/30191/) as well as Terence Broad's taxonomy on [Active Divergence](https://terencebroad.com/research/active-divergence-survey). Lastly, I was introduced to the idea of image inversion for StyleGAN and had the following plan to build meaningful control over network blending:
1. Build a user interface for fine tuned control over network blending
2. Project images of people from my life into the latent space
3. Thoughtfully blend between the real image representation and the butterfly representation for an authentic artistic expression 

In other words, my goal for this project was to use a combination of network projection and fine tuned control over network blending to create intentional, expressive portraits of real people in my life.

## Trial, But Mostly Error
### Attempt #1: StyleGAN3
_First signs of trouble_

To begin, I started with building the network bending UI workflow because I thought that would be the most challanging part of the project. The basic architectue involves a `blendModels()` function which takes a list of how much each layer in the model should be mixed (aka. a value of 0 means use only the weights from the first model, a value of 0.5 is an even mix, and a value of 1  means use the weights from the second model). I then hooked these values up to a set of sliders as a useful interface for the blend control.

![stylegan3 blend controls](images/user_interface/stylegan3_blen_ui_fine_tune.png)
*UI to control individual blend levers per layer*

The user could then select a source model and a destination model and experiment with various blend setups for a given seed. I included an EZ blending interface for simple functions like 50% interpolation or a half way crossover. Below the EZ blending controls are fine tuned controls over every layer. I find a useful workflow to be experimenting with the EZ controls to get a sense of what is possible with the models and then digging into the fine tune controls to realize a more intentional output.

![transfer learning grid](images/user_interface/../stylegan3_butterfly_blends/s3_blend_butterfly1.png)
![transfer learning grid](images/user_interface/../stylegan3_butterfly_blends/s3_blend_butterfly2.png)
*Example of fine tuned results you can get with the UI. In the first image, we are taking just the middle layers of from the butterfly model, but mainting the face structure and high level textures. In the second we are also using the butterfly model's high level textures.*

This system worked well...until I began experimenting with projections into the W+ space. I tried a couple algorithms, but had two major issues:
1. The projections were often unsatisfactory. This is known limitation of inversions, but is especially problematic when trying to create meaningful portraits
2. Using the projected W vector in the destination model resulted in really poor images. For example when working with seeds in the Z space in the case of the butterfly model, you'd always have a realistic human face on one side and a coherent butterfly on the other. But when using the projected W+ vector, you'd have a mostly coherent human face but a completely garbled butterfly on the other. Because we are essentially blending between these two image destinations, the results just weren’t as satisfying compared with the seeds.

![Example of source/destination pair in Z space](images/projections/seed_projection.png)
*Good: Source/destination pair in Z space. Both the face and butterfly are sharp.*

![Example of source/destination pair when projecting into W+ space](images/projections/w_projection.png)
*Bad: Source/destination pair when projecting into W+ space. The face is not a great match and the butterfly is a shapeless blob.*


At this point I felt confident that the UI control over the network blending was indeed useful, but needed a better solution for the projection problem. I was certain it was possible due to the results from Pinkney's [toonify yourself](https://www.justinpinkney.com/toonify-yourself/) project and was aware that the architectures of StyleGAN2 and 3 are [quite different](https://github.com/yuval-alaluf/restyle-encoder/issues/45#issuecomment-943503066). My plan was to rebuild this notebook for StyleGAN2 and hope for better results.


### Attempt #2: StyleGAN2
I rebuilt the network blending UI notebook for StyleGAN2, using [dvschultz's blending logic](https://github.com/dvschultz/stylegan2-ada-pytorch/blob/main/Network_Blending_ADA_PT.ipynb) as a starting off point. The structure was essentially the same, but the blend code needed to account for the different internal structure of StyleGAN2's layers.

![StyleGAN2 EZ blend controls](images/user_interface/stylegan_2_blend2.png)
*StyleGAN2 EZ blend controls*

**The good news:** 
- With fewer levers to control, interacting with the blending sliders in the StyleGAN2 model felt more meaningful and predictable. 

![StyleGAN2 EZ blend controls](images/stylegan2_butterfly_blends/butterfly_blend_sg2.png)
*StyleGAN2 buttefly blend with Z seed*

**The medium news:**
- Projections in the source and destination networks were indeed more reliable. But the results still were quite lackluster. The people in the projections were just slightly too off from the original to be useful as the starting point for a portrait and the corresponding image in the destination model were less interesting than when starting with a Z seed.

![StyleGAN2 EZ blend controls](images/stylegan2_butterfly_blends/projection_blend1.png)



**The worse news:**
- When fine tuning the FFHQ network on butterflies, the aesthetics of StyleGAN2 are just less expressive than StyleGAN3, so the blended results from these early checkpoints still didn’t match my goals.

![StyleGAN2 EZ blend controls](images/stylegan2_butterfly_blends/projection_blend2.png)
*StyleGAN2 buttefly blend with projections*

Once again, I was satisfied with the network blending aspect of the project, but not with the projection/portrait workflow. However, before moving on from this point, I decided to have some fun with this UI.

#### **Ukiyo-e Face**
To begin, I tested my setup with the Ukiyo-e face models provided by Pinkney to ensure my setup worked as expected. It was satisfying to get similar resutls and even more fun to start playing with levers more intetionally to fine tune results.

![ukiyo-e face forward blend](images/ukiyo_faces/ukiyo_forward.png)
![ukiyo-e face inverse blend](images/ukiyo_faces/ukiyo_inverse.png)
*Results with fine tuning blending on ukiyo-e face model blending*

#### **PokéPeople**
I recently trained a StyleGAN model on Pokémon for fun and dropped that model into this notebook. I then challenged myself to try to create some PokéPeople, with both the projections of friends and random seeds:


![PokePeople 2](images/pokemon/pokepeople/pokeperson2.png)
![PokePeople 3](images/pokemon/pokepeople/pokeperson3.png)
![PokePeople 4](images/pokemon/pokepeople/pokeperson4.png)
*Right image is produced by source model, left image is corresponding image in destination model, center is the blended image*

<br/>

![PokePeople 1](images/pokemon/pokepeople/pokeperson1.png)
![PokePeople 1 variation](images/pokemon/pokepeople/pokeperson1_variation.png)
*Different blends of same W+ project input*


#### **FleshéMon**
The much more fun experiment was reversing the blend direction and discovering **FleshéMon**: creatures in the shape of PokéMon but the texture of human flesh. The fine tuned controls over the network blend allowed me to find the ideal spot between familiar and grotesque.

![Fleshemon 1](images/pokemon/fleshemon/fleshemon1.png)
![Fleshemon 2](images/pokemon/fleshemon/fleshemon2.png)
![Fleshemon 3](images/pokemon/fleshemon/fleshemon3.png)
![Fleshemon 4](images/pokemon/fleshemon/fleshemon4.png)


#### **Experimental Overblending**
One more experiment, in the spirit of Terence Broad's [network bending](https://terencebroad.com/research/network-bending), was the technique of *overblending*. A blend value for any given layer should be between `[0-1]`. However, nothing is stopping us from giving the function values outside those bounds! I built another UI for overblending to see what would happen if you move above and below those bounds and while a lot of the outputs collapse, there is a fun, expressive element to it.

![Overblend 1](images/overblend/overblend1.png)
![Overblend 2](images/overblend/overblend2.png)
*Example of some effects achievable when overblending*


### Attempt #3: Pix2Pix
_A possible solution_
I reviewed the [Making Toonify Yourself](https://www.justinpinkney.com/making-toonify/) blogpost and read that to create a more efficient toonify process, the author built a Pix2Pix translation between the original image and blended image.

I began an exploration into doing the same, but training is still in early stages. 

![Pix2Pix progress](images/pix2pix/pix2pix_progress.png)
*Right source image, middle target image, left current pix2pix progress*

Additionaly, while the results might work for real images of people in my life, I would no longer have the fine tuned control over the output image. However, I _still_ would have control over how to blend the models before creating the dataset, so it would fulfill some of my goals for this project if this direction works out.


### Attempt #4: Pixel2Style2Pixel
_The return of StyleGAN3_
When I started this project, I was somewhat naive about the possibilities of image inversion for StyleGAN. It is evidently not a solved problem, and there is active research to improve the available methods and tools. One of these is using a `Pixel2Style2Pixel` for inversion. I only began to experiment with it at the time of writing this so it's too early to tell if it's an effective method. Using the collab notebook they provide, I was excited by the speed of the inference, but the results still weren't as close to original as I would like. (It's possible that when working with images of my friends, I am just more sensitive to divergences from the original image, as opposed to working with celebrities).

Additonaly, the pair for the latent vector in the transfer-learning model, was realtivly coherent, but not as sharp as a Z seed image.

![Pix2style2Pix progress](images/pixel2style2pixel/will_smith_p2s2p.png)
*The projection is for Will Smith's face is ok, but the matching MetFace photo is still somewhat off compared to Z seed examples*


## Conclusion
When I began this project, I set out to make portraits of my friends in the style of transfer-learned butterfly people. In the process, I developed a UI for network blending to have fine tuned control over the output image. While the UI was an effective example of "meaningful human control over deep neural networks", the projections pipeline in StyleGAN3 did not quite work due to:
1. Insufficient quality of the projection in the source model
2. Really bad corresponding image in the destination model

I continued my exploration in StyleGAN2, pix2pix, and pixel2style2pixel, but have yet to found an adequete solution for my original goal.

However, the effectiveness of the network blending UI as both an educational and artistic tool is self-evident. It helped me gain an understanding of how StyleGAN works under the hood and was just plain fun to use (evident by my deep dive into PokéPeople and FleshéMon).

Overall, while I didn't quite achieve my initial goal, I believe the development of this tool will come in handy for future explorations and I hope to continue pushing forward on the topic of meaningful, expressive portraiture with GAN architectures.


## Refrences
#### StyleGAN3 Network Blending UI
- This code lives in a fork of [StyleGAN3-fun](https://github.com/PDillis/stylegan3-fun) by [@PDillis](https://github.com/Sxela) and we take advantage of their projection script and utilities
- Setup code was based on [@dvschultz](https://github.com/dvschultz) [StyleGAN notebooks](https://github.com/dvschultz/stylegan2-ada-pytorch) 
- The idea to use a "blend mask" and many helper functions were taken fully from [@Sxela](https://github.com/PDillis) [stylegan3_blending](https://github.com/Sxela/stylegan3_blending) repo
- Much of this work was inspired by Justin Pinkney's blogpost on [network blending](https://www.justinpinkney.com/stylegan-network-blending/)
  
#### StyleGAN2 Network Blending UI
- This code lives in a [fork of StyleGAN2](https://github.com/dvschultz/stylegan2-ada-pytorch) by [@dvschultz](https://github.com/dvschultz) and we take advantage of the training, projection, blending and utility functions in that repo.
- The idea to use a "blend mask" and many helper functions were taken fully from [@Sxela](https://github.com/PDillis) [stylegan3_blending](https://github.com/Sxela/stylegan3_blending) repo
- Much of this work and some of the models were taken from Justin Pinkney's blogpost on network blending.