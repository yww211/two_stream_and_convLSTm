### FlowCNN

Fine-tune the model 
Train a CNN on the flow map generated by FlowNet. The model is trained using Lua language with Torch7 library.

Codes were initially adapted/modified from an imagenet example from [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

---
### Installation

Download our pre-generated UCF-101 flow maps from the [Dropbox link](https://www.dropbox.com/s/lmo0pp9ivyb1162/FlowMap-M-frame.zip?dl=0) (37.16GB). The flow maps have already separated into *train* and *val* folders according to the training list from UCF-101 dataset.

We are also using [pastalog](https://github.com/rewonc/pastalog) and its Torch interface [torch-pastalog](https://github.com/Kaixhin/torch-pastalog). Please make sure you follow the [installation](https://github.com/rewonc/pastalog#installation) steps provided before running the code. Also, please refer to the usage section for both github website for using the **pastalog**.

---
### Usage
simply run *main.lua* with specified directory to our generated flow map dataset, which you just downloaded.
```
th main.lua -data [dataset folder]
```

---
### Note


---


---

