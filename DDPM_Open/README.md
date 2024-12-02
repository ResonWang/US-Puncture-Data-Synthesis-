# CUID-HO-X-T
## 1. Part of CUID-HO can be downloaded from [here](https://pan.baidu.com/s/1Mq_Hln70MUp5LoXal58BPw?pwd=9vho) (9vho).
For the whole dataset, please contact the author.
## 2. Tip generation with DDPM
run image_sample.py
+ Model weight can be downloaded from [here](https://pan.baidu.com/s/1eV66iW5R7s7zVAipPXw84A?pwd=g78f) (g78f).
+ Please change the log dir in `image_sample.py` to your logging path.
```python
    logger.configure('your logging path')
```
+ Please change the generation_dir in `gaussian_diffusion.py` to your save path.
```python
    generation_dir = "your generation path"
```
## 3. US puncture image synthesis
use the function in `data_synthesis.py`
## 4. References
1. https://github.com/openai/improved-diffusion
