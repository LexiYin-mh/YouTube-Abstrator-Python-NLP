# YouTube Abstractor

This **python** tool is trained based on **Google Pegasus**, an NLP with specialty on generating abstract based on the article input. 

Counting on `youtubeCaptionFetcher`.py, we can generate the abstract for a video by inputting a video ID. 

## Model based on Google Pegasus

This project is inspired by [a google blog about Pegasus](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html).

Research paper can be found at [arXiv](https://arxiv.org/abs/1912.08777). 

Model is trained at **Google Cloud**, guidance can be found in [Google Pegasus Github](https://github.com/google-research/pegasus). 

### Project Directory

Because of the 200M limit of Github, model is available [here in Google Drive Folder](https://drive.google.com/drive/folders/1OcntRJuJJ2V_lED7AKQ9U0arScHMgNoJ?usp=sharing).   

As for where to put the model folder, project directory should be like: 
```plaintext
Project Root Directory
│
├── README.md
├── __pycache__
├── bin
├── ckpt
├── input_article
├── model
```
## Example Result

My result of testing using input_article is as below: 
```plaintext
The hacking group known as NC29 is largely believed to operate as part of Russia's security services .<n>The three countries allege that it is carrying out a persistent and ongoing cyber campaign to steal intellectual property about a possible coronavirus vaccine .
```

## Contact 
- Lexi Yin - [mhyin08@gmail.com](mailto:mhyin08@gmail.com)   
- Linkedin: [www.linkedin.com/in/lexi-m-yin](www.linkedin.com/in/lexi-m-yin)

## Acknowledgments
- Inspiration: Google Pegasus
- Tools: Tensorflow, Google Cloud
- Special thanks to [Lian Duan](https://www.linkedin.com/in/lian-duan-8aa69b239/) for his invaluable contributions and insights.