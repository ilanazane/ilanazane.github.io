---
layout: post 
title: "How to Include Jupyter Notebooks in Jekyll" 
date: 2022-11-17
---

I wanted to create a tutorial (mostly for myself because I am going to forget) on how to include jupyter notebooks with jekyll. 

Here is my current file structure: 

![image]({{site.url}}/assets/images/JupyterInJekyll_files/FileStructure.png){: width="300" }

In my docs folder I created a new folder called _notebooks. I did my project in jupyter notebook and when I was done, just moved the file into this directory. I just learned that you can run jupyter notebooks through visual studio code, but I am a fan of the jn interface and refuse to code anywhere else. 

Once the notebook is in the directory, open terminal, cd into the _notebooks folder and use the command 

```python 

jupyter nbconvert --to markdown yourNotebookName.ipynb

```

You will see a markdown file with the same name in your folder. Move this new markdown file to your _posts folder and rename the file to follow the post convention: (YYYY-MM-D-fileName.md)

Add to the beginning of your file the front matter: 

```python
---
layout: post 
title: "Your Title" 
date: YYYY-MM-D
--- 
```

Any images that are included in the notebook will show up under yourNotebookName_files in the notebooks directory. Move this entire folder to your assets section and create a new folder called images (or whatever you want). Your directory should look something like this: 

![image]({{site.url}}/assets/images/JupyterInJekyll_files/FileStructure2.png){: width="300" }

Wherever there an image is being rendered in your post use the following line to import the image: 

![image]({{site.url}}/assets/images/JupyterInJekyll_files/IncludeImage.png){: width="500" }

Run it in your local host and yay you have code blocks from your jn file on github pages! 

