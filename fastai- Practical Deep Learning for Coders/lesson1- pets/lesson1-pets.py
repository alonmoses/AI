
# coding: utf-8

# # Lesson 1 - What's your pet

# Welcome to lesson 1! For those of you who are using a Jupyter Notebook for the first time, you can learn about this useful tool in a tutorial we prepared specially for you; click `File`->`Open` now and click `00_notebook_tutorial.ipynb`. 
# 
# In this lesson we will build our first image classifier from scratch, and see if we can achieve world-class results. Let's dive in!
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[1]:


from zipfile import ZipFile
with ZipFile('/notebooks/course-v3/nbs/dl1/data/lesson1-self dataset.zip', 'r') as zf:
    zf.extractall('/notebooks/course-v3/nbs/dl1/data/')


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[3]:


from fastai.vision import *
from fastai.metrics import error_rate


# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[4]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Looking at the data

# We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. According to their paper, the best accuracy they could get in 2012 was 59.21%, using a complex model that was specific to pet detection, with separate "Image", "Head", and "Body" models for the pet photos. Let's see how accurate we can be using deep learning!
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[5]:


help(untar_data)


# In[6]:


path = untar_data('/notebooks/course-v3/nbs/dl1/data/lesson1-self dataset')
path


# In[7]:


path.ls()


# In[8]:


path_anno = path/'annotations'
path_img = path/'images'


# The first thing we do when we approach a problem is to take a look at the data. We _always_ need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, `ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).

# In[14]:


fnames = get_image_files(path_img)
fnames[:5]


# In[15]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[22]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=9
                                  ).normalize(imagenet_stats)


# In[23]:


data.show_batch(rows=3, figsize=(7,6))


# In[24]:


print(data.classes)
len(data.classes),data.c


# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[25]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[26]:


learn.model


# In[27]:


learn.fit_one_cycle(4)


# In[28]:


learn.save('stage-1')


# ## Results

# Let's see what results we have got. 
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. 
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[29]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[31]:


interp.plot_top_losses(2, figsize=(15,11))


# In[32]:


doc(interp.plot_top_losses)


# In[33]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[34]:


interp.most_confused(min_val=2)


# ## Unfreezing, fine-tuning, and learning rates

# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[35]:


learn.unfreeze()


# In[36]:


learn.fit_one_cycle(1)


# In[37]:


learn.load('stage-1');


# In[38]:


learn.lr_find()


# In[39]:


learn.recorder.plot()


# In[40]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# That's a pretty accurate model!

# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[44]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=10//2).normalize(imagenet_stats)


# In[45]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[46]:


learn.lr_find()
learn.recorder.plot()


# In[47]:


learn.fit_one_cycle(8)


# In[48]:


learn.save('stage-1-50')


# It's astonishing that it's possible to recognize pet breeds so accurately! Let's see if full fine-tuning helps:

# In[53]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))


# If it doesn't, you can always go back to your previous model.

# In[54]:


learn.load('stage-1-50');


# In[55]:


interp = ClassificationInterpretation.from_learner(learn)


# In[56]:


interp.most_confused(min_val=2)


# ## Other data formats

# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)


# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()


# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))
data.classes


# In[ ]:


data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


fn_paths = [path/name for name in df['name']]; fn_paths[:2]


# In[ ]:


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes


# In[ ]:


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]


# In[ ]:


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes

