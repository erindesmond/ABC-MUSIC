# DJ ABC LSTM
##### Traditional Irish Music with a Modern Twist

##### Goal
The aim of this project is to train a LSTM RNN to write music.

##### Data
A corpus of traditional Irish folk songs, dances, reels, and jigs contained in the Nottingham Music Database that were translated into ABC format by the [ABC Music Project](http://abc.sourceforge.net/NMD/). I found clean versions of these songs [here](http://abc.sourceforge.net/NMD/).
For this particular project, I chose ABC format because it was the simplest form of music notation to use in a neural network.


##### About ABC
ABC is a textual representation of music notation. This limited the amount of training data I could use for a model because of its simple format.

![ABC](images/abc.png) ![staff](images/abc_music.png)

You can see that the music is rather simple. This is why my data is limited - there is simply a lot of nuance that cannot be translated into ABC format. For example, imagine how Liszt's Transcendental Etude No. 5 would look in ABC format:

![Lizst](images/transcendental_etude.png)

Therefore, for this particular project, my data is limited to simpler melodic songs, and the Nottingham Music Database fit the bill.

##### The Model

I am using a [Keras LSTM](https://keras.io/layers/recurrent/#lstm) model with only one LSTM cell that has 10 memory units with a dropout rate of 0.3, an RMSprop optimizer with a learning rate of 0.01, and final softmax activation function.

This model randomly chooses a string of 25 characters, and computes the probability distribution of these characters. Then through a function that normalizes the distributions by numbers that represent the standard deviation from the mean of that distribution, it randomly picks a character to append to the original 25 characters, at which point it takes a step forward in time, dropping the first character from the original string. In this way, it makes an 'educated' guess as to what the next character should be. As weights are updated at the end of each batch size, the model will learn probabilities over time. But because of the random choice from the distribution of the original 25 characters, the predicted text will not exactly match the original text.

##### Tensorboard Images     

##### Results
After 1 epoch with a batch size of 100:
![one_epoch](images/one_epoch.png)

After 10 epochs:
![ten_epoch](images/ten_epochs.png)

After 20 epochs:
![twenty_epochs](images/twenty_epochs.png)



##### References
1. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
2. [Training a Recurrent Neural Network to Compose Music](https://maraoz.com/2016/02/02/abc-rnn/)
3. [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
4. [Creating A Text Generator Using Recurrent Neural Network](https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/)
5. [About ABC Notation](http://abcnotation.com/about)

##### In Conclusion
"Never get one of those cheap tin whistles. It leads to much harder drugs like pipes and flutes." -Irish Proverb
