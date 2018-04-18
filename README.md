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

##### Setting up the Model

This is where I'm working now. I have a LSTM model that is sort of working, but I simply don't understand it enough to feel comfortable so far.

##### In Conclusion
"Never get one of those cheap tin whistles. It leads to much harder drugs like pipes and flutes." -Irish Proverb
